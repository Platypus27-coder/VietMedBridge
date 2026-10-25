"""Sparse, resumable Kaggle handoff for the isolated frozen new-data batch.

CPU selection opens each frozen part once, saves its draft references and only
exports uncached texts. Neither export nor import changes the submission pointer.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vietmedbridge.artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, verify_file, utc_now
from vietmedbridge.content_embeddings import ContentBlocks, checked, seal, key
from vietmedbridge.embeddings import validate_vectors
from vietmedbridge.portable_embeddings import kaggle_assignment, validate_encoder, relative_path
from vietmedbridge.scale_vectors import bound
try:
    from .new_data_embeddings import NewDataEmbeddings, LazyEncoder, embedding_contract, FAMILIES
    from .new_data_cache_lookup import ResumableContentBlocks
except ImportError:
    from new_data_embeddings import NewDataEmbeddings, LazyEncoder, embedding_contract, FAMILIES
    from new_data_cache_lookup import ResumableContentBlocks

SOURCE_FILES = ('new_data_kaggle.py', 'new_data_embeddings.py', 'new_data_cache_lookup.py', 'new_data_drive_api.py')
TASK_ROWS = 512


def lookup(blocks, keys):
    known = {}
    keys = list(dict.fromkeys(keys))
    for start in range(0, len(keys), 900):
        selected = keys[start:start+900]
        rows = blocks.con.execute('SELECT key,path,sha,row FROM vectors WHERE key IN ('+
                                  ','.join('?' for _ in selected)+')', selected)
        known.update({k: {'path': p, 'sha256': s, 'row': r} for k, p, s, r in rows})
    return known


def copied(source, target, checksum):
    if Path(target).is_file():
        verify_file(target, checksum)
    elif publish_file(source, target) != checksum:
        raise ValueError('Artifact changed during copy.')


class NewDataKaggle:
    def __init__(self, project, work, identities, *, accounts=2, gpu_count=2, batch_loader=None):
        self.assignment = kaggle_assignment(accounts=accounts, account_id=0, gpu_count=gpu_count)
        self.view = NewDataEmbeddings(project, work, identities, batch_loader=batch_loader)
        self.work = Path(work)
        self.team = {'accounts': accounts, 'gpu_count': gpu_count}
        self.root = self.view.output / f'kaggle-{accounts}x{gpu_count}'
        self.identity = {'schema': 'new-data-selection-v1', 'embedding_binding': self.view.binding, 'team': self.team}
        self.binding = digest_json(self.identity)
        contract = self.root / 'contract.json'
        if contract.exists() and checked(read_json(contract)) != seal(self.identity):
            raise ValueError('Selection producer/team changed.')
        atomic_json(contract, seal(self.identity))
        self.sources = {s['ordinal']: s for s in self.view.sources}

    def journals(self, family):
        result = {}
        for build in dict.fromkeys(s['build']['build_run'] for s in self.view.sources):
            path = self.root / 'selection' / family / build / 'progress.json'
            if not path.exists():
                continue
            saved = checked(read_json(path))
            if saved.get('binding') != self.binding or saved.get('family') != family:
                raise ValueError('Selection checkpoint belongs to another batch/model/team.')
            for ordinal, record in saved['parts'].items():
                source = self.sources[int(ordinal)]
                if (source['build']['build_run'] != build or record['input_sha256'] != source['part']['sha256']
                        or record['ordinal'] != int(ordinal) or record['rows'] != source['part']['rows']):
                    raise ValueError('Selection checkpoint input binding changed.')
                result[int(ordinal)] = record
        return result

    def _save(self, family, source, selected):
        build = source['build']['build_run']
        record = {'binding': self.binding, 'family': family, 'parts': {
            str(k): v for k, v in selected.items() if self.sources[k]['build']['build_run'] == build}}
        atomic_json(self.root / 'selection' / family / build / 'progress.json', seal(record))

    def _map(self, family, source, refs):
        root, identity = self.view._view(family, source)
        config = root / 'config.json'
        if config.exists() and read_json(config) != identity:
            raise ValueError('Embedding map producer changed.')
        atomic_json(config, identity)
        part = source['part']
        target = root / f"map-{part['start']:012d}.parquet"
        local = self.work / 'map.parquet'
        pq.write_table(pa.Table.from_pylist(refs), local, compression='zstd')
        record = seal({'signature': digest_json(identity), 'start': part['start'], 'rows': part['rows'],
                       'input_sha256': part['sha256'], 'path': target.name, 'sha256': publish_file(local, target)})
        atomic_json(root / f"map-{part['start']:012d}.done.json", record)
        return record

    def select(self, *, receipt_api=None, blocks_factory=ResumableContentBlocks):
        """Save draft refs and missing-text tasks before committing each input part."""
        for family in FAMILIES:
            selected = self.journals(family)
            complete = self.view._journals(family)
            pending = [s for s in self.view.sources if s['ordinal'] not in selected and self.view._key(s) not in complete]
            print(f'SELECTION {family}: {len(selected)} selection parts saved | {len(complete)}/{len(self.sources)} maps complete; {len(pending)} new', flush=True)
            if not pending:
                continue
            expected = self.view.identities[family]
            blocks = blocks_factory(self.view.data, SimpleNamespace(identity=expected['encoder'], dimension=expected['dimension']),
                                    self.work / family, receipt_api=receipt_api, progress=print)
            try:
                for source in pending:
                    part, ordinal = source['part'], source['ordinal']
                    path = bound(self.view.project, source['build']['build_dir']) / 'index_inputs' / part['path']
                    local_input = self.work / 'current-input.parquet'
                    if publish_file(path, local_input) != part['sha256']:
                        raise ValueError('Frozen input checksum mismatch during staging.')
                    rows = pq.read_table(local_input, columns=['id', 'text']).to_pylist()
                    if len(rows) != part['rows'] or any(r['id'] != key(r['text']) for r in rows):
                        raise ValueError('Frozen input row/hash mismatch.')
                    known = lookup(blocks, [r['id'] for r in rows])
                    draft = [{'key': r['id'], **known.get(r['id'], {'path': None, 'sha256': None, 'row': None})} for r in rows]
                    directory = self.root / 'selection' / family / source['build']['build_run']
                    local = self.work / 'selection.parquet'
                    schema = pa.schema([('key', pa.string()), ('path', pa.string()), ('sha256', pa.string()), ('row', pa.int64())])
                    pq.write_table(pa.Table.from_pylist(draft, schema=schema), local, compression='zstd')
                    draft_path = directory / f'draft-{ordinal:06d}.parquet'
                    draft_entry = {'path': draft_path.relative_to(self.root).as_posix(), 'sha256': publish_file(local, draft_path)}
                    tasks, missing_rows = [], []
                    missing = [r for r in rows if r['id'] not in known]
                    target = directory / f'missing-{ordinal:06d}.parquet'
                    for worker in range(self.assignment['workers']):
                        group = [r for r in missing if int(r['id'][:8], 16) % self.assignment['workers'] == worker]
                        for offset in range(0, len(group), TASK_ROWS):
                            values = group[offset:offset+TASK_ROWS]
                            task_id = digest_json([self.binding, family, ordinal, worker, offset])
                            tasks.append({'selection_id': task_id, 'family': family, 'ordinal': ordinal, 'worker': worker,
                                'path': target.relative_to(self.root).as_posix(), 'offset': len(missing_rows),
                                'rows': len(values), 'keys_digest': digest_json(sorted(r['id'] for r in values))})
                            missing_rows.extend(values)
                    if missing_rows:
                        pq.write_table(pa.Table.from_pylist(missing_rows), local, compression='zstd')
                        checksum = publish_file(local, target)
                        for task in tasks:
                            task['sha256'] = checksum
                    selected[ordinal] = {'ordinal': ordinal, 'input_sha256': part['sha256'], 'rows': part['rows'],
                                         'cached_rows': len(rows)-len(missing), 'draft': draft_entry, 'tasks': tasks}
                    if not missing and self.view._key(source) not in complete:
                        complete[self.view._key(source)] = self._map(family, source, [known[r['id']] for r in rows])
                        self.view._save(family, complete)
                    self._save(family, source, selected)
                    print(f'SELECTION_CHECKPOINT {family} {len(selected)}/{len(self.sources)} | cached {len(rows)-len(missing)}, missing {len(missing)}', flush=True)
                self.view._seal_views(family, complete)
            finally:
                blocks.close()
        return {f: self.journals(f) for f in FAMILIES}

    def imported(self):
        done = set()
        for path in (self.root / 'imports').glob('*.json'):
            saved = checked(read_json(path))
            if saved.get('binding') != self.binding:
                raise ValueError('Imported output binding changed.')
            done.update(saved['selection_ids'])
        return done

    def complete_maps(self, family, ordinals, blocks):
        selected, completed = self.journals(family), self.view._journals(family)
        for ordinal in sorted(ordinals):
            source = self.sources[ordinal]
            if self.view._key(source) in completed:
                continue
            entry = selected[ordinal]['draft']
            path = bound(self.root, entry['path'])
            verify_file(path, entry['sha256'])
            draft = pq.read_table(path).to_pylist()
            if len(draft) != source['part']['rows']:
                raise ValueError('Draft map row coverage changed.')
            missing = [r['key'] for r in draft if r['path'] is None]
            known = lookup(blocks, missing)
            if all(k in known for k in missing):
                refs = [{k: r[k] for k in ('path', 'sha256', 'row')} if r['path'] is not None else known[r['key']] for r in draft]
                completed[self.view._key(source)] = self._map(family, source, refs)
                self.view._save(family, completed)
        self.view._seal_views(family, completed)

    def export(self, checkout, *, config, runtime, code_commit, receipt_api=None,
               blocks_factory=ResumableContentBlocks, output_bytes=16_000_000_000):
        active = self.root/'active_round.json'
        if active.exists():
            saved = checked(read_json(active))
            if saved['binding'] != self.binding:
                raise ValueError('Active Kaggle round binding changed.')
            imported = {r['account_id'] for p in (self.root/'imports').glob('*.json')
                        if (r := checked(read_json(p))).get('job') == saved['job'] and 'account_id' in r}
            if imported != set(range(self.team['accounts'])):
                return None, {'state': 'WAITING_FOR_KAGGLE_OUTPUTS', 'dataset_handle': saved['dataset_handle'],
                    'imported_accounts': sorted(imported), 'required_accounts': list(range(self.team['accounts'])),
                    'instruction': 'Use the existing dataset; import saved Outputs of the current round before starting another round.'}
        selected = self.select(receipt_api=receipt_api, blocks_factory=blocks_factory)
        if type(output_bytes) is not int or not 0 < output_bytes <= 16_000_000_000:
            raise ValueError('Output budget must be between 1 and 16 GB per account.')
        if not re.fullmatch('[0-9a-f]{40}', code_commit):
            raise ValueError('Pin the full native producer commit.')
        done = self.imported()
        used, tasks, remaining = [0]*self.team['accounts'], [], {f: 0 for f in FAMILIES}
        assets = self.work/'export-assets'
        assets.mkdir(parents=True, exist_ok=True)
        for family in FAMILIES:
            pending = [task for entry in selected[family].values() for task in entry['tasks'] if task['selection_id'] not in done]
            remaining[family] = len(pending)
            if not pending:
                continue
            expected = self.view.identities[family]
            blocks = blocks_factory(self.view.data, SimpleNamespace(identity=expected['encoder'], dimension=expected['dimension']),
                                    self.work/family, receipt_api=receipt_api, progress=print)
            cache_only, affected = [], set()
            last_path, last_rows = None, None
            try:
                for task in pending:
                    account = task['worker'] // self.team['gpu_count']
                    estimate = task['rows'] * (expected['dimension']*4+1024)
                    if estimate > output_bytes:
                        raise ValueError('One task exceeds the round output budget.')
                    if used[account]+estimate > output_bytes:
                        continue
                    path = bound(self.root, task['path'])
                    if path != last_path:
                        verify_file(path, task['sha256'])
                        last_rows = pq.read_table(path, columns=['id', 'text']).to_pylist()
                        last_path = path
                    rows = last_rows[task['offset']:task['offset']+task['rows']]
                    if len(rows) != task['rows'] or digest_json(sorted(r['id'] for r in rows)) != task['keys_digest']:
                        raise ValueError('Saved missing-text task changed.')
                    known = lookup(blocks, [r['id'] for r in rows])
                    rows = [r for r in rows if r['id'] not in known]
                    if not rows:
                        cache_only.append(task['selection_id'])
                        affected.add(task['ordinal'])
                        remaining[family] -= 1
                        continue
                    estimate = len(rows) * (expected['dimension']*4+1024)
                    used[account] += estimate
                    asset = 'task-'+task['selection_id']+'.parquet'
                    pq.write_table(pa.Table.from_pylist(rows), assets/asset, compression='zstd')
                    tasks.append({**task, 'id': task['selection_id'], 'asset': asset, 'rows': len(rows),
                        'sha256': sha256_file(assets/asset), 'keys_digest': digest_json(sorted(r['id'] for r in rows)),
                        'estimated_output_bytes': estimate})
                if cache_only:
                    self.complete_maps(family, affected, blocks)
                    atomic_json(self.root/'imports'/('cache-'+digest_json(cache_only)+'.json'), seal({
                        'binding': self.binding, 'selection_ids': cache_only, 'state': 'REUSED_NEWLY_AVAILABLE_CACHE'}))
            finally:
                blocks.close()
        if not tasks:
            return None, {'state': 'NO_PENDING_KAGGLE_TASKS', 'coverage': self.coverage()}
        model_code = {n: sha256_file(Path(checkout)/'src/vietmedbridge'/n) for n in ('retrieval_models.py', 'qwen_models.py')}
        scripts = Path(__file__).resolve().parent
        workers = {n: sha256_file(scripts/n) for n in SOURCE_FILES}
        frozen_sources = {source['build']['build_run']: source['build'] for source in self.view.sources}
        job = seal({'schema': 'new-data-kaggle-v1', 'selection_binding': self.binding, 'batch': self.view.batch['manifest_sha256'],
            'source_batch_dir': self.view.root.relative_to(self.view.project).as_posix(),
            'coverage_scope': {k: self.view.batch[k] for k in ('official_ids','source_builds','source_documents_sum','input_rows_before_cross_source_dedup','scope_manifest_sha256') if k in self.view.batch},
            'frozen_sources': [{'build_run': b['build_run'], 'candidate_manifest_sha256': b['candidate_manifest_sha256'],
                'index_inputs_manifest_sha256': b['index_inputs_manifest_sha256'], 'unique_dense_inputs': b['unique_dense_inputs'],
                'inherited_complete_models': sorted(b.get('completed_views', {}))} for b in frozen_sources.values()],
            'scheduled_tasks': len(tasks), 'unscheduled_tasks': sum(remaining.values())-len(tasks),
            'all_remaining_tasks_scheduled': sum(remaining.values()) == len(tasks),
            'team': self.team, 'code_commit': code_commit, 'repo_url': 'https://github.com/Platypus27-coder/VietMedBridge.git',
            'models': self.view.identities, 'config': config, 'runtime': runtime, 'model_code': model_code,
            'worker_sources': workers, 'tasks': tasks, 'remaining_tasks_before_round': remaining,
            'scheduled_bytes_per_account': used, 'scope': self.view.batch.get('scope','Frozen corpus child embedding tasks only; no submission inference.')})
        target = self.work / 'exports' / job['manifest_sha256'][:16]
        target.mkdir(parents=True, exist_ok=True)
        for i, task in enumerate(tasks, 1):
            copied(assets/task['asset'], target/task['asset'], task['sha256'])
            if i % 100 == 0:
                print(f'KAGGLE_EXPORT {i}/{len(tasks)} missing-text tasks', flush=True)
        for name, checksum in workers.items():
            copied(scripts/name, target/name, checksum)
        atomic_json(target/'job.json', job)
        atomic_json(self.root/'exports'/('job-'+job['manifest_sha256']+'.json'), job)
        if 'official_ids' in self.view.batch:
            self.coverage()
        return target, job

    def coverage(self):
        coverage = {f: {'completed_parts': len(self.view._journals(f)), 'total_parts': len(self.sources)} for f in FAMILIES}
        if 'official_ids' in self.view.batch:
            complete = all(v['completed_parts'] == v['total_parts'] for v in coverage.values())
            atomic_json(self.root/'full_embedding_status.json', seal({
                'state': 'FULL_DATA_EMBEDDINGS_COMPLETE' if complete else 'FULL_DATA_EMBEDDINGS_IN_PROGRESS',
                'official_ids_in_scope': self.view.batch['official_ids'], 'batch': self.view.batch['manifest_sha256'],
                'coverage': coverage, 'payload_verification': 'SHA256_ON_USE', 'updated_at': utc_now()}))
        return coverage

    def assigned_output_tasks(self, job, result):
        assignment = kaggle_assignment(accounts=job['team']['accounts'], account_id=result['account_id'],
                                       gpu_count=job['team']['gpu_count'])
        return {t['id'] for t in job['tasks'] if t['worker'] in assignment['worker_ids']}

    def install(self, output, *, receipt_api=None, blocks_factory=ResumableContentBlocks, cached_keys=None):
        output = Path(output)
        job = checked(read_json(output/'job.json'))
        saved_job = checked(read_json(self.root/'exports'/('job-'+job['manifest_sha256']+'.json')))
        result = checked(read_json(output/'new-data-result.json'))
        if job != saved_job or job['selection_binding'] != self.binding or job['models'] != self.view.identities or result['job'] != job['manifest_sha256']:
            raise ValueError('Kaggle output differs from the frozen export/model contract.')
        prior_path = self.root/'imports'/(result['manifest_sha256']+'.json')
        if prior_path.exists():
            prior = checked(read_json(prior_path))
            if (prior.get('binding') != self.binding or prior.get('job') != job['manifest_sha256']
                    or prior.get('result') != result['manifest_sha256']):
                raise ValueError('Committed import receipt binding changed.')
            return seal({**{k:v for k,v in prior.items() if k != 'manifest_sha256'},
                         'coverage': self.coverage(), 'payloads_reopened': False})
        allowed = self.assigned_output_tasks(job, result)
        tasks = {t['id']: t for t in job['tasks']}
        expected_keys, affected, completed_ids = {}, {f: set() for f in FAMILIES}, []
        for receipt in result['tasks']:
            task = tasks[receipt['task']]
            if receipt['task'] not in allowed or receipt['task'] in completed_ids:
                raise ValueError('Output contains duplicate/unassigned tasks.')
            keys = receipt['keys']
            if len(keys) != task['rows'] or len(set(keys)) != len(keys) or digest_json(sorted(keys)) != task['keys_digest']:
                raise ValueError('Output task key coverage changed.')
            expected_keys.setdefault(task['family'], set()).update(keys)
            completed_ids.append(receipt['task'])
            affected[task['family']].add(task['ordinal'])
        # Validate all output files before publishing any coverage receipt.
        blocks_by_family = {f: [] for f in FAMILIES}
        available = {f: set() for f in FAMILIES}
        # A full-input execution plan may deliberately omit vectors already on
        # Drive. Never trust that claim from the GPU result: verify its keys in
        # the exact paid producer's CPU lookup before allowing task coverage.
        for family, values in (cached_keys or {}).items():
            keys = set(values) & expected_keys.get(family, set())
            if family not in FAMILIES:
                raise ValueError('Unexpected cached vector family.')
            if not keys:
                continue
            expected = self.view.identities[family]
            cache = blocks_factory(self.view.data, SimpleNamespace(identity=expected['encoder'], dimension=expected['dimension']),
                                   self.work/family, receipt_api=receipt_api, progress=print)
            try:
                if set(lookup(cache, list(keys))) != keys:
                    raise ValueError('Execution plan cached vectors are not available in the verified Drive producer.')
                available[family].update(keys)
            finally:
                cache.close()
        for block in result['blocks']:
            checked(block)
            family = next((f for f in FAMILIES if block['identity'] == {
                **self.view.identities[f], 'format': 'content-float32-l2-v1'}), None)
            if family is None or not set(block['keys']) <= expected_keys.get(family, set()) or len(block['keys']) != block['rows']:
                raise ValueError('Unexpected vector block producer/keys.')
            path = bound(output, block['path'])
            verify_file(path, block['sha256'])
            validate_vectors(np.load(path, mmap_mode='r', allow_pickle=False), block['rows'], block['identity']['dimension'])
            if path.parent.relative_to(output).as_posix() != 'model_cache/'+digest_json(block['identity'])[:24]:
                raise ValueError('Output vector path/producer namespace changed.')
            blocks_by_family[family].append(block)
            available[family].update(block['keys'])
        if any(not keys <= available[f] for f, keys in expected_keys.items()):
            raise ValueError('Output tasks are missing saved vector payloads.')
        for family in FAMILIES:
            if not affected[family]:
                continue
            expected = self.view.identities[family]
            blocks = blocks_factory(self.view.data, SimpleNamespace(identity=expected['encoder'], dimension=expected['dimension']),
                                    self.work/family, receipt_api=receipt_api, progress=print)
            try:
                for block in blocks_by_family[family]:
                    path = bound(self.view.data, block['path'])
                    if path.exists():
                        verify_file(path, block['sha256'])
                    else:
                        copied(bound(output, block['path']), path, block['sha256'])
                    blocks.save_block(block['keys'], path, block['sha256'])
                self.complete_maps(family, affected[family], blocks)
                if getattr(blocks, '_snapshot_dirty', False):
                    blocks._snapshot()
            finally:
                blocks.close()
        report = seal({'state': 'NEW_DATA_KAGGLE_IMPORTED', 'binding': self.binding, 'job': job['manifest_sha256'],
            'result': result['manifest_sha256'], 'account_id': result['account_id'], 'selection_ids': completed_ids,
            'coverage': self.coverage(), 'updated_at': utc_now()})
        atomic_json(self.root/'imports'/(result['manifest_sha256']+'.json'), report)
        return report


def validate_job(directory, checkout=None):
    directory = Path(directory)
    job = checked(read_json(directory/'job.json'))
    if (job.get('schema') != 'new-data-kaggle-v1' or job.get('repo_url') != 'https://github.com/Platypus27-coder/VietMedBridge.git'
            or not re.fullmatch('[0-9a-f]{40}', job.get('code_commit', ''))):
        raise ValueError('Invalid new-data Kaggle job.')
    kaggle_assignment(accounts=job['team']['accounts'], account_id=0, gpu_count=job['team']['gpu_count'])
    if set(job['worker_sources']) != set(SOURCE_FILES) or set(job['models']) != set(FAMILIES):
        raise ValueError('Missing model/worker producer contract.')
    for name, checksum in job['worker_sources'].items():
        verify_file(directory/name, checksum)
    seen = set()
    for task in job['tasks']:
        if (task['id'] in seen or task['id'] != task['selection_id'] or task['family'] not in FAMILIES
                or type(task['worker']) is not int or not 0 <= task['worker'] < job['team']['accounts']*job['team']['gpu_count']
                or type(task['rows']) is not int or not 0 < task['rows'] <= TASK_ROWS
                or relative_path(task['asset']) != 'task-'+task['id']+'.parquet'):
            raise ValueError('Invalid sparse task assignment.')
        seen.add(task['id'])
    if checkout is not None:
        for name, checksum in job['model_code'].items():
            if name not in ('retrieval_models.py', 'qwen_models.py'):
                raise ValueError('Unexpected model code file.')
            verify_file(Path(checkout)/'src/vietmedbridge'/name, checksum)
        for family in FAMILIES:
            validate_encoder(job, family, job['models'][family]['encoder'], job['models'][family]['dimension'])
    return job


def download_outputs(api, handle, work, *, project=None, import_roots=(), allow_failed=False,
                     manifest_path=None, expected_result=None):
    """Page through the new-data manifest first, before downloading vectors."""
    from vietmedbridge.kaggle_outputs import _download_pages, output_handles
    output_handles(handle)
    transport = None
    if manifest_path is not None:
        try:
            from .kaggle_output_transfer import SavedNotebookOutputs, retry_kaggle
        except ImportError:
            from kaggle_output_transfer import SavedNotebookOutputs, retry_kaggle
        status = retry_kaggle(lambda: api.kernels_status(handle))
    else:
        status = api.kernels_status(handle)
    status = status.get('status') if isinstance(status, dict) else status.status
    status = getattr(status, 'name', str(status)).upper()
    failed = status in {'ERROR', 'FAILED', 'CANCELLED', 'CANCELED'}
    if status != 'COMPLETE' and not (allow_failed and failed):
        raise RuntimeError(f'{handle}: status={status}; wait for the saved version to finish.')
    if failed:
        print(f'KAGGLE_SESSION_{status}: inspecting saved checkpoint Outputs only; '
              'GPU work is not restarted and completion is not assumed.', flush=True)
    base = Path(work)/'downloads'/digest_json(handle)[:16]
    probe = base/('probe-'+str(time.time_ns()))
    probe.mkdir(parents=True, exist_ok=True)
    if manifest_path is not None:
        transport = SavedNotebookOutputs(api, handle, work, project=project)
        paths = [transport.manifest(manifest_path)]
    else:
        paths = _download_pages(api, handle, probe, file_pattern=r'(^|/)new-data-result[.]json$', force=True)
    manifests = [p for p in paths if p.name == 'new-data-result.json']
    if len(manifests) != 1:
        raise RuntimeError(f'{handle}: expected one new-data-result.json, found {len(manifests)}; choose the saved new-data worker version.')
    manifest = checked(read_json(manifests[0]))
    if expected_result is not None:
        expected = dict(expected_result)
        actual = {**manifest, 'completed_tasks': len(manifest.get('tasks', []))}
        if (not expected or set(expected)-{'manifest_sha256', 'job', 'execution_plan', 'account_id', 'completed_tasks'}
                or any(actual.get(name) != value for name, value in expected.items())):
            raise ValueError('Saved Outputs differ from the expected completed GPU run; no vector download started.')
    relative = Path(manifest_path) if transport is not None else manifests[0].relative_to(probe)
    if project is not None:
        from_parent = Path(project)/'data_staging'/'btc-new-71be0e12-20261019-v1'/'freeze_batch_v1'/'embeddings_v1'
        roots=[from_parent,*[bound(project,p) for p in import_roots]]
        receipts=[p for root in roots for p in root.glob('kaggle-*/imports/'+manifest['manifest_sha256']+'.json')]
        for receipt in receipts:
            prior = checked(read_json(receipt))
            if prior.get('result') != manifest['manifest_sha256'] or prior.get('job') != manifest['job']:
                raise ValueError('Previously imported output receipt changed.')
            job = checked(read_json(receipt.parent.parent/'exports'/('job-'+manifest['job']+'.json')))
            atomic_json(manifests[0].parent/'job.json', job)
            print('OUTPUT_ALREADY_IMPORTED: sealed receipt reused; no vector download', flush=True)
            return manifests[0].parent
    destination = base/manifest['manifest_sha256'][:16]
    destination.mkdir(parents=True, exist_ok=True)
    if transport is not None:
        target = transport.all(manifest_path, manifest)
        paths = [target]
    else:
        paths = _download_pages(api, handle, destination, force=False, quiet=False)
        target = destination/relative
    if target not in paths or checked(read_json(target)) != manifest:
        raise ValueError('Saved Kaggle Outputs changed after preflight; select a fixed version.')
    return target.parent


def upload_job(flow, target, job, owner, *, activation_filename='active_round.json', dataset_prefix=None):
    """A unique private dataset, with an upload pointer retained on Drive."""
    import kagglehub
    if not re.fullmatch('[A-Za-z0-9_-]+', owner):
        raise ValueError('Invalid Kaggle owner.')
    pointer = flow.root/'uploads'/(owner+'-'+job['manifest_sha256']+'.json')
    if pointer.exists():
        saved = read_json(pointer)
        if saved['job'] != job['manifest_sha256']:
            raise ValueError('Upload pointer belongs to another job.')
        handle = saved['dataset_handle']
    else:
        prefix = dataset_prefix or ('/vmb-full-' if job.get('coverage_scope', {}).get('official_ids') else '/vmb-new-')
        if not re.fullmatch(r'/[a-z0-9-]+-', prefix):
            raise ValueError('Invalid dataset prefix.')
        handle = owner+prefix+job['manifest_sha256'][:12]+'-'+str(time.time_ns())
        atomic_json(pointer, {'job': job['manifest_sha256'], 'dataset_handle': handle, 'state': 'UPLOAD_PLANNED'})
        saved = read_json(pointer)
    if saved['state'] == 'UPLOAD_PLANNED':
        # Deterministic pointer survives a failed upload; never choose a second handle.
        kagglehub.dataset_upload(handle, str(target))
        atomic_json(pointer, {'job': job['manifest_sha256'], 'dataset_handle': handle, 'state': 'VERIFY_PENDING'})
    remote = None
    for attempt in range(6):
        try:
            remote = Path(kagglehub.dataset_download(handle, path='job.json', force_download=True))
            break
        except Exception:
            if attempt == 5:
                raise RuntimeError('Dataset is processing or inaccessible; rerun to verify the same upload.') from None
            time.sleep(min(30, 5*(attempt+1)))
    if read_json(remote) != job:
        raise ValueError('Uploaded Kaggle job header differs; do not start GPU workers.')
    atomic_json(pointer, {'job': job['manifest_sha256'], 'dataset_handle': handle, 'state': 'VERIFIED'})
    if Path(activation_filename).name != activation_filename or not activation_filename.endswith('.json'):
        raise ValueError('Invalid dataset activation filename.')
    atomic_json(flow.root/activation_filename, seal({'binding': flow.binding, 'job': job['manifest_sha256'], 'dataset_handle': handle}))
    print('NEW_DATASET_READY:', 'https://www.kaggle.com/datasets/'+handle, flush=True)
    print('Team:', job['team'], '| scheduled missing-text tasks:', len(job['tasks']), flush=True)
    return handle


def run_worker(directory, output, work, *, worker_id, factories, deadline, batch_sizes=None):
    """Each GPU owns every occurrence of a given text hash, avoiding duplicate encoding."""
    directory, output, work = Path(directory), Path(output), Path(work)
    job = validate_job(directory)
    if type(worker_id) is not int or not 0 <= worker_id < job['team']['accounts']*job['team']['gpu_count']:
        raise ValueError('Worker ID is outside this job.')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'job.json').exists() and read_json(output/'job.json') != job:
        raise ValueError('Working output contains another job. Save its Outputs; start a fresh Kaggle version.')
    atomic_json(output/'job.json', job)
    journal = output/f'worker-{worker_id}.json'
    completed = []
    if journal.exists():
        saved = checked(read_json(journal))
        if saved['job'] != job['manifest_sha256'] or saved['worker_id'] != worker_id:
            raise ValueError('Worker journal belongs to another job/worker.')
        completed = saved['tasks']
    done = {r['task'] for r in completed}
    if len(done) != len(completed) or not done <= {t['id'] for t in job['tasks'] if t['worker'] == worker_id}:
        raise ValueError('Duplicate local task receipt.')
    stop_reason = 'ASSIGNMENT_COMPLETE'
    for family in FAMILIES:
        assigned = [t for t in job['tasks'] if t['worker'] == worker_id and t['family'] == family and t['id'] not in done]
        print(f'GPU worker {worker_id}: {family}, {len(assigned)} missing-text tasks', flush=True)
        if not assigned or time.monotonic() >= deadline:
            if assigned:
                stop_reason = 'SESSION_TIME_LIMIT'
            continue
        lazy = LazyEncoder(job['models'][family], factories[family])
        blocks = ContentBlocks(output, lazy, work/family)
        try:
            for task in assigned:
                if time.monotonic() >= deadline:
                    stop_reason = 'SESSION_TIME_LIMIT'
                    break
                path = directory/task['asset']
                verify_file(path, task['sha256'])
                rows = pq.read_table(path, columns=['id', 'text']).to_pylist()
                keys = [r['id'] for r in rows]
                workers = job['team']['accounts']*job['team']['gpu_count']
                if (len(rows) != task['rows'] or digest_json(sorted(keys)) != task['keys_digest']
                        or any(k != key(r['text']) or int(k[:8], 16) % workers != worker_id for k, r in zip(keys, rows))):
                    raise ValueError('Task payload/hash/ownership changed.')
                blocks.refs([r['text'] for r in rows], batch_size=(batch_sizes or {'bge': 32, 'qwen': 16})[family],
                            encode_missing=True, encode_block_rows=TASK_ROWS)
                completed.append({'task': task['id'], 'keys': keys})
                atomic_json(journal, seal({'job': job['manifest_sha256'], 'worker_id': worker_id, 'tasks': completed}))
                print(f'KAGGLE_CHECKPOINT worker {worker_id} | saved tasks {len(completed)} | {family} | /kaggle/working', flush=True)
        finally:
            blocks.close()
            lazy.close()
    return seal({'job': job['manifest_sha256'], 'worker_id': worker_id, 'tasks': completed, 'stop_reason': stop_reason})


def finalize(directory, output, account_id):
    job = validate_job(directory)
    output = Path(output)
    assignment = kaggle_assignment(accounts=job['team']['accounts'], account_id=account_id, gpu_count=job['team']['gpu_count'])
    root = output/'result'
    root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for worker in assignment['worker_ids']:
        path = root/f'worker-{worker}.json'
        if path.exists():
            saved = checked(read_json(path))
            if saved['job'] != job['manifest_sha256'] or saved['worker_id'] != worker:
                raise ValueError('Worker journal belongs to another job/assignment.')
            tasks.extend(saved['tasks'])
    task_family = {t['id']: t['family'] for t in job['tasks']}
    committed = {f: {k for task in tasks if task_family[task['task']] == f for k in task['keys']} for f in FAMILIES}
    blocks = {}
    for path in root.glob('model_cache/*/*.done.json'):
        block = checked(read_json(path))
        family = next((f for f in FAMILIES if block['identity'] == {**job['models'][f], 'format': 'content-float32-l2-v1'}), None)
        if family is not None and set(block['keys']) <= committed[family]:
            blocks[block['path']] = block
    atomic_json(root/'job.json', job)
    result = seal({'schema': 'new-data-kaggle-result-v1', 'job': job['manifest_sha256'],
        'account_id': account_id, 'tasks': tasks, 'blocks': list(blocks.values()), 'updated_at': utc_now(),
        'state': 'ROUND_COMPLETE' if len(tasks) == sum(t['worker'] in assignment['worker_ids'] for t in job['tasks']) else 'CHECKPOINTED_PARTIAL'})
    atomic_json(root/'new-data-result.json', result)
    return result


def launch(directory, checkout, output, work, *, account_id=0, hours=10.0):
    """Fresh CUDA subprocesses, one model per T4; leave time to save Kaggle Outputs."""
    import torch
    from vietmedbridge.portable_embeddings import embedding_runtime
    job = validate_job(directory, checkout)
    if embedding_runtime() != job['runtime'] or not torch.cuda.is_available() or torch.cuda.device_count() != job['team']['gpu_count']:
        raise RuntimeError('CUDA/runtime mismatch: select T4 x2 and use the locked CUDA packages.')
    assignment = kaggle_assignment(accounts=job['team']['accounts'], account_id=account_id, gpu_count=job['team']['gpu_count'])
    if not 0 < hours <= 10.5:
        raise ValueError('Use a save deadline up to 10.5 hours, before the provider hard stop.')
    deadline = time.monotonic()+hours*3600-180
    processes = []
    try:
        for local_gpu, worker in enumerate(assignment['worker_ids']):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(local_gpu)
            env['PYTHONPATH'] = str(Path(checkout)/'src')+os.pathsep+str(Path(directory))
            command = [sys.executable, '-u', str(Path(directory)/'new_data_kaggle.py'), 'worker', '--job', str(directory),
                       '--output', str(Path(output)/'result'), '--work', str(Path(work)/f'worker-{worker}'),
                       '--worker-id', str(worker), '--deadline', str(deadline)]
            processes.append(subprocess.Popen(command, env=env))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A GPU worker failed; outputs of completed tasks remain available.')
            time.sleep(2)
        result = finalize(directory, output, account_id)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('A GPU worker failed; inspect the preceding traceback.')
        return result
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        # Even an exception publishes a partial result for completed tasks.
        finalize(directory, output, account_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['worker', 'launch'])
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--checkout')
    parser.add_argument('--worker-id', type=int)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--account-id', type=int, default=0)
    parser.add_argument('--hours', type=float, default=10.0)
    args = parser.parse_args()
    if args.mode == 'launch':
        result = launch(args.job, args.checkout, args.output, args.work, account_id=args.account_id, hours=args.hours)
    else:
        import torch
        from vietmedbridge.portable_embeddings import embedding_runtime
        from vietmedbridge.retrieval_models import TorchDenseEncoder
        from vietmedbridge.qwen_models import TorchQwenEncoder
        job = validate_job(args.job)
        if not torch.cuda.is_available() or embedding_runtime() != job['runtime']:
            raise RuntimeError('GPU worker runtime differs from the paid vector producer.')
        result = run_worker(args.job, args.output, args.work, worker_id=args.worker_id, deadline=args.deadline,
            factories={'bge': lambda: TorchDenseEncoder(job['config']['dense']), 'qwen': lambda: TorchQwenEncoder(job['config']['second_dense'])})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
