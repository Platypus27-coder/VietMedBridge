"""Upload all missing text once; issue small, bounded GPU execution plans.

The immutable input library and the output-capacity plan have different hashes.
Only the plan changes after an import. Native encoder and vector-cache formats
are reused unchanged, and the old bounded export/import history is retained.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import os
import re
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from vietmedbridge.artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, verify_file, utc_now
from vietmedbridge.content_embeddings import ContentBlocks, checked, key, seal
from vietmedbridge.portable_embeddings import kaggle_assignment, relative_path, validate_encoder
from vietmedbridge.scale_vectors import bound
try:
    from . import new_data_kaggle as base
    from .new_data_embeddings import LazyEncoder, FAMILIES
    from .new_data_cache_lookup import ResumableContentBlocks
except ImportError:
    import new_data_kaggle as base
    from new_data_embeddings import LazyEncoder, FAMILIES
    from new_data_cache_lookup import ResumableContentBlocks

SOURCE_FILES = (*base.SOURCE_FILES, 'full_queue_embeddings.py')
LIBRARY_SCHEMA = 'full-data-text-library-v1'
PLAN_SCHEMA = 'full-data-execution-plan-v1'
ELASTIC_PLAN_SCHEMA = 'full-data-execution-plan-v2'
MAX_OUTPUT_BYTES = 16_000_000_000
ELASTIC_OUTPUT_BYTES = 18_000_000_000
LEGACY_EXECUTOR_FILES = ('full_queue_embeddings.py', 'new_data_kaggle.py')
EXECUTOR_FILES = (*LEGACY_EXECUTOR_FILES, 'qwen_token_batching.py')
QWEN_EXECUTION = 'qwen-token-batches-v1'


def execution_worker(plan, task):
    return plan['task_workers'][task['id']] if plan['schema'] == ELASTIC_PLAN_SCHEMA else task['worker']


def execution_team(plan, library):
    return plan['team'] if plan['schema'] == ELASTIC_PLAN_SCHEMA else library['team']


def verify_executor(plan):
    if plan['schema'] == ELASTIC_PLAN_SCHEMA:
        for name, checksum in plan['executor_sources'].items():
            verify_file(Path(__file__).resolve().parent/name, checksum)


def _without_sha(value):
    return {k: v for k, v in value.items() if k != 'manifest_sha256'}


def _value(value):
    if isinstance(value, dict):
        return checked(value)
    path = Path(value)
    return checked(read_json(path/'job.json' if path.is_dir() else path))


def _is_sha(value):
    return isinstance(value, str) and bool(re.fullmatch('[0-9a-f]{64}', value))


def _validate_library_header(library):
    library = checked(library)
    if (library.get('schema') != LIBRARY_SCHEMA or library.get('repo_url') != 'https://github.com/Platypus27-coder/VietMedBridge.git'
            or not re.fullmatch('[0-9a-f]{40}', library.get('code_commit', ''))
            or set(library.get('worker_sources', {})) != set(SOURCE_FILES)
            or set(library.get('models', {})) != set(FAMILIES)):
        raise ValueError('Invalid full-data input library producer.')
    assignment = kaggle_assignment(accounts=library['team']['accounts'], account_id=0, gpu_count=library['team']['gpu_count'])
    assets, seen, occurrences = library.get('assets', {}), set(), {f: 0 for f in FAMILIES}
    for task in library.get('tasks', []):
        if (not _is_sha(task.get('id')) or task['id'] in seen or task.get('selection_id') != task['id']
                or task.get('family') not in FAMILIES or type(task.get('worker')) is not int
                or not 0 <= task['worker'] < assignment['workers'] or type(task.get('rows')) is not int
                or not 0 < task['rows'] <= base.TASK_ROWS or type(task.get('offset')) is not int or task['offset'] < 0
                or type(task.get('ordinal')) is not int or task['ordinal'] < 0
                or not _is_sha(task.get('keys_digest')) or task.get('asset') not in assets):
            raise ValueError('Invalid full-data task binding.')
        asset = assets[task['asset']]
        if (task['asset'] != f"missing-{task['family']}-{task['ordinal']:06d}.parquet"
                or task.get('sha256') != asset.get('sha256') or not _is_sha(asset.get('sha256'))
                or not _is_sha(asset.get('keys_sha256'))
                or type(asset.get('rows')) is not int or task['offset']+task['rows'] > asset['rows']):
            raise ValueError('Invalid shared text asset range.')
        relative_path(asset['source_path'])
        relative_path(asset['keys_path'])
        seen.add(task['id'])
        occurrences[task['family']] += 1
    if (not seen or library.get('scheduled_tasks') != len(seen) or library.get('unscheduled_tasks') != 0
            or library.get('input_export_complete') is not True or library.get('all_remaining_tasks_scheduled') is not True
            or library.get('remaining_tasks_before_round') != occurrences
            or set(assets) != {t['asset'] for t in library['tasks']}):
        raise ValueError('Full input export does not conserve the pending task queue.')
    return library


def validate_library(directory, checkout=None):
    directory = Path(directory)
    library = _validate_library_header(read_json(directory/'job.json'))
    for name, checksum in library['worker_sources'].items():
        verify_file(directory/name, checksum)
    # Inventory only. Every text payload is hashed when consumed by its worker.
    missing = [name for name in library['assets'] if not (directory/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Full input library is missing {len(missing)} shared text files: {missing[:3]}')
    if checkout is not None:
        if set(library['model_code']) != {'retrieval_models.py', 'qwen_models.py'}:
            raise ValueError('Unexpected model producer code.')
        for name, checksum in library['model_code'].items():
            verify_file(Path(checkout)/'src/vietmedbridge'/name, checksum)
        for family in FAMILIES:
            validate_encoder(library, family, library['models'][family]['encoder'], library['models'][family]['dimension'])
    return library


def validate_plan(plan, library):
    plan, library = _value(plan), _validate_library_header(library)
    elastic = plan.get('schema') == ELASTIC_PLAN_SCHEMA
    if (plan.get('schema') not in (PLAN_SCHEMA, ELASTIC_PLAN_SCHEMA) or plan.get('library_job') != library['manifest_sha256']
            or plan.get('selection_binding') != library['selection_binding'] or (not elastic and plan.get('team') != library['team'])
            or plan.get('coverage_scope') != library.get('coverage_scope', {})):
        raise ValueError('Execution plan belongs to another input library, scope or team.')
    team = execution_team(plan, library)
    assignment = kaggle_assignment(accounts=team['accounts'], account_id=0, gpu_count=team['gpu_count'])
    expected_sources = EXECUTOR_FILES if plan.get('qwen_execution') == QWEN_EXECUTION else LEGACY_EXECUTOR_FILES
    if plan.get('qwen_execution') not in (None, QWEN_EXECUTION) or (not elastic and plan.get('qwen_execution')):
        raise ValueError('Unknown Qwen execution policy; the native model producer must remain unchanged.')
    if elastic and (set(plan.get('executor_sources', {})) != set(expected_sources)
                    or any(not _is_sha(v) for v in plan['executor_sources'].values())
                    or plan.get('library_team') != library['team']):
        raise ValueError('Execution scheduler or original library team changed.')
    budget, tasks, entries = plan.get('output_bytes_per_account'), plan.get('tasks'), {t['id']: t for t in library['tasks']}
    if (type(budget) is not int or not 0 < budget <= (ELASTIC_OUTPUT_BYTES if elastic else MAX_OUTPUT_BYTES) or not isinstance(tasks, list)
            or not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(entries)
            or plan.get('scheduled_tasks') != len(tasks)):
        raise ValueError('Invalid execution task whitelist/output budget.')
    if elastic and (set(plan.get('task_workers', {})) != set(tasks)
                    or any(type(w) is not int or not 0 <= w < assignment['workers'] for w in plan['task_workers'].values())):
        raise ValueError('Execution task/GPU assignments are incomplete or out of range.')
    cached = plan.get('cached_keys', {})
    if (set(cached) != set(FAMILIES) or any(not isinstance(values, list) or len(set(values)) != len(values)
            or any(not _is_sha(k) for k in values) for values in cached.values())):
        raise ValueError('Invalid cached-key execution contract.')
    estimates = plan.get('task_output_bytes', {})
    if set(estimates) != set(tasks):
        raise ValueError('Missing bounded task output accounting.')
    used = [0]*team['accounts']
    for task_id in tasks:
        task = entries[task_id]
        estimate = estimates[task_id]
        minimum = library['models'][task['family']]['dimension']*4+1024
        if type(estimate) is not int or not minimum <= estimate <= task['rows']*minimum or estimate % minimum:
            raise ValueError('Execution task output estimate is invalid.')
        used[execution_worker(plan, task)//team['gpu_count']] += estimate
    if plan.get('estimated_bytes_per_account') != used or any(n > budget for n in used):
        raise ValueError('Execution plan exceeds the per-account output budget.')
    if (type(plan.get('remaining_tasks_after_plan')) is not int or plan['remaining_tasks_after_plan'] < 0
            or type(plan.get('remaining_tasks_before_plan')) is not int
            or plan['remaining_tasks_before_plan'] != len(tasks)+plan['remaining_tasks_after_plan']):
        raise ValueError('Execution plan task accounting changed.')
    return plan


class FullQueueKaggle(base.NewDataKaggle):
    def export_inputs(self, checkout, *, config, runtime, code_commit, receipt_api=None,
                      blocks_factory=ResumableContentBlocks):
        """All pending text assets, independently of any GPU output budget."""
        pointer = self.root/'input_library.json'
        if pointer.exists():
            library = _validate_library_header(read_json(pointer))
            if library['selection_binding'] != self.binding or library['models'] != self.view.identities:
                raise ValueError('Saved input library belongs to another selection/model contract.')
            print('FULL_INPUT_LIBRARY_REUSED: sealed metadata; no text selection or asset replay', flush=True)
            return self.work/'full-library'/library['manifest_sha256'][:16], library
        if not re.fullmatch('[0-9a-f]{40}', code_commit):
            raise ValueError('Pin the full native producer commit.')
        selected = self.select(receipt_api=receipt_api, blocks_factory=blocks_factory)
        done, tasks, assets = self.imported(), [], {}
        temporary = self.work/'full-library-assets'
        temporary.mkdir(parents=True, exist_ok=True)
        for family in FAMILIES:
            for ordinal, entry in sorted(selected[family].items()):
                pending = [t for t in entry['tasks'] if t['selection_id'] not in done]
                if not pending:
                    continue
                paths = {t['path'] for t in pending}
                checksums = {t['sha256'] for t in pending}
                if len(paths) != 1 or len(checksums) != 1:
                    raise ValueError('Shared missing-text input binding changed.')
                source_path, checksum = next(iter(paths)), next(iter(checksums))
                asset = f'missing-{family}-{ordinal:06d}.parquet'
                source = bound(self.root, source_path)
                local = temporary/asset
                if not local.is_file():
                    if publish_file(source, local) != checksum:
                        raise ValueError('Saved missing text changed during input export.')
                else:
                    verify_file(local, checksum)
                # A small key-only index on Drive lets future plan creation
                # avoid reopening text payloads or original frozen inputs.
                keys_path = f'library_keys/{family}-{ordinal:06d}.parquet'
                marker = self.root/(keys_path+'.done.json')
                if marker.exists():
                    keys_entry = checked(read_json(marker))
                    if keys_entry.get('source_sha256') != checksum:
                        raise ValueError('Saved library key index source changed.')
                else:
                    table = pq.read_table(local, columns=['id'])
                    keys_local = temporary/'keys.parquet'
                    pq.write_table(table, keys_local, compression='zstd')
                    keys_entry = seal({'path': keys_path, 'sha256': publish_file(keys_local, self.root/keys_path),
                                       'rows': len(table), 'source_sha256': checksum})
                    atomic_json(marker, keys_entry)
                assets[asset] = {'sha256': checksum, 'rows': keys_entry['rows'], 'source_path': source_path,
                                 'keys_path': keys_path, 'keys_sha256': keys_entry['sha256']}
                tasks.extend({**t, 'id': t['selection_id'], 'asset': asset} for t in pending)
                if len(assets) % 64 == 0:
                    print(f'FULL_INPUT_EXPORT: {len(assets):,} shared files | {len(tasks):,} tasks | no output-cap truncation', flush=True)
        if not tasks:
            return None, {'state': 'NO_PENDING_KAGGLE_TASKS', 'coverage': self.coverage()}
        model_code = {n: sha256_file(Path(checkout)/'src/vietmedbridge'/n) for n in ('retrieval_models.py', 'qwen_models.py')}
        directory = Path(__file__).resolve().parent
        workers = {n: sha256_file(directory/n) for n in SOURCE_FILES}
        sources = {s['build']['build_run']: s['build'] for s in self.view.sources}
        library = seal({'schema': LIBRARY_SCHEMA, 'selection_binding': self.binding, 'batch': self.view.batch['manifest_sha256'],
            'source_batch_dir': self.view.root.relative_to(self.view.project).as_posix(),
            'coverage_scope': {k: self.view.batch[k] for k in ('official_ids','source_builds','source_documents_sum',
                'input_rows_before_cross_source_dedup','scope_manifest_sha256') if k in self.view.batch},
            'frozen_sources': [{'build_run': b['build_run'], 'candidate_manifest_sha256': b['candidate_manifest_sha256'],
                'index_inputs_manifest_sha256': b['index_inputs_manifest_sha256'], 'unique_dense_inputs': b['unique_dense_inputs'],
                'inherited_complete_models': sorted(b.get('completed_views', {}))} for b in sources.values()],
            'input_export_complete': True, 'scheduled_tasks': len(tasks), 'unscheduled_tasks': 0,
            'all_remaining_tasks_scheduled': True,
            'remaining_tasks_before_round': {f: sum(t['family'] == f for t in tasks) for f in FAMILIES},
            'team': self.team, 'code_commit': code_commit, 'repo_url': 'https://github.com/Platypus27-coder/VietMedBridge.git',
            'models': self.view.identities, 'config': config, 'runtime': runtime, 'model_code': model_code,
            'worker_sources': workers, 'tasks': tasks, 'assets': assets,
            'scope': self.view.batch.get('scope', 'All frozen missing corpus text; no submission/query inference.')})
        _validate_library_header(library)
        target = self.work/'full-library'/library['manifest_sha256'][:16]
        target.mkdir(parents=True, exist_ok=True)
        # Move within Colab scratch avoids retaining a second full payload copy.
        for asset, entry in assets.items():
            destination = target/asset
            if destination.exists():
                verify_file(destination, entry['sha256'])
            else:
                (temporary/asset).replace(destination)
        for name, checksum in workers.items():
            base.copied(directory/name, target/name, checksum)
        atomic_json(target/'job.json', library)
        atomic_json(self.root/'exports'/('job-'+library['manifest_sha256']+'.json'), library)
        atomic_json(pointer, library)
        print(f'FULL_INPUT_LIBRARY_READY: {len(tasks):,} tasks; {len(assets):,} shared files; 0 input tasks held by output budget', flush=True)
        return target, library

    def create_plan(self, library, *, output_bytes=MAX_OUTPUT_BYTES, receipt_api=None,
                    blocks_factory=ResumableContentBlocks, supersede_unstarted_legacy=False,
                    execution_accounts=None, replace_unstarted_plan=None):
        library = _validate_library_header(_value(library))
        if library['selection_binding'] != self.binding or library['models'] != self.view.identities:
            raise ValueError('Input library differs from this frozen selection.')
        elastic = execution_accounts is not None
        team = {'accounts': execution_accounts, 'gpu_count': self.team['gpu_count']} if elastic else self.team
        assignment = kaggle_assignment(accounts=team['accounts'], account_id=0, gpu_count=team['gpu_count'])
        if type(output_bytes) is not int or not 0 < output_bytes <= (ELASTIC_OUTPUT_BYTES if elastic else MAX_OUTPUT_BYTES):
            raise ValueError('Output budget exceeds the scheduler allowance; leave room below Kaggle saved Outputs capacity.')
        active = self.root/'active_execution_plan.json'
        if active.exists():
            record = checked(read_json(active))
            if record.get('binding') != self.binding or record.get('library_job') != library['manifest_sha256']:
                raise ValueError('Active execution pointer belongs to another library/selection.')
            plan = validate_plan(self.saved_plan(record['plan']), library)
            if (replace_unstarted_plan != plan['manifest_sha256'] and _is_sha(replace_unstarted_plan)
                    and (self.root/'history'/('unstarted-'+replace_unstarted_plan+'.json')).is_file()):
                replace_unstarted_plan = None
            imported_accounts = {r['account_id'] for path in (self.root/'imports').glob('*.json')
                if (r := checked(read_json(path))).get('execution_plan') == plan['manifest_sha256']}
            if imported_accounts != set(range(plan['team']['accounts'])):
                if replace_unstarted_plan == plan['manifest_sha256'] and not imported_accounts and elastic:
                    atomic_json(self.root/'history'/('unstarted-'+plan['manifest_sha256']+'.json'), seal({
                        'state': 'SUPERSEDED_UNSTARTED', 'previous': record, 'unstarted_confirmed': True,
                        'replacement_team': team, 'updated_at': utc_now()}))
                    print(f'REPLACE_CONFIRMED_UNSTARTED_PLAN: {plan["manifest_sha256"][:12]} | old imports retained', flush=True)
                else:
                    if replace_unstarted_plan is not None:
                        raise ValueError('Only the exact confirmed unstarted plan with no imports may be replaced.')
                    if record.get('state') == 'UPLOAD_PENDING' and not imported_accounts:
                        if plan['team'] != team:
                            raise ValueError('Saved pending plan has another execution team; retain it or explicitly replace it.')
                        return self.stage_plan(plan), plan
                    return None, {'state': 'WAITING_FOR_KAGGLE_OUTPUTS', 'execution_plan': plan['manifest_sha256'],
                        'library_job': library['manifest_sha256'], 'dataset_handle': record.get('dataset_handle'),
                        'imported_accounts': sorted(imported_accounts), 'required_accounts': list(range(plan['team']['accounts']))}
        legacy = self.root/'active_round.json'
        if legacy.exists():
            saved = checked(read_json(legacy))
            if saved.get('binding') != self.binding:
                raise ValueError('Legacy round pointer belongs to another selection.')
            history = self.root/'history'/('legacy-'+saved['job']+'.json')
            if not history.exists():
                imported_accounts = {r['account_id'] for path in (self.root/'imports').glob('*.json')
                    if (r := checked(read_json(path))).get('job') == saved['job'] and 'account_id' in r}
                if imported_accounts != set(range(self.team['accounts'])) and not supersede_unstarted_legacy:
                    return None, {'state': 'LEGACY_ROUND_REQUIRES_EXPLICIT_SUPERSEDE', 'dataset_handle': saved['dataset_handle'],
                                  'instruction': 'Keep/import any GPU Outputs; explicitly confirm legacy GPU did not run before promoting the full library.'}
                atomic_json(history, seal({'legacy': saved, 'superseded_by_library': library['manifest_sha256'],
                    'unstarted_confirmed': supersede_unstarted_legacy, 'updated_at': utc_now()}))
        done = self.imported()
        pending = [t for t in library['tasks'] if t['id'] not in done]
        used, scheduled, estimates = [0]*team['accounts'], [], {}
        task_workers, loads, affinity = {}, [0]*assignment['workers'], {}
        assets_read = 0
        cached, cache_only, affected = {f: set() for f in FAMILIES}, [], {f: set() for f in FAMILIES}
        for family in FAMILIES:
            family_tasks = [t for t in pending if t['family'] == family]
            if not family_tasks:
                continue
            expected = self.view.identities[family]
            blocks = blocks_factory(self.view.data, SimpleNamespace(identity=expected['encoder'], dimension=expected['dimension']),
                                    self.work/family, receipt_api=receipt_api, progress=print)
            previous, keys = None, None
            try:
                for task in family_tasks:
                    account = task['worker']//self.team['gpu_count']
                    unit_bytes = expected['dimension']*4+1024
                    # No need to read later key files once an account cannot
                    # fit even one new vector. Those tasks remain in the library.
                    if (all(n+unit_bytes > output_bytes for n in used) if elastic else used[account]+unit_bytes > output_bytes):
                        continue
                    asset = library['assets'][task['asset']]
                    if previous != task['asset']:
                        path = bound(self.root, asset['keys_path'])
                        verify_file(path, asset['keys_sha256'])
                        keys = pq.read_table(path, columns=['id']).column('id').to_pylist()
                        if len(keys) != asset['rows']:
                            raise ValueError('Saved task-key index row coverage changed.')
                        previous = task['asset']
                        assets_read += 1
                        if elastic and assets_read % 32 == 0:
                            print(f'PLAN_KEY_METADATA: {assets_read} key indexes | no corpus/text replay; imported tasks excluded', flush=True)
                    selected = keys[task['offset']:task['offset']+task['rows']]
                    if len(selected) != task['rows'] or digest_json(sorted(selected)) != task['keys_digest']:
                        raise ValueError('Saved library task keys changed.')
                    known = base.lookup(blocks, selected)
                    if len(known) == len(selected):
                        cache_only.append(task['id'])
                        affected[family].add(task['ordinal'])
                        continue
                    estimate = (len(selected)-len(known))*unit_bytes
                    if estimate > output_bytes:
                        raise ValueError('One missing-text task exceeds the execution budget.')
                    if elastic:
                        group = (family, task['keys_digest'])
                        candidates = [w for w in range(assignment['workers'])
                                      if used[w//team['gpu_count']]+estimate <= output_bytes]
                        if group in affinity:
                            candidates = [affinity[group]] if affinity[group] in candidates else []
                        if not candidates:
                            continue
                        worker = min(candidates, key=lambda w: (loads[w], w))
                        affinity[group] = worker
                        task_workers[task['id']] = worker
                        loads[worker] += (len(selected)-len(known))*(8 if family == 'qwen' else 1)
                        account = worker//team['gpu_count']
                    elif used[account]+estimate > output_bytes:
                        continue
                    used[account] += estimate
                    scheduled.append(task['id'])
                    estimates[task['id']] = estimate
                    cached[family].update(known)
                if affected[family]:
                    self.complete_maps(family, affected[family], blocks)
            finally:
                blocks.close()
        if cache_only:
            atomic_json(self.root/'imports'/('cache-'+digest_json(cache_only)+'.json'), seal({
                'binding': self.binding, 'selection_ids': cache_only, 'state': 'REUSED_NEWLY_AVAILABLE_CACHE'}))
        if not scheduled:
            return None, {'state': 'NO_PENDING_KAGGLE_TASKS', 'coverage': self.coverage()}
        remaining = len(pending)-len(cache_only)
        extra = {}
        if elastic:
            directory = Path(__file__).resolve().parent
            sources = {n: sha256_file(directory/n) for n in EXECUTOR_FILES}
            for name, checksum in sources.items():
                base.copied(directory/name, self.root/'executors'/checksum/name, checksum)
            extra = {'library_team': library['team'], 'task_workers': task_workers, 'executor_sources': sources,
                     'replaces_unstarted_plan': replace_unstarted_plan, 'qwen_execution': QWEN_EXECUTION}
        plan = seal({'schema': ELASTIC_PLAN_SCHEMA if elastic else PLAN_SCHEMA, 'library_job': library['manifest_sha256'],
            'selection_binding': self.binding, 'team': team, 'coverage_scope': library.get('coverage_scope', {}),
            'tasks': scheduled, 'cached_keys': {f: sorted(v) for f, v in cached.items()},
            'task_output_bytes': estimates, 'output_bytes_per_account': output_bytes, 'estimated_bytes_per_account': used,
            'scheduled_tasks': len(scheduled), 'remaining_tasks_before_plan': remaining,
            'remaining_tasks_after_plan': remaining-len(scheduled), 'created_at': utc_now(), **extra})
        validate_plan(plan, library)
        atomic_json(self.root/'plans'/(plan['manifest_sha256']+'.json'), plan)
        # Commit before upload: cold retry resumes exactly this plan rather than
        # rebuilding a fresh timestamp/whitelist and discarding its checkpoint.
        atomic_json(active, seal({'binding': self.binding, 'plan': plan['manifest_sha256'],
                                 'library_job': library['manifest_sha256'], 'state': 'UPLOAD_PENDING'}))
        return self.stage_plan(plan), plan

    def stage_plan(self, plan):
        target = self.work/'execution-plans'/plan['manifest_sha256'][:16]
        target.mkdir(parents=True, exist_ok=True)
        for name, checksum in plan.get('executor_sources', {}).items():
            base.copied(self.root/'executors'/checksum/name, target/name, checksum)
        atomic_json(target/'job.json', plan)
        return target

    def saved_plan(self, plan_sha):
        if not _is_sha(plan_sha):
            raise ValueError('Invalid execution plan hash.')
        plan = checked(read_json(self.root/'plans'/(plan_sha+'.json')))
        if plan['manifest_sha256'] != plan_sha:
            raise ValueError('Saved execution plan filename/hash binding changed.')
        return plan

    def install(self, output, *, receipt_api=None, blocks_factory=ResumableContentBlocks):
        output = Path(output)
        library = checked(read_json(output/'job.json'))
        if library.get('schema') != LIBRARY_SCHEMA:
            return super().install(output, receipt_api=receipt_api, blocks_factory=blocks_factory)
        saved = checked(read_json(self.root/'exports'/('job-'+library['manifest_sha256']+'.json')))
        result = checked(read_json(output/'new-data-result.json'))
        plan = validate_plan(self.saved_plan(result.get('execution_plan')), library)
        if result.get('plan') != plan or library != saved or result.get('job') != library['manifest_sha256']:
            raise ValueError('GPU Outputs differ from the saved full-library execution plan.')
        if result.get('state') not in ('ROUND_COMPLETE', 'CHECKPOINTED_PARTIAL'):
            raise ValueError('Invalid full-library execution result state.')
        team = execution_team(plan, library)
        assignment = kaggle_assignment(accounts=team['accounts'], account_id=result['account_id'], gpu_count=team['gpu_count'])
        tasks = {t['id']: t for t in library['tasks']}
        receipts = [r['task'] for r in result['tasks']]
        allowed = {task for task in plan['tasks'] if execution_worker(plan, tasks[task]) in assignment['worker_ids']}
        if len(set(receipts)) != len(receipts) or not set(receipts) <= allowed:
            raise ValueError('Output contains duplicate or non-whitelisted execution tasks.')
        if result.get('state') == 'ROUND_COMPLETE' and set(receipts) != allowed:
            raise ValueError('Round completion does not cover this account execution plan.')
        # Native importer verifies payload SHA, exact task keys, encoder identity
        # and skipped cached keys before it can publish a map/coverage receipt.
        report = super().install(output, receipt_api=receipt_api, blocks_factory=blocks_factory,
                                 cached_keys=plan['cached_keys'])
        receipt = self.root/'imports'/(result['manifest_sha256']+'.json')
        if report.get('execution_plan') != plan['manifest_sha256']:
            prior = checked(read_json(receipt))
            prior = seal({**_without_sha(prior), 'execution_plan': plan['manifest_sha256'], 'execution_team': team})
            atomic_json(receipt, prior)
            report = prior
        return report

    def assigned_output_tasks(self, job, result):
        if result.get('plan', {}).get('schema') != ELASTIC_PLAN_SCHEMA:
            return super().assigned_output_tasks(job, result)
        plan = validate_plan(self.saved_plan(result.get('execution_plan')), job)
        if result.get('plan') != plan:
            raise ValueError('Elastic output scheduler differs from the saved execution plan.')
        team = execution_team(plan, job)
        assignment = kaggle_assignment(accounts=team['accounts'], account_id=result['account_id'], gpu_count=team['gpu_count'])
        entries = {t['id']: t for t in job['tasks']}
        return {tid for tid in plan['tasks'] if execution_worker(plan, entries[tid]) in assignment['worker_ids']}


def upload_library(flow, target, library, owner):
    # Metadata-only reuse after a successful upload. If the first upload was
    # interrupted by a fresh Colab runtime, rebuild only its missing staging
    # files from the saved selection; never reopen/chunk the source corpus.
    library = _validate_library_header(library)
    saved = checked(read_json(flow.root/'exports'/('job-'+library['manifest_sha256']+'.json')))
    if saved != library or library['selection_binding'] != flow.binding:
        raise ValueError('Full-library upload differs from its saved selection contract.')
    pointer = flow.root/'uploads'/(owner+'-'+library['manifest_sha256']+'.json')
    planned = not pointer.exists() or read_json(pointer).get('state') == 'UPLOAD_PLANNED'
    if planned:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        for name, entry in library['assets'].items():
            if not (target/name).is_file():
                if publish_file(bound(flow.root, entry['source_path']), target/name) != entry['sha256']:
                    raise ValueError('Saved full-library input changed during upload staging.')
        scripts = Path(__file__).resolve().parent
        for name, checksum in library['worker_sources'].items():
            base.copied(scripts/name, target/name, checksum)
        atomic_json(target/'job.json', library)
    return base.upload_job(flow, target, library, owner, activation_filename='full_package.json', dataset_prefix='/vmb-full-inputs-')


def upload_plan(flow, target, plan, owner):
    plan = checked(plan)
    library = checked(read_json(flow.root/'exports'/('job-'+plan['library_job']+'.json')))
    validate_plan(plan, library)
    if plan != flow.saved_plan(plan['manifest_sha256']) or plan['selection_binding'] != flow.binding:
        raise ValueError('Execution upload differs from its saved plan/selection.')
    if plan['schema'] == ELASTIC_PLAN_SCHEMA:
        target = flow.stage_plan(plan)
    handle = base.upload_job(flow, target, plan, owner, activation_filename='uploaded_execution_plan.json', dataset_prefix='/vmb-full-plan-')
    atomic_json(flow.root/'active_execution_plan.json', seal({'binding': flow.binding, 'plan': plan['manifest_sha256'],
        'library_job': plan['library_job'], 'dataset_handle': handle, 'state': 'VERIFIED'}))
    return handle


def run_worker(directory, plan, output, work, *, worker_id, factories, deadline, batch_sizes=None):
    directory, output, work = Path(directory), Path(output), Path(work)
    library = validate_library(directory)
    plan = validate_plan(plan, library)
    verify_executor(plan)
    team = execution_team(plan, library)
    workers = team['accounts']*team['gpu_count']
    original_workers = library['team']['accounts']*library['team']['gpu_count']
    if type(worker_id) is not int or not 0 <= worker_id < workers:
        raise ValueError('Worker ID outside the full-data team.')
    output.mkdir(parents=True, exist_ok=True)
    for name, record in (('job.json', library), ('execution-plan.json', plan)):
        if (output/name).exists() and read_json(output/name) != record:
            raise ValueError('Working output contains another job or execution plan; preserve its Outputs.')
        atomic_json(output/name, record)
    journal = output/f'worker-{worker_id}.json'
    completed = []
    if journal.exists():
        saved = checked(read_json(journal))
        if saved.get('job') != library['manifest_sha256'] or saved.get('execution_plan') != plan['manifest_sha256'] or saved.get('worker_id') != worker_id:
            raise ValueError('Worker journal belongs to another job/plan/worker.')
        completed = saved['tasks']
    entries = {t['id']: t for t in library['tasks']}
    allowed = {tid for tid in plan['tasks'] if execution_worker(plan, entries[tid]) == worker_id}
    done = {r['task'] for r in completed}
    if len(done) != len(completed) or not done <= allowed:
        raise ValueError('Worker journal contains duplicate or non-whitelisted tasks.')
    stop_reason = 'ASSIGNMENT_COMPLETE'
    cached = {f: set(values) for f, values in plan['cached_keys'].items()}
    previous, rows = None, None
    for family in FAMILIES:
        assigned = [entries[tid] for tid in plan['tasks'] if tid in allowed-done and entries[tid]['family'] == family]
        print(f'GPU worker {worker_id}: {family}, {len(assigned)} bounded tasks; full input library already present', flush=True)
        if not assigned:
            continue
        if time.monotonic() >= deadline:
            stop_reason = 'SESSION_TIME_LIMIT'
            continue
        lazy = LazyEncoder(library['models'][family], factories[family])
        blocks = ContentBlocks(output, lazy, work/family)
        try:
            for task in assigned:
                if time.monotonic() >= deadline:
                    stop_reason = 'SESSION_TIME_LIMIT'
                    break
                if previous != task['asset']:
                    path = directory/task['asset']
                    verify_file(path, task['sha256'])
                    rows = pq.read_table(path, columns=['id', 'text']).to_pylist()
                    if len(rows) != library['assets'][task['asset']]['rows']:
                        raise ValueError('Shared text asset row coverage changed.')
                    previous = task['asset']
                values = rows[task['offset']:task['offset']+task['rows']]
                keys = [r['id'] for r in values]
                if (len(values) != task['rows'] or len(set(keys)) != len(keys) or digest_json(sorted(keys)) != task['keys_digest']
                        or any(k != key(r['text']) or int(k[:8], 16) % original_workers != task['worker'] for k, r in zip(keys, values))):
                    raise ValueError('Full-library task text/key/ownership changed.')
                new_values = [r for r in values if r['id'] not in cached[family]]
                expected_bytes = len(new_values)*(library['models'][family]['dimension']*4+1024)
                if expected_bytes != plan['task_output_bytes'][task['id']]:
                    raise ValueError('Cached-key skip differs from the bounded task estimate.')
                defaults = {'bge':32,'qwen':64 if plan.get('qwen_execution') == QWEN_EXECUTION else 16}
                blocks.refs([r['text'] for r in new_values], batch_size=(batch_sizes or defaults)[family],
                            encode_missing=True, encode_block_rows=base.TASK_ROWS)
                completed.append({'task': task['id'], 'keys': keys})
                atomic_json(journal, seal({'job': library['manifest_sha256'], 'execution_plan': plan['manifest_sha256'],
                    'worker_id': worker_id, 'tasks': completed}))
                print(f'KAGGLE_CHECKPOINT worker {worker_id} | saved tasks {len(completed)} | {family} | plan {plan["manifest_sha256"][:12]}', flush=True)
        finally:
            blocks.close()
            lazy.close()
    return seal({'job': library['manifest_sha256'], 'execution_plan': plan['manifest_sha256'], 'worker_id': worker_id,
                 'tasks': completed, 'stop_reason': stop_reason})


def finalize(directory, plan, output, account_id):
    library = validate_library(directory)
    plan = validate_plan(plan, library)
    root = Path(output)/'result'
    root.mkdir(parents=True, exist_ok=True)
    team = execution_team(plan, library)
    assignment = kaggle_assignment(accounts=team['accounts'], account_id=account_id, gpu_count=team['gpu_count'])
    entries, tasks = {t['id']: t for t in library['tasks']}, []
    allowed = {tid for tid in plan['tasks'] if execution_worker(plan, entries[tid]) in assignment['worker_ids']}
    for worker in assignment['worker_ids']:
        path = root/f'worker-{worker}.json'
        if not path.exists():
            continue
        saved = checked(read_json(path))
        if saved.get('job') != library['manifest_sha256'] or saved.get('execution_plan') != plan['manifest_sha256'] or saved.get('worker_id') != worker:
            raise ValueError('Worker journal belongs to another job/plan/assignment.')
        for receipt in saved['tasks']:
            if receipt['task'] not in allowed or execution_worker(plan, entries[receipt['task']]) != worker:
                raise ValueError('Worker returned a non-whitelisted execution task.')
            tasks.append(receipt)
    if len({r['task'] for r in tasks}) != len(tasks):
        raise ValueError('Duplicate completed execution task.')
    committed = {f: {k for receipt in tasks if entries[receipt['task']]['family'] == f for k in receipt['keys']} for f in FAMILIES}
    blocks = {}
    for path in root.glob('model_cache/*/*.done.json'):
        block = checked(read_json(path))
        family = next((f for f in FAMILIES if block['identity'] == {**library['models'][f], 'format': 'content-float32-l2-v1'}), None)
        if family is not None and set(block['keys']) <= committed[family]:
            blocks[block['path']] = block
    atomic_json(root/'job.json', library)
    atomic_json(root/'execution-plan.json', plan)
    result = seal({'schema': 'new-data-kaggle-result-v1', 'job': library['manifest_sha256'],
        'execution_plan': plan['manifest_sha256'], 'plan': plan, 'account_id': account_id,
        'tasks': tasks, 'blocks': list(blocks.values()), 'updated_at': utc_now(),
        'state': 'ROUND_COMPLETE' if {r['task'] for r in tasks} == allowed else 'CHECKPOINTED_PARTIAL'})
    atomic_json(root/'new-data-result.json', result)
    return result


def launch(directory, plan, checkout, output, work, *, account_id=0, hours=10.0):
    import torch
    from vietmedbridge.portable_embeddings import embedding_runtime
    library = validate_library(directory, checkout)
    plan_path = Path(plan)/'job.json' if Path(plan).is_dir() else Path(plan)
    plan = validate_plan(plan_path, library)
    verify_executor(plan)
    team = execution_team(plan, library)
    if embedding_runtime() != library['runtime'] or not torch.cuda.is_available() or torch.cuda.device_count() != team['gpu_count']:
        raise RuntimeError('CUDA/runtime mismatch: select the locked T4 x2 runtime.')
    assignment = kaggle_assignment(accounts=team['accounts'], account_id=account_id, gpu_count=team['gpu_count'])
    if not 0 < hours <= 11.5:
        raise ValueError('Use a save deadline up to 11.5 hours, below the 12-hour session limit.')
    deadline, processes = time.monotonic()+hours*3600-180, []
    try:
        for gpu, worker in enumerate(assignment['worker_ids']):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['PYTHONPATH'] = os.pathsep.join((str(Path(checkout)/'src'), str(Path(__file__).resolve().parent), str(Path(directory))))
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--job', str(directory),
                '--plan', str(plan_path), '--output', str(Path(output)/'result'), '--work', str(Path(work)/f'worker-{worker}'),
                '--worker-id', str(worker), '--deadline', str(deadline)]
            processes.append(subprocess.Popen(command, env=env))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A GPU worker failed; preserve completed task Outputs.')
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('A GPU worker failed; inspect its preceding traceback.')
        return finalize(directory, plan, output, account_id)
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
        finalize(directory, plan, output, account_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['worker', 'launch'])
    for name in ('job', 'plan', 'output', 'work'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--checkout')
    parser.add_argument('--worker-id', type=int)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--account-id', type=int, default=0)
    parser.add_argument('--hours', type=float, default=10.0)
    args = parser.parse_args()
    if args.mode == 'launch':
        result = launch(args.job, args.plan, args.checkout, args.output, args.work, account_id=args.account_id, hours=args.hours)
    else:
        import torch
        from vietmedbridge.portable_embeddings import embedding_runtime
        from vietmedbridge.retrieval_models import TorchDenseEncoder
        from vietmedbridge.qwen_models import TorchQwenEncoder
        library = validate_library(args.job)
        if not torch.cuda.is_available() or embedding_runtime() != library['runtime']:
            raise RuntimeError('GPU worker runtime differs from the paid vector producer.')
        plan = validate_plan(args.plan,library)
        def qwen_factory():
            owner = TorchQwenEncoder(library['config']['second_dense'])
            if plan.get('qwen_execution') == QWEN_EXECUTION:
                from qwen_token_batching import TokenBatchOwner
                return TokenBatchOwner(owner,Path(args.output)/'qwen-tuning'/f'worker-{args.worker_id}.json',
                                       deadline=args.deadline)
            return owner
        result = run_worker(args.job, plan, args.output, args.work, worker_id=args.worker_id, deadline=args.deadline,
            factories={'bge': lambda: TorchDenseEncoder(library['config']['dense']), 'qwen': qwen_factory})
    print('GPU_WORKER_RESULT:', base.json.dumps({k: v for k, v in result.items()
          if k not in ('tasks', 'blocks', 'plan')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
