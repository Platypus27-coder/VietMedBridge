"""All validated sources, with a durable audit of current BTC ID coverage.

No crawl, extraction, chunking, catalog rebuild or vector payload scan. Legacy
validated sources are retained as requested; current BTC IDs have a separate
membership manifest so document-row totals cannot masquerade as URL coverage.
"""
import gzip
import hashlib
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

from vietmedbridge.artifacts import atomic_json, digest_json, publish_file, read_json, utc_now, verify_file
from vietmedbridge.content_embeddings import checked, seal
from vietmedbridge.portable_embeddings import resources_relative
from vietmedbridge.scale_vectors import bound

try:
    from .new_data_embeddings import load_new_batch
    from .recovery_kaggle_batch import load_recovery_batch
except ImportError:
    from new_data_embeddings import load_new_batch
    from recovery_kaggle_batch import load_recovery_batch

PARENT = 'data_staging/btc-full-embeddings-71be0e12-v1'
COLLECTION = 'data_staging/btc-all-ai-guru-71be0e12-v1'
MAXIMIZE = 'data_staging/btc-maximize-71be0e12-v1'
NEW = 'data_staging/btc-new-71be0e12-20261019-v1'
MEMBERS = pa.schema([('official_id', pa.int64()), ('url', pa.string()),
                     ('source', pa.string()), ('source_doc_id', pa.int64())])


def _source(project, run, build_dir, candidate_name, candidate_sha, *, mapping=None, alias_sha=None, views=None):
    build = bound(project, build_dir)
    candidate = read_json(bound(build, candidate_name))
    if (candidate.get('candidate_manifest_sha256') != candidate_sha
            or digest_json({k: v for k, v in candidate.items() if k != 'candidate_manifest_sha256'}) != candidate_sha
            or candidate.get('state') != 'FROZEN_CANDIDATE'
            or not candidate.get('golden', {}).get('passed')
            or not candidate.get('health', {}).get('integrity', {}).get('passed')
            or candidate['health']['integrity'].get('official_membership') != 'VERIFIED'):
        raise ValueError('Unvalidated source in full-data handoff: '+run)
    if alias_sha is not None and candidate['health']['files']['canonical_aliases']['sha256'] != alias_sha:
        raise ValueError('Baseline alias mapping does not belong to the frozen candidate.')
    inputs = checked(read_json(build / 'index_inputs/units.json'))
    if inputs.get('state') != 'COMPLETE' or inputs['candidate_manifest_sha256'] != candidate_sha:
        raise ValueError('Complete native model inputs are required: '+run)
    record = {'build_run': run, 'build_dir': build.relative_to(project).as_posix(),
              'candidate_name': candidate_name, 'candidate_manifest_sha256': candidate_sha,
              'snapshot_sha256': candidate['snapshot_sha256'], 'documents': candidate['counts']['documents'],
              'index_inputs_manifest_sha256': inputs['manifest_sha256'], 'unique_dense_inputs': inputs['input_count'],
              'mapping': mapping, 'mapping_sha256': candidate['health']['files']['canonical_aliases']['sha256'] if mapping else None,
              'completed_views': {}}
    for family, directory in (views or {}).items():
        path = directory / 'embeddings.json'
        if path.is_file():
            saved = checked(read_json(path))
            if saved.get('state') == 'COMPLETE':
                if saved.get('input_manifest_sha256') != inputs['manifest_sha256'] or saved.get('input_count') != inputs['input_count']:
                    raise ValueError('Inherited model view belongs to another input manifest.')
                record['completed_views'][family] = {'path': directory.relative_to(project).as_posix(),
                    'manifest_sha256': saved['manifest_sha256']}
    return record


def inventory(project, checkout):
    """Metadata only: baseline + ten frozen intake builds + ALL frozen recovery deltas."""
    project, checkout = Path(project).resolve(), Path(checkout)
    scope = read_json(project / NEW / 'full_intake_scope.json')
    baseline = scope['base_plan']['baseline']
    active = read_json(project / 'data/active_data_candidate.json')
    if active['candidate_manifest_sha256'] != baseline['candidate_manifest_sha256']:
        raise ValueError('Active candidate differs from the audited baseline; retain both before changing scope.')
    resources = project / 'data' / resources_relative(active, read_json(checkout / 'configs/retrieval_full.json'))
    records = [_source(project, active['build_run'], 'data/processed/'+active['build_run'], active['candidate_name'],
        active['candidate_manifest_sha256'], mapping=NEW+'/references/baseline_aliases.parquet', alias_sha=baseline['sha256'],
        views={family: resources / family for family in ('bge', 'qwen')})]
    new_root, new_batch, _ = load_new_batch(project)
    for item in new_batch['builds']:
        build = bound(project, item['build_dir'])
        candidate = read_json(bound(build, item['candidate_name']))
        alias = candidate['health']['files']['canonical_aliases']
        records.append(_source(project, item['build_run'], item['build_dir'], item['candidate_name'],
            item['candidate_manifest_sha256'], mapping=bound(build, alias['path']).relative_to(project).as_posix(),
            views={family: new_root / 'embeddings_v1' / family / item['build_run'] for family in ('bge', 'qwen')}))
    recovery_root, recovery_batch, _ = load_recovery_batch(project)
    for item in recovery_batch['builds']:
        records.append(_source(project, item['build_run'], item['build_dir'], item['candidate_name'],
            item['candidate_manifest_sha256'], views={family: recovery_root / 'embeddings_v1' / family / item['build_run']
                                                    for family in ('bge', 'qwen')}))
    if len({r['build_run'] for r in records}) != len(records):
        raise ValueError('Duplicate build names in full-data handoff.')
    report = checked(read_json(project / COLLECTION / 'collection_report.json'))
    if report['archive_errors'] or report['counts_provisional']:
        raise ValueError('Resolve collection coverage before full-data export.')
    return {'workflow': 'FULL_VALIDATED_BTC_EMBEDDINGS_V1', 'sources': records,
            'collection_report': report['manifest_sha256'], 'expected_reused': report['counts_by_official_id']['REUSED_VALIDATED_CONTENT'],
            'official_total': report['official_ids'],
            'collection_pointer': checked(read_json(project / COLLECTION / 'checkpoint.json')),
            'maximize_pointer': checked(read_json(project / MAXIMIZE / 'checkpoint.json')),
            'scope': 'All frozen validated data, including retained legacy content; exact current BTC membership audited separately.'}


def _restore(root, pointer, work):
    """Restore one metadata SQL snapshot in Colab scratch, never on the user's PC."""
    work.mkdir(parents=True, exist_ok=True)
    target = work / (pointer['sqlite_sha256']+'.sqlite')
    marker = target.with_suffix('.verified.json')
    if target.is_file() and marker.is_file():
        proof = read_json(marker)
        if proof.get('sha256') == pointer['sqlite_sha256'] and proof.get('bytes') == target.stat().st_size:
            return target
    print('FULL_SCOPE restore metadata SQL:', root.name, '| no TAR/corpus/vector reads', flush=True)
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        compressed, database = Path(temporary)/'sql.gz', Path(temporary)/'sql.sqlite'
        sha, copied, logged = hashlib.sha256(), 0, time.monotonic()
        with bound(root, pointer['path']).open('rb') as incoming, compressed.open('wb') as out:
            for block in iter(lambda: incoming.read(1024**2), b''):
                sha.update(block); out.write(block); copied += len(block)
                if time.monotonic()-logged >= 15:
                    print('FULL_SCOPE metadata transfer:', root.name, '|', round(copied/1024**2, 1), 'MiB', flush=True)
                    logged = time.monotonic()
        if sha.hexdigest() != pointer['sha256']:
            raise ValueError('Metadata SQL snapshot changed during restore.')
        with gzip.open(compressed, 'rb') as incoming, database.open('wb') as out:
            shutil.copyfileobj(incoming, out, 1024**2)
        verify_file(database, pointer['sqlite_sha256'])
        database.replace(target)
    atomic_json(marker, {'sha256': pointer['sqlite_sha256'], 'bytes': target.stat().st_size})
    return target


def prepare_full_scope(project, checkout, work, *, minimum_ids=802028):
    project, work = Path(project).resolve(), Path(work)
    parent = project / PARENT
    active_path = parent / 'active_scope.json'
    if active_path.is_file():
        active = checked(read_json(active_path))
        root = bound(project, active['root'])
        contract = checked(read_json(root / 'source_contract.json'))
        if contract['manifest_sha256'] != active['contract']:
            raise ValueError('Pinned full-data scope changed.')
    else:
        contract = seal(inventory(project, checkout))
        root = parent / contract['manifest_sha256'][:16]
        atomic_json(root / 'source_contract.json', contract)
        atomic_json(active_path, seal({'root': root.relative_to(project).as_posix(), 'contract': contract['manifest_sha256']}))
    if root.parent != parent:
        raise ValueError('Unexpected full-scope output directory.')
    ready = root / 'scope.json'
    if ready.is_file():
        report = checked(read_json(ready))
        if report.get('state') != 'FULL_SCOPE_READY' or report['contract'] != contract['manifest_sha256'] or report['official_ids'] < minimum_ids:
            raise ValueError('Full-data scope is incomplete or below the requested BTC coverage.')
        print('FULL_SCOPE_REUSED:', report['official_ids'], 'BTC IDs | no SQL or corpus replay', flush=True)
        return root, report
    collection = _restore(project / COLLECTION, contract['collection_pointer'], work)
    maximum = _restore(project / MAXIMIZE, contract['maximize_pointer'], work)
    con = sqlite3.connect(collection.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        con.execute('ATTACH DATABASE ? AS maximum', [maximum.resolve().as_uri()+'?mode=ro'])
        con.execute('PRAGMA query_only=ON'); con.execute('PRAGMA cache_size=-32768'); con.execute('PRAGMA temp_store=FILE')
        for database, key in (('main', 'collection_pointer'), ('maximum', 'maximize_pointer')):
            if con.execute('SELECT value FROM '+database+'.contract').fetchone() != (digest_json(contract[key]['binding']),):
                raise ValueError('Metadata SQL has a different source binding.')
        reused = con.execute('SELECT count(*) FROM official o JOIN content c ON o.fetch=c.fetch WHERE c.existing IS NOT NULL').fetchone()[0]
        added = con.execute("SELECT count(*) FROM maximum.aliases a JOIN maximum.todo t USING(fetch) WHERE t.state='CHUNKED'").fetchone()[0]
        overlap = con.execute("""SELECT count(*) FROM maximum.aliases a JOIN maximum.todo t USING(fetch)
            JOIN official o ON o.id=a.id JOIN content c ON o.fetch=c.fetch
            WHERE t.state='CHUNKED' AND (c.existing IS NOT NULL OR o.url!=a.url)""").fetchone()[0]
        expected = reused+added
        if reused != contract['expected_reused'] or overlap or expected < minimum_ids:
            raise ValueError(f'Full scope audit failed: reused={reused}, added={added}, overlap={overlap}; required >= {minimum_ids}.')
        mappings = {r['mapping']: r['build_run'] for r in contract['sources'] if r['mapping']}
        proofs = dict(con.execute("SELECT json_extract(receipt,'$.mapping'),json_extract(receipt,'$.sha256') FROM completed WHERE kind='existing'"))
        references = {r['mapping']: r['mapping_sha256'] for r in contract['sources'] if r['mapping']}
        for label, in con.execute("SELECT DISTINCT json_extract(existing,'$.mapping') FROM content WHERE existing IS NOT NULL"):
            if label not in mappings or proofs.get(label) != references[label]:
                raise ValueError('Validated alias provenance is missing from full scope: '+str(label))
        names = {r['build_run'] for r in contract['sources']}
        phase = """CASE WHEN coalesce(json_extract(t.candidate,'$.recovery_root'),'')!='maximize'
            THEN 'metadata-text-v1' ELSE coalesce((SELECT m.phase FROM maximum.ready_members m
                WHERE m.fetch=a.fetch LIMIT 1),'raw-text-v1') END"""
        sql = """SELECT o.id,o.url,'existing',json_extract(c.existing,'$.mapping'),json_extract(c.existing,'$.original_doc_id')
            FROM official o JOIN content c ON o.fetch=c.fetch WHERE c.existing IS NOT NULL AND o.id>?
            UNION ALL SELECT a.id,a.url,'delta',"""+phase+""",a.id FROM maximum.aliases a JOIN maximum.todo t USING(fetch)
            JOIN official o ON o.id=a.id AND o.url=a.url JOIN content c ON o.fetch=c.fetch
            WHERE t.state='CHUNKED' AND c.existing IS NULL AND a.id>? ORDER BY 1"""
        progress_path = root / 'membership_progress.json'
        progress = checked(read_json(progress_path)) if progress_path.is_file() else seal({
            'contract': contract['manifest_sha256'], 'last_id': -1, 'rows': 0, 'parts': [], 'by_source': {}})
        if progress['contract'] != contract['manifest_sha256']:
            raise ValueError('Membership checkpoint belongs to another scope.')
        cursor = con.execute(sql, [progress['last_id'], progress['last_id']])
        while rows := cursor.fetchmany(8192):
            values = []
            for doc_id, url, kind, label, original in rows:
                source = mappings.get(label) if kind == 'existing' else label
                if source not in names or original is None:
                    raise ValueError('BTC ID lacks a frozen source: '+str(doc_id)+' | '+str(label))
                values.append({'official_id': doc_id, 'url': url, 'source': source, 'source_doc_id': original})
                progress['by_source'][source] = progress['by_source'].get(source, 0)+1
            with tempfile.TemporaryDirectory(dir=work) as temporary:
                path = Path(temporary)/'membership.parquet'
                pq.write_table(pa.Table.from_pylist(values, schema=MEMBERS), path, compression='zstd')
                relative = f"membership/part-{progress['rows']:09d}.parquet"
                checksum = publish_file(path, root / relative)
            progress['parts'].append({'path': relative, 'sha256': checksum, 'rows': len(values)})
            progress.update(last_id=rows[-1][0], rows=progress['rows']+len(values))
            progress = seal({k: v for k, v in progress.items() if k != 'manifest_sha256'})
            atomic_json(progress_path, progress)
            print('FULL_SCOPE membership:', f"{progress['rows']:,}/{expected:,}", '| checkpoint saved; no document/vector reads', flush=True)
        if progress['rows'] != expected or sum(progress['by_source'].values()) != expected:
            raise ValueError('Full-data membership is incomplete; no GPU export is allowed.')
        for source in contract['sources']:
            if not source['mapping'] and progress['by_source'].get(source['build_run'], 0) != source['documents']:
                raise ValueError('Frozen recovery documents are missing from BTC membership: '+source['build_run'])
        result = seal({'state': 'FULL_SCOPE_READY', 'contract': contract['manifest_sha256'],
            'official_ids': expected, 'official_total': contract['official_total'], 'reused_ids': reused, 'new_chunked_ids': added,
            'source_builds': len(contract['sources']), 'source_documents_sum': sum(r['documents'] for r in contract['sources']),
            'input_rows_before_cross_source_dedup': sum(r['unique_dense_inputs'] for r in contract['sources']),
            'membership': progress, 'scope': contract['scope'], 'submission_candidate_changed': False, 'updated_at': utc_now()})
        atomic_json(ready, result)
        print('FULL_SCOPE_READY:', expected, 'BTC IDs |', result['source_builds'], 'frozen sources', flush=True)
        return root, result
    finally:
        con.close()


def load_full_batch(project, batch_dir, *, minimum_ids=802028):
    project = Path(project).resolve()
    root = bound(project, batch_dir)
    if root.parent != project / PARENT:
        raise ValueError('Unexpected full-data batch directory.')
    contract, report = checked(read_json(root / 'source_contract.json')), checked(read_json(root / 'scope.json'))
    if (report.get('state') != 'FULL_SCOPE_READY' or report['contract'] != contract['manifest_sha256']
            or report['official_ids'] < minimum_ids):
        raise ValueError('Full-data BTC membership has not been verified; no partial fallback.')
    sources = []
    for record in contract['sources']:
        build = bound(project, record['build_dir'])
        inputs = checked(read_json(build / 'index_inputs/units.json'))
        if (inputs['state'] != 'COMPLETE' or inputs['manifest_sha256'] != record['index_inputs_manifest_sha256']
                or inputs['candidate_manifest_sha256'] != record['candidate_manifest_sha256']
                or inputs['input_count'] != record['unique_dense_inputs']):
            raise ValueError('Full-data frozen model inputs changed.')
        cursor = 0
        for part in inputs['parts']:
            if part['start'] != cursor or part['rows'] != min(4096, inputs['input_count']-cursor) or part['signature'] != inputs['signature']:
                raise ValueError('Full-data input part coverage changed.')
            bound(build / 'index_inputs', part['path'])
            sources.append({'ordinal': len(sources), 'build': record, 'inputs': inputs, 'part': part})
            cursor += part['rows']
        if cursor != inputs['input_count']:
            raise ValueError('Full-data input range is incomplete.')
    return root, seal({'state': 'FULL_VALIDATED_INPUTS_READY', 'scope_manifest_sha256': report['manifest_sha256'],
        'scope': contract['scope']+' Corpus child embeddings only; query/search/reranker/submission remain separate.',
        'source_builds': report['source_builds'], 'source_documents_sum': report['source_documents_sum'],
        'builds': contract['sources'], 'official_ids': report['official_ids'], 'input_parts': len(sources),
        'input_rows_before_cross_source_dedup': report['input_rows_before_cross_source_dedup']}), sources
