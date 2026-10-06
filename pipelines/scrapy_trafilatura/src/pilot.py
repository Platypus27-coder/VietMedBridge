"""Reviewed, explicitly bounded Stage B pilot. Never advances to 100k/full."""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.inventory.build_inventory import snapshot_dataset
from src.reports.build_report import review_gates
from src.storage.io import atomic_json, parquet_rows, write_parquet
from src.utils.config import output_path, prepare_directories
from src.utils.environment import environment_info
from src.utils.monitor import PilotMonitor, storage_measurement


def require_pilot_review(config: dict, stage_a_root: Path, size: int) -> dict:
    if not 0 < size <= 10000:
        raise ValueError('This pilot is capped at 10,000 URLs; it cannot run 100k/full.')
    if stage_a_root.resolve() == Path(config['_output_root']).resolve():
        raise ValueError('Stage B requires a separate output root to preserve Stage A.')
    report_root = stage_a_root/'reports/stage_a'
    review = json.loads((report_root/'review_decision.json').read_text(encoding='utf-8'))
    domains = json.loads((report_root/'coverage_decisions.json').read_text(encoding='utf-8'))['domains']
    report = json.loads((report_root/'report.json').read_text(encoding='utf-8'))
    integrity = json.loads((stage_a_root/'reports/integrity.json').read_text(encoding='utf-8'))
    gates = review_gates(review, domains, report['gold_coverage'].get('available', False), integrity.get('passed', False))
    deployment = config['environment']['deployment']
    if not gates['stage_b_ready'] or deployment.get('stage_b_authorized') is not True:
        raise ValueError(f'Stage B review/authorization incomplete: {gates}')
    if size > review['stage_b_pilot']['max_urls']:
        raise ValueError('Sample exceeds the approved pilot size.')
    if deployment['target'] != review['deployment']['target']:
        raise ValueError('Deployment target differs from the user decision.')
    config.setdefault('coverage', {})['deferred_domains'] = sorted(d['domain'] for d in domains if d['decision'] == 'defer_for_later')
    from src.utils.config import config_hash
    config['_config_sha256'] = config_hash(config)
    return {'review': review, 'domains': domains, 'snapshot_id': report['snapshot_id'], 'gates': gates}


def reuse_inventory(config: dict, stage_a_root: Path, snapshot_id: str):
    """Independent SQLite backup, immutable snapshots and Parquet copies."""
    manifest = snapshot_dataset(config)
    if manifest['snapshot_id'] != snapshot_id:
        raise ValueError('Stage A review belongs to a different dataset snapshot.')
    source_config = yaml.safe_load((stage_a_root/'checkpoints/effective_config.yaml').read_text(encoding='utf-8'))
    for section, keys in (('inventory', ('remove_fragments', 'remove_tracking_params')),
                          ('dataset', ('id_column', 'url_column', 'query_id_column', 'gold_doc_ids_column'))):
        if any(source_config[section][k] != config[section][k] for k in keys):
            raise ValueError('Inventory adapter/normalization changed; rebuild with a new Stage A.')
    checkpoint = output_path(config, 'checkpoints/inventory_state.json')
    if checkpoint.exists():
        if json.loads(checkpoint.read_text())['snapshot_id'] != snapshot_id:
            raise ValueError('Existing pilot inventory belongs to another snapshot.')
        return
    destination = output_path(config, 'data/inventory/inventory.sqlite')
    temporary = destination.with_suffix('.backup.sqlite')
    source_path = stage_a_root/'data/inventory/inventory.sqlite'
    with closing(sqlite3.connect(source_path.as_uri()+'?mode=ro', uri=True)) as source:
        if source.execute("SELECT value FROM meta WHERE key='snapshot_id'").fetchone()[0] != snapshot_id:
            raise ValueError('Source inventory snapshot mismatch.')
        with closing(sqlite3.connect(temporary)) as target:
            source.backup(target, pages=4096)
    temporary.replace(destination)
    for name in ('unique_urls', 'url_doc_map', 'invalid_urls', 'domain_stats', 'gold_doc_map'):
        target = output_path(config, f'data/inventory/{name}.parquet')
        tmp = target.with_suffix('.copy.tmp')
        shutil.copyfile(stage_a_root/f'data/inventory/{name}.parquet', tmp)
        tmp.replace(target)
    for name in ('gold_availability.json', 'inventory_state.json'):
        atomic_json(output_path(config, f'checkpoints/{name}'), json.loads((stage_a_root/f'checkpoints/{name}').read_text()))
    atomic_json(output_path(config, 'checkpoints/inventory_reuse.json'), {'source_root': str(stage_a_root),
                'snapshot_id': snapshot_id, 'method': 'independent_sqlite_backup_and_parquet_copy'})


def save_deferred_urls(config: dict, stage='stage_b') -> int:
    schema = pa.schema([(k, pa.string()) for k in ('crawl_url_id', 'fetch_url', 'domain', 'status', 'error_message', 'snapshot_id')]
                       + [('http_status', pa.int64()), ('attempt_no', pa.int64())])
    def rows():
        for row in parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet')):
            if row['status'] != 'SUCCESS_RAW':
                yield {name: row.get(name) for name in schema.names}
    count = write_parquet(output_path(config, f'reports/{stage}/deferred_urls.parquet'), rows(), schema)
    atomic_json(output_path(config, f'reports/{stage}/deferred_summary.json'), {
        'urls': count, 'policy': 'retain_for_later', 'known_deferred_domains': config.get('coverage', {}).get('deferred_domains', []),
        'automatic_archive_fetch': False, 'gold_coverage_decision': 'proceed_without_qrels'})
    return count


def run_stage_b(config: dict, stage_a_root: Path, size=10000):
    decision = require_pilot_review(config, stage_a_root, size)
    prepare_directories(config)
    info = environment_info(output_path(config, '.'))
    if info['disk_available_bytes'] < 10*1024**3:
        raise RuntimeError('Pilot needs at least 10 GiB free disk for independent inventory and outputs.')
    atomic_json(output_path(config, 'checkpoints/stage_b_authorization.json'), {
        **decision, 'stage_a_root': str(stage_a_root), 'max_urls_this_run': size, 'environment': info,
        'started_at': datetime.now(timezone.utc).isoformat(), 'advance_to_100k_authorized': False})
    atomic_json(output_path(config, 'reports/stage_b/coverage_decisions.json'), {'domains': decision['domains']})
    stages = {}
    from src.inventory.build_shards import build_shards, sample_inventory
    from src.crawler.run import run_crawl
    from src.extraction.worker import extract_raw
    from src.preprocessing.dedup import build_documents
    from src.reports.build_report import build_report
    from src.validation import validate_ingestion

    def perform(name, operation):
        print(f'Stage B: {name}', flush=True)
        monitor.stage = name
        start = time.monotonic()
        result = operation()
        stages[name] = {'elapsed_seconds': time.monotonic()-start, 'result': result}
        atomic_json(output_path(config, 'checkpoints/stage_b_progress.json'), {'stage': name, 'status': 'COMPLETE', 'stages': stages})
        return result

    with PilotMonitor(output_path(config, '.')) as monitor:
        perform('reuse_inventory', lambda: reuse_inventory(config, stage_a_root, decision['snapshot_id']))
        sample_checkpoint = output_path(config, 'checkpoints/sample.json')
        sample = output_path(config, 'data/inventory/stage_b_sample.parquet')
        if sample_checkpoint.exists() and sample.exists():
            existing = json.loads(sample_checkpoint.read_text())
            if existing['sample_size'] != size or existing['seed'] != config['dataset']['sample_seed']:
                raise ValueError('Pilot sample changed; choose a new output root.')
        else:
            perform('sample', lambda: str(sample_inventory(config, size, config['dataset']['sample_seed'], stage='stage_b')))
        for shard in build_shards(config, sample):
            perform('crawl_'+shard.stem, lambda: run_crawl(config, shard, retry_failed=False))
        perform('extraction', lambda: extract_raw(config))
        perform('documents', lambda: build_documents(config))
        perform('integrity', lambda: validate_ingestion(config))
        if not stages['integrity']['result']['passed']:
            raise RuntimeError('Pilot integrity failed; inspect reports/integrity.json.')
        perform('deferred_urls', lambda: save_deferred_urls(config))
        report = perform('report', lambda: build_report(config, stage='stage_b'))
    benchmark = {'scope': 'stage_b_10k_pilot', 'stages': stages, 'resources': monitor.summary(),
                 'storage': storage_measurement(output_path(config, '.')), 'advance_to_100k_authorized': False}
    # The report is already on disk; avoid repeating it inside the benchmark JSON.
    benchmark['stages']['report'].pop('result', None)
    atomic_json(output_path(config, 'reports/stage_b/benchmark.json'), benchmark)
    report['stage'] = 'stage_b'
    report['stage_a_acceptance'] = decision
    report['benchmark'] = {k: v for k, v in benchmark.items() if k != 'stages'}
    report['benchmark']['resources'] = {k: v for k, v in benchmark['resources'].items() if k != 'observations'}
    report['next_stage'] = {'100k_authorized': False, 'decision': 'review_10k_report_before_scaling'}
    atomic_json(output_path(config, 'reports/stage_b/report.json'), report)
    return report
