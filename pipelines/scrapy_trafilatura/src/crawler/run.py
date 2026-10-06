from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.storage.io import atomic_bytes, atomic_json, parquet_rows, write_parquet
from src.storage.manifests import EventWriter, compact_events
from src.storage.schemas import CRAWL_SCHEMA
from src.storage.state import StateStore
from src.storage.index import ManifestIndex
from src.utils.config import output_path
from src.utils.progress import activity, progress_callback

RETRYABLE = {'HTTP_429', 'HTTP_5XX', 'TIMEOUT', 'DNS_ERROR', 'CONNECTION_ERROR', 'RAW_WRITE_FAILED'}


def compact_crawl(config):
    return compact_events(output_path(config, 'data/manifests/crawl'), output_path(config, 'data/manifests/crawl_manifest.parquet'), CRAWL_SCHEMA)


def run_crawl(config: dict, shard: Path, *, stop_after: int | None = None, test_origins=None, retry_failed=True, _recovery=False) -> dict:
    """Subprocess isolates reactors and permits notebook/Windows restarts."""
    from src.utils.config import config_hash
    frontier_rows = pq.ParquetFile(shard).metadata.num_rows
    activity(config, 'crawl_prepare', inspected_urls=0, frontier_urls=frontier_rows)
    progress = progress_callback(config, 'crawl_prepare')
    config['_config_sha256'] = config_hash(config)
    compact_crawl(config)
    latest = ManifestIndex(output_path(config, 'checkpoints/crawl_index.sqlite'), output_path(config, 'data/manifests/crawl_manifest.parquet'))
    state = StateStore(output_path(config, 'checkpoints/crawl_state.sqlite'))
    pending = []
    deferred = []
    deferred_domains = set(config.get('coverage', {}).get('deferred_domains', []))
    active_path = output_path(config, f'checkpoints/{shard.stem}_active.json')
    active = json.loads(active_path.read_text()) if active_path.exists() else None
    resume = bool(active and active['status'] == 'STOPPED' and Path(active['batch']).exists()
                  and output_path(config, f'crawl_jobs/{Path(active["batch"]).stem}').is_dir())
    if resume:
        stats_path = output_path(config, f'reports/crawl_stats/{Path(active["batch"]).stem}.json')
        if stats_path.exists() and json.loads(stats_path.read_text()).get('finish_reason') == 'finished':
            resume = False  # Finished queues contain fingerprints but no pending work.
    try:
        for inspected, row in enumerate(parquet_rows(shard), 1):
            progress(inspected_urls=inspected, frontier_urls=frontier_rows)
            old = latest.get(row['crawl_url_id'])
            if old and old['status'] == 'SUCCESS_RAW':
                from .raw_store import RawStore
                RawStore(config).read(old, config['crawler']['max_response_bytes'])
                continue
            if old and not retry_failed:
                continue  # User's pilot policy retains completed failures for later.
            if row['domain'] in deferred_domains:
                if old is None:
                    attempt = state.allocate_attempt(row['crawl_url_id'])
                    snapshot = json.loads(output_path(config, 'data/source/dataset_manifest.json').read_text())
                    deferred.append({'crawl_url_id': row['crawl_url_id'], 'event_id': f'{row["crawl_url_id"]}:{attempt}:1',
                                     'attempt_no': attempt, 'event_seq': 1, 'run_id': 'deferred-policy',
                                     'snapshot_id': snapshot['snapshot_id'], 'config_sha256': config['_config_sha256'],
                                     'shard_id': shard.stem, 'fetch_url': row['fetch_url'], 'final_url': row['fetch_url'],
                                     'domain': row['domain'], 'status': 'DEFERRED_DOMAIN', 'attempt_count': 0,
                                     'elapsed_ms': 0.0, 'error_message': 'User deferred this domain for later handling.',
                                     'crawl_timestamp': datetime.now(timezone.utc).isoformat(), 'source_provenance': 'deferred_by_user'})
                continue
            redirect_pending = old and old['status'] == 'HTTP_OTHER' and old['http_status'] in (301, 302, 303, 307, 308)
            if old and (old['status'] not in RETRYABLE and not redirect_pending or old['attempt_no'] > config['crawler']['rate_limit']['max_batch_retries']):
                continue
            if not resume:
                row['attempt_no'] = state.allocate_attempt(row['crawl_url_id'], old['attempt_no'] if old else 0)
            pending.append(row)
    finally:
        state.close()
        latest.close()
    if deferred:
        writer = EventWriter(output_path(config, 'data/manifests/crawl'), config['storage']['jsonl_records_per_part'])
        try:
            for record in deferred:
                writer.append(record)
        finally:
            writer.close()
        compact_crawl(config)
    if not pending:
        return {'pending': 0, 'exit_code': 0}
    from src.utils.session_budget import stage_time_limit
    remaining = stage_time_limit(config, 'crawl', config['crawler'].get('session_time_limit_seconds', 0))
    if remaining is not None and remaining <= 0:
        atomic_json(output_path(config, 'checkpoints/run_state.json'), {
            'stage': 'crawl', 'shard': shard.stem, 'status': 'STOPPED',
            'completed_this_run': 0, 'counts_this_run': {}, 'scheduled_this_run': len(pending),
            'finish_reason': 'session_deadline'})
        activity(config, 'crawl', scheduled_this_run=len(pending), finish_reason='session_deadline')
        return {'pending': len(pending), 'exit_code': 0, 'stopped': True, 'finish_reason': 'session_deadline'}
    if resume:
        batch = Path(active['batch'])
    else:
        identity = hashlib.sha256(json.dumps([(r['crawl_url_id'], r['attempt_no']) for r in pending]).encode()).hexdigest()[:16]
        batch = output_path(config, f'data/crawl_shards/{shard.stem}-{identity}.parquet')
        schema = pq.ParquetFile(shard).schema_arrow.append(pa.field('attempt_no', pa.int64()))
        write_parquet(batch, pending, schema)
    atomic_json(active_path, {'status': 'STOPPED', 'batch': str(batch)})
    effective = output_path(config, 'checkpoints/effective_config.yaml')
    values = {k: v for k, v in config.items() if not k.startswith('_')}
    values['environment'] = dict(values['environment'], output_dir=config['_output_root'])
    for key in ('links_path', 'query_path'):
        from src.utils.config import input_path
        values['dataset'][key] = str(input_path(config, key))
    atomic_bytes(effective, yaml.safe_dump(values, allow_unicode=True, sort_keys=False).encode())
    command = [sys.executable, '-m', 'src.crawler.entry', '--config', str(effective), '--shard', str(batch)]
    if stop_after:
        command.extend(['--stop-after', str(stop_after)])
    if test_origins:
        # Internal harness only. entry has no public CLI flag to disable safety.
        command = [sys.executable, '-m', 'tests.integration.crawl_entry', '--config', str(effective), '--shard', str(batch), '--origins', json.dumps(list(test_origins))]
        if stop_after:
            command.extend(['--stop-after', str(stop_after)])
    atomic_json(output_path(config, 'checkpoints/run_state.json'), {
        'stage': 'crawl', 'shard': shard.stem, 'status': 'RUNNING',
        'completed_this_run': 0, 'counts_this_run': {}, 'scheduled_this_run': len(pending)})
    if config.get('kaggle_batch'):
        atomic_json(output_path(config, 'checkpoints/crawl_heartbeat.json'), {})
    activity(config, 'crawl', scheduled_this_run=len(pending))
    result = subprocess.run(command, cwd=config['_project_root'], check=False)
    activity(config, 'crawl_compaction')
    compact_crawl(config)
    committed = json.loads(output_path(config, 'checkpoints/run_state.json').read_text(encoding='utf-8'))
    if result.returncode:
        atomic_json(output_path(config, 'checkpoints/run_state.json'), {'stage': 'crawl', 'shard': shard.stem,
                    'status': 'FAILED', 'error_message': f'Crawler subprocess exited {result.returncode}.'})
        raise RuntimeError(f'Crawler subprocess failed ({result.returncode}); manifest recovery remains available.')
    stats_path = output_path(config, f'reports/crawl_stats/{batch.stem}.json')
    finish_reason = json.loads(stats_path.read_text()).get('finish_reason') if stats_path.exists() else None
    if stop_after or finish_reason and finish_reason != 'finished':
        # A session budget is a pause, not a finished frontier. Do not start
        # another recovery subprocess that would consume the budget again.
        atomic_json(active_path, {'status': 'STOPPED', 'batch': str(batch)})
        atomic_json(output_path(config, 'checkpoints/run_state.json'), {
            **committed,
            'stage': 'crawl', 'shard': shard.stem, 'status': 'STOPPED',
            'finish_reason': finish_reason, 'scheduled_this_run': len(pending)})
        return {'pending': len(pending), 'exit_code': result.returncode,
                'stopped': True, 'finish_reason': finish_reason}
    if not stop_after:
        atomic_json(active_path, {'status': 'COMPLETE', 'batch': str(batch)})
    # A hard crash can lose requests removed from Scrapy's disk queue before
    # their outcome was committed. Reconcile the frontier against the durable
    # manifest and use a fresh JOBDIR only for genuinely missing outcomes.
    current = ManifestIndex(output_path(config, 'checkpoints/crawl_index.sqlite'), output_path(config, 'data/manifests/crawl_manifest.parquet'))
    try:
        missing = [r for r in parquet_rows(shard) if current.get(r['crawl_url_id']) is None]
    finally:
        current.close()
    if missing and not stop_after:
        if _recovery:
            atomic_json(output_path(config, 'checkpoints/run_state.json'), {'stage': 'crawl', 'shard': shard.stem,
                        'status': 'FAILED', 'missing_outcomes': len(missing)})
            raise RuntimeError(f'{len(missing)} frontier URLs still lack outcomes after recovery; inspect crawler logs.')
        atomic_json(active_path, {'status': 'COMPLETE', 'batch': str(batch)})
        recovery = output_path(config, f'data/crawl_shards/{shard.stem}-recovery.parquet')
        write_parquet(recovery, missing, pq.ParquetFile(shard).schema_arrow)
        run_crawl(config, recovery, test_origins=test_origins, retry_failed=retry_failed, _recovery=True)
    atomic_json(active_path, {'status': 'STOPPED' if stop_after else 'COMPLETE', 'batch': str(batch)})
    atomic_json(output_path(config, 'checkpoints/run_state.json'), {**committed, 'stage': 'crawl', 'shard': shard.stem, 'status': 'STOPPED' if stop_after else 'COMPLETE', 'scheduled_this_run': len(pending)})
    return {'pending': len(pending), 'exit_code': result.returncode}
