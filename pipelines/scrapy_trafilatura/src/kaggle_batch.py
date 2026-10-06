"""Run one frozen frontier per Kaggle session; global ID mapping stays in inputs."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq

from src.crawler.raw_store import RawStore
from src.crawler.run import run_crawl
from src.extraction.worker import extract_raw
from src.storage.index import ManifestIndex
from src.storage.io import atomic_json, parquet_rows, sha256_file, write_parquet
from src.storage.session_bundle import make_bundle, restore_bundle
from src.utils.config import output_path, prepare_directories
from src.utils.progress import activity, progress_callback
from src.utils.session_budget import stage_time_limit


@contextmanager
def single_writer(root):
    path = root/'checkpoints/kaggle_writer.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if __import__('os').name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if __import__('os').name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def summarize(config, frontier):
    manifest = output_path(config, 'data/manifests/crawl_manifest.parquet')
    statuses, domains, checked = Counter(), defaultdict(Counter), set()
    rows = urls_with_raw = wire_bytes = 0
    frontier_rows = pq.ParquetFile(frontier).metadata.num_rows
    progress = progress_callback(config, 'validation')
    latest = ManifestIndex(output_path(config, 'checkpoints/crawl_index.sqlite'), manifest)
    try:
        for row in parquet_rows(frontier):
            rows += 1
            outcome = latest.get(row['crawl_url_id'])
            status = outcome['status'] if outcome else 'PENDING'
            statuses[status] += 1
            domains[row['domain']][status] += 1
            if outcome and outcome['status'] == 'SUCCESS_RAW':
                if outcome['snapshot_id'] != config['kaggle_batch']['snapshot_id']:
                    raise ValueError('Crawl outcome belongs to a different dataset snapshot.')
                urls_with_raw += 1
                wire_bytes += outcome['raw_size_bytes']
                if outcome['raw_sha256'] not in checked:
                    RawStore(config).read(outcome, config['crawler']['max_response_bytes'])
                    checked.add(outcome['raw_sha256'])
            progress(validated_urls=rows, verified_raw_files=len(checked), frontier_urls=frontier_rows)
    finally:
        latest.close()
    if pq.ParquetFile(manifest).metadata.num_rows != rows-statuses['PENDING']:
        raise ValueError('Batch manifest contains missing/foreign frontier outcomes.')
    extracted = output_path(config, 'data/manifests/extraction_manifest.parquet')
    extraction_counts = Counter(r['extract_status'] for r in parquet_rows(extracted)) if extracted.exists() else Counter()
    return {'frontier_urls': rows, 'crawl_counts': dict(statuses),
            'crawl_remaining': statuses['PENDING'], 'successful_raw_urls': urls_with_raw,
            'unique_raw_checksums_verified': len(checked), 'raw_size_bytes': wire_bytes,
            'extraction_counts': dict(extraction_counts),
            'extraction_remaining': urls_with_raw-sum(extraction_counts.values()),
            'domains': {d: dict(c) for d, c in sorted(domains.items())}, 'integrity_passed': True}


def run_kaggle_batch(config, campaign_root: Path, shard_id=0, *, restore=None,
                     compact_output=False, test_origins=None, stop_after=None):
    started = time.monotonic()
    manifest = json.loads((campaign_root/'campaign_manifest.json').read_text(encoding='utf-8'))
    if not 0 <= shard_id < manifest['shard_count']:
        raise ValueError('Shard ID is outside the frozen campaign.')
    item = manifest['shards'][shard_id]
    source = campaign_root/item['name']
    if source.name != f'shard_{shard_id:05d}.parquet' or item['rows'] > 100000:
        raise ValueError('Invalid/oversized Kaggle frontier.')
    if sha256_file(source) != item['sha256']:
        raise ValueError('Frozen frontier checksum mismatch.')
    identity = {k: v for k, v in item.items()}
    identity['snapshot_id'] = manifest['snapshot_id']
    root = Path(config['_output_root']).resolve()
    if compact_output and (root.parent != Path('/kaggle/working') or root.name != 'vibiomir_batch'):
        raise ValueError('Compaction is restricted to the owned Kaggle batch directory.')
    if restore is not None:
        restore_bundle(Path(restore), root)
    checkpoint = root/'checkpoints/kaggle_batch_identity.json'
    if root.exists() and any(root.iterdir()) and not checkpoint.exists():
        raise ValueError('Existing session directory is not owned by this campaign.')
    prepare_directories(config)
    if checkpoint.exists() and json.loads(checkpoint.read_text(encoding='utf-8')) != identity:
        raise ValueError('Output/restore belongs to another shard or dataset snapshot.')
    with single_writer(root):
        atomic_json(checkpoint, identity)
        config.setdefault('kaggle_batch', {})['snapshot_id'] = manifest['snapshot_id']
        atomic_json(output_path(config, 'data/source/dataset_manifest.json'),
                    json.loads((campaign_root/'dataset_manifest.json').read_text(encoding='utf-8')))
        frontier = output_path(config, f'data/crawl_shards/{source.name}')
        if not frontier.exists():
            shutil.copyfile(source, frontier)
        elif sha256_file(frontier) != item['sha256']:
            raise ValueError('Restored frontier differs from campaign.')
        crawl_started = time.monotonic()
        atomic_json(root/'checkpoints/kaggle_session_state.json', {'status': 'RUNNING', 'stage': 'crawl', **identity})
        crawl = run_crawl(config, frontier, retry_failed=False, test_origins=test_origins, stop_after=stop_after)
        crawl_elapsed = time.monotonic()-crawl_started
        extraction = None
        if config['kaggle_batch'].get('extract', True):
            atomic_json(root/'checkpoints/kaggle_session_state.json', {'status': 'RUNNING', 'stage': 'extraction', **identity})
            atomic_json(root/'checkpoints/extraction_state.json', {'committed_this_run': 0})
            activity(config, 'extraction')
            limits = config['kaggle_batch']
            remaining = stage_time_limit(config, 'extraction', limits.get('extraction_time_limit_seconds', 0))
            if remaining is not None and remaining <= 0:
                extraction = {'processed_this_run': 0, 'stopped_by_time_budget': True,
                              'stopped_by_output_budget': False}
            else:
                limits['extraction_time_limit_seconds'] = remaining or 0
                extraction = extract_raw(config)
        activity(config, 'validation')
        report = summarize(config, frontier)
        extract_pending = config['kaggle_batch'].get('extract', True) and report['extraction_remaining']
        report.update(status='PARTIAL' if report['crawl_remaining'] or extract_pending else 'COMPLETED',
                      snapshot_id=manifest['snapshot_id'], shard_id=shard_id,
                      campaign_shards=manifest['shard_count'], campaign_unique_urls=manifest['unique_urls'],
                      crawl_elapsed_seconds=crawl_elapsed, elapsed_seconds=time.monotonic()-started,
                      crawler_finish_reason=crawl.get('finish_reason', 'finished'),
                      extraction_run=extraction,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      gold_coverage_available=False, failure_policy='retain_for_later',
                      global_native_id_mapping='campaign input url_doc_map.parquet',
                      global_dedup='after_all_shards', raw_retained=True)
        if config['kaggle_batch'].get('session_deadline_unix') is not None:
            report['session_budget'] = {key: config['kaggle_batch'].get(key, 0) for key in (
                'session_deadline_unix', 'extraction_reserve_seconds', 'export_reserve_seconds')}
        atomic_json(root/'reports/kaggle_batch_report.json', report)
        atomic_json(root/'checkpoints/kaggle_session_state.json', {'status': report['status'], 'stage': 'bundling', **identity})
        activity(config, 'bundling')
        bundle_progress = {}
        def update_bundle(stage, **metrics):
            if stage not in bundle_progress:
                bundle_progress[stage] = progress_callback(config, stage)
            bundle_progress[stage](**metrics)
        bundle_started = time.monotonic()
        bundle = make_bundle(root, root.parent/f'vibiomir_shard_{shard_id:05d}.tar', progress=update_bundle)
        report['bundle'] = bundle
        report['bundle_seconds'] = time.monotonic()-bundle_started
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(root.parent/f'vibiomir_shard_{shard_id:05d}_report.json', report)
        activity(config, 'finished', shard_status=report['status'],
                 crawl_remaining=report['crawl_remaining'], extraction_remaining=report['extraction_remaining'])
    if compact_output:
        # All produced bytes have a verified durable copy in the sibling tar.
        # Never remove source inputs, other working directories or unverified raw.
        if root.resolve().parent != Path('/kaggle/working') or json.loads(checkpoint.read_text()) != identity:
            raise ValueError('Owned directory changed before compaction.')
        shutil.rmtree(root)
    return report
