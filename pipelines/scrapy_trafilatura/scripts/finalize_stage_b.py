"""Measure packing/local copying and retain the user's deferred-domain policy."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crawler.raw_store import RawStore, pack_raw
from src.storage.io import atomic_json, parquet_rows, sha256_file, write_parquet
from src.utils.config import load_config, output_path


def finalize(config):
    import pyarrow as pa
    target = output_path(config, 'reports/stage_b')
    report_path = target/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    integrity = json.loads(output_path(config, 'reports/integrity.json').read_text())
    if not integrity['passed'] or not report.get('next_stage'):
        raise RuntimeError('Complete the bounded Stage B runner and integrity checks first.')
    decisions_path = target/'coverage_decisions.json'
    decisions = json.loads(decisions_path.read_text())
    for row in decisions['domains']:
        if row['decision'] == 'pending_review' and config['coverage'].get('inaccessible_policy') == 'defer_for_later':
            row.update(decision='defer_for_later', decided_by='user_policy', decided_at=datetime.now(timezone.utc).isoformat(),
                       notes='User chose to retain inaccessible URLs for later; individual successful URLs remain usable.')
    atomic_json(decisions_path, decisions)
    report['gates']['blocked_domains_decision_complete'] = all(r['decision'] != 'pending_review' for r in decisions['domains'])
    report['gates']['stage_b_completed'] = True
    records = list(parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet')))
    times = defaultdict(list)
    counts = defaultdict(int)
    for row in records:
        if row['status'] != 'DEFERRED_DOMAIN':
            times[row['domain']].append(datetime.fromisoformat(row['crawl_timestamp']).timestamp())
        counts[row['domain']] += row['status'] == 'SUCCESS_RAW'
    with sqlite3.connect(output_path(config, 'data/inventory/inventory.sqlite')) as database:
        estimates = []
        for domain, observed in sorted(times.items()):
            span = max(observed)-min(observed)
            rate = (len(observed)-1)/span if span > 0 else None
            projected = database.execute('SELECT COUNT(*) FROM urls WHERE domain=?', (domain,)).fetchone()[0]
            estimates.append({'domain': domain, 'terminal_urls': len(observed), 'success_raw': counts[domain],
                              'observed_completion_span_seconds': span, 'observed_urls_per_second': rate,
                              'projected_hours_at_observed_rate': projected/rate/3600 if rate else None})
    write_parquet(target/'observed_domain_throughput.parquet', estimates, pa.schema([
        ('domain', pa.string()), ('terminal_urls', pa.int64()), ('success_raw', pa.int64()),
        ('observed_completion_span_seconds', pa.float64()), ('observed_urls_per_second', pa.float64()),
        ('projected_hours_at_observed_rate', pa.float64())]))
    raw_root = output_path(config, 'data/raw')
    loose = [p for p in raw_root.rglob('*') if p.is_file() and 'packed' not in p.relative_to(raw_root).parts]
    start = time.monotonic()
    packed_manifest = pack_raw(config, output_path(config, 'data/manifests/crawl_manifest.parquet'))
    packing_seconds = time.monotonic()-start
    checked = raw_bytes = 0
    start = time.monotonic()
    for row in parquet_rows(packed_manifest):
        if row['status'] == 'SUCCESS_RAW':
            raw_bytes += len(RawStore(config).read(row, config['crawler']['max_response_bytes']))
            checked += 1
    verify_seconds = time.monotonic()-start
    containers = list((raw_root/'packed').glob('*.tar'))
    copy_start = time.monotonic()
    copied_bytes = 0
    for container in containers:
        copied = container.with_suffix('.copy-benchmark.tmp')
        if not copied.resolve().is_relative_to(raw_root.resolve()):
            raise ValueError('Copy benchmark target escaped raw root.')
        shutil.copyfile(container, copied)
        if sha256_file(copied) != sha256_file(container):
            raise ValueError('Local container copy checksum mismatch.')
        copied_bytes += copied.stat().st_size
        copied.unlink()
    copy_seconds = time.monotonic()-copy_start
    measurement = {'loose_raw_files': len(loose), 'loose_raw_bytes': sum(p.stat().st_size for p in loose),
                   'containers': len(containers), 'container_bytes': sum(p.stat().st_size for p in containers),
                   'packing_seconds': packing_seconds, 'packed_raw_checksums_checked': checked,
                   'verified_raw_bytes': raw_bytes, 'packed_verification_seconds': verify_seconds,
                   'local_copy_including_checksum_seconds': copy_seconds, 'local_copy_bytes': copied_bytes,
                   'external_transfer_measured': False, 'original_loose_files_retained': True,
                   'full_scale_packer_note': 'Metadata grouping uses RAM; use disk spool/index before packing millions of files.'}
    atomic_json(target/'packing_benchmark.json', measurement)
    stats = [json.loads(p.read_text()) for p in output_path(config, 'reports/crawl_stats').glob('*.json')]
    traffic = {'stats_files': len(stats), 'scrapy_response_bytes': sum(r.get('downloader/response_bytes', 0) for r in stats) if stats else None,
               'request_count_including_robots_redirects_retries': sum(r.get('downloader/request_count', 0) for r in stats) if stats else None,
               'note': 'Scrapy response accounting; actual wire traffic, TLS/DNS overhead and external transfer are not measured.'}
    report['packing_benchmark'] = measurement
    report['network_accounting'] = traffic
    report['throughput_note'] = 'Observed domain rates share the pilot downloader and stratified sample; projections are scenarios, not full-corpus forecasts.'
    report['finalized_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(report_path, report)
    status_path = Path(config['_project_root'])/'IMPLEMENTATION_STATUS.md'
    if status_path.exists():
        marker = '## Stage B Windows 10k — measured result'
        previous = status_path.read_text(encoding='utf-8').split(marker)[0].rstrip()
        previous = previous.replace('## Stage B Windows 10k — đang chạy theo quyết định mới', '## Stage B Windows 10k — quyết định và vận hành')
        previous = previous.replace('Pilot **10.000 URL** đang chạy tại', 'Pilot **10.000 URL** đã hoàn tất tại')
        previous = previous.replace('Detached supervisor đang theo dõi runner hiện tại, tối đa hai lần phục hồi nếu runner lỗi.',
                                    'Detached supervisor đã xử lý runner, tối đa hai lần phục hồi nếu runner lỗi.')
        overview = (f'\n\n{marker}\n\n'
                    f'Completed at `{report["finalized_at"]}`. Sample: **{report["unique_urls"]:,} URLs**. '
                    f'Crawl outcomes: `{json.dumps(report["crawl"], ensure_ascii=False)}`. '
                    f'Extraction success: **{report["extraction"]["success"]}**; quality: `{json.dumps(report["quality"])}`.\n\n'
                    f'Integrity: **passed**; all packed raw checksums verified: **{checked}**. '
                    f'Packing: {packing_seconds:.2f}s; local copy including checksum: {copy_seconds:.2f}s. '
                    'External transfer and gold coverage remain unmeasured. 100k/full remains unauthorized. '
                    'Detailed evidence: `outputs/stage_b_windows_10k/reports/stage_b/report.json`, '
                    '`benchmark.json`, `packing_benchmark.json`, `observed_domain_throughput.parquet`, '
                    '`deferred_urls.parquet`, and `reports/integrity.json`.\n')
        status_path.write_text(previous+overview, encoding='utf-8')
    return {'packing': measurement, 'network_accounting': traffic}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='outputs/stage_b_windows_10k/checkpoints/effective_config.yaml')
    parser.add_argument('--output-dir', default='outputs/stage_b_windows_10k')
    args = parser.parse_args()
    print(json.dumps(finalize(load_config(args.config, args.output_dir)), indent=2))
