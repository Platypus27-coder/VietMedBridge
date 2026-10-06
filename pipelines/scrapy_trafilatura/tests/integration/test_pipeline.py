import json
import shutil
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.inventory.build_inventory import build_inventory
from src.inventory.build_shards import build_shards, sample_inventory
from src.crawler.run import run_crawl
from src.extraction.worker import extract_raw
from src.preprocessing.dedup import build_documents
from src.reports.build_report import build_report
from src.storage.io import parquet_rows
from src.utils.config import output_path
from src.validation import validate_ingestion
from .server import mock_server


def make_source(config, tmp_path, urls):
    links, query = tmp_path/'links.parquet', tmp_path/'query.parquet'
    ids = [f'btc-{i}' for i in range(len(urls))]
    pq.write_table(pa.table({'id': ids, 'url': urls}), links)
    pq.write_table(pa.table({'query_id': ['q1'], 'gold_doc_ids': [[ids[0], ids[-1], 'missing-btc-id']]}), query)
    config['dataset'].update(links_path=str(links), query_path=str(query))
    build_inventory(config)
    sample = sample_inventory(config, len(urls), 42)
    return build_shards(config, sample)[0]


def test_end_to_end_errors_retry_robots_and_integrity(config, tmp_path):
    with mock_server() as (base, counts):
        paths = ['/ok-html', '/redirect', '/not-found', '/forbidden', '/rate-limit', '/server-error',
                 '/slow', '/pdf', '/empty', '/large', '/robots-blocked', '/wrong-header-html']
        shard = make_source(config, tmp_path, [base+p for p in paths] + [base+'/ok-html#duplicate', ''])
        origin = {('http', '127.0.0.1', urlsplit(base).port)}
        run_crawl(config, shard, test_origins=origin)
        status = {r['fetch_url'].removeprefix(base): r['status'] for r in parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet'))}
        assert status['/ok-html'] == 'SUCCESS_RAW'
        assert status['/redirect'] == 'SUCCESS_RAW'
        redirected = next(r for r in parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet'))
                          if r['fetch_url'] == base+'/redirect')
        assert redirected['final_url'] == base+'/ok-html' and redirected['redirect_count'] == 1
        assert counts['/ok-html'] == 2
        assert status['/robots-blocked'] == 'ROBOTS_DENIED'
        assert counts['/robots-blocked'] == 0
        assert status['/large'] == 'RESPONSE_TOO_LARGE'
        assert status['/slow'] == 'TIMEOUT'
        assert status['/server-error'] == 'HTTP_5XX' and counts['/server-error'] == 2
        assert status['/rate-limit'] == 'HTTP_429'
        import time
        time.sleep(1.1)
        run_crawl(config, shard, test_origins=origin)
        status = {r['fetch_url'].removeprefix(base): r['status'] for r in parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet'))}
        assert status['/rate-limit'] == 'SUCCESS_RAW'
    extract_raw(config)
    build_documents(config)
    # An earlier failed pilot may have created a header-only review sheet.
    review = output_path(config, 'reports/stage_a/manual_review.csv')
    review.write_text('crawl_url_id,url,notes\n', encoding='utf-8')
    report = build_report(config)
    assert report['gold_coverage']['missing_from_inventory'] == 1
    assert report['gold_coverage']['invalid_url_doc_ids'] == 1
    assert validate_ingestion(config)['passed']
    assert output_path(config, 'reports/stage_a/manual_review.csv').exists()
    import csv
    with review.open(encoding='utf-8', newline='') as handle:
        manual = list(csv.DictReader(handle))
    assert manual and all(r['content_ok'] == '' for r in manual)
    assert len(list(parquet_rows(output_path(config, 'data/mappings/doc_outcomes.parquet')))) == 14


def test_redirect_cycles_have_terminal_records_and_resume_keeps_completed_failures(config, tmp_path):
    with mock_server() as (base, counts):
        paths = ['/redirect-self', '/redirect-a', '/refresh-self', '/redirect', '/server-error']
        shard = make_source(config, tmp_path, [base+p for p in paths])
        origin = {('http', '127.0.0.1', urlsplit(base).port)}
        run_crawl(config, shard, test_origins=origin, retry_failed=False)
        records = {r['fetch_url'].removeprefix(base): r for r in parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet'))}
        assert len(records) == len(paths)
        for path in ('/redirect-self', '/redirect-a', '/refresh-self'):
            assert records[path]['status'] == 'REDIRECT_LOOP'
            assert counts[path] == 1
        assert counts['/redirect-b'] == 1
        assert records['/redirect-self']['http_status'] == 302
        assert records['/redirect']['status'] == 'SUCCESS_RAW'
        before = dict(counts)
        assert run_crawl(config, shard, test_origins=origin, retry_failed=False)['pending'] == 0
        assert dict(counts) == before


@pytest.mark.parametrize('loss', ['none', 'jobdir', 'queue'])
def test_jobdir_resume_and_manifest_recovery(config, tmp_path, loss):
    config['crawler']['concurrent_requests'] = 1
    config['crawler']['concurrent_requests_per_domain'] = 1
    with mock_server() as (base, counts):
        shard = make_source(config, tmp_path, [f'{base}/document-{i}' for i in range(100)])
        origin = {('http', '127.0.0.1', urlsplit(base).port)}
        run_crawl(config, shard, stop_after=35, test_origins=origin)
        first = list(parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet')))
        # Closing drains responses already received by the scraper; Scrapy's
        # CLOSESPIDER_ITEMCOUNT is a trigger, not an exact response cap.
        assert 35 <= len(first) < 100
        active = json.loads(output_path(config, f'checkpoints/{shard.stem}_active.json').read_text())
        job_name = __import__('pathlib').Path(active['batch']).stem
        jobdir = output_path(config, f'crawl_jobs/{job_name}')
        assert jobdir.exists()
        if loss == 'jobdir':
            shutil.rmtree(jobdir)
        elif loss == 'queue':
            # Simulate a lost in-flight/queued request while retaining seen
            # fingerprints; manifest reconciliation must recover it.
            shutil.rmtree(jobdir/'requests.queue')
        run_crawl(config, shard, test_origins=origin)
        assert len(list(parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet')))) == 100
        assert all(counts[f'/document-{i}'] == 1 for i in range(100))
        # JOBDIR may disappear; all completed outcomes are reconstructed from manifest.
        shutil.rmtree(output_path(config, 'crawl_jobs'))
        assert run_crawl(config, shard, test_origins=origin)['pending'] == 0
    assert len(list(output_path(config, 'data/raw').rglob('*.zst'))) == 1
