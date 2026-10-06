import copy
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import pyarrow.parquet as pq

from scripts.prepare_kaggle_campaign import prepare_campaign
from src.kaggle_batch import run_kaggle_batch
from src.storage.io import parquet_rows
from src.storage.io import write_parquet
from src.storage.schemas import CRAWL_SCHEMA
from src.crawler.raw_store import RawStore
from src.extraction.worker import extract_raw
from src.storage.session_bundle import make_bundle, restore_bundle
from tests.integration.server import mock_server
from tests.integration.server import HTML
from tests.integration.test_pipeline import make_source


def test_campaign_partition_preserves_urls_and_native_ids(config, tmp_path):
    make_source(config, tmp_path, [f'https://example.com/doc-{i}' for i in range(7)])
    source = Path(config['_output_root'])
    destination = tmp_path/'campaign'
    campaign = prepare_campaign(source, destination, urls_per_shard=3)
    assert [s['rows'] for s in campaign['shards']] == [3, 3, 1]
    expected = {r['crawl_url_id'] for r in parquet_rows(source/'data/inventory/unique_urls.parquet')}
    exported = [r['crawl_url_id'] for s in campaign['shards'] for r in parquet_rows(destination/s['name'])]
    assert set(exported) == expected and len(exported) == len(expected)
    assert pq.read_table(destination/'url_doc_map.parquet').equals(pq.read_table(source/'data/inventory/url_doc_map.parquet'))
    assert prepare_campaign(source, destination, 3) == campaign
    (destination/campaign['shards'][0]['name']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        prepare_campaign(source, destination, 3)


def test_bundle_roundtrip_and_tamper_detection(tmp_path):
    source = tmp_path/'source'
    (source/'data/raw/aa').mkdir(parents=True)
    (source/'data/raw/aa/body.zst').write_bytes(b'raw-content')
    (source/'checkpoints').mkdir()
    (source/'checkpoints/state.json').write_text('{"status":"PARTIAL"}')
    (source/'crawl_jobs').mkdir()
    (source/'crawl_jobs/queue').write_bytes(b'reconstructed-from-manifest')
    archive = tmp_path/'session.tar'
    bundle = make_bundle(source, archive)
    assert bundle['verified'] and bundle['files'] == 2
    restored = tmp_path/'restored'
    restore_bundle(archive, restored)
    assert (restored/'data/raw/aa/body.zst').read_bytes() == b'raw-content'
    assert not (restored/'crawl_jobs').exists()
    with archive.open('r+b') as handle:
        handle.seek(1024)
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='SHA256'):
        restore_bundle(archive, tmp_path/'broken')


def test_timed_batch_restore_skips_committed_urls(config, tmp_path):
    config['crawler'].update(concurrent_requests=1, concurrent_requests_per_domain=1)
    with mock_server() as (base, counts):
        make_source(config, tmp_path, [f'{base}/document-{i}' for i in range(100)])
        campaign_root = tmp_path/'campaign'
        prepare_campaign(Path(config['_output_root']), campaign_root)
        runtime = copy.deepcopy(config)
        runtime['_output_root'] = str(tmp_path/'session1')
        runtime['crawler']['session_time_limit_seconds'] = 0.15
        runtime['kaggle_batch'] = {'extract': False}
        origin = {('http', '127.0.0.1', urlsplit(base).port)}
        first = run_kaggle_batch(runtime, campaign_root, test_origins=origin)
        assert first['status'] == 'PARTIAL' and 0 < first['successful_raw_urls'] < 100
        assert first['bundle']['verified']
        live = json.loads((Path(runtime['_output_root'])/'checkpoints/run_state.json').read_text())
        assert live['completed_this_run'] > 0
        assert sum(live['counts_this_run'].values()) == live['completed_this_run']
        assert live['counts_this_run']['SUCCESS_RAW'] == first['successful_raw_urls']
        completed = {r['fetch_url'].removeprefix(base) for r in parquet_rows(Path(runtime['_output_root'])/'data/manifests/crawl_manifest.parquet')}
        before = {p: counts[p] for p in completed}
        resumed = copy.deepcopy(runtime)
        resumed['_output_root'] = str(tmp_path/'session2')
        resumed['crawler']['session_time_limit_seconds'] = 0
        second = run_kaggle_batch(resumed, campaign_root, restore=Path(first['bundle']['archive']), test_origins=origin)
        assert second['status'] == 'COMPLETED' and second['crawl_remaining'] == 0
        assert second['successful_raw_urls'] == second['unique_raw_checksums_verified']*100
        assert {p: counts[p] for p in completed} == before
        assert second['frontier_urls'] == 100


def test_extraction_output_budget_retains_pending_raw_and_resumes(config):
    config['extraction']['workers'] = 1
    config['kaggle_batch'] = {'max_extraction_event_bytes': 1}
    root = Path(config['_output_root'])
    rows = []
    for i in range(2):
        rows.append({'crawl_url_id': f'url-{i}', 'event_id': f'url-{i}:1:1',
            'attempt_no': 1, 'event_seq': 1, 'status': 'SUCCESS_RAW', 'snapshot_id': 'fixture',
            'effective_content_type': 'text/html', 'final_url': f'https://example.com/doc-{i}',
            **RawStore(config).write(HTML, 'text/html')})
    write_parquet(root/'data/manifests/crawl_manifest.parquet', rows, CRAWL_SCHEMA)
    first = extract_raw(config)
    assert first['processed_this_run'] == 1 and first['stopped_by_output_budget']
    first_row = next(parquet_rows(root/'data/manifests/extraction_manifest.parquet'))
    config['kaggle_batch']['max_extraction_event_bytes'] = 0
    second = extract_raw(config)
    assert second['processed_this_run'] == 1
    results = list(parquet_rows(root/'data/manifests/extraction_manifest.parquet'))
    assert len(results) == 2 and all(r['extract_status'] == 'EXTRACT_SUCCESS' for r in results)
    assert next(r for r in results if r['crawl_url_id'] == first_row['crawl_url_id']) == first_row


def test_batch_rejects_unowned_existing_output(config, tmp_path):
    make_source(config, tmp_path, ['https://example.com/article'])
    campaign_root = tmp_path/'campaign'
    prepare_campaign(Path(config['_output_root']), campaign_root)
    runtime = copy.deepcopy(config)
    root = tmp_path/'unowned'
    root.mkdir()
    personal_file = root/'keep.txt'
    personal_file.write_text('existing user file')
    runtime['_output_root'] = str(root)
    with pytest.raises(ValueError, match='not owned'):
        run_kaggle_batch(runtime, campaign_root)
    assert personal_file.read_text() == 'existing user file'
    assert list(root.iterdir()) == [personal_file]
