import json
from pathlib import Path

import pytest

from scripts.prepare_kaggle_campaign import prepare_campaign
from scripts.prepare_kaggle_shard import prepare_shard
from src.storage.io import atomic_json
from tests.integration.test_pipeline import make_source


def campaign_fixture(config, tmp_path):
    make_source(config, tmp_path, [f'https://example.com/doc-{i}' for i in range(7)])
    campaign_root = tmp_path/'campaign'
    manifest = prepare_campaign(Path(config['_output_root']), campaign_root/'input', urls_per_shard=3)
    previous = {'status': 'COMPLETE', 'snapshot_id': manifest['snapshot_id'],
        'completed_shards': [0], 'kernel': 'tester/vibiomir-shard-0',
        'dataset': 'tester/campaign', 'runner_dataset': 'tester/runner',
        'source_sha256': 'fixture-source', 'report': 'previous-report.json'}
    atomic_json(campaign_root/'deployment.json', previous)
    return campaign_root, manifest, previous


def test_fresh_shard_selects_requested_frontier_without_previous_output(config, tmp_path):
    root, manifest, _ = campaign_fixture(config, tmp_path)
    destination = tmp_path/'launch'
    plan = prepare_shard(root, destination, 1)
    notebook = json.loads((destination/'kernel/02_kaggle_crawl_shard.ipynb').read_text())
    parameters = {}
    exec(''.join(notebook['cells'][0]['source']), parameters)
    assert parameters['SHARD_ID'] == 1 and parameters['AUTO_RESTORE'] is False
    assert parameters['REQUIRE_RESTORE'] is False and parameters['MAX_EXTRACTION_EVENT_GIB'] == 2
    assert parameters['CRAWL_HOURS'] is None and parameters['EXTRACTION_HOURS'] is None
    assert parameters['SESSION_HOURS'] == 11 and parameters['EXPORT_RESERVE_MINUTES'] == 30
    assert plan['frontier_sha256'] == manifest['shards'][1]['sha256'] and plan['frontier_urls'] == 3
    metadata = json.loads((destination/'kernel/kernel-metadata.json').read_text())
    assert metadata['kernel_sources'] == [] and metadata['dataset_sources'] == ['tester/campaign', 'tester/runner']


def test_fresh_shard_can_publish_updated_runner_without_mutating_previous_deployment(config, tmp_path):
    from zipfile import ZipFile
    root, _, previous = campaign_fixture(config, tmp_path)
    destination = tmp_path/'updated'
    plan = prepare_shard(root, destination, 1, runner_tag='session-deadline')
    assert plan['runner_dataset'] == 'tester/vibiomir-runner-session-deadline'
    assert plan['reviewed_runner_reused'] is False and plan['kernel_timeout_seconds'] == 43200
    assert json.loads((root/'deployment.json').read_text()) == previous
    code_root = destination/'code_input'
    marker = json.loads((code_root/'source_package_manifest.json').read_text())
    assert marker['sha256'] == plan['source_sha256']
    with ZipFile(code_root/'vibiomir_source.zip') as archive:
        assert 'src/utils/session_budget.py' in archive.namelist()


def test_shard_launch_guards_running_completed_and_tampered_frontiers(config, tmp_path):
    root, manifest, previous = campaign_fixture(config, tmp_path)
    with pytest.raises(ValueError, match='already completed'):
        prepare_shard(root, tmp_path/'duplicate', 0)
    previous['status'] = 'RUNNING'
    atomic_json(root/'deployment.json', previous)
    with pytest.raises(ValueError, match='must stop'):
        prepare_shard(root, tmp_path/'parallel', 1)
    previous['status'] = 'COMPLETE'
    atomic_json(root/'deployment.json', previous)
    (root/'input'/manifest['shards'][1]['name']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        prepare_shard(root, tmp_path/'bad', 1)
