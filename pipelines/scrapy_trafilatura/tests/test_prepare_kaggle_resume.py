import json

import pytest

from scripts.prepare_kaggle_resume import prepare_resume
from src.storage.io import atomic_json


def partial_deployment(tmp_path):
    root = tmp_path / 'campaign'
    report_path = root / 'saved/vibiomir_shard_00007_report.json'
    report = {
        'status': 'PARTIAL', 'shard_id': 7, 'snapshot_id': 'snapshot',
        'frontier_urls': 100000, 'crawl_remaining': 36050,
        'integrity_passed': True, 'bundle': {'verified': True, 'sha256': 'a' * 64},
        'campaign_shards': 44, 'campaign_unique_urls': 4392746,
    }
    atomic_json(report_path, report)
    report_path.with_name('vibiomir_shard_00007.tar.sha256').write_text('a' * 64 + '  checkpoint.tar\n')
    previous = {
        'status': 'COMPLETE', 'report': str(report_path), 'snapshot_id': 'snapshot',
        'shard_id': 7, 'kernel': 'tester/shard-seven', 'kernel_version': 1,
        'dataset': 'tester/campaign', 'runner_dataset': 'tester/reviewed-runner',
        'source_sha256': 'original-source', 'completed_shards': [0, 1, 2, 3, 4, 5, 6],
    }
    atomic_json(root / 'deployment.json', previous)
    return root, report_path, report, previous


def test_resume_preserves_nonzero_shard_and_pinned_checkpoint(tmp_path):
    root, _, _, previous = partial_deployment(tmp_path)
    destination = tmp_path / 'resume'
    plan = prepare_resume(root, destination, 'resume-seven', reuse_runner=True)
    notebook = json.loads((destination / 'kernel/02_kaggle_crawl_shard.ipynb').read_text())
    parameters = {}
    exec(''.join(notebook['cells'][0]['source']), parameters)
    assert parameters['SHARD_ID'] == 7
    assert parameters['AUTO_RESTORE'] and parameters['REQUIRE_RESTORE']
    assert parameters['EXPECTED_RESTORE_SHA256'] == 'a' * 64
    assert plan['pending_urls'] == 36050 and plan['expected_skipped_urls'] == 63950
    assert plan['source_sha256'] == previous['source_sha256']
    assert plan['completed_shards'] == previous['completed_shards']
    metadata = json.loads((destination / 'kernel/kernel-metadata.json').read_text())
    assert metadata['kernel_sources'] == ['tester/shard-seven/1']
    assert metadata['dataset_sources'] == ['tester/campaign', 'tester/reviewed-runner']
    assert metadata['enable_gpu'] == 'false'
    assert not (destination / 'code_input').exists()


def test_resume_rejects_running_or_mismatched_previous_state(tmp_path):
    root, report_path, report, previous = partial_deployment(tmp_path)
    previous['status'] = 'RUNNING'
    atomic_json(root / 'deployment.json', previous)
    with pytest.raises(ValueError, match='must stop'):
        prepare_resume(root, tmp_path / 'running', 'resume', reuse_runner=True)
    previous['status'] = 'COMPLETE'
    atomic_json(root / 'deployment.json', previous)
    report['shard_id'] = 0
    atomic_json(report_path, report)
    with pytest.raises(ValueError, match='different shard'):
        prepare_resume(root, tmp_path / 'wrong-shard', 'resume', reuse_runner=True)
    report['shard_id'] = 7
    atomic_json(report_path, report)
    report_path.with_name('vibiomir_shard_00007.tar.sha256').write_text('b' * 64 + '\n')
    with pytest.raises(ValueError, match='checksum disagree'):
        prepare_resume(root, tmp_path / 'wrong-checksum', 'resume', reuse_runner=True)
