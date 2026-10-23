"""A verbose subprocess must not flood the notebook or lose saved checkpoints."""
import json
import sys

import pytest

from scripts.kaggle_output_guard import run_with_bounded_output
from vietmedbridge.artifacts import atomic_json
from vietmedbridge.content_embeddings import seal


def result_file(tmp_path):
    value = seal({'schema': 'new-data-kaggle-result-v1', 'job': 'a'*64,
                  'execution_plan': 'b'*64, 'account_id': 0,
                  'state': 'CHECKPOINTED_PARTIAL',
                  'tasks': [{'task': 'c'*64, 'keys': ['d'*64]}], 'blocks': []})
    path = tmp_path/'result/new-data-result.json'
    atomic_json(path, value)
    return path


def run(tmp_path, program, *, result=None):
    result = result or result_file(tmp_path)
    reports = []
    summary = run_with_bounded_output([sys.executable, '-u', '-c', program], env=None,
        log_path=tmp_path/'gpu-worker.log', result_path=result,
        job='a'*64, plan='b'*64, account=0, report=reports.append,
        poll_seconds=0.01, heartbeat_seconds=0.02)
    return summary, reports


def test_large_json_stays_in_output_file_and_manifest_is_unchanged(tmp_path):
    result = result_file(tmp_path)
    before = result.read_bytes()
    program = ("print('BAAI/bge-m3 | saved mini-block 512/512 new texts', flush=True);"
               "print('\\n'.join(['    \\\"'+'f'*64+'\\\",']*50000));"
               "print('KAGGLE_CHECKPOINT worker 0 | saved tasks 1', flush=True)")
    summary, reports = run(tmp_path, program, result=result)
    assert summary['process_state'] == 'COMPLETE'
    assert summary['completed_tasks'] == 1
    assert (tmp_path/'gpu-worker.log').stat().st_size > 3_000_000
    assert sum(len(message) for message in reports) < 20000
    assert not any('f'*64 in message for message in reports)
    assert result.read_bytes() == before


def test_failed_worker_is_explicit_and_partial_artifacts_can_be_saved(tmp_path):
    summary, reports = run(tmp_path, "raise RuntimeError('synthetic worker failure')")
    assert summary['process_state'] == 'FAILED' and summary['returncode'] != 0
    assert summary['state'] == 'CHECKPOINTED_PARTIAL'
    assert any('GPU_RUN_FAILED' in message for message in reports)
    assert any('PARTIAL_OUTPUTS_READY_FOR_SAVE' in message for message in reports)
    assert 'synthetic worker failure' in (tmp_path/'gpu-worker.log').read_text()


def test_missing_result_does_not_claim_checkpoint_success(tmp_path):
    with pytest.raises(RuntimeError, match='without a result manifest'):
        run(tmp_path, 'print("no checkpoint")', result=tmp_path/'missing.json')


@pytest.mark.parametrize('field,value', [('job','e'*64), ('execution_plan','e'*64),
                                        ('account_id',1), ('manifest_sha256','e'*64)])
def test_result_must_match_job_plan_account_and_seal(tmp_path, field, value):
    path = result_file(tmp_path)
    saved = json.loads(path.read_text())
    saved[field] = value
    if field != 'manifest_sha256':
        saved = seal({key:item for key,item in saved.items() if key != 'manifest_sha256'})
    atomic_json(path, saved)
    with pytest.raises(ValueError, match='another job/plan/account or changed'):
        run(tmp_path, 'print("finished")', result=path)
