"""Migrate a saved two-account library to six GPUs without exporting text again."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import pyarrow.parquet as pq
import pytest

from test_new_data_kaggle import project, Encoder
from test_full_queue_embeddings import library, run_account, resealed
from scripts.full_queue_embeddings import (FullQueueKaggle, ELASTIC_PLAN_SCHEMA,
    ELASTIC_OUTPUT_BYTES, finalize, run_worker, validate_plan, execution_worker)
from scripts.new_data_embeddings import FAMILIES
from scripts.write_full_queue_notebooks import full_queue_gpu, three_account_transfer
from vietmedbridge.artifacts import read_json, atomic_json, sha256_file


def finish_account(target, job, plan, output, work, identities, account):
    calls = {f: [] for f in FAMILIES}
    factories = {f: lambda f=f: Encoder(identities[f], calls[f]) for f in FAMILIES}
    for worker in range(account*2, account*2+2):
        run_worker(target, plan, output/'result', work/str(worker), worker_id=worker,
                   factories=factories, deadline=time.monotonic()+60)
    return calls, finalize(target, plan, output, account)


def test_two_account_imports_survive_six_gpu_migration_and_account_two_import(project, tmp_path, monkeypatch):
    flow, target, job = library(project, tmp_path)
    job_before = (target/'job.json').read_bytes()
    active_before = (project[0]/'data/active_data_candidate.json').read_bytes()
    _, old = flow.create_plan(job, output_bytes=2100)
    for account in range(2):
        out = tmp_path/f'old-{account}'
        run_account(target, job, old, out, tmp_path/f'old-gpu-{account}', project[1], account)
        flow.install(out/'result')
    imported = flow.imported()
    old_receipts = {p.name: p.read_bytes() for p in (flow.root/'imports').glob('*.json')}
    _, unstarted = flow.create_plan(job, output_bytes=2100)
    before_text = pq.read_table
    def no_corpus_or_export_replay(path, *args, **kwargs):
        p = Path(path)
        if 'index_inputs' in p.parts or p.parent == target or p.name.startswith('missing-'):
            pytest.fail('Migration reopened original inputs/exported text instead of the saved key indexes.')
        return before_text(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(pq, 'read_table', no_corpus_or_export_replay)
        plan_dir, plan = flow.create_plan(job, execution_accounts=3,
            output_bytes=ELASTIC_OUTPUT_BYTES, replace_unstarted_plan=unstarted['manifest_sha256'])
    assert job['team'] == {'accounts': 2, 'gpu_count': 2}
    assert plan['team'] == {'accounts': 3, 'gpu_count': 2}
    assert plan['schema'] == ELASTIC_PLAN_SCHEMA
    assert not imported & set(plan['tasks'])
    assert set(plan['task_workers'].values()) == set(range(6))
    assert (target/'job.json').read_bytes() == job_before
    assert (plan_dir/'full_queue_embeddings.py').is_file()
    assert (plan_dir/'new_data_kaggle.py').is_file()
    assert (plan_dir/'qwen_token_batching.py').is_file()
    assert plan['qwen_execution']=='qwen-token-batches-v1'
    assert not list(plan_dir.glob('*.parquet'))
    assert (flow.root/'history'/('unstarted-'+unstarted['manifest_sha256']+'.json')).is_file()
    # Cold retry uses the same small plan, no task/key/cache replay.
    cold = FullQueueKaggle(project[0], tmp_path/'cold', project[1], accounts=2)
    with monkeypatch.context() as m:
        m.setattr(pq, 'read_table', lambda *a, **k: pytest.fail('Pending plan was recalculated.'))
        _, again = cold.create_plan(job, execution_accounts=3, output_bytes=ELASTIC_OUTPUT_BYTES,
                                   replace_unstarted_plan=unstarted['manifest_sha256'])
    assert again == plan
    receipts = set()
    for account in range(3):
        out = tmp_path/f'new-{account}'
        _, result = finish_account(target, job, plan, out, tmp_path/f'new-gpu-{account}', project[1], account)
        assert result['state'] == 'ROUND_COMPLETE'
        returned = {r['task'] for r in result['tasks']}
        assert not receipts & returned
        receipts.update(returned)
        report = flow.install(out/'result')
        assert report['account_id'] == account and report['execution_team'] == plan['team']
        assert not flow.install(out/'result').get('payloads_reopened', True)
        if account < 2:
            _, waiting = flow.create_plan(job, execution_accounts=3, output_bytes=ELASTIC_OUTPUT_BYTES)
            assert waiting['state'] == 'WAITING_FOR_KAGGLE_OUTPUTS'
            assert waiting['required_accounts'] == [0, 1, 2]
    assert receipts == set(plan['tasks'])
    assert all(p.read_bytes() == old_receipts[p.name] for p in (flow.root/'imports').glob('*.json') if p.name in old_receipts)
    assert (project[0]/'data/active_data_candidate.json').read_bytes() == active_before
    assert all(v['completed_parts'] == v['total_parts'] == 10 for v in flow.coverage().values())


def test_unstarted_confirmation_is_exact_and_cannot_replace_partly_imported_plan(project, tmp_path):
    flow, target, job = library(project, tmp_path)
    _, old = flow.create_plan(job, output_bytes=2100)
    with pytest.raises(ValueError, match='exact confirmed unstarted'):
        flow.create_plan(job, execution_accounts=3, replace_unstarted_plan='f'*64)
    output = tmp_path/'out'
    run_account(target, job, old, output, tmp_path/'gpu', project[1], 0)
    flow.install(output/'result')
    with pytest.raises(ValueError, match='exact confirmed unstarted'):
        flow.create_plan(job, execution_accounts=3, replace_unstarted_plan=old['manifest_sha256'])
    assert read_json(flow.root/'active_execution_plan.json')['plan'] == old['manifest_sha256']


@pytest.mark.parametrize('change', ['duplicate-owner', 'bad-worker', 'bad-team', 'bad-producer', 'over-budget'])
def test_elastic_plan_rejects_wrong_assignment_and_model_contract(project, tmp_path, change):
    flow, target, job = library(project, tmp_path)
    _, plan = flow.create_plan(job, execution_accounts=3, output_bytes=ELASTIC_OUTPUT_BYTES)
    if change == 'duplicate-owner':
        workers = dict(plan['task_workers']); workers['f'*64] = 0
        plan = resealed(plan, task_workers=workers)
    elif change == 'bad-worker':
        workers = dict(plan['task_workers']); workers[plan['tasks'][0]] = 6
        plan = resealed(plan, task_workers=workers)
    elif change == 'bad-team':
        plan = resealed(plan, library_team={'accounts': 3, 'gpu_count': 2})
    elif change == 'bad-producer':
        plan = resealed(plan, executor_sources={})
    else:
        plan = resealed(plan, output_bytes_per_account=20_000_000_000)
    with pytest.raises(ValueError):
        validate_plan(plan, job)


def test_budget_is_enforced_across_three_accounts_and_no_task_has_two_workers(project, tmp_path):
    flow, target, job = library(project, tmp_path)
    _, plan = flow.create_plan(job, execution_accounts=3, output_bytes=2100)
    assert plan['remaining_tasks_after_plan'] > 0
    assert all(n <= 2100 for n in plan['estimated_bytes_per_account'])
    assert len(plan['task_workers']) == len(set(plan['tasks']))
    entries = {t['id']: t for t in job['tasks']}
    groups = [{tid for tid in plan['tasks'] if execution_worker(plan, entries[tid])//2 == a} for a in range(3)]
    assert set.union(*groups) == set(plan['tasks'])
    assert all(not groups[a] & groups[b] for a in range(3) for b in range(a))


def test_three_notebooks_and_cpu_migration_are_preconfigured_and_compile():
    notebook = three_account_transfer()
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<cpu-three>', 'exec')
    assert 'EXECUTION_ACCOUNTS = 3' in notebook.cells[1].source
    assert 'KAGGLE_ACCOUNTS = 2' in notebook.cells[1].source
    assert 'output_bytes=queue.ELASTIC_OUTPUT_BYTES' in notebook.cells[3].source
    assert 'replace_unstarted_plan=CONFIRMED_UNSTARTED_PLAN' in notebook.cells[3].source
    assert 'execution_accounts=EXECUTION_ACCOUNTS' in notebook.cells[3].source
    for account in range(3):
        gpu = full_queue_gpu(account, expected_accounts=3)
        for cell in gpu.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<gpu-three>', 'exec')
        assert f'KAGGLE_ACCOUNT_ID={account}' in gpu.cells[1].source
        assert 'EXPECTED_EXECUTION_ACCOUNTS=3' in gpu.cells[1].source
        assert "PLAN['team']['accounts']" in gpu.cells[1].source
        assert "EXECUTOR_DIR/'full_queue_embeddings.py'" in gpu.cells[3].source
        assert 'SESSION_HOURS=11.5' in gpu.cells[1].source
        assert 'LEGACY_EXECUTION_PLAN:' in gpu.cells[1].source


def test_previous_two_file_executor_plan_remains_valid_and_cannot_claim_new_tuning(project,tmp_path):
    flow,target,job=library(project,tmp_path)
    _,plan=flow.create_plan(job,execution_accounts=3,output_bytes=ELASTIC_OUTPUT_BYTES)
    files={k:v for k,v in plan['executor_sources'].items() if k!='qwen_token_batching.py'}
    with pytest.raises(ValueError,match='scheduler'):
        validate_plan(resealed(plan,executor_sources=files),job)
    legacy=resealed(plan,executor_sources=files,qwen_execution=None)
    assert validate_plan(legacy,job)==legacy
    with pytest.raises(ValueError,match='Unknown Qwen execution'):
        validate_plan(resealed(plan,qwen_execution='unsafe-different-model'),job)


def test_fresh_worker_uses_small_plan_executor_with_original_published_library(project, tmp_path):
    flow, target, job = library(project, tmp_path)
    repo = Path(__file__).resolve().parents[1]
    # Production keeps the original f759b59 worker files inside its 5.5 GB
    # read-only dataset. The new scheduler must coexist with those exact files.
    sources = {}
    for name in job['worker_sources']:
        value = subprocess.check_output(['git', 'show', 'f759b59:scripts/'+name], cwd=repo)
        (target/name).write_bytes(value)
        sources[name] = sha256_file(target/name)
    job = resealed(job, worker_sources=sources)
    atomic_json(target/'job.json', job)
    atomic_json(flow.root/'input_library.json', job)
    atomic_json(flow.root/'exports'/('job-'+job['manifest_sha256']+'.json'), job)
    plan_dir, plan = flow.create_plan(job, execution_accounts=3, output_bytes=ELASTIC_OUTPUT_BYTES)
    assert plan['executor_sources']['full_queue_embeddings.py'] != sources['full_queue_embeddings.py']
    out = tmp_path/'saved'
    script = '''
import sys,time,json
from pathlib import Path
import full_queue_embeddings as q
from test_new_data_kaggle import Encoder
from vietmedbridge.artifacts import read_json
job=q.validate_library(sys.argv[1])
factories={f:lambda f=f:Encoder(job['models'][f],[]) for f in ('bge','qwen')}
q.run_worker(sys.argv[1],read_json(Path(sys.argv[2])/'job.json'),Path(sys.argv[3])/'result',
             Path(sys.argv[4]),worker_id=4,factories=factories,deadline=time.monotonic()+60)
q.finalize(sys.argv[1],read_json(Path(sys.argv[2])/'job.json'),sys.argv[3],2)
print('ORIGINAL_LIBRARY_AND_NEW_EXECUTOR_OK')
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(plan_dir), str(target), env.get('PYTHONPATH', '')))
    result = subprocess.run([sys.executable, '-c', script, str(target), str(plan_dir), str(out), str(tmp_path/'gpu')],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'ORIGINAL_LIBRARY_AND_NEW_EXECUTOR_OK' in result.stdout
    path = out/'result/new-data-result.json'
    receipt = read_json(path)
    assert receipt['tasks']
    atomic_json(path, resealed(receipt, account_id=0))
    with pytest.raises(ValueError, match='non-whitelisted'):
        flow.install(out/'result')
    assert not flow.imported()
    atomic_json(path, receipt)
    assert flow.install(out/'result')['account_id'] == 2
