"""Continue imported three-account work with eight disjoint GPU assignments."""
from itertools import combinations
from pathlib import Path

import nbformat as nb
import pytest

from test_new_data_kaggle import project
from test_full_queue_embeddings import library, resealed
from test_full_queue_three_accounts import finish_account
from test_pending_account_continuation import source_round
from scripts import full_queue_embeddings as q
from scripts.write_full_queue_notebooks import ROOT, four_account_transfer, full_queue_gpu
from vietmedbridge.artifacts import read_json


def account_groups(plan):
    return [{tid for tid in plan['tasks'] if plan['task_workers'][tid]//2 == a}
            for a in range(4)]


def test_three_imported_accounts_continue_on_four_without_reexport_or_duplicate_tasks(project, tmp_path, monkeypatch):
    flow, target, job, old = source_round(project, tmp_path, monkeypatch, accounts=(0, 1, 2))
    job_before = (target/'job.json').read_bytes()
    old_path = flow.root/'plans'/(old['manifest_sha256']+'.json')
    old_before = old_path.read_bytes()
    candidate = project[0]/'data/active_data_candidate.json'
    candidate_before = candidate.read_bytes()
    receipts_before = {p.name: p.read_bytes() for p in (flow.root/'imports').glob('*.json')}
    imported_before = flow.imported()
    assert imported_before and len(imported_before) < len(job['tasks'])

    original_read = q.pq.read_table
    def keys_only(path, *args, **kwargs):
        p = Path(path)
        if 'index_inputs' in p.parts or p.parent == target or p.name.startswith('missing-') or p.suffix == '.npy':
            pytest.fail('Four-account planning replayed corpus text or vector payloads.')
        # A cache-only task may complete its affected draft/map metadata.
        return original_read(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(q.pq, 'read_table', keys_only)
        directory, plan = flow.create_plan(job, execution_accounts=4, output_bytes=q.ELASTIC_OUTPUT_BYTES)
    assert directory is not None
    assert job['team'] == {'accounts': 2, 'gpu_count': 2}
    assert plan['team'] == {'accounts': 4, 'gpu_count': 2}
    assert plan['qwen_execution'] == 'qwen-token-batches-v2'
    assert not set(plan['tasks']) & flow.imported()
    assert plan['remaining_tasks_before_plan'] == len(job['tasks'])-len(flow.imported())
    assert not plan['reserved_task_ids']
    assert set(plan['tasks']) <= set(old['tasks'])-imported_before
    groups = account_groups(plan)
    assert all(groups)
    assert all(not a & b for a, b in combinations(groups, 2))
    assert set.union(*groups) == set(plan['tasks'])
    assert not list(directory.glob('*.parquet'))

    cold = q.FullQueueKaggle(project[0], tmp_path/'cold', project[1], accounts=2)
    with monkeypatch.context() as m:
        m.setattr(q.pq, 'read_table', lambda *a, **k: pytest.fail('Retry replayed key inputs.'))
        _, retry = cold.create_plan(job, execution_accounts=4, output_bytes=q.ELASTIC_OUTPUT_BYTES)
    assert retry == plan

    returned = set()
    for account in range(4):
        output = tmp_path/f'four-{account}'
        _, result = finish_account(target, job, plan, output, tmp_path/f'gpu-{account}', project[1], account)
        ids = {r['task'] for r in result['tasks']}
        assert ids == groups[account] and not returned & ids
        returned.update(ids)
        report = flow.install(output/'result')
        assert report['execution_team'] == {'accounts': 4, 'gpu_count': 2}
        if account < 3:
            _, waiting = flow.create_plan(job, execution_accounts=4, output_bytes=q.ELASTIC_OUTPUT_BYTES)
            assert waiting['state'] == 'WAITING_FOR_KAGGLE_OUTPUTS'
            assert waiting['required_accounts'] == [0, 1, 2, 3]
    assert returned == set(plan['tasks'])
    assert all(v['completed_parts'] == v['total_parts'] for v in flow.coverage().values())
    assert (target/'job.json').read_bytes() == job_before
    assert old_path.read_bytes() == old_before
    assert candidate.read_bytes() == candidate_before
    assert all((flow.root/'imports'/name).read_bytes() == data for name, data in receipts_before.items())


@pytest.mark.parametrize('budget', [2100, q.ELASTIC_OUTPUT_BYTES])
def test_four_accounts_enforce_per_account_budget_and_all_eight_worker_owners(project, tmp_path, budget):
    flow, _, job = library(project, tmp_path)
    _, plan = flow.create_plan(job, execution_accounts=4, output_bytes=budget)
    assert set(plan['task_workers'].values()) <= set(range(8))
    assert len(plan['estimated_bytes_per_account']) == 4
    assert all(0 < n <= budget for n in plan['estimated_bytes_per_account'])
    if budget == q.ELASTIC_OUTPUT_BYTES:
        assert set(plan['task_workers'].values()) == set(range(8))
    else:
        assert plan['remaining_tasks_after_plan'] > 0
    groups = account_groups(plan)
    assert all(not a & b for a, b in combinations(groups, 2))
    owners = dict(plan['task_workers'])
    owners[plan['tasks'][0]] = 8
    with pytest.raises(ValueError, match='assignments'):
        q.validate_plan(resealed(plan, task_workers=owners), job)


def test_four_account_request_does_not_replace_an_unfinished_three_account_plan(project, tmp_path):
    flow, _, job = library(project, tmp_path)
    _, old = flow.create_plan(job, execution_accounts=3, output_bytes=q.ELASTIC_OUTPUT_BYTES)
    active = flow.root/'active_execution_plan.json'
    before = active.read_bytes()
    with pytest.raises(ValueError, match='another execution team'):
        flow.create_plan(job, execution_accounts=4, output_bytes=q.ELASTIC_OUTPUT_BYTES)
    assert active.read_bytes() == before
    assert read_json(active)['plan'] == old['manifest_sha256']


def test_delivered_four_account_notebooks_have_correct_presets_and_current_sources():
    generated = [four_account_transfer(), *[full_queue_gpu(a, expected_accounts=4) for a in range(4)]]
    paths = [ROOT/'notebooks/04_colab_prepare_4_accounts_kaggle.ipynb',
             *[ROOT/f'notebooks/kaggle/04_full_data_4acc_account_{a}.ipynb' for a in range(4)]]
    for expected, path in zip(generated, paths):
        saved = nb.read(path, as_version=4)
        assert [c.source for c in saved.cells] == [c.source for c in expected.cells]
        for c in saved.cells:
            if c.cell_type == 'code':
                compile(c.source, str(path), 'exec')
    cpu = generated[0]
    assert 'EXECUTION_ACCOUNTS = 4' in cpu.cells[1].source
    assert 'KAGGLE_ACCOUNTS = 2' in cpu.cells[1].source
    assert 'CONFIRMED_UNSTARTED_PLAN = None' in cpu.cells[1].source
    assert 'KAGGLE_NOTEBOOK_OUTPUT = ""' in cpu.cells[1].source
    assert 'continue_completed_from=' not in cpu.cells[3].source
    assert 'execution_accounts=EXECUTION_ACCOUNTS' in cpu.cells[3].source
    assert 'GPU_PLAN_ASSIGNMENTS:' in cpu.cells[3].source
    for a, gpu in enumerate(generated[1:]):
        assert f'KAGGLE_ACCOUNT_ID={a}' in gpu.cells[1].source
        assert 'EXPECTED_EXECUTION_ACCOUNTS=4' in gpu.cells[1].source
        assert 'SESSION_HOURS=11.5' in gpu.cells[1].source
        assert 'qwen-token-batches-v2' in gpu.cells[1].source
