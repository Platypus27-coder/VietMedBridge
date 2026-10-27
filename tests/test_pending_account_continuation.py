"""Finished accounts can continue while an unknown account keeps its assignment."""
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from test_new_data_kaggle import project,Encoder
from test_full_queue_embeddings import library,resealed
from test_full_queue_three_accounts import finish_account
from scripts import full_queue_embeddings as q
from scripts.new_data_embeddings import FAMILIES
from scripts.write_full_queue_notebooks import two_account_continuation,full_queue_gpu
from vietmedbridge.artifacts import atomic_json,read_json


def partial_account(target,job,plan,output,work,identities,account,monkeypatch):
    factories={f:lambda f=f:Encoder(identities[f],[]) for f in FAMILIES}
    for worker in range(account*2,account*2+2):
        ticks=iter([0,0]+[float('inf')]*20)
        with monkeypatch.context() as m:
            m.setattr(q,'time',SimpleNamespace(monotonic=lambda:next(ticks)))
            q.run_worker(target,plan,output/'result',work/str(worker),worker_id=worker,
                factories=factories,deadline=1)
    return q.finalize(target,plan,output,account)


def source_round(project,tmp_path,monkeypatch,*,accounts=(0,1),execution_accounts=3):
    flow,target,job=library(project,tmp_path)
    _,old=flow.create_plan(job,execution_accounts=execution_accounts,output_bytes=q.ELASTIC_OUTPUT_BYTES)
    for account in accounts:
        out=tmp_path/f'old-{account}'
        partial_account(target,job,old,out,tmp_path/f'old-gpu-{account}',project[1],account,monkeypatch)
        flow.install(out/'result')
    return flow,target,job,old


def test_two_accounts_continue_without_waiting_and_late_account_two_still_imports(project,tmp_path,monkeypatch):
    flow,target,job,old=source_round(project,tmp_path,monkeypatch)
    original_job=(target/'job.json').read_bytes()
    old_plan=(flow.root/'plans'/(old['manifest_sha256']+'.json')).read_bytes()
    entries={t['id']:t for t in job['tasks']}
    account_two={tid for tid in old['tasks'] if q.execution_worker(old,entries[tid])//2==2}
    imported=flow.imported()
    assert account_two and len(imported)<len(old['tasks'])
    directory,new=flow.create_plan(job,execution_accounts=2,output_bytes=q.ELASTIC_OUTPUT_BYTES,
                                  continue_completed_from=old['manifest_sha256'])
    assert directory is not None and new['team']=={'accounts':2,'gpu_count':2}
    assert not set(new['tasks'])&(imported|account_two)
    assert set(new['reserved_task_ids'])==account_two
    # Newly available cache may also close other tasks without GPU inference.
    assert new['remaining_tasks_before_plan']==len(job['tasks'])-len(flow.imported())
    assert not set(new['tasks'])&flow.imported()
    assert set(new['task_workers'].values())<=set(range(4))
    assert any(t in new['tasks'] for t in old['tasks'] if t not in imported|account_two)
    # A cold CPU restart restores the identical pending plan, not another round.
    cold=q.FullQueueKaggle(project[0],tmp_path/'cold',project[1],accounts=2)
    with monkeypatch.context() as m:
        m.setattr(q.pq,'read_table',lambda *a,**k:pytest.fail('Cold continuation replayed key files.'))
        _,again=cold.create_plan(job,execution_accounts=2,output_bytes=q.ELASTIC_OUTPUT_BYTES,
                                continue_completed_from=old['manifest_sha256'])
    assert again==new
    for account in (0,1):
        out=tmp_path/f'new-{account}'
        finish_account(target,job,new,out,tmp_path/f'new-gpu-{account}',project[1],account)
        flow.install(out/'result')
    _,waiting=flow.create_plan(job,execution_accounts=2,output_bytes=q.ELASTIC_OUTPUT_BYTES,
                              continue_completed_from=old['manifest_sha256'])
    assert waiting['state']=='WAITING_FOR_RESERVED_KAGGLE_OUTPUTS'
    assert waiting['reserved_tasks']==len(account_two)
    # Partial late Outputs release only the uncompleted remainder after import.
    out=tmp_path/'late-two'
    late=partial_account(target,job,old,out,tmp_path/'late-gpu',project[1],2,monkeypatch)
    assert late['state']=='CHECKPOINTED_PARTIAL'
    report=flow.install(out/'result')
    assert report['account_id']==2 and report['execution_team']==old['team']
    assert read_json(flow.root/'active_execution_plan.json')['plan']==new['manifest_sha256']
    held,summaries=flow.reserved_tasks(job,flow.account_receipts(job))
    assert not held and summaries[0]['pending_accounts']==[]
    _,next_plan=flow.create_plan(job,execution_accounts=2,output_bytes=q.ELASTIC_OUTPUT_BYTES,
                                continue_completed_from=old['manifest_sha256'])
    if next_plan.get('tasks'):
        assert not set(next_plan['tasks'])&flow.imported()
        assert not next_plan['reserved_task_ids']
    else:
        assert next_plan['state']=='NO_PENDING_KAGGLE_TASKS'
    assert (target/'job.json').read_bytes()==original_job
    assert (flow.root/'plans'/(old['manifest_sha256']+'.json')).read_bytes()==old_plan


def test_colab_one_gpu_per_account_reserves_only_still_running_kaggle_account(project,tmp_path,monkeypatch):
    flow,target,job,old=source_round(project,tmp_path,monkeypatch,accounts=(0,1,3),execution_accounts=4)
    entries={t['id']:t for t in job['tasks']}
    account_two={tid for tid in old['tasks'] if q.execution_worker(old,entries[tid])//2==2}
    imported=flow.imported()
    _,colab=flow.create_plan(job,execution_accounts=4,execution_gpu_count=1,
        output_bytes=q.ELASTIC_OUTPUT_BYTES,continue_completed_from=old['manifest_sha256'])
    assert colab['team']=={'accounts':4,'gpu_count':1}
    assert set(colab['reserved_task_ids'])==account_two
    assert not set(colab['tasks'])&(imported|account_two)
    assert set(colab['task_workers'].values())<=set(range(4))
    reservation=read_json(flow.root/'reservations'/(old['manifest_sha256']+'.json'))
    assert reservation['released_accounts']==[0,1,3]
    assert reservation['held_accounts']==[2]


def test_partial_source_import_releases_only_that_account(project,tmp_path,monkeypatch):
    flow,target,job,old=source_round(project,tmp_path,monkeypatch,accounts=(0,))
    entries={t['id']:t for t in job['tasks']}
    held={tid for tid in old['tasks'] if q.execution_worker(old,entries[tid])//2 in (1,2)}
    _,new=flow.create_plan(job,execution_accounts=2,output_bytes=q.ELASTIC_OUTPUT_BYTES,
        continue_completed_from=old['manifest_sha256'])
    assert new['team']=={'accounts':2,'gpu_count':2}
    assert set(new['reserved_task_ids'])==held
    assert not set(new['tasks'])&(flow.imported()|held)
    with pytest.raises(ValueError,match='exact source execution-plan SHA'):
        flow.create_plan(job,execution_accounts=2,continue_completed_from='guess')


def test_missing_or_changed_reservation_never_reassigns_unknown_account(project,tmp_path,monkeypatch):
    flow,target,job,old=source_round(project,tmp_path,monkeypatch)
    _,new=flow.create_plan(job,execution_accounts=2,output_bytes=q.ELASTIC_OUTPUT_BYTES,
                           continue_completed_from=old['manifest_sha256'])
    path=flow.root/'reservations'/(old['manifest_sha256']+'.json')
    saved=read_json(path)
    atomic_json(path,resealed(saved,held_task_ids=[]))
    with pytest.raises(ValueError,match='whitelist changed'):
        flow.create_plan(job,execution_accounts=2,continue_completed_from=old['manifest_sha256'])
    atomic_json(path,saved)
    path.unlink()
    with pytest.raises(ValueError,match='reservation is missing'):
        flow.create_plan(job,execution_accounts=2)
    with pytest.raises(ValueError,match='Reserved pending-account tasks overlap'):
        q.validate_plan(resealed(new,reserved_task_ids=[new['tasks'][0]]),job)


def test_delivered_two_account_notebook_keeps_library_and_has_exact_source_handoff():
    n=two_account_continuation()
    for c in n.cells:
        if c.cell_type=='code':compile(c.source,'<two-account-continuation>','exec')
    assert 'EXECUTION_ACCOUNTS = 2' in n.cells[1].source
    assert 'KAGGLE_ACCOUNTS = 2' in n.cells[1].source
    assert 'c8d94f5f75f79ac89bd110b74b33befc3cbcf519be4fed2cf67be4df58fe7f56' in n.cells[1].source
    assert 'continue_completed_from=CONTINUE_FROM_PLAN' in n.cells[3].source
    assert 'RESERVED_PENDING_ACCOUNTS:' in n.cells[3].source
    for account in (0,1):
        gpu=full_queue_gpu(account,expected_accounts=2)
        assert f'KAGGLE_ACCOUNT_ID={account}' in gpu.cells[1].source
        assert 'EXPECTED_EXECUTION_ACCOUNTS=2' in gpu.cells[1].source
