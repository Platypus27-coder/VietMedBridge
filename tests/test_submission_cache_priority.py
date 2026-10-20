"""A warmed cache smaller than the corpus must not cause a full second transfer."""
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
from contextlib import contextmanager
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.submission_cache_priority import install_cache_first_io, resume_cache_first
from vietmedbridge.artifacts import sha256_file
from vietmedbridge.content_embeddings import seal


@pytest.fixture
def shipped():
    # The independent overlay must work with the adapter actually sent to Colab,
    # regardless of concurrent, uncommitted changes to other workspace scripts.
    repo=Path(__file__).resolve().parents[1]
    pinned=subprocess.run(['git','show','5315b1d:scripts/submission_vector_io.py'],cwd=repo,capture_output=True)
    # Shallow CI clones may omit the prior release object. Also keep exercising
    # forward compatibility with the checkout's adapter in that environment.
    code=pinned.stdout if pinned.returncode==0 else (repo/'scripts/submission_vector_io.py').read_bytes()
    module=ModuleType('shipped_submission_io')
    exec(compile(code,'shipped_submission_io.py','exec'),module.__dict__)
    return module


def make_store(base,adapter):
    root,work,view=base/'data',base/'work',base/'data/view'
    view.mkdir(parents=True)
    values=np.random.default_rng(14).normal(size=(16,4)).astype(np.float32)
    values/=np.linalg.norm(values,axis=1,keepdims=True)
    references=[]
    for start in range(0,16,2):
        path=root/'blocks'/f'block-{start}.npy'
        path.parent.mkdir(exist_ok=True)
        np.save(path,values[start:start+2],allow_pickle=False)
        references.extend({'path':path.relative_to(root).as_posix(),'sha256':sha256_file(path),'row':row}
            for row in range(2))
    mapping=view/'map-000000000000.parquet'
    pq.write_table(pa.Table.from_pylist(references),mapping)
    part=seal({'start':0,'rows':16,'signature':'fixed','path':mapping.name,'sha256':sha256_file(mapping)})
    manifest=seal({'state':'COMPLETE','input_count':16,'dimension':4,'encoder':{'test':'normalized'},
        'signature':'fixed','parts':[part]})
    budget=mapping.stat().st_size+4*(root/references[0]['path']).stat().st_size
    transport=adapter.VerifiedSubmissionIO(work,progress=lambda _:None)
    current=adapter.SubmissionVectorParts(root,view,manifest,work/'vectors',io=transport,local_cache_bytes=budget)
    # Simulate a previous pass retaining the last four blocks, with no live mmap
    # objects shielding them from eviction in the new consumer.
    shutil.copyfile(mapping,current.map_path(0))
    for ref in references[8::2]:
        shutil.copyfile(root/ref['path'],current.block_path(ref))
    transport.con.close()
    transport=adapter.VerifiedSubmissionIO(work,progress=lambda _:None)
    current=adapter.SubmissionVectorParts(root,view,manifest,work/'vectors',io=transport,local_cache_bytes=budget)
    return current,transport,values


def test_cache_first_avoids_rereading_unconsumed_warm_blocks_and_preserves_order(tmp_path,shipped):
    old,old_io,values=make_store(tmp_path/'old',shipped)
    old_downloads=[]
    real_old=old_io.fetch
    def count_old(*args,**kwargs):
        result=real_old(*args,**kwargs)
        if result[0] and Path(args[1]).suffix=='.npy':
            old_downloads.append(Path(args[0]).name)
        return result
    old_io.fetch=count_old
    positions=np.array([15,0,7,7,2,12,10,4,9,6,3,5,11,1,8,13,14])
    expected=old.gather(positions)
    assert len(old_downloads)==8  # Prefix downloads evicted the unread suffix.
    install_cache_first_io(shipped.VerifiedSubmissionIO)
    new,new_io,other=make_store(tmp_path/'new',shipped)
    messages=[]
    new_io.progress=messages.append
    downloads=[]
    real=new_io.fetch
    def count(*args,**kwargs):
        result=real(*args,**kwargs)
        if result[0] and Path(args[1]).suffix=='.npy':
            downloads.append(Path(args[0]).name)
        return result
    new_io.fetch=count
    actual=new.gather(positions)
    assert len(downloads)==4  # Only the four initially missing blocks transferred.
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual,other[positions])
    assert any('Vector cache plan: 4/8 local files' in value for value in messages)
    old_io.con.close();new_io.con.close()


def test_install_is_idempotent_and_other_stages_keep_original_order(tmp_path):
    visits=[]
    class IO:
        def stage(self,items,label,**kwargs):
            visits.append((items,label))
            return 'same'
    install_cache_first_io(IO)
    method=IO.stage
    install_cache_first_io(IO)
    assert IO.stage is method
    items=[('source',tmp_path/'not-present','hash')]
    assert IO().stage(items,'Mapping cache')=='same'
    assert visits==[(items,'Mapping cache')]


def test_prioritization_keeps_checksums_and_source_payloads_bound(tmp_path,shipped):
    install_cache_first_io(shipped.VerifiedSubmissionIO)
    current,transport,_=make_store(tmp_path/'bad',shipped)
    path=next(current.cache.glob('*.npy'))
    path.write_bytes(b'corrupt cached vector')
    with pytest.raises(ValueError,match='changed artifact'):
        current.gather(range(16))
    transport.con.close()


def test_resume_reuses_loaded_adapter_and_calls_only_baseline(tmp_path,monkeypatch):
    from vietmedbridge import submission_workflow
    events=[]
    class IO:
        def stage(self,*args,**kwargs):
            pass
    @contextmanager
    def progress(*args,**kwargs):
        events.append('adapter')
        yield SimpleNamespace(report=lambda _:None)
    def run(root,checkout,work,**kwargs):
        assert root==tmp_path and kwargs=={'code_commit':'same-pinned-producer'}
        events.append('baseline')
        return {'ready':{'state':'READY_FOR_MANUAL_UPLOAD'}}
    monkeypatch.setattr(submission_workflow,'run_current_baseline',run)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
    source=SimpleNamespace(verify_candidate=lambda sha:events.append(sha))
    ns={'VerifiedSubmissionIO':IO,'submission_gpu_progress':progress,
        'prepare_drive_receipt_lookup':lambda *a,**k:events.append('JSON handoff'),
        'ensure_submission_mount':lambda root,sha:root,'DRIVE_RECEIPTS':source,
        'DATA_ROOT':tmp_path,'WORK_DIR':tmp_path/'work','CHECKOUT':tmp_path/'checkout',
        'PREPARATION':{'candidate_manifest_sha256':'same-candidate'},'COMMIT':'same-pinned-producer'}
    result=resume_cache_first(ns)
    assert result is ns['BASELINE']
    assert events==['same-candidate','adapter','JSON handoff','baseline']


def test_fresh_runtime_refuses_to_start_preparation():
    with pytest.raises(RuntimeError,match='đúng runtime'):
        resume_cache_first({})
