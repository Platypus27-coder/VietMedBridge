"""Execution changes keep every frozen text/key/vector row and fail closed."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.qwen_token_batching import TokenBatchEncoder,token_groups,vector_agreement,benchmark_texts
from vietmedbridge.artifacts import read_json


class Clock:
    def __init__(self): self.t=0.0
    def __call__(self): return self.t


class OOM(RuntimeError): pass


def vectors(texts):
    a=np.array([[int(t.split(':')[0])+1,2,3,4] for t in texts],dtype=np.float32)
    return a/np.linalg.norm(a,axis=1,keepdims=True)


class Native:
    def __init__(self,clock,*,drift=False,slow=False,limit=100):
        self.clock,self.drift,self.slow,self.limit=clock,drift,slow,limit
        self.identity={'model_id':'Qwen/Qwen3-Embedding-8B','input_role':'corpus','frozen':'unchanged'}
        self.dimension,self.spec,self.encoder=4,{'max_length':1536},self
        self.torch=SimpleNamespace(OutOfMemoryError=OOM,cuda=SimpleNamespace(empty_cache=lambda:None))
        self.in_native,self.calls,self.clears=False,[],0
        self.tokenizer=lambda texts,**kw:{'input_ids':[[0]*int(t.split(':')[1]) for t in texts]}

    def _batch(self,texts):
        if len(texts)>self.limit: raise OOM('capacity')
        self.calls.append(list(texts))
        self.clock.t+=(0.03+len(texts)*max(int(t.split(':')[1]) for t in texts)*0.00001)*(3 if self.slow and not self.in_native else 1)
        v=vectors(texts)
        if self.drift and not self.in_native: v=np.roll(v,1,axis=1)
        return v

    def encode(self,texts,*,batch_size):
        self.in_native=True
        try:return np.concatenate([self._batch(texts[i:i+batch_size]) for i in range(0,len(texts),batch_size)])
        finally:self.in_native=False


def wrapper(tmp_path,**kw):
    clock=Clock(); native=Native(clock,**kw)
    e=TokenBatchEncoder(native,tmp_path/'tuning.json',clock=clock,report=lambda *a,**k:None)
    return e,native


def sample(): return [f'{i}:{32 if i%2 else 768}' for i in range(64)]


def test_token_groups_use_token_lengths_and_bounded_padding_without_dropping_rows():
    lengths=[3,1500,7,12,1500,8,5]*20
    groups=list(token_groups(lengths,64,24_576))
    assert sorted(i for g in groups for i in g)==list(range(len(lengths)))
    assert all(len(g)<=64 and len(g)*max(lengths[i] for i in g)<=24_576 for g in groups)


def test_canary_selects_faster_equivalent_execution_and_restores_original_order(tmp_path):
    e,n=wrapper(tmp_path)
    before=dict(n.identity)
    texts=sample()
    result=e.encode(texts,batch_size=64)
    assert np.allclose(result,vectors(texts))
    saved=read_json(e.path)
    assert saved['mode']=='token_batches' and saved['measured_sample_speedup']>=1.05
    assert saved['sample_texts']==64 and saved['trials']
    assert n.identity==e.identity==before
    calls=len(n.calls)
    e.encode(texts[::-1],batch_size=64)
    assert len(n.calls)-calls<len(texts)  # no repeated calibration or one-text inference


@pytest.mark.parametrize('setting',['drift','slow'])
def test_changed_vectors_or_slower_candidates_fall_back_to_native(tmp_path,setting):
    e,n=wrapper(tmp_path,**{setting:True})
    result=e.encode(sample(),batch_size=64)
    assert np.allclose(result,vectors(sample()))
    assert e.selected is None and read_json(e.path)['mode']=='native'
    assert e.safe_batch_size==16


def test_oom_retries_same_rows_and_remembers_only_the_affected_length_bucket(tmp_path):
    e,n=wrapper(tmp_path,limit=8)
    e.attempted,e.selected=True,64
    texts=[f'{i}:20' for i in range(40)]
    result=e.encode(texts,batch_size=64)
    assert np.allclose(result,vectors(texts)) and e.ooms>0
    before=e.ooms
    assert np.allclose(e.encode(texts,batch_size=64),vectors(texts))
    assert e.ooms==before
    assert all(len(c)<=8 for c in n.calls)


def test_duplicate_texts_keep_all_output_rows(tmp_path):
    e,n=wrapper(tmp_path)
    e.attempted,e.selected=True,32
    texts=['9:5','2:90','9:5','7:10']
    assert np.array_equal(e.encode(texts,batch_size=64),vectors(texts))


def test_over_length_input_is_never_truncated(tmp_path):
    e,n=wrapper(tmp_path)
    e.attempted,e.selected=True,32
    with pytest.raises(ValueError,match='no text was truncated'):
        e.encode(['1:1600'],batch_size=64)
    assert not n.calls


def test_malformed_or_drifting_vectors_fail_agreement_gate():
    a=vectors(['1:5','2:8'])
    assert vector_agreement(a,a)[0]
    assert not vector_agreement(a,np.roll(a,1,axis=1))[0]
    assert not vector_agreement(a,np.full(a.shape,np.nan,dtype=np.float32))[0]


def attention_native(clock,*,drift=False,bad_later=False):
    n=Native(clock)
    state={'enabled':False,'calls':[],'bad_later':bad_later}
    def setter(value):
        state['calls'].append(value)
        state['enabled']=value
    n.torch.backends=SimpleNamespace(cuda=SimpleNamespace(
        allow_fp16_bf16_reduction_math_sdp=setter,
        fp16_bf16_reduction_math_sdp_allowed=lambda:state['enabled']))
    original=n._batch
    def fast(texts):
        before=clock.t
        v=original(texts)
        if state['enabled']:
            clock.t=before+(clock.t-before)/3
            if drift: v=np.roll(v,1,axis=1)
            if state['bad_later']: v[:]=np.nan
        return v
    n._batch=fast
    return n,state


def test_faster_attention_requires_second_canary_and_always_restores_backend_flag(tmp_path):
    clock=Clock();n,state=attention_native(clock)
    e=TokenBatchEncoder(n,tmp_path/'tuned.json',attention_tuning=True,clock=clock,report=lambda *a,**k:None)
    assert np.allclose(e.encode(sample()),vectors(sample()))
    report=read_json(e.path)
    assert report['kernel']=='fp16_math' and report['confirmation']['compatible']
    assert report['confirmation']['selected_seconds']<=report['confirmation']['native_seconds']/1.05
    assert not state['enabled'] and True in state['calls']
    assert n.identity==e.identity


def test_numerically_bad_attention_is_rejected_even_when_faster(tmp_path):
    clock=Clock();n,state=attention_native(clock,drift=True)
    e=TokenBatchEncoder(n,tmp_path/'tuned.json',attention_tuning=True,clock=clock,report=lambda *a,**k:None)
    assert np.allclose(e.encode(sample()),vectors(sample()))
    report=read_json(e.path)
    assert report['kernel']=='native'
    assert all(not t['compatible'] for t in report['trials'] if t['kernel']=='fp16_math')
    assert not state['enabled']


def test_later_invalid_attention_retries_only_current_uncommitted_task_natively(tmp_path):
    clock=Clock();n,state=attention_native(clock)
    e=TokenBatchEncoder(n,tmp_path/'tuned.json',attention_tuning=True,clock=clock,report=lambda *a,**k:None)
    e.encode(sample())
    assert e.selected_kernel=='fp16_math'
    state['bad_later']=True
    texts=['101:90','102:91']
    assert np.allclose(e.encode(texts),vectors(texts))
    assert e.selected is None and e.selected_kernel=='native' and not state['enabled']


def test_attention_flag_restored_after_oom_and_other_exceptions(tmp_path):
    clock=Clock();n,state=attention_native(clock)
    n.limit=4
    e=TokenBatchEncoder(n,tmp_path/'tuned.json',attention_tuning=True,clock=clock,report=lambda *a,**k:None)
    e.attempted,e.selected,e.selected_kernel=True,32,'fp16_math'
    assert np.allclose(e.encode(['1:4']*8),vectors(['1:4']*8))
    assert not state['enabled']
    def fail(texts): raise RuntimeError('unexpected kernel failure')
    n._batch=fail
    with pytest.raises(RuntimeError,match='unexpected kernel'):
        e.encode(['2:5'])
    assert not state['enabled']


def test_benchmark_notebook_needs_no_execution_plan_and_never_marks_tasks_done():
    from scripts.write_full_queue_notebooks import qwen_benchmark_notebook
    n=qwen_benchmark_notebook()
    for cell in n.cells:
        if cell.cell_type=='code':compile(cell.source,'<benchmark>','exec')
    assert 'd6ab9a432ac64e5' in n.cells[1].source
    assert 'PLAN_PATH' not in n.cells[1].source+n.cells[3].source
    assert 'BENCHMARK_COMPLETE_NO_TASKS_COMMITTED' in n.cells[3].source
    assert 'CUDA_VISIBLE_DEVICES' in n.cells[3].source
    assert '.install(' not in n.cells[3].source and '.create_plan(' not in n.cells[3].source


def test_benchmark_samples_only_three_assets_and_checks_exact_task_keys(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vietmedbridge.artifacts import digest_json,sha256_file
    from vietmedbridge.content_embeddings import key
    assets,tasks={},[]
    for i in range(9):
        name=f'missing-qwen-{i:06d}.parquet'
        rows=[{'id':key(f'{i}-{j} text'),'text':f'{i}-{j} text'} for j in range(32)]
        pq.write_table(pa.Table.from_pylist(rows),tmp_path/name)
        checksum=sha256_file(tmp_path/name)
        assets[name]={'rows':len(rows),'sha256':checksum}
        tasks.append({'family':'qwen','asset':name,'offset':0,'rows':len(rows),
            'sha256':checksum,'keys_digest':digest_json(sorted(r['id'] for r in rows))})
    original=pq.read_table
    opened=[]
    def read(path,**kw):
        opened.append(path.name)
        return original(path,**kw)
    monkeypatch.setattr(pq,'read_table',read)
    texts=benchmark_texts(tmp_path,{'tasks':tasks,'assets':assets})
    assert len(texts)==72 and opened==[tasks[i]['asset'] for i in (0,4,8)]
    tasks[0]['keys_digest']='f'*64
    with pytest.raises(ValueError,match='text/key binding'):
        benchmark_texts(tmp_path,{'tasks':tasks,'assets':assets})
