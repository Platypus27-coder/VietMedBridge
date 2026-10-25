"""Execution changes keep every frozen text/key/vector row and fail closed."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.qwen_token_batching import TokenBatchEncoder,token_groups,vector_agreement
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
