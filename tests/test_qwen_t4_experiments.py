"""Benchmark-only branches restore execution state and compare against v2."""
from contextlib import contextmanager
from types import ModuleType,SimpleNamespace
import sys

import numpy as np
import pytest

from scripts import qwen_t4_experiments as experiments
from scripts.qwen_token_batching import TokenBatchEncoder
from test_colab_qwen_benchmark import Cuda,sample_pair
from test_qwen_token_batching import Clock,Native,OOM
from vietmedbridge.artifacts import sha256_file
from scripts.write_colab_qwen_benchmark_notebook import ROOT,notebook


def configured(tmp_path,monkeypatch,*,drift=False,unavailable=False):
    clock=Clock();native=Native(clock)
    state={'threads':2,'efficient':False}
    native.torch.get_num_threads=lambda:state['threads']
    native.torch.set_num_threads=lambda value:state.update(threads=value)
    @contextmanager
    def efficient():
        old=state['efficient']
        state['efficient']=True
        try:
            if unavailable:raise RuntimeError('No available kernel')
            yield
        finally:state['efficient']=old
    monkeypatch.setattr(experiments,'efficient_attention',efficient)
    original=native._batch
    def batch(texts):
        start=clock.t
        value=original(texts)
        if state['efficient']:
            clock.t=start+(clock.t-start)/3
            if drift:value=np.roll(value,1,axis=1)
        return value
    native._batch=batch
    reference=TokenBatchEncoder(native,tmp_path/'reference.json',clock=clock,report=lambda *a,**k:None)
    return native,reference,clock,state


def test_faster_branch_must_beat_existing_v2_and_match_original_native_vectors(tmp_path,monkeypatch):
    native,reference,clock,state=configured(tmp_path,monkeypatch)
    result=experiments.run_suite(native,reference,sample_pair(),Cuda(),clock=clock,report=lambda *a,**k:None)
    assert result['candidate_accepted']
    assert result['additional_speedup_over_v2']>=1.05
    assert result['selected_config']['backend']=='efficient'
    assert [r['mode'] for r in result['runs']]==[
        'original_native','reference_v2','candidate','candidate','reference_v2','original_native']
    assert all(a['compatible'] for a in result['agreements'])
    assert result['measurement_scope']=='held_out_two_passes_against_existing_v2'
    assert state=={'threads':2,'efficient':False}


@pytest.mark.parametrize('setting',['drift','unavailable'])
def test_incompatible_or_unavailable_backend_keeps_v2_and_skips_unneeded_holdout(tmp_path,monkeypatch,setting):
    native,reference,clock,state=configured(tmp_path,monkeypatch,**{setting:True})
    result=experiments.run_suite(native,reference,sample_pair(),Cuda(),clock=clock,report=lambda *a,**k:None)
    assert result['candidate_accepted'] is False and result['selected_config'] is None
    assert result['recommendation']=='KEEP_EXISTING_V2'
    assert result['additional_speedup_over_v2']==1 and result['evaluation_texts']==0
    assert result['measurement_scope']=='calibration_only_no_faster_candidate'
    assert state=={'threads':2,'efficient':False}
    if setting=='unavailable':
        assert any(t['state']=='UNAVAILABLE' for t in result['trials'])


def test_cpu_thread_count_restored_after_unexpected_kernel_failure(tmp_path,monkeypatch):
    native,reference,clock,state=configured(tmp_path,monkeypatch)
    config=next(c for c in experiments.CONFIGS if c['name']=='threads1-b16')
    with pytest.raises(RuntimeError,match='illegal memory'):
        with experiments.execution_context(native.torch,config):
            assert state['threads']==1
            raise RuntimeError('CUDA illegal memory access')
    assert state['threads']==2
    assert experiments._known_unavailable(OOM('capacity'),native.torch)
    assert experiments._known_unavailable(RuntimeError('No available kernel'),native.torch)
    assert experiments._known_unavailable(RuntimeError('cutlass: no kernel found to launch!'),native.torch)
    assert not experiments._known_unavailable(RuntimeError('CUDA illegal memory access'),native.torch)


@pytest.mark.parametrize('failure',['drift','unsupported','fatal'])
def test_held_out_candidate_cannot_hide_new_shape_failures_or_vector_drift(tmp_path,monkeypatch,failure):
    native,reference,clock,state=configured(tmp_path,monkeypatch)
    original=native._batch
    def batch(texts):
        held_out=any(int(text.split(':')[0])>=100 for text in texts)
        if held_out and state['efficient']:
            if failure=='unsupported':raise RuntimeError('No kernel found to launch!')
            if failure=='fatal':raise RuntimeError('CUDA illegal memory access')
            return np.roll(original(texts),1,axis=1)
        return original(texts)
    native._batch=batch
    if failure=='fatal':
        with pytest.raises(RuntimeError,match='illegal memory'):
            experiments.run_suite(native,reference,sample_pair(),Cuda(),clock=clock,report=lambda *a,**k:None)
    else:
        result=experiments.run_suite(native,reference,sample_pair(),Cuda(),clock=clock,report=lambda *a,**k:None)
        assert not result['candidate_accepted'] and result['selected_config'] is None
        assert result['recommendation']=='KEEP_EXISTING_V2'
        if failure=='unsupported':
            assert result['measurement_scope']=='held_out_candidate_failed'
            assert result['additional_speedup_over_v2'] is None
        else:assert not all(a['compatible'] for a in result['agreements'])
    assert state=={'threads':2,'efficient':False}


def test_real_backend_context_restores_hf_dispatch_function_after_failure(monkeypatch):
    original=lambda *a,**k:True
    integration=SimpleNamespace(use_gqa_in_sdpa=original)
    parent=ModuleType('transformers.integrations');parent.sdpa_attention=integration
    monkeypatch.setitem(sys.modules,'transformers.integrations',parent)
    state={'backend':None}
    @contextmanager
    def kernel(backend):
        state['backend']=backend
        try:yield
        finally:state['backend']=None
    attention=ModuleType('torch.nn.attention')
    attention.sdpa_kernel=kernel
    attention.SDPBackend=SimpleNamespace(EFFICIENT_ATTENTION='efficient')
    monkeypatch.setitem(sys.modules,'torch.nn.attention',attention)
    with pytest.raises(RuntimeError,match='synthetic'):
        with experiments.efficient_attention():
            assert integration.use_gqa_in_sdpa(None,None) is False
            assert state['backend']=='efficient'
            raise RuntimeError('synthetic')
    assert integration.use_gqa_in_sdpa is original and state['backend'] is None


def test_extra_notebook_cells_preserve_production_tuner_and_do_not_dispatch_tasks():
    n=notebook()
    assert len(n.cells)==7
    for c in n.cells:
        if c.cell_type=='code':compile(c.source,'<extra-t4>','exec')
    assert "WORK/'qwen_t4_experiments.py'" in n.cells[5].source
    assert "WORK/'benchmark-extended.json'" in n.cells[5].source
    assert 'T4_EXTRA_SUMMARY:' in n.cells[6].source
    assert 'additional_speedup_over_v2' in n.cells[6].source
    assert 'Popen' not in n.cells[6].source
    assert sha256_file(ROOT/'scripts/qwen_token_batching.py')=='a7f242c9fc9a53ba6acd006e1b4d38d79d4140f54a944be06c4922bab0f42c20'
