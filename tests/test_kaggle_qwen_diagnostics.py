"""Tuning reports stay pinned, small, resumable, and separate from vector import."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from scripts.kaggle_qwen_diagnostics import read_qwen_tuning, tuning_summary
from vietmedbridge.artifacts import atomic_json, read_json
from vietmedbridge.content_embeddings import seal


ENCODER={'model_id':'Qwen/Qwen3-Embedding-8B','revision':'locked','role':'corpus'}


def diagnostic_fixture(tmp_path, *, account=0):
    plan=seal({'team':{'accounts':4,'gpu_count':2},'qwen_execution':'qwen-token-batches-v2'})
    result=seal({'job':'a'*64,'execution_plan':plan['manifest_sha256'],'plan':plan,'account_id':account})
    root=tmp_path/'remote'
    value={'encoder':ENCODER,'policy':'qwen-token-batches-v2','mode':'native','kernel':'native',
           'batch_cap':16,'baseline_seconds':10,'selected_seconds':10,'measured_sample_speedup':1,
           'trials':[{'batch_cap':32,'compatible':False,'reason':'vector_validation_failed'}],
           'sample_texts':64,'baseline_texts_per_second':6.4,'selected_texts_per_second':6.4}
    for w in range(account*2,account*2+2):
        atomic_json(root/f'vmb_full_queue/result/qwen-tuning/worker-{w}.json',value)
    calls=[]
    def download(*,path):
        calls.append(path)
        assert path.startswith('vmb_full_queue/result/qwen-tuning/') and path.endswith('.json')
        return root/path
    transport=SimpleNamespace(pinned='owner/notebook/versions/3',_download=download)
    return result,root,value,transport,calls


def test_reports_for_each_account_use_execution_workers_and_restore_from_one_drive_summary(tmp_path):
    result,root,value,transport,calls=diagnostic_fixture(tmp_path,account=3)
    messages=[]
    project=tmp_path/'drive-project'
    saved=read_qwen_tuning(transport,result,project,ENCODER,report=messages.append)
    assert calls == [f'vmb_full_queue/result/qwen-tuning/worker-{w}.json' for w in (6,7)]
    assert [w['worker_id'] for w in saved['workers']] == [6,7]
    assert saved['tasks_committed'] == 0 and saved['state'] == 'QWEN_TUNING_REPORTS_SAVED'
    assert saved['pinned_handle'] == transport.pinned
    assert all(w['mode'] == 'native' and w['batch_cap'] == 16 for w in saved['workers'])
    assert saved['workers'][0]['trials'][0]['compatible'] is False
    path=project/'data_staging/kaggle_qwen_diagnostics'/(result['manifest_sha256']+'.json')
    assert read_json(path) == saved
    transport._download=lambda **k:pytest.fail('A cold retry queried immutable sidecars again.')
    assert read_qwen_tuning(transport,result,project,ENCODER,report=messages.append) == saved
    assert any('REPORTS_REUSED' in m for m in messages)


def test_missing_report_does_not_discard_the_other_report_or_expose_signed_urls(tmp_path):
    result,root,value,transport,calls=diagnostic_fixture(tmp_path)
    original=transport._download
    def download(*,path):
        if path.endswith('worker-1.json'):
            error=requests.HTTPError('https://files.test/?secret-signed-token=hidden')
            error.response=SimpleNamespace(status_code=404)
            raise error
        return original(path=path)
    transport._download=download
    messages=[]
    saved=read_qwen_tuning(transport,result,tmp_path/'drive',ENCODER,report=messages.append)
    assert saved['state'] == 'QWEN_TUNING_REPORTS_INCOMPLETE'
    assert [w['state'] for w in saved['workers']] == ['READ','UNAVAILABLE']
    assert saved['workers'][1]['http_status'] == 404
    assert 'hidden' not in json.dumps(saved) and all('hidden' not in m for m in messages)


@pytest.mark.parametrize('change',[{'encoder':{'model_id':'wrong'}},{'baseline_seconds':float('nan')},
                                  {'policy':'unknown'},{'mode':'unsupported'}])
def test_foreign_or_invalid_tuning_report_is_not_presented_as_a_valid_measurement(tmp_path,change):
    result,root,value,transport,calls=diagnostic_fixture(tmp_path)
    bad={**value,**change}
    with pytest.raises(ValueError):
        tuning_summary(bad,ENCODER)
    atomic_json(root/'vmb_full_queue/result/qwen-tuning/worker-0.json',bad)
    saved=read_qwen_tuning(transport,result,tmp_path/'drive',ENCODER,report=lambda *a:None)
    assert saved['workers'][0]['state'] == 'UNAVAILABLE'
    assert saved['workers'][1]['state'] == 'READ'


def test_changed_execution_account_assignment_is_rejected_before_any_sidecar_fetch(tmp_path):
    result,root,value,transport,calls=diagnostic_fixture(tmp_path)
    bad=seal({k:v for k,v in {**result,'account_id':4}.items() if k!='manifest_sha256'})
    with pytest.raises(ValueError,match='execution account'):
        read_qwen_tuning(transport,bad,tmp_path/'drive',ENCODER)
    assert not calls
