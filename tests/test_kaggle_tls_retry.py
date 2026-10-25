"""Retry the observed TLS protocol alert without changing certificate checks."""
import hashlib
from pathlib import Path
import ssl
import sys
from types import SimpleNamespace

import pytest
import requests
from urllib3.exceptions import MaxRetryError, SSLError as UrllibSSLError

from scripts import kaggle_output_transfer as transfer
from scripts.kaggle_output_paged import _asset


def alert():
    nested = ssl.SSLError(1,'[SSL: TLSV1_ALERT_DECODE_ERROR] tlsv1 alert decode error')
    return requests.exceptions.SSLError(MaxRetryError(None,'/redacted',reason=UrllibSSLError(nested)))


def test_wrapped_tls_alert_retries_same_operation():
    calls,waits,messages = [],[],[]
    def operation():
        calls.append(1)
        if len(calls)<3: raise alert()
        return 'ready'
    assert transfer.retry_kaggle(operation,sleep=waits.append,report=messages.append)=='ready'
    assert len(calls)==3 and waits==[15,30]
    assert all('SSLError' in m and '/redacted' not in m for m in messages)


@pytest.mark.parametrize('reason',[
    'CERTIFICATE_VERIFY_FAILED', 'certificate verification failed', 'hostname mismatch',
    'certificate expired', 'WRONG_VERSION_NUMBER',
    'TLSV1_ALERT_DECODE_ERROR with CERTIFICATE_VERIFY_FAILED',
])
def test_other_tls_and_certificate_failures_are_not_retried(reason):
    calls=[]
    def operation():
        calls.append(1)
        raise requests.exceptions.SSLError(reason)
    with pytest.raises(requests.exceptions.SSLError):
        transfer.retry_kaggle(operation,sleep=lambda n:pytest.fail('must not retry'),report=lambda m:None)
    assert calls==[1]


def test_repeated_decode_alert_is_bounded():
    calls=[]
    def operation():
        calls.append(1)
        raise alert()
    with pytest.raises(requests.exceptions.SSLError):
        transfer.retry_kaggle(operation,attempts=3,sleep=lambda n:None,report=lambda m:None)
    assert len(calls)==3


def test_signed_file_retries_handshake_then_publishes_only_verified_payload(tmp_path,monkeypatch):
    payload=b'verified vector fixture'
    calls=[]
    class Response:
        status_code=200
        headers={'Content-Length':str(len(payload))}
        def raise_for_status(self): pass
        def iter_content(self,size): yield payload
        def __enter__(self): return self
        def __exit__(self,*args): pass
    def get(url,**options):
        calls.append(1)
        assert options.get('verify',True) is True
        if len(calls)==1: raise alert()
        return Response()
    monkeypatch.setattr(requests,'get',get)
    target=tmp_path/'vector.npy'
    result=transfer.retry_kaggle(lambda:_asset('https://files.example.test/fixture',target,
        hashlib.sha256(payload).hexdigest()),sleep=lambda n:None,report=lambda m:None)
    assert result==len(payload) and target.read_bytes()==payload and len(calls)==2


def test_live_tls_patch_is_idempotent(monkeypatch):
    waits,calls=[],[]
    def original(operation,**options): return operation()
    live=SimpleNamespace(retry_kaggle=original)
    monkeypatch.setitem(sys.modules,'kaggle_output_transfer',live)
    monkeypatch.setattr('time.sleep',waits.append)
    source=(Path(__file__).parents[1]/'scripts/resume_kaggle_tls_cell.py').read_text(encoding='utf-8')
    for _ in range(2): exec(compile(source,'<tls-resume-cell>','exec'),{})
    assert live._before_tls_retry is original
    def operation():
        calls.append(1)
        if len(calls)<3: raise alert()
        return 'ready'
    assert live.retry_kaggle(operation)=='ready' and waits==[1,2] and len(calls)==3


def test_live_tls_patch_does_not_wrap_new_runtime_with_existing_tls_support(monkeypatch):
    live=SimpleNamespace(retry_kaggle=transfer.retry_kaggle)
    monkeypatch.setitem(sys.modules,'kaggle_output_transfer',live)
    source=(Path(__file__).parents[1]/'scripts/resume_kaggle_tls_cell.py').read_text(encoding='utf-8')
    exec(compile(source,'<tls-resume-cell>','exec'),{})
    assert live.retry_kaggle is transfer.retry_kaggle and not hasattr(live,'_before_tls_retry')
