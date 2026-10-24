"""Direct pinned downloads avoid list-page floods and preserve native validation."""
from contextlib import contextmanager
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest
import requests

from test_new_data_kaggle import project
from test_full_queue_embeddings import library, run_account
from scripts import kaggle_output_transfer as transfer
from scripts.new_data_kaggle import download_outputs
from vietmedbridge.artifacts import read_json, atomic_json
from vietmedbridge.content_embeddings import seal


def http_error(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers['Retry-After'] = str(retry_after)
    return requests.HTTPError('synthetic API failure', response=response)


def test_rate_limit_retries_same_operation_and_respects_retry_after():
    calls, waits, messages = [], [], []
    def operation():
        calls.append('same operation')
        if len(calls) == 1:
            raise http_error(429, 3)
        return 'ready'
    assert transfer.retry_kaggle(operation, sleep=waits.append, report=messages.append) == 'ready'
    assert len(calls) == 2 and waits == [3]
    assert '429' in messages[0]


@pytest.mark.parametrize('status', [401,403,404])
def test_access_and_missing_file_errors_are_not_retried(status):
    calls = []
    def operation():
        calls.append(1)
        raise http_error(status)
    with pytest.raises(requests.HTTPError):
        transfer.retry_kaggle(operation, sleep=lambda n:pytest.fail('must not sleep'))
    assert len(calls) == 1


def test_long_cooldown_does_not_spin_or_drop_cache():
    with pytest.raises(RuntimeError, match='retain this runtime/cache'):
        transfer.retry_kaggle(lambda:(_ for _ in ()).throw(http_error(429, 3600)),
                             sleep=lambda n:pytest.fail('must not sleep an hour'))


def api(monkeypatch, number=7):
    monkeypatch.setattr(transfer, '_request', SimpleNamespace)
    class API:
        def kernels_status(self, handle):
            return {'status':'ERROR'}
        def kernels_output(self, *args, **kwargs):
            pytest.fail('The optimized importer must never enumerate output pages')
        @contextmanager
        def build_kaggle_client(self):
            def metadata(request):
                assert request.user_name == 'owner' and request.kernel_slug == 'saved'
                return SimpleNamespace(metadata=SimpleNamespace(current_version_number=number))
            yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(get_kernel=metadata)))
    return API()


def test_full_import_downloads_named_manifest_and_one_pinned_bundle(project,tmp_path,monkeypatch):
    flow,target,job = library(project,tmp_path)
    _,plan = flow.create_plan(job)
    saved = tmp_path/'remote'
    _,result = run_account(target,job,plan,saved/'vmb_full_queue',tmp_path/'gpu',project[1],0)
    calls = []
    relative = 'vmb_full_queue/result/new-data-result.json'
    prior_cache = os.environ.get('KAGGLEHUB_CACHE')
    def download(handle,*,path=None,force_download=False):
        calls.append((handle,path,force_download))
        assert os.environ['KAGGLEHUB_CACHE'] == str(tmp_path/'work/kagglehub-output-cache')
        return str(saved/path) if path else str(saved)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    restored = download_outputs(api(monkeypatch),'owner/saved',tmp_path/'work',project=project[0],
        allow_failed=True,manifest_path=relative)
    assert restored == (saved/'vmb_full_queue/result')
    assert calls == [('owner/saved/versions/7',relative,True),('owner/saved/versions/7',None,False)]
    assert os.environ.get('KAGGLEHUB_CACHE') == prior_cache
    record = next((project[0]/'data_staging/kaggle_output_downloads').glob('*.json'))
    assert read_json(record)['state'] == 'BUNDLE_READY_FOR_NATIVE_VALIDATION'
    flow.install(restored)
    assert {r['task'] for r in result['tasks']} <= flow.imported()


def test_partial_archive_does_not_pass_as_success(project,tmp_path,monkeypatch):
    flow,target,job = library(project,tmp_path)
    _,plan = flow.create_plan(job)
    saved = tmp_path/'remote'
    _,result = run_account(target,job,plan,saved/'vmb_full_queue',tmp_path/'gpu',project[1],0)
    assert result['blocks']
    (saved/'vmb_full_queue/result'/result['blocks'][0]['path']).unlink()
    def download(handle,*,path=None,force_download=False):
        return str(saved/path) if path else str(saved)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    with pytest.raises(RuntimeError,match='archive is incomplete'):
        download_outputs(api(monkeypatch),'owner/saved',tmp_path/'work',allow_failed=True,
            manifest_path='vmb_full_queue/result/new-data-result.json')
    assert not flow.imported()


def test_changed_bundle_manifest_cannot_be_imported(project,tmp_path,monkeypatch):
    flow,target,job = library(project,tmp_path)
    _,plan = flow.create_plan(job)
    saved = tmp_path/'remote'
    _,result = run_account(target,job,plan,saved/'vmb_full_queue',tmp_path/'gpu',project[1],0)
    relative = 'vmb_full_queue/result/new-data-result.json'
    def download(handle,*,path=None,force_download=False):
        if path is None:
            changed = seal({**{k:v for k,v in result.items() if k!='manifest_sha256'},'updated_at':'changed'})
            atomic_json(saved/relative,changed)
        return str(saved/path) if path else str(saved)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    with pytest.raises(ValueError,match='changed after preflight'):
        download_outputs(api(monkeypatch),'owner/saved',tmp_path/'work',allow_failed=True,manifest_path=relative)
    assert not flow.imported()


def test_explicit_version_is_not_replaced_by_current_metadata(tmp_path):
    transport = transfer.SavedNotebookOutputs(object(),'owner/saved/3',tmp_path,report=lambda m:None)
    assert transport.pinned == 'owner/saved/versions/3'
