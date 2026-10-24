"""Direct pinned downloads avoid list-page floods and preserve native validation."""
from contextlib import contextmanager
from pathlib import Path
import os
import sys
import hashlib
import base64
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


@pytest.mark.parametrize('error_type', [requests.exceptions.ChunkedEncodingError,
                                      requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout])
def test_interrupted_network_is_retried_without_resetting_the_operation(error_type):
    calls, waits, messages = [], [], []
    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise error_type('synthetic interrupted stream')
        return 'same pinned transfer completed'
    assert transfer.retry_kaggle(operation, sleep=waits.append, report=messages.append).endswith('completed')
    assert len(calls) == 3 and waits == [15, 30]
    assert all('partial cache retained' in message for message in messages)


def test_persistent_network_failure_is_bounded():
    calls = []
    def operation():
        calls.append(1)
        raise requests.exceptions.ChunkedEncodingError('still broken')
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        transfer.retry_kaggle(operation, attempts=3, sleep=lambda n:None, report=lambda m:None)
    assert len(calls) == 3


@pytest.mark.parametrize('error', [ValueError('invalid manifest'), OSError(28, 'No space left on device'),
                                   requests.exceptions.SSLError('certificate verification failed')])
def test_local_integrity_disk_and_certificate_errors_are_not_retried(error):
    with pytest.raises(type(error)):
        transfer.retry_kaggle(lambda:(_ for _ in ()).throw(error),
                             sleep=lambda n:pytest.fail('must not retry a permanent failure'))


@pytest.mark.parametrize('corrupt', [False, True])
def test_hub_103_resumes_interrupted_stream_and_checks_complete_checksum(tmp_path,monkeypatch,corrupt):
    # Exercise the actual pinned downloader with a synthetic HTTP stream, not
    # just a fake retry callback. No corpus or remote network is used here.
    from kagglehub.clients import download_file, CHUNK_SIZE
    from kagglehub.exceptions import DataCorruptionError
    from kagglehub.handle import parse_notebook_handle
    payload = b'a'*CHUNK_SIZE+b'b'*CHUNK_SIZE
    expected = base64.b64encode(hashlib.md5(payload).digest()).decode()
    destination = tmp_path/'output.archive'
    calls, ranges = [], []
    class Response:
        url = 'https://storage.example.test/pinned-output'
        def __init__(self, resumed=False):
            self.resumed = resumed
            self.headers = requests.structures.CaseInsensitiveDict({
                'Content-Length':str(len(payload)), 'Accept-Ranges':'bytes', 'X-Goog-Hash':'md5='+expected})
            self.status_code = 206 if resumed else 200
        def iter_content(self, chunk_size):
            assert chunk_size == CHUNK_SIZE
            if self.resumed:
                yield (b'c'*CHUNK_SIZE if corrupt else payload[CHUNK_SIZE:])
            else:
                yield payload[:CHUNK_SIZE]
                raise requests.exceptions.ChunkedEncodingError('stream closed mid-transfer')
        def __enter__(self): return self
        def __exit__(self,*args): pass
    def get(url, **options):
        assert url == Response.url
        ranges.append(options['headers']['Range'])
        assert destination.stat().st_size == CHUNK_SIZE
        return Response(resumed=True)
    monkeypatch.setattr(requests,'get',get)
    def operation():
        calls.append(1)
        return download_file(Response(),str(destination),parse_notebook_handle('owner/saved/versions/2'))
    if corrupt:
        with pytest.raises(DataCorruptionError):
            transfer.retry_kaggle(operation, sleep=lambda n:None, report=lambda m:None)
        assert not destination.exists()
    else:
        assert transfer.retry_kaggle(operation, sleep=lambda n:None, report=lambda m:None)
        assert destination.read_bytes() == payload
    assert len(calls) == 2 and ranges == [f'bytes={CHUNK_SIZE}-']


def test_recovery_cell_is_idempotent_and_retries_existing_download(monkeypatch):
    calls, waits = [], []
    def original(operation, **options):
        return operation()
    module = SimpleNamespace(retry_kaggle=original, SavedNotebookOutputs=type('Saved',(),{}))
    monkeypatch.setitem(sys.modules,'kaggle_output_transfer',module)
    monkeypatch.setattr('time.sleep',waits.append)
    source = (Path(__file__).parents[1]/'scripts/resume_kaggle_output_download_cell.py').read_text(encoding='utf-8')
    for _ in range(2):
        exec(compile(source,'<live-recovery-cell>','exec'),{})
    assert module._before_stream_retry is original
    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ChunkedEncodingError('interrupted archive')
        return 'ready'
    assert module.retry_kaggle(operation) == 'ready'
    assert len(calls) == 3 and waits == [5,10]


def test_live_recovery_cell_imports_native_vectors_without_archive(project,tmp_path,monkeypatch):
    flow,target,job = library(project,tmp_path)
    _,plan = flow.create_plan(job)
    remote = tmp_path/'remote'
    _,result = run_account(target,job,plan,remote/'vmb_full_queue',tmp_path/'gpu',project[1],0)
    calls = []
    def download(handle,*,path=None,force_download=False):
        assert path is not None, 'Live patch must not request an archive'
        calls.append(path)
        return str(remote/path)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    # A separate module namespace emulates the helper already loaded in Colab.
    live = SimpleNamespace(retry_kaggle=transfer.retry_kaggle,
        SavedNotebookOutputs=type('SavedInColab',(transfer.SavedNotebookOutputs,),{}))
    monkeypatch.setitem(sys.modules,'kaggle_output_transfer',live)
    source = (Path(__file__).parents[1]/'scripts/resume_kaggle_output_download_cell.py').read_text(encoding='utf-8')
    exec(compile(source,'<live-recovery-cell>','exec'),{})
    saved = live.SavedNotebookOutputs(object(),'owner/saved/2',tmp_path/'work',report=lambda m:None)
    output = saved.all('vmb_full_queue/result/new-data-result.json',result).parent
    flow.install(output)
    assert calls and {r['task'] for r in result['tasks']} <= flow.imported()


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


def test_full_import_downloads_only_named_files_and_never_archive_or_pages(project,tmp_path,monkeypatch):
    flow,target,job = library(project,tmp_path)
    _,plan = flow.create_plan(job)
    saved = tmp_path/'remote'
    _,result = run_account(target,job,plan,saved/'vmb_full_queue',tmp_path/'gpu',project[1],0)
    calls = []
    relative = 'vmb_full_queue/result/new-data-result.json'
    prior_cache = os.environ.get('KAGGLEHUB_CACHE')
    def download(handle,*,path=None,force_download=False):
        assert path is not None, 'An archive request must never be made'
        calls.append((handle,path,force_download))
        assert os.environ['KAGGLEHUB_CACHE'] == str(tmp_path/'work/kagglehub-output-cache')
        return str(saved/path) if path else str(saved)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    restored = download_outputs(api(monkeypatch),'owner/saved',tmp_path/'work',project=project[0],
        allow_failed=True,manifest_path=relative)
    assert restored == (saved/'vmb_full_queue/result')
    assert calls[:2] == [('owner/saved/versions/7',relative,True),('owner/saved/versions/7',relative,False)]
    required = {'job.json','execution-plan.json',*[block['path'] for block in result['blocks']]}
    assert {path for _,path,_ in calls[2:]} == {'vmb_full_queue/result/'+p for p in required}
    assert all(handle == 'owner/saved/versions/7' for handle,_,_ in calls)
    assert os.environ.get('KAGGLEHUB_CACHE') == prior_cache
    record = next((project[0]/'data_staging/kaggle_output_downloads').glob('*.json'))
    assert read_json(record)['state'] == 'BUNDLE_READY_FOR_NATIVE_VALIDATION'
    flow.install(restored)
    assert {r['task'] for r in result['tasks']} <= flow.imported()


def test_missing_named_file_does_not_pass_as_success(project,tmp_path,monkeypatch):
    flow,target,job = library(project,tmp_path)
    _,plan = flow.create_plan(job)
    saved = tmp_path/'remote'
    _,result = run_account(target,job,plan,saved/'vmb_full_queue',tmp_path/'gpu',project[1],0)
    assert result['blocks']
    (saved/'vmb_full_queue/result'/result['blocks'][0]['path']).unlink()
    def download(handle,*,path=None,force_download=False):
        assert path is not None
        return str(saved/path) if path else str(saved)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    with pytest.raises(RuntimeError,match='required output file'):
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
        assert path is not None
        if path.endswith('/job.json'):
            changed = seal({**{k:v for k,v in result.items() if k!='manifest_sha256'},'updated_at':'changed'})
            atomic_json(saved/relative,changed)
        return str(saved/path) if path else str(saved)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    with pytest.raises(ValueError,match='changed after preflight'):
        download_outputs(api(monkeypatch),'owner/saved',tmp_path/'work',allow_failed=True,manifest_path=relative)
    assert not flow.imported()


def test_named_files_resume_reuses_completed_cache_and_retries_only_broken_asset(tmp_path,monkeypatch):
    remote = tmp_path/'cache'
    relative = 'vmb_full_queue/result/new-data-result.json'
    root = remote/'vmb_full_queue/result'
    root.mkdir(parents=True)
    (remote/relative).write_text('manifest')
    calls, waits, complete = [], [], set()
    fail = {'active':True}
    def download(handle,*,path=None,force_download=False):
        assert path is not None
        calls.append(path)
        if path == relative:
            return str(remote/path)
        if path not in complete:
            if path.endswith('/vector-1.npy') and fail['active']:
                raise requests.exceptions.ChunkedEncodingError('one file repeatedly interrupted')
            target = remote/path
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('downloaded named asset')
            complete.add(path)
        return str(remote/path)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    # Keep retry limits and avoid sleeping for the synthetic permanent failure.
    original = transfer.retry_kaggle
    monkeypatch.setattr(transfer,'retry_kaggle',lambda op,**kw:original(op,**kw,attempts=2,sleep=waits.append))
    saved = transfer.SavedNotebookOutputs(object(),'owner/saved/2',tmp_path/'work',report=lambda m:None)
    manifest = {'manifest_sha256':'a'*64,'blocks':[{'path':'vector-0.npy'},{'path':'vector-1.npy'}]}
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        saved.all(relative,manifest)
    assert 'vmb_full_queue/result/vector-0.npy' in complete
    before = complete.copy()
    fail['active'] = False
    assert saved.all(relative,manifest) == remote/relative
    assert before <= complete and 'vmb_full_queue/result/vector-1.npy' in complete
    assert all(p is not None for p in calls)
    assert read_json(saved.record)['state'] == 'BUNDLE_READY_FOR_NATIVE_VALIDATION'


@pytest.mark.parametrize('unsafe', ['../outside.npy','/absolute.npy','x\\y.npy','x/../y.npy','C:/file.npy'])
def test_unsafe_manifest_asset_is_rejected_before_additional_downloads(tmp_path,monkeypatch,unsafe):
    root = tmp_path/'saved'
    root.mkdir()
    manifest = root/'new-data-result.json'
    manifest.write_text('manifest')
    calls = []
    def download(handle,*,path=None,force_download=False):
        calls.append(path)
        return str(manifest)
    monkeypatch.setitem(sys.modules,'kagglehub',SimpleNamespace(notebook_output_download=download))
    saved = transfer.SavedNotebookOutputs(object(),'owner/saved/2',tmp_path/'work',report=lambda m:None)
    with pytest.raises(ValueError,match='Unsafe saved vector path'):
        saved.all('new-data-result.json',{'manifest_sha256':'a'*64,'blocks':[{'path':unsafe}]})
    assert calls == ['new-data-result.json']


def test_explicit_version_is_not_replaced_by_current_metadata(tmp_path):
    transport = transfer.SavedNotebookOutputs(object(),'owner/saved/3',tmp_path,report=lambda m:None)
    assert transport.pinned == 'owner/saved/versions/3'
