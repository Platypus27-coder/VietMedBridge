"""Fetch signed URLs in pages; retain old Hub cache and strict native import."""
from contextlib import contextmanager
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import sys
from types import SimpleNamespace

import pytest
import requests

from test_new_data_kaggle import project
from test_full_queue_embeddings import library, run_account
from scripts import kaggle_output_paged as pages
from scripts import kaggle_output_transfer as transfer
from scripts.new_data_kaggle import download_outputs
from vietmedbridge.artifacts import read_json


def setup_remote(project,tmp_path,monkeypatch,*,missing=None,corrupt=False):
    from kagglehub.cache import Cache
    from kagglehub.handle import parse_notebook_handle
    flow,target,job = library(project,tmp_path)
    _,plan = flow.create_plan(job)
    remote = tmp_path/'remote'
    _,manifest = run_account(target,job,plan,remote/'vmb_full_queue',tmp_path/'gpu',project[1],0)
    work = tmp_path/'work'
    pinned = 'owner/saved/versions/2'
    relative = 'vmb_full_queue/result/new-data-result.json'
    assets = [f'vmb_full_queue/result/{name}' for name in
              ['job.json','execution-plan.json',*[b['path'] for b in manifest['blocks']]]]
    if missing:
        assets.remove(next(p for p in assets if p.endswith(missing)))
    listed,downloads,hub_calls = [],[],[]
    def hub(handle,*,path=None,force_download=False):
        assert handle == pinned and path == relative, 'No per-vector download API or archive allowed'
        hub_calls.append(path)
        cache = Cache()
        dst = Path(cache.get_path(parse_notebook_handle(handle),path))
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(remote/path,dst)
        cache.mark_as_complete(parse_notebook_handle(handle),path)
        return str(dst)
    monkeypatch.setitem(sys.modules,'kagglehub',__import__('kagglehub'))
    monkeypatch.setattr(sys.modules['kagglehub'],'notebook_output_download',hub)
    class API:
        def kernels_status(self,handle): return {'status':'ERROR'}
        @contextmanager
        def build_kaggle_client(self):
            def output(request):
                assert (request.user_name,request.kernel_slug,request.version_label,request.page_size) == (
                    'owner','saved','v2',200)
                offset = int(request.page_token or '0')
                listed.append(offset)
                selected = assets[offset:offset+2]
                items = [SimpleNamespace(file_name=p,url='https://files.example.test/'+p) for p in selected]
                # Unreferenced logs must never be downloaded.
                items.append(SimpleNamespace(file_name='gpu-worker.log',url='https://files.example.test/log'))
                return SimpleNamespace(files=items,next_page_token=str(offset+2) if offset+2<len(assets) else '')
            yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
                list_kernel_session_output=output)))
    class Response:
        status_code = 200
        def __init__(self,payload):
            self.payload = payload
            self.headers = {'Content-Length':str(len(payload))}
        def raise_for_status(self): pass
        def iter_content(self,size): yield self.payload
        def __enter__(self): return self
        def __exit__(self,*args): pass
    def get(url,**options):
        name = url.removeprefix('https://files.example.test/')
        assert name in assets and options['stream']
        downloads.append(name)
        payload = (remote/name).read_bytes()
        if corrupt and name.endswith('.npy'):
            payload = b'bad vector'
        return Response(payload)
    monkeypatch.setattr(requests,'get',get)
    monkeypatch.setattr(transfer.SavedNotebookOutputs,'all',pages.download_paged_outputs)
    return flow,remote,manifest,work,API(),listed,downloads,hub_calls,pinned,relative


def test_native_import_uses_paged_signed_urls_and_reuses_previously_downloaded_hub_assets(project,tmp_path,monkeypatch):
    from kagglehub.cache import Cache
    from kagglehub.handle import parse_notebook_handle
    flow,remote,result,work,api,listed,downloads,hub,pinned,relative = setup_remote(project,tmp_path,monkeypatch)
    old = 'vmb_full_queue/result/'+result['blocks'][0]['path']
    with transfer._cache_at(work/'kagglehub-output-cache'):
        cache = Cache()
        dst = Path(cache.get_path(parse_notebook_handle(pinned),old))
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(remote/old,dst)
        cache.mark_as_complete(parse_notebook_handle(pinned),old)
    output = download_outputs(api,'owner/saved/2',work,allow_failed=True,manifest_path=relative)
    assert old not in downloads and hub == [relative]
    assert listed and len(listed)<len(downloads)+2
    flow.install(output)
    assert {r['task'] for r in result['tasks']} <= flow.imported()
    before = (len(listed),len(downloads))
    # A second attempt resolves only cached named files, with no listing or payload fetch.
    download_outputs(api,'owner/saved/2',work,allow_failed=True,manifest_path=relative)
    assert (len(listed),len(downloads)) == before


def test_changed_vector_checksum_never_publishes_a_cache_marker_or_import(project,tmp_path,monkeypatch):
    flow,remote,result,work,api,listed,downloads,hub,pinned,relative = setup_remote(project,tmp_path,monkeypatch,corrupt=True)
    with pytest.raises(ValueError,match='checksum differs'):
        download_outputs(api,'owner/saved/2',work,allow_failed=True,manifest_path=relative)
    assert not flow.imported()
    from kagglehub.cache import Cache
    from kagglehub.handle import parse_notebook_handle
    with transfer._cache_at(work/'kagglehub-output-cache'):
        assert Cache().load_from_cache(parse_notebook_handle(pinned),
            'vmb_full_queue/result/'+result['blocks'][0]['path']) is None


def test_missing_file_stops_before_native_import_and_retains_completed_files(project,tmp_path,monkeypatch):
    flow,remote,result,work,api,listed,downloads,hub,pinned,relative = setup_remote(project,tmp_path,monkeypatch,missing='job.json')
    with pytest.raises(RuntimeError,match='required files absent'):
        download_outputs(api,'owner/saved/2',work,allow_failed=True,manifest_path=relative)
    assert downloads and not flow.imported()


def test_page_cursor_resumes_after_last_completed_page(project,tmp_path,monkeypatch):
    flow,remote,result,work,api,listed,downloads,hub,pinned,relative = setup_remote(project,tmp_path,monkeypatch)
    original = requests.get
    failing = {'enabled':True}
    old_retry = transfer.retry_kaggle
    monkeypatch.setattr(transfer,'retry_kaggle',lambda op,**kw:old_retry(op,**kw,attempts=1))
    def get(url,**options):
        if failing['enabled'] and url.endswith('.npy'):
            raise requests.exceptions.ChunkedEncodingError('synthetic cutoff')
        return original(url,**options)
    monkeypatch.setattr(requests,'get',get)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download_outputs(api,'owner/saved/2',work,allow_failed=True,manifest_path=relative)
    cursor = next((work/'kagglehub-output-cache/vmb-output-pages').glob('*.json'))
    assert read_json(cursor)['next_token'] == '2'
    cutoff = len(listed)
    failing['enabled'] = False
    output = download_outputs(api,'owner/saved/2',work,allow_failed=True,manifest_path=relative)
    assert listed[cutoff] == 2
    flow.install(output)
    assert flow.imported()


@pytest.mark.parametrize('ignore_range',[False,True])
def test_partial_stream_resumes_or_safely_restarts_tempfile(tmp_path,monkeypatch,ignore_range):
    payload = b'a'*1024+b'b'*1024
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path/'asset.npy'
    calls = []
    class Response:
        def __init__(self,first):
            self.first = first
            self.status_code = 200 if first or ignore_range else 206
            self.headers = {'Content-Length':str(len(payload))}
            if self.status_code == 206:
                self.headers['Content-Range'] = 'bytes 1024-2047/2048'
        def raise_for_status(self): pass
        def iter_content(self,size):
            if self.first:
                yield payload[:1024]
                raise requests.exceptions.ChunkedEncodingError('cutoff')
            yield payload if ignore_range else payload[1024:]
        def __enter__(self): return self
        def __exit__(self,*args): pass
    def get(url,**options):
        calls.append(options['headers'])
        return Response(len(calls)==1)
    monkeypatch.setattr(requests,'get',get)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        pages._asset('https://files.example.test/a',destination,expected)
    assert not destination.exists()
    assert pages._asset('https://files.example.test/a',destination,expected) == len(payload)
    assert calls[1]['Range'] == 'bytes=1024-'
    assert destination.read_bytes() == payload
