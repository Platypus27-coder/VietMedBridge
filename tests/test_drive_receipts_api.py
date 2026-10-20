"""API pagination, bounded media and offline lookup resume with no FUSE reads."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.drive_receipts_api import DriveReceiptSource, RemoteReceipt
import scripts.drive_receipts_api as transport
import scripts.submission_gpu_progress as adapter
from test_submission_cache_lookup import blocks
from vietmedbridge.artifacts import digest_json, read_json
from vietmedbridge.content_embeddings import seal
from test_submission_workflow import prepared, frozen, external, setup_workflow
from vietmedbridge.scale_benchmark import load_handoff


class Response:
    def __init__(self, value=None, payload=b'', status=200):
        self.value, self.payload, self.status_code = value, payload, status
    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()
    def close(self):
        pass
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')
    def json(self):
        return self.value
    def iter_content(self, **kwargs):
        yield self.payload


def client_with_responses(tmp_path, responses):
    calls = []
    class Session:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return responses.pop(0)
    return DriveReceiptSource(Session, tmp_path, tmp_path, data_folder_id='data-id'), calls


def test_api_follows_all_pages_and_refuses_incomplete_search(tmp_path):
    source, calls = client_with_responses(tmp_path, [
        Response({'files':[{'id':'a'}], 'nextPageToken':'next'}),
        Response({'files':[{'id':'b'}]})])
    assert list(source._list("'folder' in parents")) == [{'id':'a'}, {'id':'b'}]
    assert calls[1][1]['params']['pageToken'] == 'next'
    source, _ = client_with_responses(tmp_path, [Response({'files':[], 'incompleteSearch':True})])
    with pytest.raises(ValueError, match='incomplete'):
        list(source._list('query'))


def test_payload_directory_listing_is_shared_by_many_vector_downloads(tmp_path,monkeypatch):
    source,_ = client_with_responses(tmp_path,[])
    source.folders[source.folder_id,'model_cache']='cache'
    source.folders['cache','producer']='producer'
    calls=[]
    def listing(query):
        calls.append(query)
        return iter([{'id':'a','name':'a.npy','size':'10'}, {'id':'b','name':'b.npy','size':'20'}])
    monkeypatch.setattr(source,'_list',listing)
    assert source.payload_metadata('model_cache/producer/a.npy')['id']=='a'
    assert source.payload_metadata('model_cache/producer/b.npy')['id']=='b'
    assert len(calls)==1


def test_api_transient_http_retries_but_permission_errors_do_not_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.time, 'sleep', lambda _:None)
    source, calls = client_with_responses(tmp_path, [Response(status=429), Response({'files':[]})])
    assert list(source._list('query')) == [] and len(calls) == 2
    source, calls = client_with_responses(tmp_path, [Response(status=403)])
    with pytest.raises(RuntimeError, match='403'):
        list(source._list('query'))
    assert len(calls) == 1


def test_api_media_is_bounded_and_checksum_verified(tmp_path):
    payload = b'{"hello": "receipt"}\n'
    metadata = {'id':'receipt', 'name':'r.json', 'size':str(len(payload)),
        'md5Checksum':hashlib.md5(payload).hexdigest()}
    source, _ = client_with_responses(tmp_path, [Response(payload=payload)])
    assert source.read_json(metadata) == {'hello':'receipt'}
    source, _ = client_with_responses(tmp_path, [Response(payload=b'corrupt')])
    with pytest.raises(ValueError, match='size/checksum'):
        source.read_json(metadata)
    source, calls = client_with_responses(tmp_path, [])
    with pytest.raises(ValueError, match='bounded JSON'):
        source.read_json(metadata | {'size':str(17*1024*1024)})
    assert calls == []


def test_api_wrong_candidate_source_is_refused(tmp_path, monkeypatch):
    source, _ = client_with_responses(tmp_path, [])
    monkeypatch.setattr(source, 'relative_json', lambda _: {'candidate_manifest_sha256':'other'})
    with pytest.raises(ValueError, match='expected frozen'):
        source.verify_candidate('expected')


def test_api_candidate_digest_is_verified(tmp_path, monkeypatch):
    source, _ = client_with_responses(tmp_path, [])
    candidate = {'state':'FROZEN_CANDIDATE'}
    expected = digest_json(candidate)
    values = [{'candidate_manifest_sha256':expected, 'build_run':'build', 'candidate_name':'candidate.json'},
        candidate | {'candidate_manifest_sha256':expected}]
    monkeypatch.setattr(source, 'relative_json', lambda _: values.pop(0))
    source.verify_candidate(expected)


def test_receipt_listing_filters_suffix_locally_after_complete_parent_listing(tmp_path, monkeypatch):
    source, _ = client_with_responses(tmp_path, [])
    monkeypatch.setattr(source, 'child', lambda parent, name, **kw: {'id':name})
    names = ['block-'+'a'*64+'.done.json', 'block-'+'b'*64+'.done.json', 'block-'+'a'*64+'.npy']
    def listing(query):
        assert 'name contains' not in query, 'Drive name-contains cannot match suffixes'
        assert 'in parents' in query
        return [{'id':str(i), 'name':name, 'size':'10'} for i,name in enumerate(names)]
    monkeypatch.setattr(source, '_list', listing)
    assert [p.name for p in source.receipts({'producer':'test'})] == names[:2]


def api_from_blocks(blocks):
    root, _, _, _, _, markers = blocks
    values = {path.name:read_json(path) for path in markers}
    class API:
        requested_root = data_root = root
        def __init__(self):
            self.reads = []
        def relative_json(self, _):
            return None
        def receipts(self, identity):
            return [RemoteReceipt(self, {'name':name}) for name in sorted(values)]
        def read_json(self, metadata):
            self.reads.append(metadata['name'])
            return copy.deepcopy(values[metadata['name']])
    return API(), values


def test_api_inventory_and_reuse_never_read_or_write_mount_receipts(blocks, monkeypatch):
    root, work, encoder, texts, expected, _ = blocks
    api, values = api_from_blocks(blocks)
    monkeypatch.setattr(adapter, 'read_json', lambda _:pytest.fail('FUSE JSON read during API inventory'))
    monkeypatch.setattr(adapter, 'publish_file', lambda *a, **k:pytest.fail('FUSE snapshot publish during API inventory'))
    monkeypatch.setattr(adapter.NativeContentBlocks, 'register', lambda *a:pytest.fail('Native register resolves FUSE paths'))
    lookup = adapter.ResumableContentBlocks(root, encoder, work / 'api', receipt_api=api)
    try:
        assert lookup.refs(texts,batch_size=1,encode_missing=False) == (expected,0)
        assert encoder.calls == 3 and len(api.reads) == 3
    finally:
        lookup.close()
    api.reads.clear()
    resumed = adapter.ResumableContentBlocks(root, encoder, work / 'api', receipt_api=api)
    try:
        assert resumed.refs(texts,batch_size=1,encode_missing=False) == (expected,0)
        assert api.reads == []
    finally:
        resumed.close()


def test_api_transport_failure_keeps_committed_prefix_for_next_attempt(blocks):
    import requests
    root, work, encoder, texts, expected, _ = blocks
    api, values = api_from_blocks(blocks)
    first, second, *_ = sorted(values)
    original = api.read_json
    def fail(metadata):
        if metadata['name'] == second:
            raise requests.ConnectionError('API request interrupted')
        return original(metadata)
    api.read_json = fail
    with pytest.raises(requests.ConnectionError):
        adapter.ResumableContentBlocks(root, encoder, work / 'partial-api', receipt_api=api)
    def recover(metadata):
        assert metadata['name'] != first, 'Committed API prefix downloaded again'
        return original(metadata)
    api.read_json = recover
    lookup = adapter.ResumableContentBlocks(root, encoder, work / 'partial-api', receipt_api=api)
    try:
        assert lookup.refs(texts,batch_size=1,encode_missing=False) == (expected,0)
        assert encoder.calls == 3
    finally:
        lookup.close()


def test_api_restores_existing_published_lookup_without_receipt_redownload(blocks):
    import shutil
    from vietmedbridge.artifacts import verify_file
    root, work, encoder, texts, expected, _ = blocks
    existing = adapter.ResumableContentBlocks(root, encoder, work / 'previous-runtime')
    remote = existing.remote
    existing.close()
    api, _ = api_from_blocks(blocks)
    api.relative_json = lambda name: read_json(root / name) if (root / name).is_file() else None
    api.relative = lambda name: {'path':root / name}
    def download(metadata, target, *, expected_sha256):
        shutil.copyfile(metadata['path'], target)
        verify_file(target, expected_sha256)
    api.download = download
    api.read_json = lambda _:pytest.fail('Old API receipt redownloaded after snapshot restore')
    restored = adapter.ResumableContentBlocks(root, encoder, work / 'new-api-runtime', receipt_api=api)
    try:
        assert restored.refs(texts,batch_size=1,encode_missing=False) == (expected,0)
    finally:
        restored.close()


@pytest.mark.parametrize('fault', ['identity', 'escape'])
def test_api_receipt_producer_and_reference_paths_remain_guarded(blocks, fault):
    root, work, encoder, *_ = blocks
    api, values = api_from_blocks(blocks)
    name = sorted(values)[0]
    value = values[name]
    if fault == 'identity':
        value['identity'] = {'other':'producer'}
    else:
        value['path'] = '../outside.npy'
    values[name] = seal({k:v for k,v in value.items() if k != 'manifest_sha256'})
    with pytest.raises(ValueError, match='producer/key|escapes'):
        adapter.ResumableContentBlocks(root, encoder, work / 'bad-api', receipt_api=api)


def test_api_preparation_uses_saved_producer_without_allocating_a_model(prepared, monkeypatch):
    p = prepared
    setup_workflow(p, monkeypatch)
    _, candidate, inputs = load_handoff(p.root)
    config = read_json(p.checkout / 'configs/retrieval_full.json')
    view = p.root / adapter.resources_relative(candidate, config) / 'bge'
    header = read_json(view / 'config.json')
    identity = {'encoder':header['encoder'], 'dimension':1024, 'format':'content-float32-l2-v1'}
    folder = p.root / 'model_cache' / digest_json(identity)[:24]
    values = {path.name:read_json(path) for path in folder.glob('*.done.json')}
    class API:
        requested_root = data_root = p.root
        active = {'index_inputs_manifest_sha256':inputs['manifest_sha256']}
        def relative_json(self, name):
            return header if name.endswith('bge/config.json') else None
        def receipts(self, _):
            return [RemoteReceipt(self, {'name':name}) for name in sorted(values)]
        def read_json(self, metadata):
            return values[metadata['name']]
    api = API()
    api.candidate = candidate
    from vietmedbridge import retrieval_models
    monkeypatch.setattr(retrieval_models, 'TorchDenseEncoder', lambda *a, **k:pytest.fail('Weights allocated for API preparation'))
    before = dict(p.counters)
    adapter.prepare_drive_receipt_lookup(api, p.checkout, p.work)
    assert p.counters == before
    lookup = adapter.ResumableContentBlocks(p.root,
        type('Producer', (), {'identity':header['encoder'], 'dimension':1024})(),
        p.work, receipt_api=api)
    try:
        assert lookup.con.execute('SELECT count(*) FROM receipts').fetchone()[0] == len(values)
    finally:
        lookup.close()


def test_api_stage_failure_diagnostics_do_not_touch_the_dead_mount(tmp_path, monkeypatch):
    real_atomic = adapter.atomic_json
    root, work = tmp_path / 'drive', tmp_path / 'work'
    def local_only(path, value):
        assert not Path(path).is_relative_to(root), 'Diagnostic write hit the unavailable Drive'
        real_atomic(path, value)
    monkeypatch.setattr(adapter, 'atomic_json', local_only)
    with pytest.raises(ValueError, match='API failed'):
        with adapter.submission_gpu_progress(root, work, receipt_api=object()) as progress:
            progress.report('API inventory running')
            raise ValueError('API failed')
    assert read_json(work / 'submission_progress.json')['state'] == 'STOPPED'
