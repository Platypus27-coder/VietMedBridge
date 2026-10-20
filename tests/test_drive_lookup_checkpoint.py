"""A new runtime/account must recover paid references from one saved lookup."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys

import pytest

import scripts.submission_gpu_progress as adapter
from scripts.drive_receipts_api import DriveReceiptSource, LookupPublishUnavailable
from test_drive_receipts_api import api_from_blocks
from test_submission_cache_lookup import blocks
from vietmedbridge.artifacts import digest_json, read_json, verify_file
from vietmedbridge.content_embeddings import seal


def snapshot_api(blocks, *, root=None, saved=None):
    api, values = api_from_blocks(blocks)
    api.folder_id = 'same-shared-data-id'
    api.requested_root = api.data_root = root or api.data_root
    api.saved = {} if saved is None else saved
    api.published = []
    api.relative_json = lambda path: copy.deepcopy(api.saved.get(path))
    api.relative = lambda path: {'path':path} if path in api.saved else None
    def download(metadata, target, *, expected_sha256):
        Path(target).write_bytes(api.saved[metadata['path']])
        verify_file(target, expected_sha256)
    api.download = download
    def publish(directory, payload, marker, md5):
        api.saved[directory+'/'+marker['path']] = payload.read_bytes()
        api.saved[directory+'/current.json'] = copy.deepcopy(marker)
        api.published.append(marker['receipts'])
        return marker
    api._publish_lookup_files = publish
    api.publish_lookup = lambda *a: DriveReceiptSource.publish_lookup(api, *a)
    return api, values


def test_api_snapshot_survives_new_account_path_without_receipt_reads_or_encoding(blocks):
    root, work, encoder, texts, expected, _ = blocks
    api, _ = snapshot_api(blocks)
    first = adapter.ResumableContentBlocks(root, encoder, work/'first-api', receipt_api=api)
    first.close()
    assert api.published == [3]
    other_root = work/'other-account/shared-data'
    other, _ = snapshot_api(blocks, root=other_root, saved=api.saved)
    other.read_json = lambda _: pytest.fail('Previously indexed receipt downloaded again')
    resumed = adapter.ResumableContentBlocks(other_root, encoder, work/'new-runtime', receipt_api=other)
    try:
        assert resumed.refs(texts, batch_size=1, encode_missing=False) == (expected, 0)
        assert encoder.calls == 3
        assert resumed.remote.relative_to(other_root) == first.remote.relative_to(root)
    finally:
        resumed.close()


def test_api_snapshot_partial_prefix_survives_vm_loss(blocks):
    root, work, encoder, texts, expected, _ = blocks
    api, values = snapshot_api(blocks)
    names = sorted(values)
    original = api.read_json
    def interrupted(meta):
        if meta['name'] == names[1]:
            raise OSError('Read interrupted after committed prefix')
        return original(meta)
    api.read_json = interrupted
    with pytest.raises(OSError):
        adapter.ResumableContentBlocks(root, encoder, work/'interrupted', receipt_api=api)
    assert api.published == [1]
    new, _ = snapshot_api(blocks, saved=api.saved)
    read = new.read_json
    def missing_only(meta):
        assert meta['name'] != names[0], 'Saved prefix downloaded again after VM loss'
        return read(meta)
    new.read_json = missing_only
    resumed = adapter.ResumableContentBlocks(root, encoder, work/'new-vm', receipt_api=new)
    try:
        assert resumed.refs(texts, batch_size=1, encode_missing=False) == (expected, 0)
        assert new.published == [3] and encoder.calls == 3
    finally:
        resumed.close()


def test_lookup_is_saved_before_inventory_finishes_not_only_at_close(blocks):
    from vietmedbridge.content_embeddings import key
    root, work, encoder, *_ = blocks
    api, values = snapshot_api(blocks)
    identity = next(iter(values.values()))['identity']
    for i in range(260):
        keys = [key('existing paid block '+str(i))]
        name = 'block-'+digest_json(keys)+'.done.json'
        values[name] = seal({'identity':identity, 'keys':keys, 'rows':1,
            'path':'model_cache/existing.npy', 'sha256':'a'*64})
    lookup = adapter.ResumableContentBlocks(root, encoder, work/'periodic', receipt_api=api)
    try:
        assert api.published == [256, 263], 'No intermediate checkpoint survived an abrupt VM loss'
    finally:
        lookup.close()


def test_checkpoint_existing_old_cell_never_lists_receipts(blocks):
    root, work, encoder, texts, expected, _ = blocks
    from test_drive_receipts_api import api_from_blocks
    old, _ = api_from_blocks(blocks)
    lookup = adapter.ResumableContentBlocks(root, encoder, work/'old-cell', receipt_api=old)
    lookup.close()
    api, _ = snapshot_api(blocks)
    api.receipts = lambda _: pytest.fail('Save-only rescanned receipts')
    api.read_json = lambda _: pytest.fail('Save-only reread receipt JSON')
    save = adapter.ResumableContentBlocks(root, encoder, work/'old-cell', receipt_api=api, inventory=False)
    save.close()
    assert api.published == [3] and encoder.calls == 3
    with pytest.raises(FileNotFoundError, match='No local lookup'):
        adapter.ResumableContentBlocks(root, encoder, work/'missing-vm', receipt_api=api, inventory=False)


def test_old_local_lookup_is_auto_published_on_next_resume_even_with_no_new_receipts(blocks):
    root, work, encoder, *_ = blocks
    old, _ = api_from_blocks(blocks)
    previous = adapter.ResumableContentBlocks(root, encoder, work/'old-local', receipt_api=old)
    previous.close()
    api, _ = snapshot_api(blocks)
    api.read_json = lambda _:pytest.fail('Already indexed receipt read while upgrading local lookup')
    resumed = adapter.ResumableContentBlocks(root, encoder, work/'old-local', receipt_api=api)
    resumed.close()
    assert api.published == [3]


def test_save_only_cell_is_self_contained_and_refuses_active_processing_cell():
    from scripts.write_available_submission_notebook import save_lookup_cell
    cell = save_lookup_cell()
    compile(cell, 'save-existing-lookup', 'exec')
    import threading
    from unittest.mock import patch
    with patch.object(threading, 'enumerate', return_value=[SimpleNamespace(name='submission-progress', is_alive=lambda:True)]):
        with pytest.raises(RuntimeError, match='đang chạy'):
            exec(cell, {})


def test_upload_failure_retains_local_lookup_and_does_not_claim_drive_saved(blocks):
    root, work, encoder, texts, expected, _ = blocks
    api, _ = snapshot_api(blocks)
    api.publish_lookup = lambda *a: (_ for _ in ()).throw(LookupPublishUnavailable('network'))
    logs = []
    lookup = adapter.ResumableContentBlocks(root, encoder, work/'offline', receipt_api=api, progress=logs.append)
    try:
        assert lookup.refs(texts, batch_size=1, encode_missing=False) == (expected,0)
    finally:
        lookup.close()
    assert any('deferred' in line for line in logs)
    assert not any('SAVED_ON_DRIVE' in line for line in logs)
    assert not api.saved


def test_corrupt_uploaded_lookup_is_rejected_before_using_any_vectors(blocks):
    root, work, encoder, *_ = blocks
    api, _ = snapshot_api(blocks)
    first = adapter.ResumableContentBlocks(root, encoder, work/'first', receipt_api=api)
    first.close()
    payload = next(name for name in api.saved if name.endswith('.gz'))
    api.saved[payload] = b'corrupt'
    with pytest.raises(ValueError, match='changed artifact'):
        adapter.ResumableContentBlocks(root, encoder, work/'second', receipt_api=api)


def test_lookup_publisher_refuses_paths_outside_its_shared_checkpoint_folder(blocks):
    api, _ = snapshot_api(blocks)
    with pytest.raises(ValueError, match='outside'):
        DriveReceiptSource.publish_lookup(api, 'model_cache/other/current.json', Path('unused'), {})


def fake_sdk(monkeypatch, source, *, corrupt_payload=False):
    """Exercise real publisher ordering against a resumable-media API double."""
    objects, events = {}, []
    class Execute:
        def __init__(self, result): self.result = result
        def execute(self, **kw): return self.result
    class Media:
        def __init__(self, path, **kw):
            self.data = path.getvalue() if hasattr(path, 'getvalue') else Path(path).read_bytes()
    class Upload:
        def __init__(self, identifier, name, media):
            self.identifier, self.name, self.media = identifier, name, media
        def next_chunk(self, **kw):
            events.append(self.name)
            data = self.media.data
            objects[self.identifier] = {'id':self.identifier, 'name':self.name, 'size':str(len(data)),
                'md5Checksum':hashlib.md5(data).hexdigest(), 'data':data}
            if corrupt_payload and self.name.endswith('.gz'):
                objects[self.identifier]['md5Checksum'] = 'bad'
            return None, {'id':self.identifier}
    class Files:
        cursor = 0
        def generateIds(self, **kw):
            self.cursor += 1
            return Execute({'ids':[str(self.cursor)]})
        def create(self, body, media_body=None, **kw):
            if media_body:
                return Upload(body['id'], body['name'], media_body)
            objects[body['id']] = body
            return Execute(body)
        def update(self, fileId, media_body, **kw):
            return Upload(fileId, objects[fileId]['name'], media_body)
        def get(self, fileId, **kw): return Execute(objects[fileId])
    files = Files()
    discovery, http = ModuleType('googleapiclient.discovery'), ModuleType('googleapiclient.http')
    discovery.build = lambda *a, **kw: SimpleNamespace(files=lambda:files)
    http.MediaFileUpload = http.MediaIoBaseUpload = Media
    monkeypatch.setitem(sys.modules, 'googleapiclient', ModuleType('googleapiclient'))
    monkeypatch.setitem(sys.modules, 'googleapiclient.discovery', discovery)
    monkeypatch.setitem(sys.modules, 'googleapiclient.http', http)
    source._session = lambda: SimpleNamespace(credentials=object())
    def child(parent, name, **kw):
        values = [v for v in objects.values() if v['name'] == name and
            (not kw.get('folder') or v.get('mimeType') == 'application/vnd.google-apps.folder')]
        return values[0] if values else None
    source.child = child
    source.read_json = lambda meta:json.loads(meta['data'])
    return objects, events


@pytest.mark.parametrize('corrupt', [False, True])
def test_publisher_verifies_payload_before_pointer_and_never_replaces_newer_prefix(tmp_path, monkeypatch, corrupt):
    source = DriveReceiptSource(lambda:None, tmp_path, tmp_path, data_folder_id='shared-id')
    objects, events = fake_sdk(monkeypatch, source, corrupt_payload=corrupt)
    payload = tmp_path/'lookup.gz'
    payload.write_bytes(b'compressed sqlite payload')
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest()
    binding = {'schema_version':2, 'data_folder_id':'shared-id', 'identity':{'producer':'bge'}}
    marker = seal({'binding':binding, 'path':'lookup-'+checksum+'.sqlite.gz', 'sha256':checksum, 'receipts':3})
    destination = 'retrieval/content_lookups/'+digest_json(binding)[:16]+'/current.json'
    if corrupt:
        with pytest.raises(ValueError, match='pointer not advanced'):
            source.publish_lookup(destination, payload, marker)
        assert events == [marker['path']]
        assert not any(v['name'] == 'current.json' for v in objects.values())
    else:
        source.publish_lookup(destination, payload, marker)
        assert events == [marker['path'], 'current.json']
        smaller = seal({k:v for k,v in marker.items() if k != 'manifest_sha256'} | {'receipts':1})
        assert source.publish_lookup(destination, payload, smaller) == marker
        assert events == [marker['path'], 'current.json']
