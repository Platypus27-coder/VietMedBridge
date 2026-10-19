"""Drive fault recovery must retain paid content keys and native validation."""
from pathlib import Path
from contextlib import closing
import sqlite3

import pytest

import scripts.submission_gpu_progress as adapter
from scripts.submission_gpu_progress import NativeContentBlocks, ResumableContentBlocks
from test_scale_baseline import Encoder
from test_submission_workflow import prepared, frozen, external
from test_submission_workflow import test_primary_honors_bge_batch_and_reuses_existing_vectors_when_remapping as exercise_remap
from vietmedbridge.artifacts import atomic_json, read_json
from vietmedbridge.content_embeddings import seal


@pytest.fixture
def blocks(tmp_path_factory):
    base = tmp_path_factory.mktemp('cl')
    root, work = base / 'd', base / 'w'
    encoder = Encoder()
    native = NativeContentBlocks(root, encoder, work / 'seed')
    texts = ['old Vietnamese paid text', 'old English paid text', 'partial mini block']
    expected = []
    try:
        for text in texts:
            refs, _ = native.refs([text], batch_size=1, encode_missing=True)
            expected.extend(refs)
        markers = sorted(native.directory.glob('*.done.json'))
    finally:
        native.close()
    return root, work, encoder, texts, expected, markers


def test_reconnect_resumes_after_committed_receipt_without_rereading_prefix(blocks, monkeypatch):
    root, work, encoder, texts, expected, markers = blocks
    real_read = adapter.read_json
    def disconnected(path):
        if Path(path) == markers[1]:
            raise OSError(107, 'Transport endpoint is not connected', str(path))
        return real_read(path)
    monkeypatch.setattr(adapter, 'read_json', disconnected)
    with pytest.raises(OSError):
        ResumableContentBlocks(root, encoder, work / 'resume')
    database = next((work / 'resume/receipt_lookups').glob('*/lookup.sqlite'))
    with closing(sqlite3.connect(database)) as con:
        assert con.execute('SELECT count(*) FROM receipts').fetchone()[0] == 1
    def resumed(path):
        assert Path(path) != markers[0], 'Previously committed receipt reread after recovery'
        return real_read(path)
    monkeypatch.setattr(adapter, 'read_json', resumed)
    lookup = ResumableContentBlocks(root, encoder, work / 'resume')
    try:
        refs, missing = lookup.refs(texts, batch_size=1, encode_missing=False)
        assert refs == expected and missing == 0 and encoder.calls == 3
    finally:
        lookup.close()


def test_fresh_runtime_restores_compressed_lookup_without_old_receipt_reads(blocks, monkeypatch):
    root, work, encoder, texts, expected, _ = blocks
    first = ResumableContentBlocks(root, encoder, work / 'first')
    first.close()
    real_read = adapter.read_json
    def no_old_receipts(path):
        assert not Path(path).name.endswith('.done.json'), 'Fresh runtime scanned old receipt JSON'
        return real_read(path)
    monkeypatch.setattr(adapter, 'read_json', no_old_receipts)
    second = ResumableContentBlocks(root, encoder, work / 'fresh')
    try:
        refs, missing = second.refs(texts, batch_size=1, encode_missing=False)
        assert refs == expected and missing == 0 and encoder.calls == 3
    finally:
        second.close()


def test_new_receipts_are_added_after_snapshot_without_reencoding_old_text(blocks):
    root, work, encoder, texts, expected, _ = blocks
    first = ResumableContentBlocks(root, encoder, work / 'first')
    first.close()
    native = NativeContentBlocks(root, encoder, work / 'seed-new')
    try:
        refs, _ = native.refs(['additional paid block'], batch_size=1, encode_missing=True)
        expected.extend(refs)
    finally:
        native.close()
    second = ResumableContentBlocks(root, encoder, work / 'fresh')
    try:
        refs, missing = second.refs(texts+['additional paid block'], batch_size=1, encode_missing=False)
        assert refs == expected and missing == 0 and encoder.calls == 4
    finally:
        second.close()


def test_corrupt_snapshot_is_refused_before_lookup_use(blocks):
    root, work, encoder, *_ = blocks
    first = ResumableContentBlocks(root, encoder, work / 'first')
    marker = first.remote / 'current.json'
    first.close()
    snapshot = read_json(marker)
    (marker.parent / snapshot['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed artifact'):
        ResumableContentBlocks(root, encoder, work / 'fresh')


def test_changed_receipt_identity_does_not_advance_inventory(blocks):
    root, work, encoder, _, _, markers = blocks
    changed = read_json(markers[0])
    changed['identity'] = {'another': 'producer'}
    atomic_json(markers[0], seal({k:v for k,v in changed.items() if k != 'manifest_sha256'}))
    with pytest.raises(ValueError, match='producer/key mismatch'):
        ResumableContentBlocks(root, encoder, work / 'bad')
    database = next((work / 'bad/receipt_lookups').glob('*/lookup.sqlite'))
    with closing(sqlite3.connect(database)) as con:
        assert con.execute('SELECT count(*) FROM receipts').fetchone()[0] == 0


def test_newly_encoded_blocks_commit_to_lookup_and_survive_close(blocks):
    root, work, encoder, texts, _, _ = blocks
    first = ResumableContentBlocks(root, encoder, work / 'first')
    try:
        refs, count = first.refs(['new encoded text'], batch_size=1, encode_missing=True)
        assert count == 1
        database = first.database
    finally:
        first.close()
    with closing(sqlite3.connect(database)) as con:
        assert con.execute('SELECT count(*) FROM receipts').fetchone()[0] == 4
    second = ResumableContentBlocks(root, encoder, work / 'first')
    try:
        actual, count = second.refs(['new encoded text'], batch_size=1, encode_missing=False)
        assert actual == refs and count == 0 and encoder.calls == 4
    finally:
        second.close()


def test_native_paid_cache_remap_works_under_scoped_adapter(prepared, monkeypatch):
    original = adapter.NativeContentBlocks
    from vietmedbridge import content_embeddings
    with adapter.submission_gpu_progress(prepared.root, prepared.work):
        exercise_remap(prepared, monkeypatch)
    assert content_embeddings.ContentBlocks is original


def test_preflight_recovers_dead_mount_and_preserves_expected_candidate(monkeypatch):
    from vietmedbridge import colab_storage, scale_benchmark
    root = Path('/content/drive/MyDrive/VietMedBridge/data')
    calls = []
    def handoff(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError(107, 'Transport endpoint is not connected', str(root / 'active_data_candidate.json'))
        return None, {'candidate_manifest_sha256': 'same-frozen-candidate'}, None
    remounts = []
    monkeypatch.setattr(scale_benchmark, 'load_handoff', handoff)
    monkeypatch.setattr(colab_storage, '_remount', lambda *a, **k: remounts.append((a,k)))
    adapter.ensure_submission_mount(root, 'same-frozen-candidate')
    assert len(calls) == 2 and remounts == [((Path('/content/drive'),), {'readonly': False})]


def test_preflight_refuses_candidate_change_without_restarting_preparation(monkeypatch):
    from vietmedbridge import colab_storage, scale_benchmark
    monkeypatch.setattr(scale_benchmark, 'load_handoff',
        lambda _: (None, {'candidate_manifest_sha256': 'another-candidate'}, None))
    monkeypatch.setattr(colab_storage, '_remount', lambda *a, **k: pytest.fail('Remounted a healthy corpus'))
    with pytest.raises(ValueError, match='Active candidate changed'):
        adapter.ensure_submission_mount(Path('/content/drive/MyDrive/VietMedBridge/data'), 'original')
