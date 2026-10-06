import json

import pyarrow as pa
import pytest

from src.crawler.raw_store import RawStore, pack_raw
from src.crawler.router import sniff_content
from src.storage.io import parquet_rows, write_parquet
from src.storage.schemas import CRAWL_SCHEMA
from src.utils.config import output_path
from src.storage.manifests import EventWriter, compact_events, iter_events


def test_raw_bytes_checksum_and_cap(config):
    store = RawStore(config)
    body = '<html><body>Điều trị 病人 HbA1c SpO₂ Na+</body></html>'.encode()
    record = store.write(body, 'text/html')
    assert store.read(record, len(body)) == body
    assert store.write(body, 'text/html') == record
    with pytest.raises(OverflowError):
        store.read(record, len(body)-1)
    record['raw_sha256'] = 'bad'
    with pytest.raises(ValueError, match='checksum'):
        store.read(record, len(body))


def test_manifest_latest_and_truncated_tail(tmp_path):
    directory = tmp_path/'events'
    writer = EventWriter(directory, 10)
    writer.append({'crawl_url_id': 'u', 'event_id': 'u:1', 'attempt_no': 1, 'event_seq': 1, 'status': 'HTTP_429'})
    writer.handle.write(b'{"crawl_url_id":"cut')
    writer.handle.close()  # Simulate abrupt process loss before seal.
    recovered = list(iter_events(directory))
    assert len(recovered) == 1
    writer = EventWriter(directory, 10)
    writer.append({'crawl_url_id': 'u', 'event_id': 'u:2', 'attempt_no': 2, 'event_seq': 1, 'status': 'SUCCESS_RAW'})
    writer.close()
    schema = pa.schema([('crawl_url_id', pa.string()), ('status', pa.string())])
    compact_events(directory, tmp_path/'latest.parquet', schema)
    assert list(parquet_rows(tmp_path/'latest.parquet')) == [{'crawl_url_id': 'u', 'status': 'SUCCESS_RAW'}]
    assert list(directory.glob('*.truncated'))


def test_malformed_middle_line_is_not_ignored(tmp_path):
    path = tmp_path/'part-00000.jsonl'
    path.write_bytes(b'{bad}\n{}\n')
    with pytest.raises(ValueError, match='Corrupt manifest'):
        list(iter_events(tmp_path))


@pytest.mark.parametrize('body,declared,effective', [(b'%PDF-1.7\n', 'text/html', 'application/pdf'),
    (b'<html><body>text</body></html>', 'application/octet-stream', 'text/html'),
    (b'\x89PNG\x00binary', 'text/html', None)])
def test_mime_sniff(body, declared, effective):
    assert sniff_content(body, declared)['effective_content_type'] == effective


def test_packed_raw_preserves_bytes_and_loose_recovery(config):
    bodies = [b'<html>first biomedical document</html>', b'<html>second biomedical document</html>']
    records = [dict(RawStore(config).write(body, 'text/html'), crawl_url_id=str(n), shard_id='s1', status='SUCCESS_RAW')
               for n, body in enumerate(bodies)]
    manifest = output_path(config, 'data/manifests/crawl_manifest.parquet')
    write_parquet(manifest, records, CRAWL_SCHEMA)
    packed = list(parquet_rows(pack_raw(config, manifest)))
    assert len({row['raw_path'] for row in packed}) == 1
    for record, archived, body in zip(records, packed, bodies):
        assert archived['raw_member_offset'] >= 0 and archived['raw_member_size'] > 0
        assert RawStore(config).read(archived, 1000) == body
        legacy = {k: v for k, v in archived.items() if k not in ('raw_member_offset', 'raw_member_size')}
        assert RawStore(config).read(legacy, 1000) == body
        damaged = dict(archived, raw_member_offset=archived['raw_member_offset']+1)
        with pytest.raises(ValueError, match='checksum'):
            RawStore(config).read(damaged, 1000)
        assert output_path(config, record['raw_path']).exists()


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows sharing violation')
def test_atomic_commit_survives_transient_windows_reader(tmp_path, monkeypatch):
    from src.storage import io
    original = io.os.replace
    attempts = []
    def sometimes_locked(source, destination):
        attempts.append(1)
        if len(attempts) <= 2:
            error = PermissionError('temporary sharing violation')
            error.winerror = 32
            raise error
        return original(source, destination)
    monkeypatch.setattr(io.os, 'replace', sometimes_locked)
    target = tmp_path/'checkpoint.json'
    io.atomic_json(target, {'committed': True})
    assert json.loads(target.read_text()) == {'committed': True}
    assert len(attempts) == 3 and not list(tmp_path.glob('*.tmp'))


def test_status_snapshot_allows_atomic_replacement_while_parsing(tmp_path):
    from src.storage.io import atomic_bytes, snapshot_binary_reader
    target = tmp_path/'checkpoint.json'
    atomic_bytes(target, b'old')
    with snapshot_binary_reader(target) as handle:
        atomic_bytes(target, b'new')
        assert handle.read() == b'old'
    assert target.read_bytes() == b'new'


def test_live_status_reader_does_not_repair_a_writers_partial_tail(tmp_path):
    from scripts.pilot_status import status
    directory = tmp_path/'data/manifests/crawl'
    directory.mkdir(parents=True)
    path = directory/'part-00000.jsonl.open'
    body = json.dumps({'crawl_url_id': 'one', 'attempt_no': 1, 'event_seq': 1, 'status': 'SUCCESS_RAW'}).encode()+b'\n{unfinished'
    path.write_bytes(body)
    assert status(tmp_path)['crawl_completed'] == 1
    assert path.read_bytes() == body
