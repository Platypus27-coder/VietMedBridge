from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

import pyarrow as pa

from .io import atomic_bytes, write_parquet

LOGGER = logging.getLogger(__name__)


def recover_open_tail(path: Path) -> None:
    if not path.exists() or not path.stat().st_size:
        return
    with path.open('rb+') as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        handle.seek(max(0, end - 1024 * 1024))
        tail = handle.read()
        if tail.endswith(b'\n'):
            return
        start = end - len(tail) + tail.rfind(b'\n') + 1
        candidate = tail[tail.rfind(b'\n') + 1:]
        try:
            json.loads(candidate)
        except (UnicodeDecodeError, json.JSONDecodeError):
            atomic_bytes(path.with_suffix('.truncated'), candidate)
            handle.truncate(start)
            LOGGER.warning('Recovered truncated manifest tail %s at byte %d (%d bytes)', path, start, len(candidate))
        else:
            handle.seek(end)
            handle.write(b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def iter_events(directory: Path):
    for path in sorted(directory.glob('part-*.jsonl*')):
        if not (path.name.endswith('.jsonl') or path.name.endswith('.jsonl.open')):
            continue
        if path.name.endswith('.open'):
            recover_open_tail(path)
        with path.open('rb') as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f'Corrupt manifest {path}:{line_no}') from error
                if not isinstance(record, dict):
                    raise ValueError(f'Manifest event must be an object: {path}:{line_no}')
                yield record


class EventWriter:
    def __init__(self, directory: Path, records_per_part: int):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.limit = records_per_part
        opens = sorted(directory.glob('part-*.jsonl.open'))
        if len(opens) > 1:
            raise ValueError('Multiple active manifest parts; writers must not share an output directory.')
        if opens:
            self.path = opens[0]
            recover_open_tail(self.path)
        else:
            existing = list(directory.glob('part-*.jsonl'))
            numbers = [int(p.name.split('-')[1].split('.')[0]) for p in existing]
            self.path = directory / f'part-{max(numbers, default=-1)+1:05d}.jsonl.open'
        self.count = sum(1 for _ in self.path.open('rb')) if self.path.exists() else 0
        self.handle = self.path.open('ab')

    def append(self, record: dict):
        if self.count >= self.limit:
            self.close()
            self.__init__(self.directory, self.limit)
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.count += 1

    def close(self):
        if not self.handle.closed:
            self.handle.close()
            self.path.replace(self.path.with_suffix(''))


def compact_events(directory: Path, target: Path, schema: pa.Schema, *, key='crawl_url_id', attempt='attempt_no') -> int:
    """Disk-backed latest view; detect conflicting replay instead of picking randomly."""
    temporary_db = target.with_suffix('.compact.sqlite')
    if temporary_db.exists():
        temporary_db.unlink()
    db = sqlite3.connect(temporary_db)
    try:
        db.executescript('CREATE TABLE events(id TEXT PRIMARY KEY, payload TEXT); CREATE TABLE latest(id TEXT PRIMARY KEY, attempt INTEGER, seq INTEGER, payload TEXT);')
        for record in iter_events(directory):
            payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
            event_id = record['event_id']
            old_event = db.execute('SELECT payload FROM events WHERE id=?', (event_id,)).fetchone()
            if old_event:
                if old_event[0] != payload:
                    raise ValueError(f'Conflicting event replay {event_id}')
                continue
            db.execute('INSERT INTO events VALUES (?,?)', (event_id, payload))
            ordering = (record[attempt], record['event_seq'])
            previous = db.execute('SELECT attempt,seq,payload FROM latest WHERE id=?', (record[key],)).fetchone()
            if previous and tuple(previous[:2]) == ordering and previous[2] != payload:
                raise ValueError(f'Conflicting ordering key for {record[key]}')
            if not previous or tuple(previous[:2]) < ordering:
                db.execute('INSERT OR REPLACE INTO latest VALUES (?,?,?,?)', (record[key], *ordering, payload))
        db.commit()
        return write_parquet(target, (json.loads(row[0]) for row in db.execute('SELECT payload FROM latest ORDER BY id')), schema)
    finally:
        db.close()
        temporary_db.unlink(missing_ok=True)
