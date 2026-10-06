from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .io import parquet_rows, sha256_file


class ManifestIndex:
    """Bounded-memory lookups over a compact Parquet view."""
    def __init__(self, path: Path, parquet: Path, key='crawl_url_id'):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload TEXT); CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);')
        checksum = sha256_file(parquet)
        previous = self.db.execute("SELECT value FROM meta WHERE key='sha256'").fetchone()
        if not previous or previous[0] != checksum:
            with self.db:
                self.db.execute('DELETE FROM records')
                self.db.executemany('INSERT INTO records VALUES (?,?)', ((row[key], json.dumps(row, ensure_ascii=False)) for row in parquet_rows(parquet)))
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('sha256',?)", (checksum,))

    def get(self, key: str):
        row = self.db.execute('SELECT payload FROM records WHERE id=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def __iter__(self):
        return (json.loads(row[0]) for row in self.db.execute('SELECT payload FROM records ORDER BY id'))

    def close(self):
        self.db.close()
