from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class StateStore:
    """Transactional ordering and cooldowns; manifests remain the outcome truth."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS attempts (url_id TEXT PRIMARY KEY, attempt INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS cooldowns (domain TEXT PRIMARY KEY, until REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')

    def allocate_attempt(self, url_id: str, previous: int = 0) -> int:
        with self.db:
            row = self.db.execute('SELECT attempt FROM attempts WHERE url_id=?', (url_id,)).fetchone()
            attempt = max(row[0] if row else 0, previous) + 1
            self.db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?)', (url_id, attempt))
        return attempt

    def cooldown(self, domain: str, until: float) -> None:
        with self.db:
            self.db.execute('INSERT INTO cooldowns VALUES (?,?) ON CONFLICT(domain) DO UPDATE SET until=MAX(until,excluded.until)', (domain, until))

    def cooldown_until(self, domain: str) -> float:
        row = self.db.execute('SELECT until FROM cooldowns WHERE domain=?', (domain,)).fetchone()
        return row[0] if row else 0.0

    def set_metadata(self, key: str, value) -> None:
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, json.dumps(value)))

    def get_metadata(self, key: str):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.db.close()
