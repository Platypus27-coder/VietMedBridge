"""Disk-backed deduplication with stable hash order and minimum-child ties.

Only bounded Parquet batches and a 32 MiB SQLite page cache are retained. Source
parts commit independently on Colab local disk; model-input receipts stay on
Drive. No global hash aggregate or repeated corpus-sized text join is needed.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from .artifacts import digest_json, verify_file

CACHE_KIB = 32768
BATCH_ROWS = 1024
FORMAT = "sqlite-min-child-representatives-v1"


class InputSpool:
    def __init__(self, root, candidate, directory, signature):
        self.root, self.candidate = Path(root).resolve(), candidate
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path = self.directory / "representatives.sqlite"
        self.con = sqlite3.connect(self.path)
        try:
            self.con.executescript(f"""
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=NORMAL;
                PRAGMA cache_size=-{CACHE_KIB};
                PRAGMA temp_store=FILE;
                PRAGMA mmap_size=0;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS completed(
                    kind TEXT,path TEXT,sha TEXT NOT NULL,rows INTEGER NOT NULL,
                    PRIMARY KEY(kind,path)) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS documents(doc_id INTEGER PRIMARY KEY,title TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS representatives(
                    id TEXT PRIMARY KEY,chunk_id TEXT NOT NULL,doc_id INTEGER NOT NULL,
                    text TEXT NOT NULL,heading TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)) WITHOUT ROWID;
            """)
            identity = digest_json({"format":FORMAT,"input_signature":signature,
                                   "candidate":candidate["candidate_manifest_sha256"]})
            old = self.con.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if old is not None and old[0] != identity:
                raise ValueError("Local model-input spool identity changed.")
            with self.con:
                self.con.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)",(identity,))
            print(f"Index dedup staging: local disk | SQLite page cache {CACHE_KIB//1024} MiB | {self.path}",flush=True)
        except BaseException:
            self.con.close()
            raise

    def close(self):
        self.con.close()

    def discard_completed(self):
        """Release only our generated local DB after the durable input manifest."""
        for suffix in ("","-journal","-wal","-shm"):
            self.path.with_name(self.path.name+suffix).unlink(missing_ok=True)
        try:
            self.directory.rmdir()  # Never recursively remove unexpected files.
        except OSError:
            pass

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Input spool source escapes its frozen build.")
        return path

    def stage(self):
        for kind, columns in (("documents",["doc_id","title"]),
            ("children",["retrieval_representation_hash","chunk_id","doc_id","text","heading"])):
            processed = 0
            for ordinal, part in enumerate(self.candidate["parts"],1):
                entry = part["files"][kind]
                old = self.con.execute("SELECT sha,rows FROM completed WHERE kind=? AND path=?",
                                       (kind,entry["path"])).fetchone()
                if old is not None:
                    if old[0] != entry["sha256"]:
                        raise ValueError("Input spool source checkpoint changed.")
                    processed += old[1]
                    action = "reused"
                else:
                    path = self._path(entry["path"])
                    verify_file(path,entry["sha256"])
                    source = pq.ParquetFile(path)
                    seen = 0
                    # A failed/aborted source part rolls back completely. Earlier
                    # committed parts remain reusable in this runtime.
                    with self.con:
                        for batch in source.iter_batches(batch_size=BATCH_ROWS,columns=columns,use_threads=False):
                            rows = batch.to_pylist()
                            if kind == "documents":
                                self.con.executemany("INSERT INTO documents VALUES (?,?)",
                                    [(r["doc_id"],r["title"]) for r in rows])
                            else:
                                self.con.executemany("""INSERT INTO representatives VALUES (?,?,?,?,?)
                                    ON CONFLICT(id) DO UPDATE SET chunk_id=excluded.chunk_id,
                                        doc_id=excluded.doc_id,text=excluded.text,heading=excluded.heading
                                    WHERE excluded.chunk_id<representatives.chunk_id""",
                                    [(r["retrieval_representation_hash"],r["chunk_id"],r["doc_id"],
                                      r["text"],r["heading"]) for r in rows])
                            seen += len(rows)
                        if seen != source.metadata.num_rows:
                            raise ValueError("Input spool source row count changed.")
                        self.con.execute("INSERT INTO completed VALUES (?,?,?,?)",
                                         (kind,entry["path"],entry["sha256"],seen))
                    processed += seen
                    action = "new"
                if ordinal % 16 == 0 or ordinal == len(self.candidate["parts"]):
                    print(f"Index staging {kind}: {ordinal:,}/{len(self.candidate['parts']):,} source parts"
                          f" | {processed:,} rows | {action}",flush=True)
            if processed != self.candidate["counts"][kind]:
                raise ValueError(f"Input spool {kind} coverage differs from the frozen candidate.")
        count = self.con.execute("SELECT count(*) FROM representatives").fetchone()[0]
        previous = self.con.execute("SELECT value FROM metadata WHERE key='ready_count'").fetchone()
        if previous is not None and int(previous[0]) != count:
            raise ValueError("Completed local input spool count changed.")
        with self.con:
            self.con.execute("INSERT OR IGNORE INTO metadata VALUES ('ready_count',?)",(str(count),))
        print(f"Index dedup ready: {count:,} unique texts on local disk",flush=True)
        return count

    def keys_after(self, last, count):
        return [r[0] for r in self.con.execute(
            "SELECT id FROM representatives WHERE id>? ORDER BY id LIMIT ?",(last,count))]

    def rows_after(self, last, count):
        return self.con.execute("""SELECT r.id,r.text,r.heading,d.title FROM representatives r
            JOIN documents d ON d.doc_id=r.doc_id WHERE r.id>? ORDER BY r.id LIMIT ?""",
            (last,count)).fetchall()
