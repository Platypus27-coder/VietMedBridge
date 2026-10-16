"""Parallel lexical preparation with durable receipts and a resumable local DB.

Keep disk_catalog's schema, analyzer and source policy. Only execution changes:
bounded CPU processes, cached token hashes, batched SQL and atomic shard commits.
Lexical receipts survive a lost VM; a local SQLite cursor survives cell failure.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time

import pyarrow as pa
import pyarrow.parquet as pq

from . import disk_catalog as native
from .artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, verify_file
from .medical_lexical import MedicalAnalyzer
from .scale_vectors import bound, check_manifest

_ANALYZER = None
_LEXICAL_SCHEMA = pa.schema([
    ("doc_id",pa.int64()),("language",pa.string()),("title",pa.string()),
    ("headings",pa.string()),("body",pa.string()),("aliases",pa.string()),
])


def cpu_workers():
    available = os.cpu_count() or 1
    if hasattr(os,"sched_getaffinity"):
        available = min(available,len(os.sched_getaffinity(0)))
    try:
        quota,period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            available = min(available,max(1,int(quota)//int(period)))
    except (OSError,ValueError):
        pass
    requested = int(os.environ.get("VMB_CATALOG_WORKERS",min(4,available)))
    if not 1 <= requested <= available:
        raise ValueError(f"VMB_CATALOG_WORKERS must be in 1..{available}")
    return requested


def _init_analyzer(identity,entries):
    global _ANALYZER
    # Worker BLAS threads must not compete with the other CPU processes.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    analyzer = MedicalAnalyzer(segmentation=identity["segmentation"])
    analyzer.entries = entries
    analyzer.identity.update(glossary_sha256=digest_json(entries),aliases=len(entries))
    if analyzer.identity != identity:
        raise ValueError("Catalog worker analyzer differs from the query analyzer.")
    _ANALYZER = analyzer


@lru_cache(maxsize=65536)
def _term(token):
    return "t"+hashlib.sha256(token.encode()).hexdigest()


def _field(analyzer,text,language):
    return " ".join(map(_term,analyzer.analyze(text,language)))


def _analyze_document(document,analyzer=None):
    analyzer = _ANALYZER if analyzer is None else analyzer
    language = document["language"]
    languages = (language,) if language in ("vi","en","zh") else ("vi","en","zh")
    fields = (document.get("title",""),"\n".join(document.get("heading_hints") or []),
              document["source_text"],analyzer.aliases(document["source_text"]))
    return [{"doc_id":document["doc_id"],"language":lang,
        **dict(zip(("title","headings","body","aliases"),
                   (_field(analyzer,text,lang) for text in fields),strict=True))} for lang in languages]


def _lexical_checkpoint(docs,part,policy,local_dir,remote_dir,analyzer,pool):
    # Tokenization is independent of candidate membership and dense positions.
    # Reuse unchanged source parts after another batch of URLs is added.
    contract = {"format":"catalog-lexical-v1","analyzer":policy["analyzer"],
        "field_encoding_code_sha256":policy["code_sha256"],
        "source_sha256":part["files"]["documents"]["sha256"]}
    name = "lexical-"+digest_json(contract)
    payload,receipt = remote_dir / (name+".parquet"),remote_dir / (name+".json")
    local = local_dir / (name+".parquet")
    if receipt.is_file():
        saved = read_json(receipt)
        check_manifest(saved)
        expected_rows = sum(1 if d["language"] in ("vi","en","zh") else 3 for d in docs)
        if (any(saved.get(k) != v for k,v in contract.items()) or saved.get("path") != payload.name
            or saved.get("rows") != expected_rows):
            raise ValueError("Catalog lexical checkpoint changed.")
        if not local.is_file() or sha256_file(local) != saved["sha256"]:
            verify_file(payload,saved["sha256"])
            if publish_file(payload,local) != saved["sha256"]:
                raise ValueError("Catalog lexical copy changed.")
        return local,True
    temporary = local.with_suffix(".pending.parquet")
    count = 0
    try:
        with pq.ParquetWriter(temporary,_LEXICAL_SCHEMA,compression="zstd") as output:
            for start in range(0,len(docs),32):
                batch = docs[start:start+32]
                results = (map(lambda d:_analyze_document(d,analyzer),batch) if pool is None else
                           pool.map(_analyze_document,batch,chunksize=4))
                records = [row for result in results for row in result]
                output.write_table(pa.Table.from_pylist(records,schema=_LEXICAL_SCHEMA))
                count += len(records)
        os.replace(temporary,local)
        checksum = publish_file(local,payload)
        verify_file(payload,checksum)
        saved = contract | {"path":payload.name,"sha256":checksum,"rows":count,"state":"COMPLETE"}
        saved["manifest_sha256"] = digest_json(saved)
        atomic_json(receipt,saved)
    finally:
        temporary.unlink(missing_ok=True)
    return local,False


def _create(con,policy):
    con.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE documents(doc_id INTEGER PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE children(chunk_id TEXT PRIMARY KEY,doc_id INTEGER NOT NULL,
            representation TEXT NOT NULL,parent_id TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE parents(chunk_id TEXT PRIMARY KEY,doc_id INTEGER NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE units(position INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL);
        CREATE TABLE metadata(payload TEXT NOT NULL);
    """)
    for lang in ("vi","en","zh"):
        con.execute(f"CREATE VIRTUAL TABLE fts_{lang} USING fts5(title,headings,body,aliases,tokenize='ascii',content='')")
    state = {"builder":"parallel-catalog-v1","policy":policy,"units_cursor":0,"next_part":0,
             "counts":{"documents":0,"children":0,"parents":0}}
    _state(con,state)
    con.commit()
    return state


def _state(con,state):
    con.execute("DELETE FROM metadata")
    con.execute("INSERT INTO metadata VALUES (?)",(native.packed(state),))


def _insert_spans(con,kind,path,docs):
    batch = []
    count = 0
    for row in native.rows(path):
        d = docs[row["doc_id"]]
        a,b = row["start_char"],row["end_char"]
        if (not 0 <= a < b <= len(d["source_text"]) or row["text"] != d["source_text"][a:b]
            or row["source_text_sha256"] != d["source_text_sha256"]):
            raise ValueError("Frozen source span changed.")
        payload = native.packed({k:v for k,v in row.items() if k not in ("text","retrieval_text")})
        batch.append((row["chunk_id"],row["doc_id"],row["retrieval_representation_hash"],row["parent_id"],payload)
                     if kind == "children" else (row["chunk_id"],row["doc_id"],payload))
        if len(batch) == 256:
            con.executemany(f"INSERT INTO {kind} VALUES ({','.join('?' for _ in batch[0])})",batch)
            batch.clear()
        count += 1
    if batch:
        con.executemany(f"INSERT INTO {kind} VALUES ({','.join('?' for _ in batch[0])})",batch)
    return count


def prepare_catalog(build,candidate,inputs,output_dir,analyzer,*,work_dir,policy,checkpoint_dir,workers=None):
    """Build a catalog with the native reader contract; never touch model caches."""
    signature = digest_json(policy)
    target = Path(output_dir) / ("catalog-"+signature[:16])
    target.mkdir(parents=True,exist_ok=True)
    marker,local = target / "catalog.json",target / "catalog.sqlite"
    if marker.is_file():
        saved = read_json(marker)
        check_manifest(saved)
        if saved["signature"] != signature or saved["state"] != "COMPLETE":
            raise ValueError("Disk catalog contract changed.")
        verify_file(local,saved["sha256"])
        return native.DiskCatalog(local,candidate,inputs,analyzer,saved)
    building = target / "building.sqlite"
    remote = Path(checkpoint_dir)
    remote.mkdir(parents=True,exist_ok=True)
    workers = cpu_workers() if workers is None else workers
    if type(workers) is not int or workers < 1:
        raise ValueError("Catalog workers must be a positive integer.")
    con = sqlite3.connect(building)
    pool = None
    try:
        con.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
                          "PRAGMA cache_size=-131072; PRAGMA temp_store=FILE;")
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE name='metadata'").fetchone()
        state = read_json_value(con.execute("SELECT payload FROM metadata").fetchone()[0]) if exists else _create(con,policy)
        if state != policy and (state.get("builder") != "parallel-catalog-v1" or state.get("policy") != policy
            or not 0 <= state.get("units_cursor",-1) <= inputs["input_count"]
            or not 0 <= state.get("next_part",-1) <= len(candidate["parts"])):
            raise ValueError("Partial catalog source/analyzer policy changed.")
        if state != policy:
            print(f"Catalog CPU workers: {workers} | local SQL resume {state['next_part']}/{len(candidate['parts'])} shards"
                  " | lexical checkpoints on Drive",flush=True)
            cursor = state["units_cursor"]
            if cursor and con.execute("SELECT count(*) FROM units").fetchone()[0] != cursor:
                raise ValueError("Partial catalog unit cursor changed.")
            for unit_index,part in enumerate(inputs["parts"],1):
                if part["start"]+part["rows"] <= cursor:
                    continue
                if part["start"] != cursor:
                    raise ValueError("Input ordering changed.")
                path = bound(Path(build) / "index_inputs",part["path"])
                verify_file(path,part["sha256"])
                ids = pq.read_table(path,columns=["id"]).column("id").to_pylist()
                if len(ids) != part["rows"]:
                    raise ValueError("Input count changed.")
                with con:
                    con.executemany("INSERT INTO units VALUES (?,?)",((cursor+i,key) for i,key in enumerate(ids)))
                    cursor += len(ids)
                    state["units_cursor"] = cursor
                    _state(con,state)
                if unit_index % 64 == 0 or cursor == inputs["input_count"]:
                    print(f"Catalog unit map: {cursor:,}/{inputs['input_count']:,} rows | SQL checkpoint saved",flush=True)
            if cursor != inputs["input_count"]:
                raise ValueError("Incomplete unit mapping.")
            if workers > 1 and state["next_part"] < len(candidate["parts"]):
                pool = ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_analyzer,initargs=(analyzer.identity,analyzer.entries))
            for index in range(state["next_part"],len(candidate["parts"])):
                started = time.perf_counter()
                part = candidate["parts"][index]
                paths = {kind:bound(build,part["files"][kind]["path"]) for kind in state["counts"]}
                for kind,path in paths.items():
                    verify_file(path,part["files"][kind]["sha256"])
                docs = list(native.rows(paths["documents"]))
                for d in docs:
                    if hashlib.sha256(d["source_text"].encode()).hexdigest() != d["source_text_sha256"]:
                        raise ValueError("Frozen document source changed.")
                lexical,reused = _lexical_checkpoint(docs,part,policy,target,remote,analyzer,pool)
                analyzed = time.perf_counter()
                by_id = {d["doc_id"]:d for d in docs}
                expected = {(d["doc_id"],lang) for d in docs for lang in (
                    (d["language"],) if d["language"] in ("vi","en","zh") else ("vi","en","zh"))}
                with con:
                    con.executemany("INSERT INTO documents VALUES (?,?)",((d["doc_id"],native.packed(d)) for d in docs))
                    for batch in pq.ParquetFile(lexical).iter_batches(batch_size=64):
                        groups = {lang:[] for lang in ("vi","en","zh")}
                        for row in batch.to_pylist():
                            pair = (row["doc_id"],row["language"])
                            if pair not in expected:
                                raise ValueError("Lexical checkpoint document/language changed.")
                            expected.remove(pair)
                            groups[row["language"]].append(tuple(row[k] for k in ("doc_id","title","headings","body","aliases")))
                        for lang,entries in groups.items():
                            con.executemany(f"INSERT INTO fts_{lang}(rowid,title,headings,body,aliases) VALUES (?,?,?,?,?)",entries)
                    if expected:
                        raise ValueError("Incomplete lexical checkpoint.")
                    added = {"documents":len(docs)}
                    for kind in ("parents","children"):
                        added[kind] = _insert_spans(con,kind,paths[kind],by_id)
                    state["counts"] = {k:state["counts"][k]+added[k] for k in state["counts"]}
                    state["next_part"] = index+1
                    _state(con,state)
                elapsed = time.perf_counter()-started
                print(f"Disk catalog/BM25: {index+1}/{len(candidate['parts'])} shards |"
                      f" lexical {'reused' if reused else 'new'} {analyzed-started:.1f}s |"
                      f" SQL {time.perf_counter()-analyzed:.1f}s | {elapsed:.1f}s/shard",flush=True)
            counts = state["counts"]
            if any(counts[k] != candidate["counts"][k]
                   or con.execute(f"SELECT count(*) FROM {k}").fetchone()[0] != counts[k] for k in counts):
                raise ValueError("Disk catalog counts differ from frozen snapshot.")
            print("Catalog: create lookup indexes and validate source/parent/unit mappings",flush=True)
            con.executescript("""
                CREATE INDEX IF NOT EXISTS child_doc ON children(doc_id);
                CREATE INDEX IF NOT EXISTS child_representation ON children(representation);
                CREATE INDEX IF NOT EXISTS parent_doc ON parents(doc_id);
            """)
            invalid = con.execute("""SELECT count(*) FROM children c LEFT JOIN units u ON u.id=c.representation
                LEFT JOIN parents p ON p.chunk_id=c.parent_id WHERE u.id IS NULL OR p.chunk_id IS NULL
                OR p.doc_id <> c.doc_id""").fetchone()[0]
            unused = con.execute("SELECT count(*) FROM units u WHERE NOT EXISTS(SELECT 1 FROM children c WHERE c.representation=u.id)").fetchone()[0]
            if invalid or unused:
                raise ValueError("Unit/official alias/parent mapping is incomplete.")
            with con:
                _state(con,policy)
        if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed.")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("PRAGMA journal_mode=DELETE")
    finally:
        con.close()
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True)
    checksum = sha256_file(building)
    # Keep the finalized file at building.sqlite until the completion marker is
    # durable. A lost process can finalize again without replaying source rows.
    if local.is_file():
        verify_file(local,checksum)
    else:
        try:
            # Same local filesystem: avoid temporarily duplicating a large DB.
            os.link(building,local)
        except OSError:
            if publish_file(building,local) != checksum:
                raise ValueError("Catalog local copy changed.")
    saved = policy | {"signature":signature,"counts":{k:candidate["counts"][k] for k in ("documents","children","parents")},
        "unit_count":inputs["input_count"],"path":"catalog.sqlite","sha256":checksum,"state":"COMPLETE"}
    saved["manifest_sha256"] = digest_json(saved)
    atomic_json(marker,saved)
    building.unlink()
    return native.DiskCatalog(local,candidate,inputs,analyzer,saved)


def read_json_value(value):
    import json
    return json.loads(value)
