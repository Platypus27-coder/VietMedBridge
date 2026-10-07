"""Read frozen source rows lazily and keep multilingual BM25 postings on disk."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, sha256_file, verify_file
from .scale_vectors import bound, check_manifest


def packed(value):
    return json.dumps(value,ensure_ascii=False,separators=(",",":"))


def lexical_field(analyzer,text,language):
    # The tokenizer sees one ASCII term per medical token, retaining accents/CJK
    # distinctions and segmentation boundaries without FTS query syntax injection.
    return " ".join("t"+hashlib.sha256(t.encode()).hexdigest() for t in analyzer.analyze(text,language))


def rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        yield from batch.to_pylist()


def prepare_disk_catalog(build, candidate, inputs, output_dir, analyzer, *, work_dir):
    check_manifest(inputs)
    if (candidate.get("state") != "FROZEN_CANDIDATE" or not candidate.get("selected_range_complete")
        or not candidate.get("integrity",{}).get("passed")
        or candidate["integrity"].get("official_membership") != "VERIFIED"
        or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]
        or digest_json({"signature":candidate["signature"],"parts":candidate["parts"]}) != candidate["snapshot_sha256"]
        or inputs["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]):
        raise ValueError("Disk catalog requires matching, validated frozen inputs.")
    build_config = read_json(Path(build) / "config.json")
    if (build_config["signature"] != candidate["signature"] or
        digest_json({k:v for k,v in build_config.items() if k != "signature"}) != candidate["signature"]):
        raise ValueError("Frozen build config changed.")
    policy = {"candidate":candidate["candidate_manifest_sha256"],"inputs":inputs["manifest_sha256"],
        "analyzer":analyzer.identity,"sqlite":sqlite3.sqlite_version,"schema":"disk-source-fts5-v1",
        "code_sha256":sha256_file(Path(__file__)),"field_weights":[2.,1.5,1.,1.5]}
    signature = digest_json(policy)
    target = Path(output_dir) / ("catalog-"+signature[:16])
    marker = target / "catalog.json"
    local_dir = Path(work_dir) / ("catalog-"+signature[:16])
    local_dir.mkdir(parents=True,exist_ok=True)
    local = local_dir / "catalog.sqlite"
    if marker.exists():
        saved = read_json(marker)
        check_manifest(saved)
        if saved["signature"] != signature:
            raise ValueError("Disk catalog contract changed.")
        remote = bound(target,saved["path"])
        if local.is_file():
            verify_file(local,saved["sha256"])
        else:
            verify_file(remote,saved["sha256"])
            if publish_file(remote,local) != saved["sha256"]:
                raise ValueError("Catalog cache copy changed.")
        return DiskCatalog(local,candidate,inputs,analyzer,saved)
    # Work in a private local file. A partial DB is never published as complete.
    with local_workspace(work_dir) as temporary:
        database = Path(temporary) / "catalog.sqlite"
        con = sqlite3.connect(database)
        try:
            con.executescript("""
                PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
                PRAGMA cache_size=-65536; PRAGMA temp_store=FILE;
                CREATE TABLE documents(doc_id INTEGER PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE children(chunk_id TEXT PRIMARY KEY,doc_id INTEGER NOT NULL,
                    representation TEXT NOT NULL,parent_id TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE parents(chunk_id TEXT PRIMARY KEY,doc_id INTEGER NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE units(position INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL);
                CREATE TABLE metadata(payload TEXT NOT NULL);
            """)
            for lang in ("vi","en","zh"):
                con.execute(f"CREATE VIRTUAL TABLE fts_{lang} USING fts5(title,headings,body,aliases,tokenize='ascii',content='')")
            cursor = 0
            for part in inputs["parts"]:
                path = bound(Path(build) / "index_inputs",part["path"])
                verify_file(path,part["sha256"])
                if part["start"] != cursor:
                    raise ValueError("Input ordering changed.")
                unit_rows = pq.read_table(path,columns=["id"]).column("id").to_pylist()
                if len(unit_rows) != part["rows"]:
                    raise ValueError("Input count changed.")
                con.executemany("INSERT INTO units VALUES (?,?)",[(cursor+i,key) for i,key in enumerate(unit_rows)])
                cursor += len(unit_rows)
            if cursor != inputs["input_count"]:
                raise ValueError("Incomplete unit mapping.")
            counts = {"documents":0,"children":0,"parents":0}
            for part_index,part in enumerate(candidate["parts"]):
                paths = {}
                for kind in counts:
                    entry = part["files"][kind]
                    paths[kind] = bound(build,entry["path"])
                    verify_file(paths[kind],entry["sha256"])
                docs = {}
                for d in rows(paths["documents"]):
                    if hashlib.sha256(d["source_text"].encode()).hexdigest() != d["source_text_sha256"]:
                        raise ValueError("Frozen document source changed.")
                    docs[d["doc_id"]] = d
                    con.execute("INSERT INTO documents VALUES (?,?)",(d["doc_id"],packed(d)))
                    language = d["language"] if d["language"] in ("vi","en","zh") else "unknown"
                    for lang in ((language,) if language != "unknown" else ("vi","en","zh")):
                        fields = [d.get("title", ""),"\n".join(d.get("heading_hints") or []),
                            d["source_text"],analyzer.aliases(d["source_text"])]
                        con.execute(f"INSERT INTO fts_{lang}(rowid,title,headings,body,aliases) VALUES (?,?,?,?,?)",
                            [d["doc_id"],*[lexical_field(analyzer,v,lang) for v in fields]])
                    counts["documents"] += 1
                for kind in ("parents","children"):
                    for row in rows(paths[kind]):
                        d = docs[row["doc_id"]]
                        a,b = row["start_char"],row["end_char"]
                        if (not 0 <= a < b <= len(d["source_text"]) or row["text"] != d["source_text"][a:b]
                            or row["source_text_sha256"] != d["source_text_sha256"]):
                            raise ValueError("Frozen source span changed.")
                        payload = packed({k:v for k,v in row.items() if k not in ("text","retrieval_text")})
                        if kind == "children":
                            con.execute("INSERT INTO children VALUES (?,?,?,?,?)",(row["chunk_id"],row["doc_id"],
                                row["retrieval_representation_hash"],row["parent_id"],payload))
                        else:
                            con.execute("INSERT INTO parents VALUES (?,?,?)",(row["chunk_id"],row["doc_id"],payload))
                        counts[kind] += 1
                con.commit()
                print(f"Disk catalog/BM25: {part_index+1}/{len(candidate['parts'])} shards",flush=True)
            if any(counts[k] != candidate["counts"][k] for k in counts):
                raise ValueError("Disk catalog counts differ from frozen snapshot.")
            con.executescript("""
                CREATE INDEX child_doc ON children(doc_id);
                CREATE INDEX child_representation ON children(representation);
                CREATE INDEX parent_doc ON parents(doc_id);
            """)
            invalid = con.execute("""SELECT count(*) FROM children c LEFT JOIN units u ON u.id=c.representation
                LEFT JOIN parents p ON p.chunk_id=c.parent_id WHERE u.id IS NULL OR p.chunk_id IS NULL
                OR p.doc_id <> c.doc_id""").fetchone()[0]
            unused = con.execute("SELECT count(*) FROM units u WHERE NOT EXISTS(SELECT 1 FROM children c WHERE c.representation=u.id)").fetchone()[0]
            if invalid or unused:
                raise ValueError("Unit/official alias/parent mapping is incomplete.")
            con.execute("INSERT INTO metadata VALUES (?)",(packed(policy),))
            con.commit()
            if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed.")
        finally:
            con.close()
        checksum = publish_file(database,target / database.name)
        saved = policy | {"signature":signature,"counts":counts,"unit_count":cursor,
            "path":database.name,"sha256":checksum,"state":"COMPLETE"}
        saved["manifest_sha256"] = digest_json(saved)
        atomic_json(marker,saved)
        if publish_file(database,local) != checksum:
            raise ValueError("Catalog local copy changed.")
    return DiskCatalog(local,candidate,inputs,analyzer,saved)


class _Rows:
    def __init__(self,catalog,table,key):
        self.catalog,self.table,self.key,self.cache = catalog,table,key,OrderedDict()

    def get(self,key,default=None):
        if key not in self.cache:
            found = self.catalog.con.execute(f"SELECT payload FROM {self.table} WHERE {self.key}=?",(key,)).fetchone()
            if found is None:
                return default
            row = json.loads(found[0])
            if self.table != "documents":
                row["text"] = self.catalog.documents[row["doc_id"]]["source_text"][row["start_char"]:row["end_char"]]
            self.cache[key] = row
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self,key):
        row = self.get(key)
        if row is None:
            raise KeyError(key)
        return row

    def __contains__(self,key):
        return self.catalog.con.execute(f"SELECT 1 FROM {self.table} WHERE {self.key}=?",(key,)).fetchone() is not None


class DiskCatalog:
    def __init__(self,path,candidate,inputs,analyzer,manifest):
        self.con = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)
        self.con.execute("PRAGMA cache_size=-65536")
        self.candidate,self.inputs,self.analyzer,self.manifest = candidate,inputs,analyzer,manifest
        self.documents,self.children,self.parents = (_Rows(self,"documents","doc_id"),
            _Rows(self,"children","chunk_id"),_Rows(self,"parents","chunk_id"))

    @property
    def identity(self):
        return {"snapshot_sha256":self.candidate["snapshot_sha256"],
            "candidate_manifest_sha256":self.candidate["candidate_manifest_sha256"],
            "index_inputs_manifest_sha256":self.inputs["manifest_sha256"],
            "disk_catalog_manifest_sha256":self.manifest["manifest_sha256"]}

    def close(self):
        self.con.close()

    def dense_documents(self,positions,scores,limit):
        self.con.execute("CREATE TEMP TABLE IF NOT EXISTS hits(position INTEGER PRIMARY KEY,score REAL)")
        self.con.execute("DELETE FROM hits")
        self.con.executemany("INSERT INTO hits VALUES (?,?)",[(int(p),float(s)) for p,s in zip(positions,scores,strict=True)])
        return self.con.execute("""SELECT c.doc_id,max(h.score) AS score FROM hits h
            JOIN units u USING(position) JOIN children c ON c.representation=u.id
            GROUP BY c.doc_id ORDER BY score DESC,c.doc_id LIMIT ?""",(limit,)).fetchall()

    def sparse(self,query,language,limit):
        if language not in ("vi","en","zh"):
            raise ValueError("Unsupported lexical language.")
        terms = sorted(set(lexical_field(self.analyzer,query,language).split()))
        if not terms:
            return []
        return self.con.execute(f"""SELECT rowid,-bm25(fts_{language},2.0,1.5,1.0,1.5) AS score
            FROM fts_{language} WHERE fts_{language} MATCH ? ORDER BY score DESC,rowid LIMIT ?""",
            (" OR ".join(terms),limit)).fetchall()

    def document_children(self,doc_id):
        result = self.con.execute("""SELECT c.payload,u.position FROM children c JOIN units u ON u.id=c.representation
            WHERE c.doc_id=? ORDER BY c.chunk_id""",(doc_id,)).fetchall()
        source = self.documents[doc_id]["source_text"]
        children = []
        for payload,position in result:
            row = json.loads(payload)
            row["text"] = source[row["start_char"]:row["end_char"]]
            children.append((row,position))
        return children
