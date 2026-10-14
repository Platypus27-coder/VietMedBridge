"""Disk dedup, legacy-compatible ordering, bounded batches and interruption recovery."""
from pathlib import Path
import hashlib
import sqlite3

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_scale_baseline import Tokenizer, frozen, external
from vietmedbridge import index_inputs, input_spool
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.dataset import _connection
from vietmedbridge.representation import build_dense_text
from vietmedbridge.validation import open_views


def values(root, manifest):
    return [row for part in manifest["parts"] for row in pq.read_table(root / part["path"]).to_pylist()]


def test_disk_spool_matches_legacy_group_order_and_min_child(tmp_path,frozen):
    build,candidate,_ = frozen
    with _connection(tmp_path / "reference") as con:
        open_views(con,build,candidate)
        reference = con.execute("""SELECT r.retrieval_representation_hash,c.text,c.heading,d.title,r.chunk_id
            FROM (SELECT retrieval_representation_hash,min(chunk_id) AS chunk_id
                  FROM children GROUP BY 1) r JOIN children c USING(chunk_id)
            JOIN documents d USING(doc_id) ORDER BY r.retrieval_representation_hash""").fetchall()
    target = tmp_path / "drive-inputs"
    result = index_inputs.prepare_index_inputs(build,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",
        Tokenizer(),work_dir=tmp_path / "work",checkpoint_root=target,part_size=1)
    expected = [{"id":key,"text":build_dense_text({"text":text},heading=heading,title=title,
        tokenizer=Tokenizer(),max_tokens=candidate["chunking"]["dense_max_tokens"])} for key,text,heading,title,_ in reference]
    assert values(target,result) == expected
    spool_dir = tmp_path / "work/index_spools" / result["signature"][:24]
    assert not (spool_dir / "representatives.sqlite").exists()
    spool = input_spool.InputSpool(build,candidate,spool_dir,result["signature"])
    try:
        spool.stage()
        assert spool.con.execute("SELECT id,chunk_id FROM representatives ORDER BY id").fetchall() == [(r[0],r[4]) for r in reference]
    finally:
        spool.close()


def test_input_part_resume_reuses_local_spool_and_drive_parts(tmp_path,frozen,monkeypatch):
    build,candidate,_ = frozen
    target = tmp_path / "drive-inputs"
    original = index_inputs.atomic_json
    def interrupt(path,value):
        if Path(path).name == "units-000000000001.done.json":
            raise OSError("test interrupted part publication")
        return original(path,value)
    monkeypatch.setattr(index_inputs,"atomic_json",interrupt)
    args = (build,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",Tokenizer())
    kwargs = {"work_dir":tmp_path / "work","checkpoint_root":target,"part_size":1}
    with pytest.raises(OSError,match="publication"):
        index_inputs.prepare_index_inputs(*args,**kwargs)
    first = read_json(target / "units-000000000000.done.json")
    monkeypatch.setattr(index_inputs,"atomic_json",original)
    monkeypatch.setattr(input_spool.InputSpool,"_path",lambda *a:pytest.fail("Completed source spool was reread"))
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    assert result["state"] == "COMPLETE" and result["parts"][0] == first
    assert sha256_file(target / first["path"]) == first["sha256"]


def test_source_transaction_rolls_back_incomplete_part(tmp_path,frozen,monkeypatch):
    build,candidate,_ = frozen
    original = pq.ParquetFile
    child_paths = {build / part["files"]["children"]["path"] for part in candidate["parts"]}
    class Broken:
        def __init__(self, source):
            self.source,self.metadata = source,source.metadata
        def iter_batches(self,**kwargs):
            for batch in self.source.iter_batches(**(kwargs | {"batch_size":1})):
                yield batch
                raise OSError("test interrupted source scan")
    monkeypatch.setattr(input_spool.pq,"ParquetFile",lambda path:Broken(original(path)) if Path(path) in child_paths else original(path))
    spool = input_spool.InputSpool(build,candidate,tmp_path / "spool","stable-signature")
    with pytest.raises(OSError,match="source scan"):
        spool.stage()
    assert spool.con.execute("SELECT count(*) FROM representatives").fetchone()[0] == 0
    assert spool.con.execute("SELECT count(*) FROM completed WHERE kind='children'").fetchone()[0] == 0
    spool.close()
    monkeypatch.setattr(input_spool.pq,"ParquetFile",original)
    resumed = input_spool.InputSpool(build,candidate,tmp_path / "spool","stable-signature")
    try:
        assert resumed.stage() == frozen[2]["input_count"]
    finally:
        resumed.close()


def test_million_inputs_avoid_low_memory_global_aggregate(tmp_path):
    """One million synthetic unique texts: the old query OOMs at 64 MB."""
    count = 1_000_000
    source = tmp_path / "synthetic"
    source.mkdir()
    children,documents = source / "children.parquet",source / "documents.parquet"
    spill = tmp_path / "spill"
    spill.mkdir()
    with _connection(spill,memory_limit="64MB") as con:
        con.execute(f"""CREATE VIEW children AS SELECT sha256('chunk-'||i::VARCHAR) AS retrieval_representation_hash,
            sha256('child-'||i::VARCHAR) AS chunk_id,1::BIGINT AS doc_id,
            'chunk-'||i::VARCHAR AS text,'' AS heading FROM range({count}) t(i)""")
        con.execute("COPY (SELECT * FROM children) TO ? (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 16384)",[str(children)])
        with pytest.raises(duckdb.OutOfMemoryException):
            con.execute("""CREATE TABLE representatives AS
                SELECT row_number() OVER(ORDER BY retrieval_representation_hash)-1 AS position,*
                FROM (SELECT retrieval_representation_hash,min(chunk_id) AS chunk_id FROM children GROUP BY 1)""")
    pq.write_table(pa.Table.from_pylist([{"doc_id":1,"title":""}]),documents)
    spec = {"model_id":"SYNTHETIC_TEST_ONLY","revision":"not-a-model"}
    candidate = {"state":"FROZEN_CANDIDATE","selected_range_complete":True,
        "golden":{"tokenizer":spec},"chunking":{"dense_max_tokens":512},
        "counts":{"documents":1,"children":count},"parts":[{"files":{
            "documents":{"path":documents.name,"sha256":sha256_file(documents)},
            "children":{"path":children.name,"sha256":sha256_file(children)}}}]}
    candidate["candidate_manifest_sha256"] = digest_json(candidate)
    atomic_json(source / "config.json",{"tokenizer":spec})
    atomic_json(source / "candidate.json",candidate)
    output = tmp_path / "output"
    manifest = index_inputs.prepare_index_inputs(source,"candidate.json",Tokenizer(),work_dir=tmp_path / "work",
                                                 checkpoint_root=output)
    assert manifest["state"] == "COMPLETE" and manifest["input_count"] == count
    previous, seen = "", 0
    for part in manifest["parts"]:
        for row in pq.read_table(output / part["path"]).to_pylist():
            assert row["id"] > previous and row["id"] == hashlib.sha256(row["text"].encode()).hexdigest()
            previous = row["id"]; seen += 1
    assert seen == count
    spool = input_spool.InputSpool(source,candidate,tmp_path / "work/index_spools" / manifest["signature"][:24],manifest["signature"])
    try:
        assert spool.con.execute("PRAGMA cache_size").fetchone()[0] == -32768
        assert spool.con.execute("PRAGMA temp_store").fetchone()[0] == 1
    finally:
        spool.close()
