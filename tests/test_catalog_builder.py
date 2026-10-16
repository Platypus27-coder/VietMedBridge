"""Actual SQLite/FTS parity and interruptions, without neural model inference."""
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_scale_baseline import external, frozen  # noqa: F401
from vietmedbridge import catalog_builder as fast, catalog_storage as storage, disk_catalog as old
from vietmedbridge.artifacts import digest_json, read_json, sha256_file
from vietmedbridge.medical_lexical import MedicalAnalyzer


def build(tmp_path,frozen,analyzer,work="fast",workers=1):
    source,candidate,inputs = frozen
    return fast.prepare_catalog(source,candidate,inputs,tmp_path / work,analyzer,
        work_dir=tmp_path / work,policy=storage._contract(source,candidate,inputs,analyzer),
        checkpoint_dir=tmp_path / "receipts",workers=workers)


@pytest.mark.parametrize("language",["vi","en","zh","unknown"])
def test_fast_catalog_rows_and_bm25_equal_original(tmp_path,frozen,language):
    analyzer = MedicalAnalyzer(segmentation=False)
    source,candidate,inputs = frozen
    for part in candidate["parts"]:
        entry = part["files"]["documents"]
        path = source / entry["path"]
        table = pq.read_table(path)
        if table.num_rows:
            docs = table.to_pylist()
            for document in docs:
                document["language"] = language
            pq.write_table(pa.Table.from_pylist(docs,schema=table.schema),path)
            entry.update(sha256=sha256_file(path),bytes=path.stat().st_size)
    candidate["snapshot_sha256"] = digest_json({"signature":candidate["signature"],"parts":candidate["parts"]})
    candidate["candidate_manifest_sha256"] = digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"})
    inputs["candidate_manifest_sha256"] = candidate["candidate_manifest_sha256"]
    inputs["manifest_sha256"] = digest_json({k:v for k,v in inputs.items() if k != "manifest_sha256"})
    before = old.prepare_disk_catalog(source,candidate,inputs,tmp_path / "old",analyzer,work_dir=tmp_path / "original")
    after = build(tmp_path,frozen,analyzer,workers=2)
    try:
        for table,key in (("documents","doc_id"),("children","chunk_id"),("parents","chunk_id"),("units","position"),("metadata","payload")):
            query = f"SELECT * FROM {table} ORDER BY {key}"
            assert before.con.execute(query).fetchall() == after.con.execute(query).fetchall()
        for language in ("vi","en","zh"):
            assert before.sparse("中文医学 triệu chứng Y sinh",language,100) == after.sparse("中文医学 triệu chứng Y sinh",language,100)
        assert before.document_children(583) == after.document_children(583)
        assert before.manifest["signature"] == after.manifest["signature"]
    finally:
        before.close()
        after.close()


def test_lexical_checkpoint_reused_when_only_candidate_identity_changes(tmp_path,frozen,monkeypatch):
    source,candidate,inputs = frozen
    analyzer = MedicalAnalyzer(segmentation=False)
    policy = storage._contract(source,candidate,inputs,analyzer)
    part = candidate["parts"][0]
    docs = list(old.rows(source / part["files"]["documents"]["path"]))
    local,remote = tmp_path / "cache-local",tmp_path / "cache-remote"
    local.mkdir()
    remote.mkdir()
    first,reused = fast._lexical_checkpoint(docs,part,policy,local,remote,analyzer,None)
    assert not reused
    monkeypatch.setattr(fast,"_analyze_document",lambda *a:pytest.fail("Changing candidate must not retokenize identical source"))
    second,reused = fast._lexical_checkpoint(docs,part,policy | {"candidate":"different","inputs":"different"},local,remote,analyzer,None)
    assert reused and first == second


def test_local_shard_commit_survives_later_failure(tmp_path,frozen,monkeypatch,capsys):
    analyzer = MedicalAnalyzer(segmentation=False)
    insert = fast._insert_spans
    calls = 0
    def interrupt(con,kind,path,docs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("interrupted next shard")
        return insert(con,kind,path,docs)
    monkeypatch.setattr(fast,"_insert_spans",interrupt)
    with pytest.raises(OSError,match="interrupted"):
        build(tmp_path,frozen,analyzer)
    database = next((tmp_path / "fast").glob("*/building.sqlite"))
    with sqlite3.connect(database) as con:
        state = json.loads(con.execute("SELECT payload FROM metadata").fetchone()[0])
        assert state["next_part"] == 1
    con.close()
    monkeypatch.setattr(fast,"_insert_spans",insert)
    original = old.rows
    def without_first(path):
        first = frozen[1]["parts"][0]
        if any(Path(path).name == Path(entry["path"]).name for entry in first["files"].values()):
            pytest.fail("Committed shard should not be read again")
        return original(path)
    monkeypatch.setattr(old,"rows",without_first)
    catalog = build(tmp_path,frozen,analyzer)
    catalog.close()
    assert "local SQL resume 1/" in capsys.readouterr().out


def test_new_runtime_reuses_published_lexical_receipts(tmp_path,frozen,monkeypatch):
    analyzer = MedicalAnalyzer(segmentation=False)
    first = build(tmp_path,frozen,analyzer)
    expected = first.sparse("triệu chứng","vi",100)
    first.close()
    receipts = list((tmp_path / "receipts").glob("*.json"))
    unchanged = {p.name:sha256_file(p) for p in receipts}
    monkeypatch.setattr(fast,"_analyze_document",lambda *a,**kw:pytest.fail("Durable lexical receipt should skip tokenization"))
    second = build(tmp_path,frozen,analyzer,work="new-vm")
    assert second.sparse("triệu chứng","vi",100) == expected
    second.close()
    assert unchanged == {p.name:sha256_file(p) for p in receipts}


def test_partial_lexical_publication_reused_after_sql_rollback(tmp_path,frozen,monkeypatch):
    analyzer = MedicalAnalyzer(segmentation=False)
    original = fast._insert_spans
    monkeypatch.setattr(fast,"_insert_spans",lambda *a:(_ for _ in ()).throw(OSError("SQL interrupted")))
    with pytest.raises(OSError,match="SQL interrupted"):
        build(tmp_path,frozen,analyzer)
    assert list((tmp_path / "receipts").glob("*.json"))
    monkeypatch.setattr(fast,"_insert_spans",original)
    analyzed = fast._analyze_document
    calls = []
    monkeypatch.setattr(fast,"_analyze_document",lambda d,a=None:(calls.append(d["doc_id"]),analyzed(d,a))[1])
    catalog = build(tmp_path,frozen,analyzer)
    assert catalog.manifest["counts"] == {k:frozen[1]["counts"][k] for k in ("documents","children","parents")}
    catalog.close()
    assert 583 not in calls and 921 not in calls


def test_corrupt_lexical_checkpoint_fails_before_catalog_completion(tmp_path,frozen,monkeypatch):
    analyzer = MedicalAnalyzer(segmentation=False)
    catalog = build(tmp_path,frozen,analyzer)
    catalog.close()
    receipt = read_json(next((tmp_path / "receipts").glob("*.json")))
    (tmp_path / "receipts" / receipt["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="changed artifact"):
        build(tmp_path,frozen,analyzer,work="new-vm")
    assert not list((tmp_path / "new-vm").glob("*/catalog.json"))


@pytest.mark.parametrize("language",["vi","en","zh","unknown"])
def test_parallel_segmentation_keeps_original_terms(language):
    pytest.importorskip("pyvi")
    pytest.importorskip("jieba")
    analyzer = MedicalAnalyzer(segmentation=True)
    document = {"doc_id":1,"language":language,"title":"Viêm dạ dày 医学",
        "heading_hints":["Chẩn đoán β-hCG"],"source_text":"Người bệnh bị viêm dạ dày. 中文医学治疗。 Beta-hCG 0.10 U/L."}
    result = fast._analyze_document(document,analyzer)
    for row in result:
        fields = [document["title"],"\n".join(document["heading_hints"]),document["source_text"],""]
        assert [row[k] for k in ("title","headings","body","aliases")] == [old.lexical_field(analyzer,t,row["language"]) for t in fields]


def test_real_segmenters_in_spawned_workers_keep_glossary_and_terms(tmp_path):
    pytest.importorskip("pyvi")
    pytest.importorskip("jieba")
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps({"reviewed":True,"entries":[{"aliases":["beta-hCG","hCG"],"source":"fixture"}]}),encoding="utf-8")
    analyzer = MedicalAnalyzer(glossary_path=glossary)
    docs = [{"doc_id":i,"language":lang,"source_text":"Người bệnh viêm dạ dày. 中文医学。 beta-hCG 0.10 U/L.",
             "title":"Viêm dạ dày","heading_hints":[]} for i,lang in enumerate(("vi","en","zh","unknown"))]
    expected = [fast._analyze_document(d,analyzer) for d in docs]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn"),
        initializer=fast._init_analyzer,initargs=(analyzer.identity,analyzer.entries)) as pool:
        assert list(pool.map(fast._analyze_document,docs)) == expected
