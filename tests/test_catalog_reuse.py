"""Inherit actual FTS postings, remap dense positions, and append only new IDs."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_external_import import external, run_import, SPEC, CONFIG, write_parquet
from test_scale_baseline import frozen, Tokenizer
from vietmedbridge import catalog_builder as fast, catalog_reuse as reuse, catalog_storage as storage, disk_catalog as native
from vietmedbridge import official_scope as scope
from vietmedbridge.artifacts import atomic_json, code_fingerprint, digest_json, read_json, sha256_file
from vietmedbridge.external_import import MEMBERS
from vietmedbridge.medical_lexical import MedicalAnalyzer


@pytest.fixture
def enlarged(tmp_path,frozen,external):
    analyzer = MedicalAnalyzer(segmentation=False)
    old_build,old_candidate,old_inputs = frozen
    source,links,text = external
    root = tmp_path / "catalogs"
    old = native.prepare_disk_catalog(old_build,old_candidate,old_inputs,root,analyzer,work_dir=tmp_path / "old-work")
    old.close()
    crawl_path = source / MEMBERS["crawl"]
    crawls = pq.read_table(crawl_path).to_pylist()
    for row in crawls:
        row["status"],row["http_status"] = "SUCCESS_RAW",200
    write_parquet(crawl_path,crawls)
    template = pq.read_table(source / MEMBERS["extracted"]).to_pylist()[0]
    extracted = []
    for index in range(3):
        body = text if index == 0 else text+f"\nNew evidence {index}: beta hCG diagnosis and treatment."
        extracted.append(template | {"crawl_url_id":str(index),"text":body,"text_markdown":"# Evidence\n"+body,
            "char_count":len(body),"extract_status":"EXTRACT_SUCCESS","language":"en" if index==1 else "vi"})
    write_parquet(source / MEMBERS["extracted"],extracted)
    run_import(tmp_path,external,run_name="recovered")
    snapshot = {"revision":"original-source","files":{"links_corpus.parquet":{
        "path":links.name,"sha256":sha256_file(links)}}}
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    build,candidate,inputs = scope.derive_official_scope(tmp_path,[old_build.name,"recovered"],snapshot,
        Tokenizer(),golden,work_dir=tmp_path / "scope-work",keep_all_legacy=True)
    assert candidate["counts"]["documents"] == 4
    return build,candidate,inputs,analyzer,root


def extended(tmp_path,enlarged,work="new-work"):
    build,candidate,inputs,analyzer,root = enlarged
    return storage.prepare_disk_catalog(build,candidate,inputs,root,analyzer,work_dir=tmp_path / work,
        stage_delta=lambda indexes:scope.stage_scope_catalog(build,candidate,inputs,
            work_dir=tmp_path / work,skip_parts=indexes))


def test_inherited_catalog_equals_full_rebuild_without_old_tokenization_or_source_copy(tmp_path,enlarged,monkeypatch):
    build,candidate,inputs,analyzer,root = enlarged
    original_files = {p:sha256_file(p) for p in root.rglob("*") if p.is_file()}
    expected = native.prepare_disk_catalog(build,candidate,inputs,tmp_path / "expected",analyzer,
        work_dir=tmp_path / "expected-work")
    analyze = fast._analyze_document
    calls = []
    def only_new(document,analyzer=None):
        assert document["doc_id"] not in (583,921)
        calls.append(document["doc_id"])
        return analyze(document,analyzer)
    monkeypatch.setattr(fast,"_analyze_document",only_new)
    original_copy = scope._copy_checked
    def no_old_payload(source,destination,checksum):
        if Path(source).parent.name == "parts" and Path(source).name in {
            Path(candidate["parts"][0]["files"][k]["path"]).name for k in ("documents","parents","children")}:
            pytest.fail("Inherited source payload must not be staged or analyzed again")
        return original_copy(source,destination,checksum)
    monkeypatch.setattr(scope,"_copy_checked",no_old_payload)
    actual = extended(tmp_path,enlarged)
    try:
        assert set(calls) == {4000,9999}
        assert actual.manifest["reused_catalog"]["counts"]["documents"] == 2
        for table,key in (("documents","doc_id"),("parents","chunk_id"),("children","chunk_id"),("units","position")):
            query = f"SELECT * FROM {table} ORDER BY {key}"
            assert actual.con.execute(query).fetchall() == expected.con.execute(query).fetchall()
        for lang in ("vi","en","zh"):
            assert actual.sparse("triệu chứng beta hCG 中文医学",lang,100) == expected.sparse("triệu chứng beta hCG 中文医学",lang,100)
        positions = list(range(inputs["input_count"]))
        scores = np.arange(len(positions),dtype=np.float32)
        assert actual.dense_documents(positions,scores,100) == expected.dense_documents(positions,scores,100)
        assert actual.document_children(583) == expected.document_children(583)
    finally:
        expected.close()
        actual.close()
    assert all(sha256_file(path) == value for path,value in original_files.items())


def test_upgrade_preserves_existing_unit_map_and_reuses_delta_lexical_checkpoint(tmp_path,enlarged,monkeypatch):
    build,candidate,inputs,analyzer,root = enlarged
    work = tmp_path / "partial-work"
    policy = storage._contract(build,candidate,inputs,analyzer)
    insert = fast._insert_spans
    new_parent = Path(candidate["parts"][1]["files"]["parents"]["path"]).name
    def interrupt(con,kind,path,docs):
        if Path(path).name == new_parent:
            raise OSError("interrupted delta SQL")
        return insert(con,kind,path,docs)
    monkeypatch.setattr(fast,"_insert_spans",interrupt)
    with pytest.raises(OSError,match="interrupted delta"):
        fast.prepare_catalog(build,candidate,inputs,work,analyzer,work_dir=work,policy=policy,
            checkpoint_dir=root / "lexical_cache",workers=1)
    database = next(work.glob("*/building.sqlite"))
    con = sqlite3.connect(database)
    unit_positions = con.execute("SELECT * FROM units ORDER BY position").fetchall()
    con.close()
    monkeypatch.setattr(fast,"_insert_spans",insert)
    monkeypatch.setattr(fast,"_analyze_document",lambda *a:pytest.fail("Existing lexical checkpoint must be retained"))
    catalog = extended(tmp_path,enlarged,work="partial-work")
    assert catalog.con.execute("SELECT * FROM units ORDER BY position").fetchall() == unit_positions
    assert catalog.manifest["reused_catalog"]["counts"]["documents"] == 2
    catalog.close()
    assert list(work.glob("*/before-inheritance.sqlite"))


@pytest.mark.parametrize("fault",["selection","snapshot","target"])
def test_changed_scope_receipt_never_silently_seeds_or_rebuilds(tmp_path,enlarged,fault):
    build,candidate,inputs,analyzer,root = enlarged
    receipt = build / "scope_checkpoints/source-000-000000.json"
    value = read_json(receipt)
    if fault == "selection":
        value["selected_ids"].pop()
    elif fault == "snapshot":
        value["witness"]["source_snapshot"] = "changed"
    else:
        value["part"]["files"]["children"]["sha256"] = "changed"
    value["manifest_sha256"] = digest_json({k:v for k,v in value.items() if k != "manifest_sha256"})
    atomic_json(receipt,value)
    with pytest.raises(ValueError,match="selection|whole source"):
        extended(tmp_path,enlarged)


def test_inheritance_never_applies_to_revised_official_whitelist(tmp_path,enlarged):
    build,candidate,inputs,analyzer,root = enlarged
    candidate = candidate | {"input_kind":"OFFICIAL_SCOPE"}
    policy = {"analyzer":analyzer.identity}
    assert reuse.find_seed(build,candidate,inputs,root,policy) is None


def test_restore_failure_preserves_partial_sql_and_source_catalog(tmp_path,enlarged,monkeypatch):
    build,candidate,inputs,analyzer,root = enlarged
    policy = storage._contract(build,candidate,inputs,analyzer)
    seed = reuse.find_seed(build,candidate,inputs,root,policy)
    local = tmp_path / "fresh"
    local.mkdir()
    original = {p:sha256_file(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(storage,"_restore",lambda *a:(_ for _ in ()).throw(OSError("Drive read aborted")))
    with pytest.raises(OSError,match="Drive read aborted"):
        reuse.initialize_seed(local,candidate,policy,seed,tmp_path / "no-cache")
    assert not (local / "building.sqlite").exists()
    assert all(sha256_file(p) == sha for p,sha in original.items())


def test_next_scope_can_inherit_completed_available_catalog_by_exact_parts(tmp_path,enlarged,external,monkeypatch):
    build,candidate,inputs,analyzer,root = enlarged
    first = extended(tmp_path,enlarged)
    first.close()
    snapshot = {"revision":"original-source","files":{"links_corpus.parquet":{
        "path":external[1].name,"sha256":sha256_file(external[1])}}}
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    next_build,next_candidate,next_inputs = scope.derive_official_scope(tmp_path,[build.name],snapshot,
        Tokenizer(),golden,work_dir=tmp_path / "next-scope",keep_all_legacy=True)
    assert reuse._exact_prefix(candidate,next_candidate) == list(range(len(candidate["parts"])))
    monkeypatch.setattr(fast,"_analyze_document",lambda *a:pytest.fail("Complete available catalog must be inherited"))
    next_catalog = extended(tmp_path,(next_build,next_candidate,next_inputs,analyzer,root),work="next-work")
    assert next_catalog.manifest["reused_catalog"]["counts"]["documents"] == 4
    next_catalog.close()


def test_valid_checksum_with_wrong_database_policy_cannot_be_inherited(tmp_path,enlarged):
    import json
    build,candidate,inputs,analyzer,root = enlarged
    marker = next(root.glob("catalog-*/catalog.json"))
    saved = read_json(marker)
    database = marker.parent / "catalog.sqlite"
    con = sqlite3.connect(database)
    original = json.loads(con.execute("SELECT payload FROM metadata").fetchone()[0])
    original["schema"] = "foreign"
    con.execute("UPDATE metadata SET payload=?",(json.dumps(original),))
    con.commit()
    con.close()
    saved["sha256"] = sha256_file(database)
    saved["manifest_sha256"] = digest_json({k:v for k,v in saved.items() if k != "manifest_sha256"})
    atomic_json(marker,saved)
    with pytest.raises(ValueError,match="database producer policy"):
        extended(tmp_path,enlarged)


@pytest.mark.parametrize("interruption",["during_copy","after_copy"])
def test_interrupted_inheritance_keeps_committed_units_without_restore_or_recopy(tmp_path,enlarged,monkeypatch,interruption):
    build,candidate,inputs,analyzer,root = enlarged
    policy = storage._contract(build,candidate,inputs,analyzer)
    seed = reuse.find_seed(build,candidate,inputs,root,policy)
    local = tmp_path / "migration"
    local.mkdir()
    building = local / "building.sqlite"
    con = sqlite3.connect(building)
    state = fast._create(con,policy)
    ids = [key for part in inputs["parts"] for key in pq.read_table(
        build / "index_inputs" / part["path"],columns=["id"]).column("id").to_pylist()]
    con.executemany("INSERT INTO units VALUES (?,?)",enumerate(ids))
    state["units_cursor"] = len(ids)
    fast._state(con,state)
    con.commit()
    con.close()
    original_hash = sha256_file(building)
    monkeypatch.setattr(reuse,"UNIT_COPY_ROWS",1)
    report = reuse._Progress.report
    blocks = []
    def stop(progress,phase):
        if phase.startswith("copy dense positions"):
            blocks.append(phase)
            if interruption == "during_copy" and len(blocks) == 2:
                raise KeyboardInterrupt("simulated interruption")
        if interruption == "after_copy" and phase == "preserve previous SQL checkpoint":
            raise KeyboardInterrupt("simulated interruption")
        report(progress,phase)
    monkeypatch.setattr(reuse._Progress,"report",stop)
    with pytest.raises(KeyboardInterrupt,match="simulated"):
        reuse.initialize_seed(local,candidate,policy,seed,tmp_path / "no-cache")
    assert sha256_file(building) == original_hash
    pending = sqlite3.connect(local / "seed-copy.sqlite")
    saved = json.loads(pending.execute("SELECT payload FROM metadata").fetchone()[0])
    committed = saved["units_cursor"]
    assert committed == (1 if interruption == "during_copy" else len(ids))
    assert pending.execute("SELECT count(*) FROM units").fetchone()[0] == committed
    pending.close()
    monkeypatch.setattr(storage,"_restore",lambda *a:pytest.fail("Restored BM25 must not be downloaded again"))
    remaining = []
    def resume(progress,phase):
        if phase.startswith("copy dense positions"):
            remaining.append(phase)
        report(progress,phase)
    monkeypatch.setattr(reuse._Progress,"report",resume)
    reuse.initialize_seed(local,candidate,policy,seed,tmp_path / "no-cache")
    assert len(remaining) == len(ids)-committed
    actual = sqlite3.connect(building)
    assert actual.execute("SELECT * FROM units ORDER BY position").fetchall() == list(enumerate(ids))
    assert actual.execute("SELECT count(*) FROM documents").fetchone()[0] == seed["manifest"]["counts"]["documents"]
    assert actual.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    actual.close()


def test_resume_v7_remap_committed_before_silent_quick_check(tmp_path,enlarged,monkeypatch):
    build,candidate,inputs,analyzer,root = enlarged
    policy = storage._contract(build,candidate,inputs,analyzer)
    seed = reuse.find_seed(build,candidate,inputs,root,policy)
    local = tmp_path / "v7"
    local.mkdir()
    saved = seed["manifest"]
    pending = local / "seed-copy.sqlite"
    storage._restore(seed["directory"],saved,pending)
    # v7 had already committed the entire replacement map before quick_check.
    con = sqlite3.connect(pending)
    con.executescript("DROP TABLE units; CREATE TABLE units(position INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL)")
    ids = [key for part in inputs["parts"] for key in pq.read_table(
        build / "index_inputs" / part["path"],columns=["id"]).column("id").to_pylist()]
    con.executemany("INSERT INTO units VALUES (?,?)",enumerate(ids))
    state = {"builder":"parallel-catalog-v1","policy":policy,"units_cursor":len(ids),
        "next_part":len(seed["plan"]["part_indexes"]),"counts":saved["counts"],"seed":{
            "plan_manifest_sha256":seed["plan"]["manifest_sha256"],"source_catalog_sha256":saved["sha256"],
            "source_candidate":saved["candidate"],"part_indexes":seed["plan"]["part_indexes"],"counts":saved["counts"]}}
    fast._state(con,state)
    con.commit()
    con.close()
    monkeypatch.setattr(storage,"_restore",lambda *a:pytest.fail("v7 completed seed must not restore again"))
    reuse.initialize_seed(local,candidate,policy,seed,tmp_path / "no-cache")
    assert (local / "building.sqlite").is_file() and not pending.exists()
    con = sqlite3.connect(local / "building.sqlite")
    assert con.execute("SELECT * FROM units ORDER BY position").fetchall() == list(enumerate(ids))
    con.close()
