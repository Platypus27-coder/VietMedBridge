"""Current-official filtering preserves paid vectors and exact source spans."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_scale_baseline import frozen, Tokenizer, Encoder
from test_external_import import external, SPEC, CONFIG
from vietmedbridge import official_scope as scope
from vietmedbridge.artifacts import atomic_json, code_fingerprint, read_json, sha256_file
from vietmedbridge.content_embeddings import content_embeddings, ContentVectorParts
from vietmedbridge.submission_workflow import verify_current_submission


def current(tmp_path, rows):
    path = tmp_path / "new-official.parquet"
    pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema([("id",pa.int64()),("url",pa.string())])),path)
    return {"revision":"a"*40,"files":{"links_corpus.parquet":{
        "path":path.name,"sha256":sha256_file(path),"rows":len(rows)}}}


def golden():
    return {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}


def derive(tmp_path, frozen, snapshot, **kwargs):
    return scope.derive_official_scope(tmp_path,[frozen[0].name],snapshot,Tokenizer(),golden(),
                                      work_dir=tmp_path / "work",**kwargs)


def test_scope_keeps_exact_fragment_alias_and_source_offsets(tmp_path,frozen,external):
    original = pq.read_table(frozen[0] / frozen[1]["parts"][0]["files"]["documents"]["path"]).to_pylist()
    chosen = next(x for x in original if x["doc_id"] == 921)
    snap = current(tmp_path,[{"id":921,"url":chosen["url"]},{"id":777777,"url":"https://new.example/no-source"}])
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    assert candidate["counts"]["documents"] == candidate["counts"]["input_records"] == 1
    assert candidate["counts"]["failures"] == 0
    assert candidate["integrity"]["passed"]
    doc = pq.read_table(build / candidate["parts"][0]["files"]["documents"]["path"]).to_pylist()[0]
    assert doc == chosen
    for kind in ("children","parents"):
        spans = pq.read_table(build / candidate["parts"][0]["files"][kind]["path"]).to_pylist()
        assert spans and all(x["doc_id"] == 921 and x["text"] == chosen["source_text"][x["start_char"]:x["end_char"]] for x in spans)
    assert inputs["state"] == "COMPLETE"
    assert read_json(tmp_path / "active_data_candidate.json")["build_run"] == frozen[0].name
    # A completed scope resumes from its own sealed manifests, without reading sources again.
    assert derive(tmp_path,frozen,snap)[1]["candidate_manifest_sha256"] == candidate["candidate_manifest_sha256"]


def test_removed_or_changed_url_is_not_rebound_by_id_alone(tmp_path,frozen):
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/changed"}])
    with pytest.raises(ValueError,match="coverage mismatch"):
        derive(tmp_path,frozen,snap)


def test_missing_expected_url_aborts_before_activation(tmp_path,frozen):
    url = "https://example.org/not-crawled"
    snap = current(tmp_path,[{"id":123456,"url":url}])
    expected = tmp_path / "expected.parquet"
    pq.write_table(pa.Table.from_pylist([{"doc_id":123456,"url":url}]),expected)
    with pytest.raises(ValueError,match="coverage mismatch"):
        derive(tmp_path,frozen,snap,expected_urls=expected)
    assert read_json(tmp_path / "active_data_candidate.json")["build_run"] == frozen[0].name


def test_source_corruption_cannot_be_hidden_by_scope_filter(tmp_path,frozen):
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    (frozen[0] / frozen[1]["parts"][0]["files"]["documents"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="Changed source artifact"):
        derive(tmp_path,frozen,snap)


def test_scope_part_checkpoint_resumes_after_copy_interruption(tmp_path,frozen,monkeypatch):
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    original = scope.publish_file
    failed = False
    def interrupt(source,target):
        nonlocal failed
        if Path(target).name == "build.json" and not failed:
            failed = True
            raise OSError("test publication interruption")
        return original(source,target)
    monkeypatch.setattr(scope,"publish_file",interrupt)
    with pytest.raises(OSError,match="interruption"):
        derive(tmp_path,frozen,snap)
    # Source parts have durable receipts. The second attempt must not copy original data again.
    original_copy = scope._copy_checked
    def copy(source,target,checksum):
        if frozen[0] in Path(source).parents:
            pytest.fail("Already checkpointed source part was processed twice")
        return original_copy(source,target,checksum)
    monkeypatch.setattr(scope,"_copy_checked",copy)
    _,candidate,_ = derive(tmp_path,frozen,snap)
    assert candidate["counts"]["documents"] == 1


def test_scope_reuses_unchanged_vectors_without_neural_inference(tmp_path,frozen):
    old_build,_,old_inputs = frozen
    encoder = Encoder()
    old_view = tmp_path / "old_vectors"
    old_manifest = content_embeddings(tmp_path,old_build / "index_inputs",old_inputs,encoder,old_view,work_dir=tmp_path / "work")
    calls = encoder.calls
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    view = tmp_path / "new_vectors"
    report = scope.rebind_cached_family(tmp_path,build,candidate,inputs,
        {"encoder":encoder.identity,"dimension":encoder.dimension},view,work_dir=tmp_path / "work")
    assert report["missing_inputs"] == 0 and report["cached_inputs"] == inputs["input_count"]
    assert report["neural_inference_used"] is False and encoder.calls == calls
    new_manifest = read_json(view / "embeddings.json")
    old_ids = [row["id"] for p in old_inputs["parts"] for row in pq.read_table(old_build / "index_inputs" / p["path"]).to_pylist()]
    new_ids = [row["id"] for p in inputs["parts"] for row in pq.read_table(build / "index_inputs" / p["path"]).to_pylist()]
    old_store = ContentVectorParts(tmp_path,old_view,old_manifest,tmp_path / "old_cache")
    new_store = ContentVectorParts(tmp_path,view,new_manifest,tmp_path / "new_cache")
    np.testing.assert_array_equal(new_store.gather(range(len(new_ids))),old_store.gather([old_ids.index(i) for i in new_ids]))


def test_finished_family_inventory_resumes_without_corpus_rescan(tmp_path,frozen,monkeypatch):
    old_build,_,old_inputs = frozen
    encoder = Encoder()
    content_embeddings(tmp_path,old_build / "index_inputs",old_inputs,encoder,tmp_path / "old_vectors",work_dir=tmp_path / "work")
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    producer = {"encoder":encoder.identity,"dimension":encoder.dimension}
    output = tmp_path / "new_vectors"
    report = scope.rebind_cached_family(tmp_path,build,candidate,inputs,producer,output,work_dir=tmp_path / "work")
    monkeypatch.setattr(scope,"ContentBlocks",lambda *a,**k:pytest.fail("Completed family lookup rebuilt"))
    monkeypatch.setattr(scope,"content_embeddings",lambda *a,**k:pytest.fail("Completed family remapped"))
    assert scope.rebind_cached_family(tmp_path,build,candidate,inputs,producer,output,work_dir=tmp_path / "fresh") == report
    changed = {"encoder":encoder.identity | {"revision":"other-producer"},"dimension":encoder.dimension}
    with pytest.raises(ValueError,match="checkpoint input/model changed"):
        scope.rebind_cached_family(tmp_path,build,candidate,inputs,changed,output,work_dir=tmp_path / "fresh")


def test_family_inventory_refreshes_when_gpu_maps_advance(tmp_path,frozen):
    old_build,_,old_inputs = frozen
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    encoder = Encoder()
    producer = {"encoder":encoder.identity,"dimension":encoder.dimension}
    output = tmp_path / "new_vectors"
    initial = scope.rebind_cached_family(tmp_path,build,candidate,inputs,producer,output,work_dir=tmp_path / "work")
    assert initial["cached_inputs"] == initial["complete_mapped_parts"] == 0
    content_embeddings(tmp_path,old_build / "index_inputs",old_inputs,encoder,tmp_path / "seed",work_dir=tmp_path / "work")
    content_embeddings(tmp_path,build / "index_inputs",inputs,encoder,output,work_dir=tmp_path / "work",encode_missing=False)
    refreshed = scope.rebind_cached_family(tmp_path,build,candidate,inputs,producer,output,work_dir=tmp_path / "work")
    assert refreshed["missing_inputs"] == 0 and refreshed["complete_mapped_parts"] == len(inputs["parts"])


def test_legacy_all_keeps_both_old_ids_without_new_btc_filter(tmp_path,frozen,external):
    snap = {"revision":"old-source","files":{"links_corpus.parquet":{
        "path":external[1].name,"sha256":sha256_file(external[1])}}}
    build,candidate,inputs = derive(tmp_path,frozen,snap,keep_all_legacy=True)
    docs = [r for part in candidate["parts"] for r in pq.read_table(
        build / part["files"]["documents"]["path"],columns=["doc_id"]).to_pylist()]
    assert {r["doc_id"] for r in docs} == {583,921}
    assert candidate["input_kind"] == "AVAILABLE_CORPUS" and candidate["counts"]["documents"] == 2
    assert read_json(build / "config.json")["selection"] == "all-validated-successes-keep-legacy-urls-v1"


def test_index_checkpoint_survives_fresh_local_runtime(tmp_path,frozen,monkeypatch):
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    original = scope.prepare_index_inputs
    saved = None
    def interrupt(*args,**kwargs):
        nonlocal saved
        saved = original(*args,**kwargs)
        raise OSError("test after durable index publication")
    monkeypatch.setattr(scope,"prepare_index_inputs",interrupt)
    with pytest.raises(OSError,match="durable index"):
        derive(tmp_path,frozen,snap)
    local = tmp_path / "work/official_scope"
    assert local.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(local)
    monkeypatch.setattr(scope,"prepare_index_inputs",original)
    original_read = scope.read_json
    def no_receipt_replay(path):
        if Path(path).parent.name == "scope_checkpoints":
            pytest.fail("Sealed freeze must skip source receipt replay")
        return original_read(path)
    monkeypatch.setattr(scope,"read_json",no_receipt_replay)
    original_copy = scope._copy_checked
    def metadata_only(source,target,checksum):
        if Path(source).parent.name == "parts":
            pytest.fail("Complete inputs must not restore corpus payloads")
        if "reports" in Path(source).parts:
            pytest.fail("Index input resume must not download large alias reports")
        return original_copy(source,target,checksum)
    monkeypatch.setattr(scope,"_copy_checked",metadata_only)
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    assert inputs == saved and inputs["candidate_manifest_sha256"] == candidate["candidate_manifest_sha256"]


@pytest.mark.parametrize("fresh_runtime",[False,True])
def test_partial_index_resume_skips_scope_and_keeps_published_part(tmp_path,frozen,monkeypatch,capsys,fresh_runtime):
    from vietmedbridge import index_inputs
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    original_prepare, original_atomic = scope.prepare_index_inputs,index_inputs.atomic_json
    monkeypatch.setattr(scope,"prepare_index_inputs",lambda *a,**k:original_prepare(*a,**(k | {"part_size":1})))
    def interrupt(path,value):
        if Path(path).name == "units-000000000001.done.json":
            raise OSError("partial model inputs")
        return original_atomic(path,value)
    monkeypatch.setattr(index_inputs,"atomic_json",interrupt)
    with pytest.raises(OSError,match="partial model inputs"):
        derive(tmp_path,frozen,snap)
    target = next((tmp_path / "processed").glob("official-*"))
    sealed = read_json(target / "scope_freeze.json")
    part = read_json(target / "index_inputs/units-000000000000.done.json")
    first = target / "index_inputs" / part["path"]
    before = sha256_file(first),first.stat().st_mtime_ns
    if fresh_runtime:
        shutil.rmtree(tmp_path / "work")
    monkeypatch.setattr(index_inputs,"atomic_json",original_atomic)
    original_copy, original_read = scope._copy_checked,scope.read_json
    copied = []
    def copy(source,destination,checksum):
        if Path(source).parent.name == "parts":
            assert fresh_runtime, "A complete local spool must not restore source payloads"
            assert target in Path(source).parents
            assert Path(source).name.endswith((".documents.parquet",".children.parquet"))
            copied.append(Path(source).name)
        return original_copy(source,destination,checksum)
    def read(path):
        if Path(path).parent.name == "scope_checkpoints":
            pytest.fail("Frozen source receipts replayed before partial inputs")
        return original_read(path)
    monkeypatch.setattr(scope,"_copy_checked",copy)
    monkeypatch.setattr(scope,"read_json",read)
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    assert bool(copied) == fresh_runtime and candidate == sealed["candidate"]
    assert (sha256_file(first),first.stat().st_mtime_ns) == before
    assert inputs["parts"][0] == part and inputs["state"] == "COMPLETE"
    assert "Restored frozen scope:" in capsys.readouterr().out


def test_lazy_restore_still_rejects_corrupt_frozen_payload(tmp_path,frozen,monkeypatch):
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    original = scope.prepare_index_inputs
    monkeypatch.setattr(scope,"prepare_index_inputs",lambda *a,**k:(_ for _ in ()).throw(OSError("before inputs")))
    with pytest.raises(OSError,match="before inputs"):
        derive(tmp_path,frozen,snap)
    target = next((tmp_path / "processed").glob("official-*"))
    candidate = read_json(target / "scope_freeze.json")["candidate"]
    (target / candidate["parts"][0]["files"]["children"]["path"]).write_bytes(b"corrupt")
    shutil.rmtree(tmp_path / "work")
    monkeypatch.setattr(scope,"prepare_index_inputs",original)
    with pytest.raises(ValueError,match="Changed source artifact"):
        derive(tmp_path,frozen,snap)
    assert not (target / "scope_handoff.json").exists()


def test_catalog_staging_uses_only_catalog_kinds_and_known_input_checksums(tmp_path,frozen):
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    local = scope.stage_scope_catalog(build,candidate,inputs,work_dir=tmp_path / "fresh")
    assert {p.name.split(".")[-2] for p in (local / "parts").glob("*.parquet")} == {"documents","children","parents"}
    assert read_json(local / "index_inputs/units.json") == inputs
    for part in inputs["parts"]:
        assert sha256_file(local / "index_inputs" / part["path"]) == part["sha256"]


def test_activation_records_previous_snapshot_and_candidate(tmp_path,frozen):
    old = read_json(tmp_path / "active_data_candidate.json")
    atomic_json(tmp_path / "raw/snapshot.json",{"revision":"previous","files":{}})
    snap = current(tmp_path,[{"id":921,"url":"https://example.org/article#evidence"}])
    build,candidate,inputs = derive(tmp_path,frozen,snap)
    result = scope.activate_scope(tmp_path,build,candidate,inputs,snap)
    assert result["candidate_manifest_sha256"] == candidate["candidate_manifest_sha256"]
    journal = read_json(tmp_path / "submission_scope_activation.json")
    assert journal["state"] == "COMPLETE" and journal["previous_candidate"] == old
    scope.activate_scope(tmp_path,build,candidate,inputs,snap)
    assert read_json(tmp_path / "submission_scope_activation.json")["previous_candidate"] == old
    assert read_json(tmp_path / "raw/snapshot.json") == snap


def test_final_zip_rejects_removed_id_and_preserves_query_order(tmp_path):
    path = tmp_path / "official.parquet"
    pq.write_table(pa.Table.from_pylist([{"id":921,"url":"https://example.org/current"}]),path)
    queries = [{"id":9,"query":"question"}]
    result = {"id":9,"relevant_docs":[921],"relevant_chunks":[{"doc_id":921,"chunk_text":"source"}]}
    zip_path = tmp_path / "submission.zip"
    def save(value):
        with ZipFile(zip_path,"w") as z:
            z.writestr("results.json",json.dumps([value]))
    save(result)
    assert verify_current_submission(zip_path,queries,path)["state"] == "DECLARED_SNAPSHOT_SUBMISSION_VERIFIED"
    save(result | {"relevant_docs":[583]})
    with pytest.raises(ValueError,match="nonofficial"):
        verify_current_submission(zip_path,queries,path)
