import hashlib
import tarfile
from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vietmedbridge.archive_parts import restore_archive
from vietmedbridge.artifacts import atomic_json, code_fingerprint, digest_json, read_json, sha256_file
from vietmedbridge.chunks import ChunkConfig
from vietmedbridge.external_import import MEMBERS, find_external_source, import_external_corpus
from vietmedbridge.health import freeze_candidate, health_report
from vietmedbridge.index_inputs import prepare_index_inputs, publish_data_handoff
from vietmedbridge.scale_benchmark import load_handoff, sample_inputs


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i,i+1) for i in range(len(text))]}


SPEC = {"model_id":"test", "revision":"test", "special_tokens":False}
CONFIG = ChunkConfig(child_tokens=100,overlap_tokens=20,parent_tokens=220)


def write_parquet(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),path)


@pytest.fixture
def external(tmp_path):
    source = tmp_path / "source"
    links = tmp_path / "links.parquet"
    urls = ["https://example.org/article", "https://example.org/failed", "https://example.org/parse"]
    write_parquet(links,[{"id":583,"url":urls[0]}, {"id":921,"url":urls[0]+"#evidence"},
                         {"id":4000,"url":urls[1]}, {"id":9999,"url":urls[2]}])
    write_parquet(source / MEMBERS["frontier"],[{"crawl_url_id":str(i),"fetch_url":u,
        "doc_count":2 if i==0 else 1} for i,u in enumerate(urls)])
    raw_sha = hashlib.sha256(b"original HTTP").hexdigest()
    write_parquet(source / MEMBERS["crawl"],[{"crawl_url_id":str(i),"fetch_url":u,
        "status":"HTTP_403" if i==1 else "SUCCESS_RAW","final_url":u,"content_type":"text/html",
        "raw_path":f"raw/{i}.zst","raw_sha256":raw_sha,"raw_size_bytes":1000,"elapsed_ms":100.,
        "http_status":403 if i==1 else 200,"crawl_timestamp":"2026-10-05T00:00:00+00:00",
        "snapshot_id":"team-test"} for i,u in enumerate(urls)])
    text = "Kết quả\nNgười bệnh được theo dõi triệu chứng và đáp ứng điều trị e\u0301 中文医学。\n"*10
    write_parquet(source / MEMBERS["extracted"],[{"crawl_url_id":str(i),"raw_sha256":raw_sha,
        "snapshot_id":"team-test","extract_status":"EXTRACT_SUCCESS" if i==0 else "EXTRACT_EMPTY",
        "text":text if i==0 else "","char_count":len(text) if i==0 else 0,"title":"Y sinh",
        "language":"vi","language_method":"hint","language_confidence":1.,"extractor":"test",
        "extractor_version":"v1","source_issue":None,"text_markdown":"# Kết quả\n"+text}
        for i in (0,2)])
    atomic_json(source / MEMBERS["dataset"],{"snapshot_id":"team-test",
        "sources":[{"kind":"links_path","sha256":sha256_file(links)}]})
    return source,links,text


def run_import(tmp_path, external, **kwargs):
    source,links,_ = external
    return import_external_corpus(source,links,tmp_path / "processed",CharacterTokenizer(),SPEC,
        config=CONFIG,work_dir=tmp_path / "work",shard_size=2,**kwargs)


def test_aliases_failures_exact_source_resume_and_freeze(tmp_path,external):
    first = run_import(tmp_path,external,max_shards=1)
    root = tmp_path / "processed/team-100k-data-v1"
    assert first["counts"]["documents"] == 2
    assert not first["selected_range_complete"]
    with pytest.raises(ValueError,match="Complete every"):
        freeze_candidate(root,{},golden_report={})
    first_sha = first["parts"][0]["files"]["children"]["sha256"]
    result = run_import(tmp_path,external)
    assert result["integrity"]["passed"] and result["selected_range_complete"]
    assert result["counts"]["input_records"] == 4
    assert result["counts"]["documents"] == result["counts"]["failures"] == 2
    assert result["parts"][0]["files"]["children"]["sha256"] == first_sha
    assert read_json(root / "import_runtime.json")["reused_shards"] == 1
    docs = pq.read_table(root / result["parts"][0]["files"]["documents"]["path"]).to_pylist()
    assert {d["doc_id"] for d in docs} == {583,921}
    assert all(d["source_text"] == external[2] for d in docs)
    health = health_report(root,official_links=external[1],work_dir=tmp_path / "work",audit_size=2)
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    candidate = freeze_candidate(root,health,golden_report=golden)
    inputs = prepare_index_inputs(root,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",
        CharacterTokenizer(),work_dir=tmp_path / "work",part_size=3)
    assert inputs["input_count"] < result["counts"]["children"]
    again = prepare_index_inputs(root,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",
        CharacterTokenizer(),work_dir=tmp_path / "work",part_size=3)
    assert again == inputs
    publish_data_handoff(tmp_path, "team-100k-data-v1",candidate,inputs)
    build,_,units = load_handoff(tmp_path)
    assert sample_inputs(build,units,8)
    damaged = root / "index_inputs" / inputs["parts"][0]["path"]
    damaged.write_bytes(b"damaged")
    with pytest.raises(ValueError,match="changed artifact"):
        prepare_index_inputs(root,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",
            CharacterTokenizer(),work_dir=tmp_path / "work",part_size=3)


def test_legacy_data_v2_build_can_freeze_after_current_integrity_and_golden(tmp_path,external):
    source,links,_ = external
    import_external_corpus(source,links,tmp_path / "processed",CharacterTokenizer(),SPEC,
        config=CONFIG,work_dir=tmp_path / "work",shard_size=2)
    root = tmp_path / "processed/team-100k-data-v1"
    config = read_json(root / "config.json")
    config["code_sha256"] = "legacy-producer-code-fingerprint"
    config_payload = {key:value for key,value in config.items() if key != "signature"}
    config["signature"] = digest_json(config_payload)
    atomic_json(root / "config.json",config)
    build = read_json(root / "build.json")
    build["signature"] = config["signature"]
    build["snapshot_sha256"] = digest_json({"signature":build["signature"],"parts":build["parts"]})
    atomic_json(root / "build.json",build)

    health = health_report(root,official_links=links,work_dir=tmp_path / "work",audit_size=2)
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    candidate = freeze_candidate(root,health,golden_report=golden)

    assert candidate["state"] == "FROZEN_CANDIDATE"
    assert candidate["code_provenance"]["status"] == "LEGACY_DATA_V2_VALIDATED"
    assert candidate["code_provenance"]["producer_matches_freeze_code"] is False


def test_notebook_freeze_preserves_legacy_candidate_and_existing_index_checkpoint(tmp_path,external):
    from copy import deepcopy
    from scripts.freeze_workers import freeze_build_for_notebook, verify_freeze_record

    build = run_import(tmp_path, external)
    root = tmp_path / "processed/team-100k-data-v1"
    health = health_report(root, official_links=external[1], work_dir=tmp_path / "work", audit_size=2)
    golden = {"passed": True, "code_sha256": code_fingerprint(), "tokenizer": SPEC, "chunking": asdict(CONFIG)}
    current = freeze_candidate(root, health, golden_report=golden)
    legacy = deepcopy(current)
    legacy["golden"]["code_sha256"] = "old-validator-code"
    legacy.pop("candidate_manifest_sha256")
    legacy["candidate_manifest_sha256"] = digest_json(legacy)
    legacy_name = f"candidate-{legacy['candidate_manifest_sha256'][:16]}.json"
    atomic_json(root / legacy_name, legacy)
    original_inputs = prepare_index_inputs(root, legacy_name, CharacterTokenizer(),
                                          work_dir=tmp_path / "work", part_size=4096)
    original_bytes = (root / "index_inputs/units.json").read_bytes()
    (root / f"candidate-{current['candidate_manifest_sha256'][:16]}.json").unlink()
    item = {"build_dir": root, "build": build, "config": read_json(root / "config.json"),
            "batch_item": {"candidate_name": f"candidate-{current['candidate_manifest_sha256'][:16]}.json"}}
    candidate, inputs, record = freeze_build_for_notebook(item, data_root=tmp_path,
        official_links=external[1], tokenizer=CharacterTokenizer(), golden_report=golden,
        work_dir=tmp_path / "work", audit_size=2)
    assert candidate["candidate_manifest_sha256"] == legacy["candidate_manifest_sha256"]
    assert inputs == original_inputs
    assert (root / "index_inputs/units.json").read_bytes() == original_bytes
    assert record["validation_candidate_manifest_sha256"] != legacy["candidate_manifest_sha256"]
    verified = verify_freeze_record(tmp_path, "team-100k-data-v1",
        {"snapshot_sha256": build["snapshot_sha256"]}, record, code_fingerprint())
    assert verified["validation"]["golden"]["code_sha256"] == code_fingerprint()
    _, resumed_inputs, resumed_record = freeze_build_for_notebook(item, data_root=tmp_path,
        official_links=external[1], tokenizer=CharacterTokenizer(), golden_report=golden,
        work_dir=tmp_path / "work", audit_size=2)
    assert resumed_inputs == original_inputs
    assert resumed_record == record


def test_generated_notebook_seven_builds_three_workers_coordinator_and_resume(tmp_path,external):
    import json
    from pathlib import Path
    from scripts.freeze_workers import (
        freeze_batch_signature, freeze_worker_manifest_path, partition_freeze_builds,
        verify_freeze_workers, write_freeze_worker_manifest, freeze_build_for_notebook,
        audit_freeze_batch, collect_completed_build_refs, verify_freeze_record,
    )

    active, refs = [], []
    for index in range(7):
        run = f"batch-{index}"
        build = run_import(tmp_path, external, run_name=run)
        root = tmp_path / "processed" / run
        ref = {"build_run": run, "snapshot_sha256": build["snapshot_sha256"],
               "source_path": f"/incoming/archive-{index}", "state": "COMPLETE"}
        refs.append(ref)
        active.append({"build_run": run, "build_dir": root, "build": build,
                       "config": read_json(root / "config.json"), "batch_item": ref})
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "notebooks/03_colab_validate_and_freeze.ipynb").read_text(encoding="utf-8"))
    source = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"][-1]
    source = "".join(source)
    golden = {"passed": True, "code_sha256": code_fingerprint(), "tokenizer": SPEC, "chunking": asdict(CONFIG)}
    scope = dict(Path=Path, json=json, read_json=read_json, atomic_json=atomic_json,
        digest_json=digest_json, code_fingerprint=code_fingerprint,
        freeze_batch_signature=freeze_batch_signature, freeze_worker_manifest_path=freeze_worker_manifest_path,
        partition_freeze_builds=partition_freeze_builds, verify_freeze_workers=verify_freeze_workers,
        write_freeze_worker_manifest=write_freeze_worker_manifest, freeze_build_for_notebook=freeze_build_for_notebook,
        audit_freeze_batch=audit_freeze_batch, publish_data_handoff=publish_data_handoff,
        verify_freeze_record=verify_freeze_record,
        ACTIVE_BUILDS=active, DATA_ROOT=tmp_path, OFFICIAL_LINKS=external[1], TOKENIZER=CharacterTokenizer(),
        GOLDEN=golden, WORK_DIR=tmp_path / "work", PIPELINE_CONFIG={"audit_size": 2},
        FREEZE_MODE="worker", FREEZE_TEAM_SIZE=3, BATCH={"builds": refs, "state": "COMPLETE"},
        batch_path=tmp_path / "active_data_batch.json")
    for worker_id in range(3):
        scope["FREEZE_WORKER_ID"] = worker_id
        exec(source, scope)
    assert not (tmp_path / "active_data_candidate.json").exists()
    # Resume a completed worker using the same frozen candidates/input parts.
    before = [(item["build_dir"] / "index_inputs/units.json").read_bytes() for item in active]
    scope["FREEZE_WORKER_ID"] = 0
    exec(source, scope)
    assert before == [(item["build_dir"] / "index_inputs/units.json").read_bytes() for item in active]
    scope["FREEZE_MODE"] = "coordinator"
    exec(source, scope)
    report = read_json(tmp_path / "reports/freeze_batch_coverage.json")
    assert report["archives"] == 7
    assert report["input_records"] == 28
    assert report["counts"]["documents"] == report["counts"]["failures"] == 14
    assert report["unique_official_ids"] == 4
    assert report["unaccounted_input_records"] == 0
    assert len(read_json(tmp_path / "candidate_lineage.json")["sources"]) == 7
    assert len(collect_completed_build_refs([r["source_path"] for r in refs],
        read_json(tmp_path / "active_data_batch.json"), [])) == 7
    exec(source, scope)
    assert len(read_json(tmp_path / "candidate_lineage.json")["sources"]) == 7
    scope["FREEZE_MODE"] = "serial"
    exec(source, scope)
    assert read_json(tmp_path / "reports/freeze_batch_coverage.json")["unaccounted_input_records"] == 0
    assert len(read_json(tmp_path / "candidate_lineage.json")["sources"]) == 7


def test_wrong_snapshot_and_corrupted_source_rejected(tmp_path,external):
    source,links,_ = external
    descriptor = source / MEMBERS["dataset"]
    saved = read_json(descriptor)
    saved["sources"][0]["sha256"] = "0"*64
    atomic_json(descriptor,saved)
    with pytest.raises(ValueError,match="snapshots differ"):
        run_import(tmp_path,external)


def test_raw_binding_rejected(tmp_path,external):
    path = external[0] / MEMBERS["extracted"]
    rows = pq.read_table(path).to_pylist()
    rows[0]["raw_sha256"] = "0"*64
    write_parquet(path,rows)
    with pytest.raises(ValueError,match="invalid_extraction_binding.*1"):
        run_import(tmp_path,external)


def test_parts_restore_checksum_and_automatic_discovery(tmp_path,external):
    from scripts.split_external_archive import split_archive

    archive = tmp_path / "vibiomir_shard_00000.tar"
    with tarfile.open(archive,"w") as tar:
        for member in MEMBERS.values():
            tar.add(external[0] / member,arcname=member)
    incoming = tmp_path / "data/incoming/vibiomir_shard_00000.tar.parts"
    manifest = split_archive(archive,incoming,part_size=4096)
    assert find_external_source(tmp_path / "data") == incoming.resolve()
    restored = restore_archive(incoming,tmp_path / "restore")
    assert sha256_file(restored) == sha256_file(archive)
    assert restore_archive(incoming,tmp_path / "restore") == restored
    imported = import_external_corpus(incoming,external[1],tmp_path / "parts_import",
        CharacterTokenizer(),SPEC,config=CONFIG,work_dir=tmp_path / "work",shard_size=2)
    assert imported["integrity"]["passed"] and imported["selected_range_complete"]
    part = incoming / manifest["parts"][0]["name"]
    raw = part.read_bytes()
    part.write_bytes(b"!"+raw[1:])
    with pytest.raises(ValueError,match="part checksum mismatch"):
        restore_archive(incoming,tmp_path / "fresh_restore")
    manifest["parts"][0]["name"] = "../outside"
    atomic_json(incoming / "archive_manifest.json",manifest)
    with pytest.raises(ValueError,match="part name"):
        restore_archive(incoming,tmp_path / "fresh_restore")


def test_archive_accepts_numbered_frontier_shard(tmp_path,external):
    source,links,_ = external
    archive = tmp_path / "vibiomir_shard_00001.tar"
    with tarfile.open(archive,"w") as tar:
        for kind,member in MEMBERS.items():
            arcname = ("data/crawl_shards/shard_00001.parquet" if kind == "frontier" else member)
            tar.add(source / member,arcname=arcname)
    result = import_external_corpus(archive,links,tmp_path / "processed",CharacterTokenizer(),SPEC,
        run_name="team-shard-00001",config=CONFIG,work_dir=tmp_path / "work",shard_size=2)
    assert result["integrity"]["passed"] and result["selected_range_complete"]
    assert result["counts"]["input_records"] == 4


def test_archive_rejects_multiple_numbered_frontier_shards(tmp_path,external):
    source,links,_ = external
    archive = tmp_path / "ambiguous.tar"
    with tarfile.open(archive,"w") as tar:
        for kind,member in MEMBERS.items():
            arcname = ("data/crawl_shards/shard_00001.parquet" if kind == "frontier" else member)
            tar.add(source / member,arcname=arcname)
        tar.add(source / MEMBERS["frontier"],arcname="data/crawl_shards/shard_00002.parquet")
    with pytest.raises(ValueError,match="multiple numbered crawl frontier"):
        import_external_corpus(archive,links,tmp_path / "processed",CharacterTokenizer(),SPEC,
            run_name="ambiguous",config=CONFIG,work_dir=tmp_path / "work",shard_size=2)


def test_extracted_directory_accepts_numbered_frontier_shard(tmp_path,external):
    source,links,_ = external
    (source / MEMBERS["frontier"]).rename(source / "data/crawl_shards/shard_00001.parquet")
    result = import_external_corpus(source,links,tmp_path / "processed",CharacterTokenizer(),SPEC,
        run_name="directory-shard-00001",config=CONFIG,work_dir=tmp_path / "work",shard_size=2)
    assert result["integrity"]["passed"] and result["selected_range_complete"]
    assert result["counts"]["input_records"] == 4


def test_bounded_benchmark_reuses_completed_models_without_loading_corpus(tmp_path,external,monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    from vietmedbridge import scale_benchmark as bench

    result = run_import(tmp_path,external)
    root = tmp_path / "processed/team-100k-data-v1"
    health = health_report(root,official_links=external[1],work_dir=tmp_path / "work",audit_size=2)
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    candidate = freeze_candidate(root,health,golden_report=golden)
    inputs = prepare_index_inputs(root,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",
        CharacterTokenizer(),work_dir=tmp_path / "work",part_size=3)
    publish_data_handoff(tmp_path,"team-100k-data-v1",candidate,inputs)
    atomic_json(tmp_path / "raw/snapshot.json",{"files":{"links_corpus.parquet":{
        "sha256":candidate["origin_corpus_sha256"]}}})
    config = {"dense":SPEC,"second_dense":SPEC,"reranker":SPEC | {"max_length":1024}}
    atomic_json(tmp_path / "configs/retrieval_full.json",config)
    atomic_json(tmp_path / "configs/strong_model_manifest.json",{})
    cuda = SimpleNamespace(is_available=lambda:True,get_device_name=lambda _:"GPU-TEST-DOUBLE",
        get_device_properties=lambda _:SimpleNamespace(total_memory=16*2**30),synchronize=lambda:None,
        reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:1024)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda,__version__="test"))
    monkeypatch.setattr(bench,"review_model_registry",lambda *_:None)
    monkeypatch.setattr(bench,"model_budget_report",lambda *_:{"scope":"test"})
    monkeypatch.setattr(bench,"load_queries",lambda *_,**kw:[{"id":i,"query":"Triệu chứng?"} for i in range(8)])
    monkeypatch.setattr(bench,"parquet_path",lambda *_:"unused-query-test-double")
    monkeypatch.setattr(bench,"pair_windows",lambda _,q,t,l:[{"text":t}])
    loaded,counts = [],[]

    class Model:
        def __init__(self,spec):
            loaded.append(spec)
            self.oom_backoffs = 0
            self.identity = {"test_double":True}
            self.tokenizer = None
        def encode(self,items,**kwargs):
            counts.append(len(items))
            return np.zeros((len(items),2))
        score = encode
        def close(self):
            pass

    for name in ("TorchDenseEncoder","TorchQwenEncoder","TorchQwenReranker"):
        monkeypatch.setattr(bench,name,Model)
    report = bench.run_scale_benchmark(tmp_path,tmp_path,sample_size=8)
    assert report["state"] == "RESOURCE_BENCHMARK_COMPLETE"
    assert report["submission_created"] is False and len(loaded) == 3
    assert max(counts) <= 8 and result["integrity"]["passed"]
    again = bench.run_scale_benchmark(tmp_path,tmp_path,sample_size=8)
    assert len(loaded) == 3 and again["stages"] == report["stages"]
