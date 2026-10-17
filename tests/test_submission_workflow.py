"""Real corpus/catalog/ZIP flow with CPU neural doubles; no score/GPU claims."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import sys
from zipfile import ZipFile

import nbformat
import pyarrow.parquet as pq
import pytest

from test_cached_strong import prepared, frozen, external
from test_external_import import SPEC, CONFIG
from test_full_scale import FullTokenizer
from vietmedbridge import submission_workflow as workflow
from vietmedbridge.artifacts import atomic_json, code_fingerprint, digest_json, read_json, sha256_file


def configure_workflow(p, monkeypatch):
    from vietmedbridge import chunks, golden, full_scale_runtime, qwen_models
    snapshot = read_json(p.root / "raw/snapshot.json")
    snapshot["revision"] = "original-old-snapshot"
    snapshot["files"]["links_corpus.parquet"]["path"] = "links.parquet"
    atomic_json(p.root / "raw/snapshot.json",snapshot)
    monkeypatch.setattr(chunks,"load_bge_tokenizer",lambda *_:(FullTokenizer(),SPEC))
    monkeypatch.setattr(golden,"run_golden_suite",lambda *a,**k:{"passed":True,"code_sha256":code_fingerprint(),
        "tokenizer":SPEC,"chunking":asdict(CONFIG)})
    # Test double encoder identity deliberately has no inference source hash.
    # No production model source or identity is altered by this integration test.
    real_sha = workflow.sha256_file
    monkeypatch.setattr(workflow,"sha256_file",lambda p:None if Path(p).name in {
        "retrieval_models.py","qwen_models.py"} else real_sha(p))
    monkeypatch.setattr(qwen_models,"review_model_registry",lambda *a:None)


def setup_workflow(p, monkeypatch):
    configure_workflow(p,monkeypatch)
    report = workflow.prepare_current_submission(p.root,p.checkout,p.work)
    assert report["legacy_urls_retained"] and not report["current_btc_membership_validated"]
    assert report["documents"] == 2 and report["cache"]["bge"]["missing_inputs"] == 0
    # Completed CPU preparation must preserve the corpus/input identity.
    assert workflow.prepare_current_submission(p.root,p.checkout,p.work) == report
    return report


def test_gpu_resumes_preparation_without_report(prepared,monkeypatch):
    p = prepared
    configure_workflow(p,monkeypatch)
    assert not (p.root / "retrieval/current_submission.json").exists()
    result = workflow.ensure_submission_preparation(p.root,p.checkout,p.work,gpu_session=True)
    assert result["state"] == "CPU_PREPARATION_COMPLETE" and result["documents"] == 2
    assert all(count == 0 for count in p.counters.values())  # No neural weights/encoding in preparation.
    monkeypatch.setattr(workflow,"prepare_current_submission",lambda *a,**k:pytest.fail("Completed preparation rebuilt"))
    fresh_work = p.work / "fresh-runtime"
    assert not fresh_work.exists()
    assert workflow.ensure_submission_preparation(p.root,p.checkout,fresh_work,gpu_session=True) == result


@pytest.mark.parametrize("delete_build_report",[False,True])
def test_gpu_recovers_missing_pointer_from_drive_reports(prepared,monkeypatch,delete_build_report):
    p = prepared
    result = setup_workflow(p,monkeypatch)
    (p.root / "retrieval/current_submission.json").unlink()
    if delete_build_report:
        (p.root / "processed" / result["build_run"] / "submission_preparation.json").unlink()
    monkeypatch.setattr(workflow,"prepare_current_submission",lambda *a,**k:pytest.fail("Durable report ignored"))
    restored = workflow.ensure_submission_preparation(p.root,p.checkout,p.work / "fresh",gpu_session=True)
    assert restored == result and read_json(p.root / "retrieval/current_submission.json") == result


def test_notebook_cell_two_survives_fresh_kernel(prepared,monkeypatch):
    p = prepared
    configure_workflow(p,monkeypatch)
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/04_colab_submit_available_data.ipynb",as_version=4)
    source = [c.source for c in notebook.cells if c.cell_type == "code"][1]
    cpu = {"DATA_ROOT":p.root,"CHECKOUT":p.checkout,"WORK_DIR":p.work,"GPU_SESSION":False,"json":json}
    exec(compile(source,"notebook-cell-2","exec"),cpu)
    # A new kernel starts with NO PREPARATION variable and an empty work directory.
    gpu = {"DATA_ROOT":p.root,"CHECKOUT":p.checkout,"WORK_DIR":p.work / "empty-kernel",
           "GPU_SESSION":True,"json":json}
    assert "PREPARATION" not in gpu
    monkeypatch.setattr(workflow,"prepare_current_submission",lambda *a,**k:pytest.fail("CPU work repeated after kernel change"))
    exec(compile(source,"notebook-cell-2","exec"),gpu)
    assert gpu["PREPARATION"] == cpu["PREPARATION"] and all(count == 0 for count in p.counters.values())


def test_legacy_strong_zip_and_resume(prepared,monkeypatch):
    p = prepared
    preparation = setup_workflow(p,monkeypatch)
    partial = workflow.run_current_submission(p.root,p.checkout,p.work,code_commit="CPU-test",max_new_queries=3)
    assert partial["ready"] == {"state":"QUERIES_IN_PROGRESS","query_count":3}
    result = workflow.run_current_submission(p.root,p.checkout,p.work,code_commit="CPU-test")
    assert result["ready"]["state"] == "READY_FOR_MANUAL_UPLOAD"
    assert result["status"]["strong_inference"] and p.counters["qwen"] == 0
    assert result["status"]["branches"]["qwen_child_dense"] != "COMPLETE"
    saved = read_json(p.root / "retrieval/latest_submission.json")
    assert saved["variant"] == "strong" and saved["legacy_urls_retained"]
    assert saved["candidate_manifest_sha256"] == preparation["candidate_manifest_sha256"]
    assert saved["current_btc_membership_validated"] is False
    assert Path(saved["zip_path"]).is_relative_to(p.root.parent / "submission_runs")
    with ZipFile(saved["zip_path"]) as archive:
        records = json.loads(archive.read("results.json"))
    assert [r["id"] for r in records] == [q["id"] for q in p.queries]
    assert all(set(r["relevant_docs"]) == {583,921} for r in records)
    before = dict(p.counters)
    repeated = workflow.run_current_submission(p.root,p.checkout,p.work,code_commit="CPU-test")
    assert repeated["ready"]["zip_sha256"] == saved["zip_sha256"] and p.counters == before


def test_legacy_baseline_zip_and_resume(prepared,monkeypatch):
    from vietmedbridge import scale_baseline, full_scale_runtime
    p = prepared
    preparation = setup_workflow(p,monkeypatch)
    # Keep the nested 64-character release path below Windows MAX_PATH in tests.
    # Colab uses POSIX paths; production retains the Drive sibling default.
    preparation["results_root"] = str(p.root.parent / "b")
    atomic_json(p.root / "retrieval/current_submission.json",preparation)
    repo = Path(__file__).resolve().parents[1]
    config = read_json(repo / "configs/retrieval_large_baseline.json")
    config.update(dense=p.config["dense"],lexical_segmentation=False)
    atomic_json(p.checkout / "configs/retrieval_large_baseline.json",config)
    atomic_json(p.checkout / "configs/strong_model_manifest.json",{"models":[
        p.config["dense"] | {"parameters":4,"test_double":True}]})
    monkeypatch.setattr(scale_baseline,"TorchDenseEncoder",full_scale_runtime.TorchDenseEncoder)
    monkeypatch.setattr(scale_baseline,"review_model_registry",lambda *a:None)
    real_search = scale_baseline.search_parts
    monkeypatch.setattr(scale_baseline,"search_parts",lambda *a,**k:real_search(*a,**(k | {"device":"cpu"})))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
    result = workflow.run_current_baseline(p.root,p.checkout,p.work,code_commit="CPU-test")
    assert result["ready"]["state"] == "READY_FOR_MANUAL_UPLOAD"
    saved = read_json(p.root / "retrieval/baseline_submission.json")
    assert saved["variant"] == "baseline" and saved["query_count"] == 1200
    before = dict(p.counters)
    repeated = workflow.run_current_baseline(p.root,p.checkout,p.work,code_commit="CPU-test")
    assert repeated["ready"]["zip_sha256"] == saved["zip_sha256"] and p.counters == before


def test_complete_primary_cache_skips_model_loading(prepared,monkeypatch):
    from vietmedbridge import retrieval_models
    p = prepared
    setup_workflow(p,monkeypatch)
    monkeypatch.setattr(retrieval_models,"TorchDenseEncoder",lambda *a,**k:pytest.fail("Complete BGE cache must skip model loading"))
    before = dict(p.counters)
    workflow._ensure_primary(workflow._submission_context(p.root,p.checkout),p.work)
    assert p.counters == before


def test_primary_honors_bge_batch_and_reuses_existing_vectors_when_remapping(prepared,monkeypatch):
    from vietmedbridge import retrieval_models, runtime_profile, full_scale_runtime, portable_embeddings
    p = prepared
    setup_workflow(p,monkeypatch)
    context = workflow._submission_context(p.root,p.checkout)
    build,candidate,inputs = context[3:6]
    view = p.root / portable_embeddings.resources_relative(candidate,p.config) / "bge"
    for path in view.glob("map-*.done.json"):
        path.unlink()  # Simulate missing maps while paid-for vectors remain.
    fake = full_scale_runtime.TorchDenseEncoder
    batches = []
    class NoReencode(fake):
        def encode(self,*a,**kw):
            pytest.fail("Remapping existing content vectors must not encode again")
    monkeypatch.setattr(retrieval_models,"TorchDenseEncoder",NoReencode)
    monkeypatch.setattr(runtime_profile,"inference_batches",lambda:{"embedding":16})
    from vietmedbridge import content_embeddings as content
    original = content.content_embeddings
    def record(*a,**kw):
        batches.append(kw["batch_size"])
        assert Path(a[1]) == p.work / "official_scope" / build.name / "index_inputs"
        return original(*a,**kw)
    monkeypatch.setattr(content,"content_embeddings",record)
    workflow._ensure_primary(context,p.work)
    assert batches == [32]
    manifest = read_json(view / "embeddings.json")
    assert manifest["state"] == "COMPLETE" and len(manifest["parts"]) == len(inputs["parts"])


def test_completed_catalog_releases_only_local_source_copies(prepared,monkeypatch):
    from vietmedbridge import official_scope
    p = prepared
    setup_workflow(p,monkeypatch)
    context = workflow._submission_context(p.root,p.checkout)
    build,candidate,inputs = context[3:6]
    stage = official_scope.stage_scope_catalog(build,candidate,inputs,work_dir=p.work)
    source_hashes = {f:sha256_file(f) for f in p.root.rglob("*") if f.is_file()}
    local_catalogs = {f:sha256_file(f) for f in p.work.glob("catalog-*/catalog.sqlite")}
    protected = stage / "parts/user-note.txt"
    protected.write_text("Keep this unrelated file")
    copies = [stage / part["files"][kind]["path"] for part in candidate["parts"]
              for kind in ("documents","children","parents")]
    expected = sum(f.stat().st_size for f in copies)
    assert workflow._release_catalog_staging(context,p.work) == expected and expected > 0
    assert not any(f.exists() for f in copies)
    assert protected.read_text() == "Keep this unrelated file"
    assert (stage / "config.json").exists()
    assert all((stage / "index_inputs" / part["path"]).exists() for part in inputs["parts"])
    assert source_hashes == {f:sha256_file(f) for f in p.root.rglob("*") if f.is_file()}
    assert local_catalogs and local_catalogs == {f:sha256_file(f) for f in p.work.glob("catalog-*/catalog.sqlite")}
    assert workflow._release_catalog_staging(context,p.work) == 0


@pytest.mark.parametrize("fault",["unpublished","wrong-stage","work-on-drive"])
def test_catalog_staging_cleanup_rejects_unfinished_or_unrelated_state(prepared,monkeypatch,fault):
    from vietmedbridge import official_scope
    p = prepared
    setup_workflow(p,monkeypatch)
    context = workflow._submission_context(p.root,p.checkout)
    build,candidate,inputs = context[3:6]
    work = p.root / "unsafe-work" if fault == "work-on-drive" else p.work
    stage = official_scope.stage_scope_catalog(build,candidate,inputs,work_dir=work)
    files = {f:sha256_file(f) for f in stage.rglob("*") if f.is_file()}
    if fault == "unpublished":
        for f in (p.root / "retrieval/catalogs").glob("catalog-*/catalog.json"):
            if read_json(f)["manifest_sha256"] == context[2]["catalog"]["disk_catalog_manifest_sha256"]:
                f.unlink()
    if fault == "wrong-stage":
        atomic_json(stage / "config.json",{"signature":"other-corpus"})
        files[stage / "config.json"] = sha256_file(stage / "config.json")
    with pytest.raises(ValueError,match="Publish|different candidate|work disk"):
        workflow._release_catalog_staging(context,work)
    assert files == {f:sha256_file(f) for f in stage.rglob("*") if f.is_file()}


def test_submission_notebook_has_two_phases_and_persistent_results():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/04_colab_submit_available_data.ipynb",as_version=4)
    nbformat.validate(notebook)
    codes = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in codes:
        ast.parse(source)
    joined = "\n".join(codes)
    assert joined.index("runtime_lock =") < joined.index("import torch")
    assert 'ensure_submission_preparation(DATA_ROOT,CHECKOUT,WORK_DIR,gpu_session=GPU_SESSION)' in codes[1]
    assert 'raise RuntimeError' not in codes[1] and 'all-available-legacy-20261016-v8' in codes[0]
    assert "run_current_baseline(" in codes[2] and "run_current_submission(" in codes[3]
    assert "latest_submission.json" in codes[4] and "verify_file(" in codes[4]
    assert "KAGGLE_JSON" not in joined and "BTC_REVISION" not in joined
