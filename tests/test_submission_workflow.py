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


def setup_workflow(p, monkeypatch):
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
    report = workflow.prepare_current_submission(p.root,p.checkout,p.work)
    assert report["legacy_urls_retained"] and not report["current_btc_membership_validated"]
    assert report["documents"] == 2 and report["cache"]["bge"]["missing_inputs"] == 0
    # Completed CPU preparation must preserve the corpus/input identity.
    assert workflow.prepare_current_submission(p.root,p.checkout,p.work) == report
    return report


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


def test_submission_notebook_has_two_phases_and_persistent_results():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/04_colab_submit_available_data.ipynb",as_version=4)
    nbformat.validate(notebook)
    codes = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in codes:
        ast.parse(source)
    joined = "\n".join(codes)
    assert joined.index("runtime_lock =") < joined.index("import torch")
    assert 'read_json(REPORT_PATH)' in codes[1] and 'if GPU_SESSION:' in codes[1]
    assert "run_current_baseline(" in codes[2] and "run_current_submission(" in codes[3]
    assert "latest_submission.json" in codes[4] and "verify_file(" in codes[4]
    assert "KAGGLE_JSON" not in joined and "BTC_REVISION" not in joined
