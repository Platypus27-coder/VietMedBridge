"""Full 1,200-query pilot contract with CPU test doubles, interruption and export."""
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from test_retrieval import Encoder, catalog
from test_retrieval_cascade import CascadeReranker, Tokenizer, Translator
from vietmedbridge.artifacts import atomic_json, read_json
from vietmedbridge.competition_pilot import MASTER_PLAN, bind_pilot_run, finish_pilot
from vietmedbridge.embeddings import embed_units, embedding_matrix
from vietmedbridge.query_translation import translate_queries
from vietmedbridge.retrieval_cascade import CascadeConfig, CascadeIndex, predict_cascade


def test_end_to_end_all_1200_queries_resume_export_and_score_binding(tmp_path, catalog):
    queries = [{"id":100001 + i*7, "query":f"HbA1c xét nghiệm số {i}"} for i in range(1200)]
    config = {"expected_queries":1200, "scope":"CPU_TEST_DOUBLES"}
    plan = tmp_path / MASTER_PLAN
    plan.write_text("Unit-test master plan fixture", encoding="utf-8")
    run = tmp_path / "pilot"
    contract = bind_pilot_run(run, plan_path=plan, catalog=catalog, queries=queries,
                              config=config, code_commit="test-commit")
    assert contract == bind_pilot_run(run, plan_path=plan, catalog=catalog, queries=queries,
                                      config=config, code_commit="new-export-commit")
    encoder = Encoder()
    cm = embed_units(catalog.units, encoder, tmp_path / "corpus")
    qm = embed_units([{"id":q["id"],"text":q["query"]} for q in queries], encoder, tmp_path / "qvectors")
    index = CascadeIndex(catalog, embedding_matrix(tmp_path / "corpus", cm), cm,
                         run / "index", Tokenizer())
    vectors = embedding_matrix(tmp_path / "qvectors", qm)
    translations, tr = translate_queries(queries, Translator(), run / "translations", max_new_queries=17)
    records, report = predict_cascade(index, queries, vectors, CascadeReranker(), run / "queries",
        query_embedding_manifest=qm, translations=translations, translation_signature=tr["signature"])
    assert len(records) == 17 and report["state"] == "IN_PROGRESS"
    evidence = {"inference_scope":"CPU_TEST_DOUBLES_DO_NOT_SUBMIT"}
    with pytest.raises(ValueError, match="Finish every"):
        finish_pilot(records, queries, catalog, report, run, contract=contract,
                     tokenizer=index.tokenizer, evidence=evidence)
    assert not (run / "submission/submission.zip").exists()
    translator = Translator()
    translations, tr = translate_queries(queries, translator, run / "translations")
    assert translator.calls == 1200 - 17
    reranker = CascadeReranker()
    records, report = predict_cascade(index, queries, vectors, reranker, run / "queries",
        query_embedding_manifest=qm, translations=translations, translation_signature=tr["signature"])
    assert report["state"] == "COMPLETE" and len(records) == 1200
    assert reranker.calls == 2 * (1200 - 17)
    manifest, ready = finish_pilot(records, queries, catalog, report, run, contract=contract,
        tokenizer=index.tokenizer, evidence=evidence, reference_labels_path=tmp_path / "missing-labels.json")
    assert ready["state"] == "READY_FOR_MANUAL_UPLOAD" and ready["query_count"] == 1200
    assert ready["official_score"] is None and ready["evaluation"] == "NOT_EVALUATED_NO_REFERENCE_LABELS"
    with ZipFile(ready["zip_path"]) as archive:
        assert archive.namelist() == ["results.json"] and archive.testzip() is None
        assert archive.getinfo("results.json").date_time == (1980,1,1,0,0,0)
        predictions = json.loads(archive.read("results.json"))
    assert [p["id"] for p in predictions] == [q["id"] for q in queries]
    assert all(p["relevant_docs"] and p["relevant_chunks"] for p in predictions)
    feedback_path = run / "submission/score_feedback.json"
    feedback = read_json(feedback_path)
    assert feedback["status"] == "AWAITING_BTC_SCORE" and feedback["overall_score"] is None
    feedback.update(status="UNIT_TEST_SCORE", overall_score=.123, notes="Synthetic test value, not a BTC score")
    atomic_json(feedback_path, feedback)
    bad_labels = tmp_path / "bad-labels.json"
    bad_labels.write_text("not valid JSON", encoding="utf-8")
    repeat, ready = finish_pilot(records, queries, catalog, report, run, contract=contract,
        tokenizer=index.tokenizer, evidence=evidence, reference_labels_path=bad_labels)
    assert repeat["zip_sha256"] == manifest["zip_sha256"]
    assert ready["evaluation"] == "OPTIONAL_EVALUATION_FAILED_SUBMISSION_STILL_VALID"
    assert read_json(feedback_path) == feedback
    feedback["zip_sha256"] = "another submission"
    atomic_json(feedback_path, feedback)
    with pytest.raises(ValueError, match="another ZIP"):
        finish_pilot(records, queries, catalog, report, run, contract=contract,
                     tokenizer=index.tokenizer, evidence=evidence)


def test_master_plan_and_full_query_contract_required(tmp_path, catalog):
    queries = [{"id":i,"query":"HbA1c"} for i in range(1200)]
    config = {"expected_queries":1200}
    alternative = tmp_path / "PLAN_ViBioMIR_Data_Pipeline_v2.md"
    alternative.write_text("not the authoritative plan", encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative"):
        bind_pilot_run(tmp_path / "run", plan_path=alternative, catalog=catalog,
                       queries=queries, config=config, code_commit="test")
    plan = tmp_path / MASTER_PLAN
    plan.write_text("fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="1,200"):
        bind_pilot_run(tmp_path / "run", plan_path=plan, catalog=catalog,
                       queries=queries[:25], config=config, code_commit="test")
    bind_pilot_run(tmp_path / "run", plan_path=plan, catalog=catalog,
                   queries=queries, config=config, code_commit="test")
    plan.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="plan/data/config changed"):
        bind_pilot_run(tmp_path / "run", plan_path=plan, catalog=catalog,
                       queries=queries, config=config, code_commit="test")


def test_notebook_defaults_run_all_and_download_only_verified_zip():
    path = Path(__file__).resolve().parents[1] / "notebooks/04_colab_retrieval_baseline.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    source = "\n".join(cells)
    for variable in ("MAX_NEW_EMBEDDING_PARTS", "MAX_NEW_TRANSLATIONS", "MAX_NEW_QUERIES"):
        assert f"{variable} = None" in source
    assert "bind_pilot_run(RUN_DIR, plan_path=CHECKOUT / MASTER_PLAN" in source
    assert "EXPORT, READY = finish_pilot" in source
    assert 'verify_file(READY["zip_path"], READY["zip_sha256"])' in cells[-1]
    assert "QUERIES[:25]" not in source
    assert "max_new_queries=25" not in source
