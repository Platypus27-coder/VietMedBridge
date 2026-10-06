"""Bind the authoritative plan and finish a partial-corpus competition submission."""
from __future__ import annotations

from pathlib import Path

from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .retrieval_eval import evaluate_candidate_recall, evaluate_predictions
from .submission import export_submission


MASTER_PLAN = "R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md"


def bind_pilot_run(run_dir, *, plan_path, catalog, queries, config, code_commit):
    plan_path = Path(plan_path)
    if plan_path.name != MASTER_PLAN or not plan_path.is_file():
        raise ValueError("The competition pilot must bind the authoritative Stage 3 master plan.")
    if config.get("expected_queries") != 1200 or len(queries) != 1200:
        raise ValueError("A competition pilot must target all 1,200 official queries.")
    if len({q["id"] for q in queries}) != 1200 or any(type(q["id"]) is not int or not isinstance(q["query"], str) or not q["query"].strip() for q in queries):
        raise ValueError("Official queries need unique integer IDs and nonempty text.")
    identity = {"authoritative_plan": {"name": MASTER_PLAN, "sha256": sha256_file(plan_path)},
        "catalog": catalog.identity, "queries_sha256": digest_json(queries),
        "config_sha256": digest_json(config), "expected_queries": 1200,
        "scope": "PARTIAL_CORPUS_END_TO_END_COMPETITION_PILOT",
        "pipeline": "bge-dense-multilingual-sparse-translation-document-child-parent-lcs-submission",
        "requires_full_corpus": False, "requires_reference_labels_to_export": False,
        "fine_tuned": False, "corpus_quality_promoted": False}
    signature = digest_json(identity)
    path = Path(run_dir) / "run_contract.json"
    if path.exists():
        saved = read_json(path)
        if saved.get("signature") != signature or any(saved.get(k) != v for k,v in identity.items()) or digest_json({k:v for k,v in saved.items() if k != "manifest_sha256"}) != saved["manifest_sha256"]:
            raise ValueError("Pilot plan/data/config changed; choose a new RUN_NAME.")
    else:
        saved = {**identity, "signature": signature, "initialized_with_commit": code_commit}
        saved["manifest_sha256"] = digest_json(saved)
        atomic_json(path, saved)
    return saved


def finish_pilot(records, queries, catalog, report, run_dir, *, contract, tokenizer,
                 evidence, reference_labels_path=None, work_dir=None):
    root = Path(run_dir)
    expected = contract.get("expected_queries")
    if expected != 1200 or contract.get("catalog") != catalog.identity or contract.get("queries_sha256") != digest_json(queries) or digest_json({k:v for k,v in contract.items() if k != "manifest_sha256"}) != contract["manifest_sha256"]:
        raise ValueError("Pilot contract/input integrity mismatch.")
    # Export is independent of optional offline evaluation: missing labels do not block a pilot.
    manifest = export_submission(records, queries, catalog, report, root / "submission",
        evidence={**evidence, "pilot_contract": contract}, expected_count=1200, work_dir=work_dir)
    readiness = {"state": "READY_FOR_MANUAL_UPLOAD", "query_count": 1200,
        "zip_path": str(root / "submission/submission.zip"),
        "zip_sha256": manifest["zip_sha256"], "json_sha256": manifest["json_sha256"],
        "prediction_signature": report["signature"], "pilot_contract_signature": contract["signature"],
        "evaluation": "NOT_EVALUATED_NO_REFERENCE_LABELS", "official_score": None,
        "inference_scope": evidence.get("inference_scope", "UNSPECIFIED"),
        "source_and_schema_validated": True, "full_corpus": False, "fine_tuned": False}
    if reference_labels_path is not None and Path(reference_labels_path).is_file():
        try:
            labels = read_json(reference_labels_path)
            if labels.get("reviewed") is not True:
                raise ValueError("Reference labels are not marked reviewed.")
            evaluation = evaluate_predictions([r["prediction"] for r in records], labels["queries"], tokenizer)
            evaluation["candidate_recall"] = evaluate_candidate_recall(records, labels["queries"], catalog, tokenizer)
            atomic_json(root / "evaluation/reference_report.json", evaluation)
            readiness["evaluation"] = "PLAN_DERIVED_LOCAL_PROXY_NOT_OFFICIAL_BTC"
        except (ValueError, KeyError, TypeError, OSError, AttributeError) as error:
            readiness["evaluation"] = "OPTIONAL_EVALUATION_FAILED_SUBMISSION_STILL_VALID"
            readiness["evaluation_error"] = f"{type(error).__name__}: {error}"
    atomic_json(root / "submission/ready_to_submit.json", readiness)
    feedback_path = root / "submission/score_feedback.json"
    if feedback_path.exists():
        if read_json(feedback_path).get("zip_sha256") != manifest["zip_sha256"]:
            raise ValueError("Existing score feedback belongs to another ZIP; preserve it in a separate run.")
    else:
        atomic_json(feedback_path, {"status": "AWAITING_BTC_SCORE", "zip_sha256": manifest["zip_sha256"],
            "prediction_signature": report["signature"], "submission_id": None,
            "submitted_at": None, "document_f2": None, "chunk_f2": None,
            "overall_score": None, "notes": None})
    return manifest, readiness
