"""Immutable small release bundles, score journal and verified rollback pointers.

These functions never upload to the competition or label a run as BTC-approved.
Large corpus/index/weight artifacts remain bound by the existing run contract.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from zipfile import ZipFile

from .artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, verify_file


def _checked(value):
    if value.get("manifest_sha256") != digest_json({k: v for k, v in value.items() if k != "manifest_sha256"}):
        raise ValueError("Release manifest integrity mismatch.")
    return value


def _safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release artifact path escapes its directory.")
    return path


def freeze_release(run_dir, *, registry_dir=None):
    run = Path(run_dir)
    ready = read_json(run / "submission/ready_to_submit.json")
    manifest = _checked(read_json(run / "submission/submission_manifest.json"))
    contract = _checked(read_json(run / "run_contract.json"))
    if (ready.get("state") != "READY_FOR_MANUAL_UPLOAD" or ready.get("query_count") != 1200
        or manifest.get("query_count") != 1200 or not ready.get("source_and_schema_validated")
        or ready.get("pilot_contract_signature") != contract.get("signature")
        or manifest.get("evidence", {}).get("pilot_contract") != contract
        or ready.get("prediction_signature") != manifest.get("prediction_signature")):
        raise ValueError("Only a source/schema-validated complete competition release can be archived.")
    for key, name in (("zip_sha256", "submission.zip"), ("json_sha256", "results.json")):
        if ready[key] != manifest[key]:
            raise ValueError("Readiness and submission hashes differ.")
        verify_file(run / "submission" / name, manifest[key])
    with ZipFile(run / "submission/submission.zip") as archive:
        if archive.namelist() != ["results.json"] or archive.testzip() is not None:
            raise ValueError("Invalid submission ZIP layout.")
        predictions = json.loads(archive.read("results.json"))
    if len(predictions) != 1200 or len({p["id"] for p in predictions}) != 1200:
        raise ValueError("Release ZIP must cover 1,200 unique queries.")
    if predictions != read_json(run / "submission/results.json"):
        raise ValueError("ZIP payload and standalone JSON differ.")
    registry = Path(registry_dir) if registry_dir is not None else run / "releases"
    # Same ZIP with different model/config evidence is a different release.
    release_id = digest_json([manifest["manifest_sha256"], contract["manifest_sha256"]])
    target = registry / release_id
    path = target / "release.json"
    if path.exists():
        return verify_release(path)
    files = {}
    for name, source in (("submission.zip", run / "submission/submission.zip"),
                         ("results.json", run / "submission/results.json"),
                         ("submission_manifest.json", run / "submission/submission_manifest.json"),
                         ("run_contract.json", run / "run_contract.json")):
        files[name] = {"sha256": publish_file(source, target / name), "bytes": source.stat().st_size}
    result = {"schema_version": 1, "release_id": release_id, "state": "SOURCE_SCHEMA_ARCHIVED_NOT_QUALITY_PROMOTED",
        "files": files, "source_run": str(run), "zip_sha256": manifest["zip_sha256"],
        "catalog": contract["catalog"], "queries_sha256": contract["queries_sha256"],
        "config_sha256": contract["config_sha256"], "plan": contract["authoritative_plan"],
        "prediction_signature": manifest["prediction_signature"], "official_score": None,
        "scope": "SMALL_RELEASE_BUNDLE_CORPUS_AND_MODEL_ARTIFACTS_REFERENCED_NOT_COPIED"}
    result["manifest_sha256"] = digest_json(result)
    atomic_json(path, result)
    return result


def verify_release(path):
    path = Path(path)
    result = _checked(read_json(path))
    for relative, item in result["files"].items():
        file = _safe(path.parent, relative)
        verify_file(file, item["sha256"])
        if file.stat().st_size != item["bytes"]:
            raise ValueError("Release size mismatch.")
    return result


def record_dashboard_result(release_path, journal_dir, *, submission_id, phase, submitted_at,
                            metrics, hypothesis, account):
    """Record externally observed results; count uploads separately from experiments."""
    release = verify_release(release_path)
    if phase not in ("public", "private") or not all(isinstance(v, str) and v.strip() for v in (submission_id, hypothesis, account)):
        raise ValueError("Provide phase, submission ID, account and experiment hypothesis.")
    stamp = datetime.fromisoformat(submitted_at)
    if stamp.tzinfo is None:
        raise ValueError("Submission timestamp must include a timezone.")
    names = {"final_score", "document_f2", "chunk_f2", "document_precision", "document_recall", "chunk_precision", "chunk_recall"}
    if set(metrics) != names or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in metrics.values()):
        raise ValueError("Supply all seven finite dashboard metrics in [0,1].")
    # Allow leaderboard rounding; macro-F2 is not F2 of macro P/R.
    if abs(metrics["final_score"] - (metrics["document_f2"] + metrics["chunk_f2"]) / 2) > .00011:
        raise ValueError("Final score is inconsistent with document/chunk F2.")
    result = {"release_id": release["release_id"], "zip_sha256": release["zip_sha256"],
        "submission_id": submission_id, "phase": phase, "account": account,
        "submitted_at": stamp.isoformat(), "vietnam_date": stamp.astimezone(timezone(timedelta(hours=7))).date().isoformat(),
        "metrics": metrics, "hypothesis": hypothesis, "source": "USER_REPORTED_BTC_DASHBOARD"}
    result["manifest_sha256"] = digest_json(result)
    path = Path(journal_dir) / (digest_json([account, phase, submission_id]) + ".json")
    if path.exists() and read_json(path) != result:
        raise ValueError("Submission already recorded with different evidence; retain the original event.")
    # Validate the optional per-run update before appending a team journal event.
    run = Path(release_path).resolve().parent.parent.parent
    feedback_path = run / "submission" / "score_feedback.json"
    feedback = None
    if feedback_path.is_file():
        feedback = read_json(feedback_path)
        if feedback.get("zip_sha256") != release["zip_sha256"]:
            raise ValueError("BTC score release differs from this run's score-feedback record.")
    atomic_json(path, result)
    # Keep the short per-run view in sync with the append-only team journal.
    if feedback is not None:
        feedback.update(status="BTC_SCORE_RECORDED", release_id=release["release_id"],
            submission_id=submission_id, phase=phase, account=account, submitted_at=stamp.isoformat(),
            document_f2=metrics["document_f2"], chunk_f2=metrics["chunk_f2"],
            overall_score=metrics["final_score"], metrics=metrics, hypothesis=hypothesis,
            journal_record_sha256=result["manifest_sha256"])
        atomic_json(feedback_path, feedback)
    return result


def submission_budget(journal_dir, *, vietnam_date, account):
    events = [_checked(read_json(p)) for p in sorted(Path(journal_dir).glob("*.json"))]
    public = sum(e["phase"] == "public" and e["vietnam_date"] == vietnam_date for e in events)
    private = sum(e["phase"] == "private" and e["account"] == account for e in events)
    return {"public_recorded_today_team": public, "private_recorded_for_account": private,
        "public_remaining_from_recorded_events": max(0, 10 - public),
        "private_remaining_from_recorded_events": max(0, 5 - private),
        "scope": "LOCAL_JOURNAL_ONLY_NOT_LIVE_BTC_QUOTA"}


def select_release(release_path, pointer_path, *, reason):
    """Explicit local selection/rollback; validate bytes before changing a pointer."""
    release = verify_release(release_path)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Record the reason for selection/rollback.")
    pointer = Path(pointer_path)
    previous = _checked(read_json(pointer)) if pointer.exists() else None
    result = {"release_id": release["release_id"], "release_manifest": str(Path(release_path).resolve()),
        "zip_sha256": release["zip_sha256"], "reason": reason, "previous_release_id": previous["release_id"] if previous else None,
        "state": "SELECTED_FOR_MANUAL_UPLOAD_NOT_UPLOADED"}
    if previous and previous["release_id"] == result["release_id"] and previous["reason"] == reason:
        return previous
    result["manifest_sha256"] = digest_json(result)
    history = pointer.parent / (pointer.stem + "_history") / (result["manifest_sha256"] + ".json")
    atomic_json(history, result)
    atomic_json(pointer, result)
    return result
