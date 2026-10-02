"""Scale approvals are evidence records, never implied by a completed job."""

from __future__ import annotations

from pathlib import Path

from .artifacts import atomic_json, code_fingerprint, digest_json, read_json, utc_now


def stage_for_rows(rows: int) -> str:
    if rows < 1:
        raise ValueError("A crawl needs a positive requested row count.")
    return next(stage for limit, stage in ((1000, "stage_a"), (10000, "stage_b1"),
                                          (100000, "stage_b2"), (1000000, "stage_c"),
                                          (float("inf"), "full")) if rows <= limit)


def authorize_scale(data_root: str | Path, *, rows: int, corpus_sha256: str,
                    golden_report: dict, chunking: dict | None = None) -> str:
    if not golden_report.get("passed") or golden_report.get("code_sha256") != code_fingerprint():
        raise ValueError("Run golden regression with the current code before crawling.")
    if chunking is not None and golden_report.get("chunking") != chunking:
        raise ValueError("Chunking policy changed. Rerun golden with the requested data configuration.")
    stage = stage_for_rows(rows)
    previous = {"stage_b1": "stage_a", "stage_b2": "stage_b1", "stage_c": "stage_b2", "full": "stage_c"}.get(stage)
    if previous is None:
        return stage
    path = Path(data_root) / "gates" / f"{previous}.json"
    if not path.exists():
        raise ValueError(f"Missing {previous} milestone review. Complete notebook 03 and review the source audit first.")
    record = read_json(path)
    if not record.get("approved") or record.get("corpus_sha256") != corpus_sha256:
        raise ValueError("Previous milestone is unapproved or belongs to another dataset snapshot.")
    if record.get("code_sha256") != code_fingerprint():
        raise ValueError("Code changed after milestone review. Rebuild, rerun golden and review the new candidate.")
    if record.get("chunking") != golden_report.get("chunking"):
        raise ValueError("Data policy changed after milestone review. Review the new candidate before scaling.")
    if not record.get("reviewed_golden_passed"):
        raise ValueError("Review and replay 100–500 real-source golden annotations before scaling beyond the pilot.")
    if stage in ("stage_b2", "stage_c", "full") and not record.get("retrieval_passed"):
        raise ValueError("A real retrieval regression report is required for this scale-up; self-retrieval is insufficient.")
    if stage == "full" and not record.get("full_budget_passed"):
        raise ValueError("Full run requires measured storage/GPU/deadline estimates within explicit budgets.")
    return stage


def record_milestone(data_root: str | Path, candidate: dict, *, stage: str,
                     corpus_sha256: str, reviewer: str, approved: bool,
                     retrieval_report: dict | None = None, resource_report: dict | None = None,
                     reviewed_golden_report: dict | None = None) -> dict:
    if stage not in ("stage_a", "stage_b1", "stage_b2", "stage_c"):
        raise ValueError("Unknown milestone stage.")
    if not reviewer.strip() or candidate["state"] != "FROZEN_CANDIDATE":
        raise ValueError("A named human review and a frozen candidate are required.")
    identity = {key: value for key, value in candidate.items() if key != "candidate_manifest_sha256"}
    if digest_json(identity) != candidate["candidate_manifest_sha256"]:
        raise ValueError("Candidate manifest changed after freeze.")
    if candidate.get("origin_corpus_sha256") != corpus_sha256:
        raise ValueError("Candidate and milestone belong to different official corpus snapshots.")
    if candidate["golden"].get("code_sha256") != code_fingerprint():
        raise ValueError("Golden must pass with the current code.")
    if not candidate["health"]["integrity"]["passed"] or not candidate["golden"]["passed"]:
        raise ValueError("Hard/golden gates failed.")
    expected_stage = stage_for_rows(candidate["counts"]["input_records"])
    if expected_stage != stage:
        raise ValueError("Stage label does not match this candidate's processed row count.")
    if candidate["selected_shards"] != candidate["available_crawl_shards"]:
        raise ValueError("Process all available crawl shards before approving the milestone.")
    if not candidate.get("selected_range_complete") or candidate["counts"]["input_records"] != candidate["requested_input_records"]:
        raise ValueError("The requested crawl range is incomplete. Resume missing shards before milestone review.")
    reviewed_golden_passed = bool(
        reviewed_golden_report and reviewed_golden_report.get("passed")
        and 100 <= reviewed_golden_report.get("reviewed_cases", 0) <= 500
        and reviewed_golden_report.get("annotation_sha256")
        and reviewed_golden_report.get("scope") == "reviewed_real_source_regression_not_relevance_labels"
        and reviewed_golden_report.get("code_sha256") == code_fingerprint()
        and reviewed_golden_report.get("tokenizer") == candidate["golden"]["tokenizer"]
        and reviewed_golden_report.get("chunking") == candidate["chunking"]
    )
    if approved and not reviewed_golden_passed:
        raise ValueError("100–500 reviewed real-source golden cases must pass before approving scale-up.")
    retrieval_passed = False
    if retrieval_report:
        retrieval_passed = bool(
            retrieval_report.get("passed") and retrieval_report.get("qrels_sha256")
            and retrieval_report.get("snapshot_sha256") == candidate["snapshot_sha256"]
            and retrieval_report.get("evaluation_kind") == "labeled_retrieval"
        )
        if not retrieval_passed:
            raise ValueError("Retrieval report must use labels and the exact candidate snapshot.")
    full_budget = False
    if resource_report:
        required = ("max_gpu_hours", "max_storage_bytes", "deadline_margin_hours", "estimated_gpu_hours", "estimated_storage_bytes")
        if any(resource_report.get(key) is None for key in required):
            raise ValueError("Unknown budgets cannot approve a full run.")
        full_budget = bool(
            resource_report.get("measured")
            and resource_report.get("snapshot_sha256") == candidate["snapshot_sha256"]
            and resource_report["estimated_gpu_hours"] <= resource_report["max_gpu_hours"]
            and resource_report["estimated_storage_bytes"] <= resource_report["max_storage_bytes"]
            and resource_report["deadline_margin_hours"] > 0
        )
        if not full_budget:
            raise ValueError("Resource gate failed.")
    record = {"stage": stage, "approved": approved, "reviewer": reviewer.strip(),
              "corpus_sha256": corpus_sha256, "snapshot_sha256": candidate["snapshot_sha256"],
              "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
              "code_sha256": code_fingerprint(),
              "chunking": candidate["chunking"],
              "reviewed_golden_passed": reviewed_golden_passed,
              "reviewed_golden_report": reviewed_golden_report,
              "retrieval_passed": retrieval_passed, "full_budget_passed": full_budget,
              "retrieval_report": retrieval_report, "resource_report": resource_report,
              "reviewed_at": utc_now()}
    atomic_json(Path(data_root) / "gates" / f"{stage}.json", record)
    return record
