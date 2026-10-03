"""Checkpointed post-baseline recovery for any completed crawl range."""

from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from vietmedbridge.artifacts import (atomic_json, digest_json, publish_file, read_json,
                                     sha256_file, utc_now, verify_file)
from vietmedbridge.crawl import completed_parts, iter_raw_records, range_complete
from vietmedbridge.dataset import iter_link_shards
from vietmedbridge.stage_a_recovery import recover_remaining


def prepare_crawl_recovery_source(data_root, links_path, run_name, *,
                                  expected_origin_corpus_sha256=None, work_dir=None):
    """Verify a completed crawl against its exact input and index only failed IDs.

    The index is compressed JSONL and contains failures only; successful records
    stay in their immutable crawl shards. Verification is bounded by shard size.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("Use a simple crawl run name without slashes.")
    root, links_path = Path(data_root).resolve(), Path(links_path).resolve()
    run_dir = (root / "crawl" / run_name).resolve()
    if not run_dir.is_relative_to(root) or not links_path.is_file():
        raise ValueError("Crawl run or input links file is missing/outside DATA_ROOT.")
    run = read_json(run_dir / "run.json")
    run_identity = {key: value for key, value in run.items()
                    if key not in {"signature", "created_at"}}
    if digest_json(run_identity) != run.get("signature"):
        raise ValueError("Crawl run manifest signature is invalid or incomplete.")
    links_sha = sha256_file(links_path)
    if run.get("links_sha256") != links_sha:
        raise ValueError("Selected input Parquet differs from the input used by this crawl run.")
    if (expected_origin_corpus_sha256 is not None and
            run.get("origin_corpus_sha256") != expected_origin_corpus_sha256):
        raise ValueError("Crawl run belongs to a different official corpus snapshot.")

    requested = run.get("requested_range", {})
    start, stop = requested.get("start"), requested.get("stop")
    shard_size = run.get("config", {}).get("shard_size")
    if not isinstance(start, int) or not isinstance(stop, int) or not isinstance(shard_size, int):
        raise ValueError("Crawl run lacks a valid requested row range or shard size.")
    parts = completed_parts(run_dir)
    if not range_complete(run, parts):
        raise ValueError("Baseline range is incomplete; resume Notebook 01 baseline first.")
    by_first = {part["first_input_row"]: part for part in parts}
    if len(by_first) != len(parts):
        raise ValueError("Crawl has duplicate checkpoint pointers for an input shard.")

    part_identity = [{"first_input_row": p["first_input_row"], "records": p["records"],
                      "raw_sha256": p["raw_sha256"], "input_sha256": p["input_sha256"],
                      "failed_ids_sha256": digest_json(p["failed_ids"])}
                     for p in sorted(parts, key=lambda x: x["first_input_row"])]
    source_identity = {"schema": "crawl-failures-v1", "run_name": run_name,
                       "run_signature": run["signature"], "links_sha256": links_sha,
                       "origin_corpus_sha256": run.get("origin_corpus_sha256"),
                       "requested_range": requested, "shard_size": shard_size,
                       "parts": part_identity}
    source_key = digest_json(source_identity)[:20]
    target = root / "reports" / "crawl_recovery" / run_name / source_key
    target.mkdir(parents=True, exist_ok=True)
    manifest_path, failures_path = target / "source.json", target / "failures.jsonl.gz"
    if manifest_path.exists() and failures_path.exists():
        saved = read_json(manifest_path)
        if saved.get("identity") != source_identity:
            raise ValueError("Recovery input manifest belongs to a different crawl run.")
        verify_file(failures_path, saved["failures_sha256"])
        return {**saved, "manifest_path": str(manifest_path), "failures_path": str(failures_path)}

    totals = Counter()
    with tempfile.TemporaryDirectory(prefix="vmb-recovery-index-", dir=work_dir) as temporary:
        local_path = Path(temporary) / "failures.jsonl.gz"
        with local_path.open("wb") as binary:
            with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as compressed:
                for first, rows in iter_link_shards(links_path, start=start, stop=stop,
                                                    shard_size=shard_size):
                    part = by_first.pop(first, None)
                    if part is None or part["records"] != len(rows):
                        raise ValueError(f"Missing/misaligned crawl checkpoint at input row {first}.")
                    expected = {int(row["id"]): row["url"] for row in rows}
                    pairs_sha = digest_json(sorted(rows, key=lambda row: row["id"]))
                    if (len(expected) != len(rows) or digest_json(rows) != part["input_sha256"]
                            or pairs_sha != part.get("input_pairs_sha256")):
                        raise ValueError(f"Crawl shard input ID/URL pairs changed at row {first}.")
                    observed, failures, statuses = set(), [], Counter()
                    for record in iter_raw_records(run_dir / part["raw_file"]):
                        doc_id = int(record["doc_id"])
                        if doc_id in observed or expected.get(doc_id) != record.get("url"):
                            raise ValueError(f"Crawl output ID/URL differs from official input: {doc_id}.")
                        observed.add(doc_id)
                        status = record.get("status")
                        if status not in {"ok", "error"}:
                            raise ValueError(f"Unexpected crawl status for ID {doc_id}: {status!r}.")
                        statuses[status if status == "ok" else record.get("error", "unknown_error")] += 1
                        totals["baseline_successes" if status == "ok" else "failed_ids"] += 1
                        if status != "ok":
                            totals["pending_recovery_ids" if record.get("robots_state") != "ROBOTS_OK_DISALLOWED"
                                   else "robots_policy_review_ids"] += 1
                            failures.append({"doc_id": doc_id, "url": record["url"],
                                             "original_error": record.get("error"),
                                             "http_status": record.get("http_status"),
                                             "prior_robots_state": record.get("robots_state"),
                                             "state": ("ROBOTS_POLICY_REVIEW"
                                                       if record.get("robots_state") == "ROBOTS_OK_DISALLOWED"
                                                       else "PENDING_RECOVERY")})
                    if observed != set(expected):
                        raise ValueError(f"Crawl shard outcomes omit or add IDs at input row {first}.")
                    if sorted(r["doc_id"] for r in failures) != sorted(part["failed_ids"]):
                        raise ValueError(f"Failed-ID manifest differs from raw outcomes at row {first}.")
                    if dict(statuses) != part["statuses"]:
                        raise ValueError(f"Status counts differ from raw outcomes at row {first}.")
                    totals["official_ids"] += len(rows)
                    for record in failures:
                        compressed.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        if by_first:
            raise ValueError("Crawl contains checkpoint shards outside its requested range.")
        if totals["official_ids"] != stop - start:
            raise ValueError("Verified crawl outcomes do not cover the requested physical row range.")
        if totals["baseline_successes"] + totals["failed_ids"] != totals["official_ids"]:
            raise ValueError("Official IDs are not partitioned into baseline successes and failures.")
        publish_file(local_path, failures_path)
    result = {"identity": source_identity, **totals,
              "failures_sha256": sha256_file(failures_path), "failures_path": str(failures_path),
              "created_at": utc_now(), "range_complete": True,
              "proof": "Every raw ID/URL was matched to its official input row; failures match shard manifests."}
    atomic_json(manifest_path, result)
    return {**result, "manifest_path": str(manifest_path)}


def _iter_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _batches(source, batch_size):
    batch, index = [], 0
    for record in _iter_jsonl_gz(source["failures_path"]):
        batch.append(record)
        if len(batch) >= batch_size:
            yield index, batch
            index, batch = index + 1, []
    if batch:
        yield index, batch


def _plan(rows):
    records = []
    for row in rows:
        records.append({"doc_id": row["doc_id"], "url": row["url"],
                        "original_error": row.get("original_error"),
                        "original_http_status": row.get("http_status"),
                        "prior_robots_state": row.get("prior_robots_state"),
                        "state": row["state"]})
    return {"official_pairs": [(row["doc_id"], row["url"]) for row in rows],
            "records": records, "baseline_http_captures": 0,
            "failed_ids": len(rows),
            "remaining_ids": sum(row["state"] == "PENDING_RECOVERY" for row in rows)}


def _write_batch_coverage(path, plan, result):
    records = {}
    marker_dir = Path(result["output_dir"]) / "markers"
    for row in plan["records"]:
        saved = row
        marker = marker_dir / f"{row['doc_id']}.json"
        if row["state"] == "PENDING_RECOVERY":
            if not marker.is_file():
                raise ValueError(f"Recovery claims completion but ID {row['doc_id']} has no checkpoint.")
            saved = read_json(marker)
            if saved.get("doc_id") != row["doc_id"] or saved.get("url") != row["url"]:
                raise ValueError("Recovery checkpoint differs from official ID/URL mapping.")
        records[row["doc_id"]] = saved.get("state", row["state"])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("doc_id", "url", "baseline_error",
            "baseline_http_status", "prior_robots_state", "domain", "recovery_state"))
        writer.writeheader()
        for row in plan["records"]:
            writer.writerow({"doc_id": row["doc_id"], "url": row["url"],
                "baseline_error": row.get("original_error"),
                "baseline_http_status": row.get("original_http_status"),
                "prior_robots_state": row.get("prior_robots_state"),
                "domain": urlsplit(row["url"]).hostname,
                "recovery_state": records[row["doc_id"]]})
    temporary.replace(path)
    return Counter(records.values())


def _read_done(done_path, expected_plan_sha):
    done = read_json(done_path)
    if done.get("plan_sha256") != expected_plan_sha:
        raise ValueError("Recovery batch checkpoint does not match its source IDs/URLs.")
    verify_file(done["coverage_path"], done["coverage_sha256"])
    return done


def _coverage_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _export_failure_coverage(source, batch_dir, output_path, batch_size):
    temporary = output_path.with_name(output_path.name + ".tmp")
    state_counts = Counter()
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("doc_id", "url", "baseline_error", "baseline_http_status",
                  "prior_robots_state", "domain", "recovery_state")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, rows in _batches(source, batch_size):
            batch_plan = _plan(rows)
            batch_sha = digest_json(batch_plan)
            done_path = batch_dir / f"batch-{index:07d}.done.json"
            if done_path.is_file():
                done = _read_done(done_path, batch_sha)
                batch_rows = list(_coverage_rows(done["coverage_path"]))
                if len(batch_rows) != len(rows):
                    raise ValueError("Recovery batch coverage does not include every failed input ID.")
                if any((int(saved["doc_id"]), saved["url"]) != (expected["doc_id"], expected["url"])
                       for saved, expected in zip(batch_rows, rows)):
                    raise ValueError("Recovery batch coverage changed an official ID/URL pair.")
                writer.writerows(batch_rows)
                state_counts.update(row["recovery_state"] for row in batch_rows)
            else:
                for row in rows:
                    writer.writerow({"doc_id": row["doc_id"], "url": row["url"],
                        "baseline_error": row.get("original_error"),
                        "baseline_http_status": row.get("http_status"),
                        "prior_robots_state": row.get("prior_robots_state"),
                        "domain": urlsplit(row["url"]).hostname,
                        "recovery_state": row["state"]})
                    state_counts[row["state"]] += 1
    temporary.replace(output_path)
    domain_counts = Counter()
    candidate_path = output_path.with_name("candidate_review.csv")
    candidate_temp = candidate_path.with_name(candidate_path.name + ".tmp")
    with output_path.open(encoding="utf-8-sig", newline="") as source_stream, \
            candidate_temp.open("w", encoding="utf-8-sig", newline="") as candidate_stream:
        reader = csv.DictReader(source_stream)
        candidate_writer = csv.DictWriter(candidate_stream, fieldnames=reader.fieldnames)
        candidate_writer.writeheader()
        for row in reader:
            domain_counts[(row["domain"], row["recovery_state"])] += 1
            if row["recovery_state"] in {"ARTICLE_CANDIDATE_REVIEW", "SHORT_OR_UNTITLED_REVIEW"}:
                candidate_writer.writerow(row)
    candidate_temp.replace(candidate_path)
    domain_path = output_path.with_name("coverage_by_domain.csv")
    domain_temp = domain_path.with_name(domain_path.name + ".tmp")
    with domain_temp.open("w", encoding="utf-8-sig", newline="") as stream:
        domain_writer = csv.DictWriter(stream, fieldnames=("domain", "recovery_state", "ids"))
        domain_writer.writeheader()
        for (domain, state), count in sorted(domain_counts.items()):
            domain_writer.writerow({"domain": domain, "recovery_state": state, "ids": count})
    domain_temp.replace(domain_path)
    unresolved_path = output_path.with_name("unresolved_urls.csv")
    unresolved_temp = unresolved_path.with_name(unresolved_path.name + ".tmp")
    shutil.copyfile(output_path, unresolved_temp)
    unresolved_temp.replace(unresolved_path)
    return state_counts


async def recover_crawl_failures(source, output_root, get, *, advanced_runtime,
                                 browser_probe=None, versions=None, label="advanced-v1",
                                 batch_size=512, max_batches_this_session=8,
                                 max_new_ids_per_batch=None, sleep=None):
    """Recover verified failed IDs in resumable batches without loading the range.

    `max_batches_this_session` bounds Colab work. If a batch is interrupted, its
    per-ID markers resume before later batches are started on the next run.
    """
    if batch_size < 1 or max_batches_this_session < 1:
        raise ValueError("batch_size and max_batches_this_session must be positive.")
    if max_new_ids_per_batch is not None and max_new_ids_per_batch < 1:
        raise ValueError("max_new_ids_per_batch must be positive or None.")
    source_path = Path(source["failures_path"])
    verify_file(source_path, source["failures_sha256"])
    provenance = {"source_sha256": source["failures_sha256"],
                  "source_identity": source["identity"], "label": label,
                  "batch_size": batch_size, "versions": versions or {},
                  "workflow_sha256": sha256_file(Path(__file__)),
                  "advanced_recovery": (advanced_runtime.identity if advanced_runtime else None)}
    key = digest_json(provenance)[:20]
    target = Path(output_root) / key
    batch_dir = target / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = target / "recovery.json"
    if manifest.exists() and read_json(manifest).get("identity") != provenance:
        raise ValueError("Recovery configuration changed; use a new label for a new experiment.")
    if not manifest.exists():
        atomic_json(manifest, {"identity": provenance, "created_at": utc_now()})

    completed_ids, batch_count, started_batches = 0, 0, 0
    recovered_states = Counter()
    last_output = None
    for index, rows in _batches(source, batch_size):
        plan = _plan(rows)
        plan_sha = digest_json(plan)
        done_path = batch_dir / f"batch-{index:07d}.done.json"
        if done_path.exists():
            done = _read_done(done_path, plan_sha)
            completed_ids += done["ids"]
            batch_count += 1
            recovered_states.update(done["states"])
            continue
        if started_batches >= max_batches_this_session:
            break
        if plan["remaining_ids"] and advanced_runtime is None:
            raise ValueError("Eligible failed IDs require Crawl4AI/Scrapling runtime installation.")
        result = await recover_remaining(
            plan, target / "attempts", get, browser_probe=browser_probe,
            provenance={"run_signature": source["identity"]["run_signature"],
                        "source_sha256": source["failures_sha256"], "batch_index": index},
            versions=versions, label=label, max_new_ids=max_new_ids_per_batch,
            advanced_runtime=advanced_runtime, sleep=sleep,
        )
        last_output = result["output_dir"]
        started_batches += 1
        if result["processed_remaining_ids"] < plan["remaining_ids"]:
            _compact_attempt_summary(result)
            # Leave the batch incomplete. Restarting the notebook picks up its
            # committed ID markers and never begins later input rows first.
            break
        coverage_path = Path(result["output_dir"]) / "coverage.csv"
        counts = _write_batch_coverage(coverage_path, plan, result)
        _compact_attempt_summary(result)
        done = {"batch": index, "plan_sha256": plan_sha, "ids": len(rows),
                "coverage_path": str(coverage_path), "coverage_sha256": sha256_file(coverage_path),
                "states": dict(counts), "experiment": result["experiment"],
                "completed_at": utc_now()}
        atomic_json(done_path, done)
        completed_ids += len(rows)
        batch_count += 1
        recovered_states.update(counts)

    failures_coverage = target / "failed_id_coverage.csv"
    exported_states = _export_failure_coverage(source, batch_dir, failures_coverage, batch_size)
    complete = completed_ids == source["failed_ids"]
    result = {"experiment": key, "output_dir": str(target),
              "run_name": source["identity"]["run_name"],
              "run_signature": source["identity"]["run_signature"],
              "official_ids": source["official_ids"],
              "baseline_http_captures": source["baseline_successes"],
              "baseline_failed_ids": source["failed_ids"],
              "recovery_complete": complete, "evaluated_failed_ids": completed_ids,
              "pending_or_incomplete_failed_ids": source["failed_ids"] - completed_ids,
              "completed_batches": batch_count, "batches_this_session": started_batches,
              "failure_states": dict(exported_states),
              "failed_id_coverage_csv": str(failures_coverage),
              "failed_id_coverage_sha256": sha256_file(failures_coverage),
              "coverage_by_domain_csv": str(failures_coverage.with_name("coverage_by_domain.csv")),
              "unresolved_urls_csv": str(failures_coverage.with_name("unresolved_urls.csv")),
              "candidate_review_csv": str(failures_coverage.with_name("candidate_review.csv")),
              "last_batch_output_dir": last_output,
              "canonical_corpus_modified": False, "updated_at": utc_now()}
    atomic_json(target / "summary.json", result)
    return result


def _compact_attempt_summary(result):
    """Keep per-ID markers, but avoid duplicating large batch details in JSON."""
    detail_path = Path(result["output_dir"]) / "summary.json"
    compact = {key: value for key, value in result.items()
               if key not in {"records", "coverage_ledger"}}
    atomic_json(detail_path, compact)
