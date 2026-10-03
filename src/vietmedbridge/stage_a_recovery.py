"""Recover the remaining Stage A IDs without dropping successes or source IDs."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from .artifacts import atomic_json, code_fingerprint, digest_json, read_json, utc_now, verify_file
from .browser_probe import BrowserProbeConfig, response_evidence
from .chrome_transport import ChromeRobotsTransport
from .crawl import CrawlConfig, Fetcher
from .quality import document_quality
from .recovery import RecoveryConfig, guarded_http_fetch
from .structure import parse_structure, validate_structure
from .text import EXPECTED_PARSE_ERRORS, extract_source


def _index(rows, expected):
    indexed = {}
    for row in rows:
        doc_id = int(row["doc_id"])
        if doc_id in indexed or expected.get(doc_id) != row["url"]:
            raise ValueError("Duplicate ID or URL differs from official Stage A mapping.")
        indexed[doc_id] = row
    return indexed


def select_remaining(expected, failures, robots, *, cached_browser_ids, legacy_candidate_ids):
    """Every failed ID belongs to exactly one bucket; neither prior group is lost."""
    failed, policies = _index(failures, expected), _index(robots, expected)
    if set(policies) != set(failed):
        raise ValueError("Robots ledger does not cover every failed ID.")
    cached, legacy = set(cached_browser_ids), set(legacy_candidate_ids)
    if cached & legacy or not (cached | legacy) <= set(failed):
        raise ValueError("Recovery groups overlap or contain unexpected IDs.")
    records = []
    for doc_id, row in sorted(failed.items()):
        records.append({"doc_id": doc_id, "url": row["url"], "original_error": row.get("error"),
                        "prior_robots_state": policies[doc_id]["robots_state"],
                        "state": "CACHED_BROWSER_REVIEW" if doc_id in cached else
                                 "LEGACY_CANDIDATE_REVIEW" if doc_id in legacy else "PENDING_RECOVERY"})
    return {"official_pairs": sorted(expected.items()), "records": records,
            "baseline_http_captures": len(expected) - len(failed),
            "failed_ids": len(failed), "remaining_ids": len(failed) - len(cached) - len(legacy)}


def _write_asset(target, relative, body):
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(body).hexdigest()
    if path.exists():
        verify_file(path, checksum)
    else:
        temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(body)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": relative, "sha256": checksum}


def _extract(record, body, content_type, target, asset, *, capture_kind, method):
    try:
        extracted = extract_source(body, content_type, source_url=record.get("final_url") or record["url"])
    except EXPECTED_PARSE_ERRORS as exc:
        record.update(state="EXTRACTION_REJECTED", extract_error=f"{type(exc).__name__}: {str(exc)[:300]}")
        return False
    doc = {**extracted, "doc_id": record["doc_id"], "url": record["url"],
           "final_url": record.get("final_url") or record["url"], "body_sha256": asset["sha256"],
           "source_asset": asset["path"], "capture_kind": capture_kind,
           "source_method": method, "fetched_at": record["started_at"],
           "source_experiment": target.name}
    structure = parse_structure(doc, extracted["heading_hints"])
    validate_structure(doc, structure)
    quality = document_quality(doc["source_text"], title=doc["title"],
                               section_count=structure["known_heading_count"], raw_bytes=len(body),
                               content_type=content_type)
    if doc.get("raw_has_table"):
        quality["quality_flags"].append("TABLE_TEXT_LINEARIZED_REQUIRES_AUDIT")
    doc.update(quality, section_count=structure["known_heading_count"])
    ready = len(doc["source_text"]) >= 300 and (bool(doc["title"].strip()) or "pdf" in content_type.lower())
    doc["state"] = "ARTICLE_CANDIDATE_REVIEW" if ready else "SHORT_OR_UNTITLED_REVIEW"
    variant = f"{method}-{capture_kind}"
    prefix = f"documents/{record['doc_id']}/{record['capture_attempt']}"
    document_asset = _write_asset(target, f"{prefix}/{variant}.json",
        (json.dumps({"document": doc, "structure": structure}, ensure_ascii=False, indent=2) + "\n").encode())
    text_asset = _write_asset(target, f"{prefix}/{variant}.txt", doc["source_text"].encode())
    record["assets"][f"document-{variant}"] = document_asset
    record["assets"][f"text-{variant}"] = text_asset
    record["assets"].update({"document.json": document_asset, "source.txt": text_asset})
    record.update(state=doc["state"], source_chars=len(doc["source_text"]), title=doc["title"],
                  source_text_sha256=doc["source_text_sha256"], section_count=doc["section_count"],
                  quality_flags=doc["quality_flags"], capture_kind=capture_kind, source_method=method)
    return ready


async def recover_remaining(plan, output_root, get, *, browser_probe=None, provenance=None,
                            versions=None, label="remaining-v1", max_new_ids=None, sleep=None,
                            advanced_runtime=None):
    """Run bounded HTTP then Crawl4AI for eligible HTML failures, with ID checkpoints.

    Successful HTTP captures and existing candidates are never refetched. Raw
    Stage A remains immutable. Outputs are review candidates, not promotions.
    Change label for a new retry experiment; same label resumes completed IDs.
    """
    if max_new_ids is not None and max_new_ids < 0:
        raise ValueError("max_new_ids must be nonnegative.")
    http_config = RecoveryConfig(attempts=2, impersonate="chrome", max_inline_wait_seconds=30)
    identity = {"plan_sha256": digest_json(plan), "provenance": provenance or {},
                "code_sha256": code_fingerprint(), "versions": versions or {},
                "label": label, "http": asdict(http_config), "policy": "remaining-stage-a-v1"}
    if advanced_runtime is not None:
        identity["advanced_recovery"] = advanced_runtime.identity
    key = digest_json(identity)[:16]
    target = Path(output_root) / key
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "experiment.json"
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError("Recovery experiment configuration changed.")
    if not manifest.exists():
        atomic_json(manifest, identity)
    held_hosts, robot_events = set(), []
    def robots_observer(event, body):
        if body is not None:
            relative = f"robots/{hashlib.sha256(body).hexdigest()}.bin"
            event["asset"] = _write_asset(target, relative, body)
        robot_events.append(event)
    transport_kwargs = {"sleep": sleep} if sleep else {}
    if advanced_runtime is not None:
        transport_kwargs["browser_read"] = advanced_runtime.read_robots
    transport = ChromeRobotsTransport(get, observer=robots_observer, **transport_kwargs)
    guard = Fetcher(CrawlConfig(concurrency=1, attempts=1, per_host_delay=3), transport=transport)
    if advanced_runtime is not None:
        advanced_runtime.configure_guard(guard)
    if sleep:
        async def no_pace(*_args):
            pass
        guard.pace = no_pace
        guard.robots.pace = no_pace
    completed, new_count = {}, 0
    try:
        for input_row in plan["records"]:
            if input_row["state"] != "PENDING_RECOVERY":
                continue
            doc_id, url = input_row["doc_id"], input_row["url"]
            marker = target / "markers" / f"{doc_id}.json"
            if marker.exists():
                saved = read_json(marker)
                if saved["doc_id"] != doc_id or saved["url"] != url:
                    raise ValueError("Recovery checkpoint ID/URL changed.")
                for asset in saved.get("assets", {}).values():
                    path = (target / asset["path"]).resolve()
                    if not path.is_relative_to(target.resolve()):
                        raise ValueError("Recovery asset escapes its experiment.")
                    verify_file(path, asset["sha256"])
                completed[doc_id] = saved
                if saved.get("http_status") == 429 or saved.get("state") == "RATE_LIMIT_HOLD":
                    held_hosts.add(urlsplit(saved.get("final_url") or url).hostname)
                held_hosts.update(saved.get("held_hosts", []))
                continue
            if max_new_ids is not None and new_count >= max_new_ids:
                continue
            started, event_start = time.monotonic(), len(robot_events)
            record = {**input_row, "started_at": utc_now(), "capture_attempt": uuid4().hex, "assets": {}}
            decision = await guard.robots_decision(url)
            record.update(decision.record())
            if urlsplit(url).hostname in held_hosts or decision.state == "ROBOTS_429":
                record["state"] = "RATE_LIMIT_HOLD"
                held_hosts.add(urlsplit(url).hostname)
            elif decision.state == "ROBOTS_OK_DISALLOWED" or input_row["prior_robots_state"] == "ROBOTS_OK_DISALLOWED":
                record["state"] = "ROBOTS_POLICY_REVIEW"  # Keep prior explicit policy evidence.
            elif not decision.allowed:
                record["state"] = "ROBOTS_ACCESS_HOLD"
            else:
                evidence = []
                def observed_get(request_url, **kwargs):
                    response = get(request_url, **kwargs)
                    body = bytes(response.body)
                    evidence.append({"url": request_url, "http_status": int(response.status),
                                     **response_evidence(body, response.headers)})
                    if len(body) <= http_config.max_body_bytes:
                        name = f"http-{len(evidence)}.bin"
                        record["assets"][name] = _write_asset(target, f"assets/{doc_id}/{record['capture_attempt']}/{name}", body)
                    return response
                kwargs = {"sleep": sleep} if sleep else {}
                result, body = await guarded_http_fetch(url, guard, observed_get,
                    user_agent=guard.config.user_agent, config=http_config, held_hosts=held_hosts, **kwargs)
                record.update(result, http_evidence=evidence, state="HTTP_FETCH_FAILED")
                ready = False
                if body is not None:
                    asset = record["assets"][f"http-{len(evidence)}.bin"]
                    ready = _extract(record, body, result["content_type"], target, asset,
                                     capture_kind="response_bytes", method="chrome_http")
                browser_url = record.get("final_url") or url
                eligible = result.get("http_status") == 403 or (
                    result.get("http_status") == 200 and not ready and
                    "pdf" not in result.get("content_type", "").lower())
                if advanced_runtime is not None:
                    eligible = eligible or result.get("outcome") == "fetch_error" or (
                        result.get("http_status") in {404, 408, 500, 502, 503, 504} and
                        urlsplit(browser_url).hostname not in held_hosts)
                if eligible and browser_probe and urlsplit(browser_url).scheme == "https":
                    settings = BrowserProbeConfig(allowed_hosts=(urlsplit(browser_url).hostname,),
                        document_path_prefix="/", render_wait_seconds=5)
                    observed, assets = await browser_probe(browser_url, guard, config=settings)
                    record["browser"] = observed
                    record.update(held_hosts=observed.get("held_hosts", []))
                    held_hosts.update(record["held_hosts"])
                    for name, blob in assets.items():
                        record["assets"][name] = _write_asset(target, f"assets/{doc_id}/{record['capture_attempt']}/{name}", blob)
                    valid = (observed.get("http_status") == 200 and observed.get("rendered_http_status") == 200
                             and not observed.get("guard_errors"))
                    if valid:
                        record["final_url"] = observed["final_url"]
                        for name, kind in (("response.html", "response_bytes"), ("rendered.html", "rendered_dom")):
                            if name in assets and _extract(record, assets[name], "text/html", target,
                                    record["assets"][name], capture_kind=kind,
                                    method=observed.get("method", "crawl4ai_browser")):
                                break
                    elif not ready and record["state"] != "SHORT_OR_UNTITLED_REVIEW":
                        record["state"] = "BROWSER_FETCH_FAILED"
                    if observed.get("http_status") == 429 or urlsplit(browser_url).hostname in held_hosts:
                        record["state"] = "RATE_LIMIT_HOLD"
                if result.get("outcome") in {"rate_limited_retry_later", "host_rate_limit_hold"}:
                    record["state"] = "RATE_LIMIT_HOLD"
            record["robots_transport_events"] = robot_events[event_start:]
            # Include robots assets in each checkpoint's integrity verification.
            for index, event in enumerate(record["robots_transport_events"]):
                if event.get("asset"):
                    record["assets"][f"robots-{index}"] = event["asset"]
            record.update(finished_at=utc_now(), elapsed_ms=round((time.monotonic() - started) * 1000))
            atomic_json(marker, record)
            completed[doc_id], new_count = record, new_count + 1
            print(doc_id, record["state"], record.get("source_chars", 0), flush=True)
    finally:
        await guard.client.aclose()
    states = {row["doc_id"]: completed.get(row["doc_id"], row) for row in plan["records"]}
    ledger = [{"doc_id": doc_id, "url": url, "state": states.get(doc_id, {}).get("state", "BASELINE_HTTP_CAPTURE")}
              for doc_id, url in plan["official_pairs"]]
    if len({r["doc_id"] for r in ledger}) != len(ledger):
        raise ValueError("Coverage ledger contains duplicate official IDs.")
    result = {"experiment": key, "output_dir": str(target), "official_ids": len(ledger),
              "baseline_http_captures": plan["baseline_http_captures"], "remaining_input_ids": plan["remaining_ids"],
              "processed_remaining_ids": len(completed), "states": dict(Counter(r["state"] for r in ledger)),
              "records": list(states.values()), "coverage_ledger": ledger, "created_at": utc_now(),
              "canonical_corpus_modified": False}
    atomic_json(target / "summary.json", result)
    return result


def load_remaining_plan(data_root, *,
        robots_sha256="637738e9641c7b8cfa668b9f30ea547e9de664fe8a9204458e6836bd023b3a27",
        legacy_sha256="09f3edd03c459b8791bb819805656e01626a837eec898c2a1d46130af5de5244",
        browser_sha256="c4c716d838f4c676e62cc7e9e8e9b8ba366ee04dd5f3e5d2abdc3560b3fdb807",
        work_dir=None):
    """Read and prove the original complete input/failed ledger before requests."""
    import csv
    import pyarrow.parquet as pq
    from .crawl import completed_parts, iter_raw_records, range_complete
    from .probe_review import review_browser_captures

    root = Path(data_root).resolve()
    def source_path(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source artifact escapes DATA_ROOT.")
        return path
    def csv_rows(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    report = root / "reports/crawl_recovery/stage-a-v2"
    triage = read_json(report / "triage.json")
    crawl_root = root / "crawl/stage-a-v2"
    run = read_json(crawl_root / "run.json")
    pilot = read_json(root / "reports/inventory/stage_a.json")
    link_asset = pilot["files"]["stage_a_links"]
    link_path = source_path(link_asset["path"])
    verify_file(link_path, link_asset["sha256"])
    if link_asset["sha256"] != run["links_sha256"] or triage["run_signature"] != run["signature"]:
        raise ValueError("Official Stage A sample, triage and crawl signatures differ.")
    pairs = pq.read_table(link_path, columns=["id", "url"]).to_pylist()
    expected = {int(row["id"]): row["url"] for row in pairs}
    if len(expected) != len(pairs) or run["requested_range"] != {"start": 0, "stop": len(pairs)}:
        raise ValueError("Stage A input duplicate IDs or requested range mismatch.")
    parts = completed_parts(crawl_root)
    if not range_complete(run, parts):
        raise ValueError("Stage A raw range is incomplete; resume notebook 01 first.")
    raw = _index(({k: row[k] for k in ("doc_id", "url", "status")}
                  for part in parts for row in iter_raw_records(crawl_root / part["raw_file"])), expected)
    if set(raw) != set(expected):
        raise ValueError("Original crawl does not record every official Stage A ID.")
    failures_path = source_path(triage["failures_csv"])
    verify_file(failures_path, triage["failures_sha256"])
    failures = csv_rows(failures_path)
    failed = _index(failures, expected)
    if set(failed) != {i for i, r in raw.items() if r["status"] != "ok"}:
        raise ValueError("Failure CSV differs from actual raw checkpoint outcomes.")
    if len(failed) != triage["unresolved"] or len(expected) - len(failed) != triage["successful"]:
        raise ValueError("Triage counts do not match raw IDs.")
    legacy_root = report / "experiments/6a5793cdc03be0cb"
    if read_json(legacy_root / "experiment.json")["source_run_signature"] != run["signature"]:
        raise ValueError("01c experiment does not belong to original Stage A run.")
    summary = read_json(legacy_root / "summary.json")
    if summary["robots_sha256"] != robots_sha256 or summary["attempts_sha256"] != legacy_sha256:
        raise ValueError("01c source hashes differ from the reviewed experiment.")
    verify_file(legacy_root / "robots.csv", robots_sha256)
    verify_file(legacy_root / "attempts.csv", legacy_sha256)
    legacy_rows = csv_rows(legacy_root / "attempts.csv")
    _index(legacy_rows, expected)
    candidates = [r for r in legacy_rows if r.get("article_candidate", "").lower() == "true"]
    for candidate in candidates:
        verify_file(source_path(candidate["asset_path"]), candidate["body_sha256"])
    if len(candidates) != summary["article_candidates_needing_human_review"]:
        raise ValueError("Legacy candidate count differs from source summary.")
    browser_root = report / "experiments/4c49d3358aa818b7"
    browser_summary = read_json(browser_root / "summary.json")
    browser_ids = browser_summary["candidate_ids_needing_review"]
    if not set(browser_ids) <= set(failed):
        raise ValueError("Browser IDs do not belong to original failed Stage A IDs.")
    cached = review_browser_captures(browser_root, report / "extraction_review",
        expected_links={i: expected[i] for i in browser_ids},
        expected_attempts_sha256=browser_sha256, work_dir=work_dir)
    ready = [r["doc_id"] for r in cached["records"] if r["state"] == "READY_FOR_EXTRACTION_REVIEW"]
    plan = select_remaining(expected, failures, csv_rows(legacy_root / "robots.csv"),
        cached_browser_ids=ready, legacy_candidate_ids=[int(r["doc_id"]) for r in candidates])
    provenance = {"run_signature": run["signature"], "links_sha256": link_asset["sha256"],
                  "failures_sha256": triage["failures_sha256"], "robots_sha256": robots_sha256,
                  "legacy_attempts_sha256": legacy_sha256, "browser_attempts_sha256": browser_sha256,
                  "cached_extraction_signature": cached["signature"]}
    sources = {"official_stage_a": link_path, "failures_after_retry.csv": failures_path,
               "robots_01c.csv": legacy_root / "robots.csv", "attempts_01c.csv": legacy_root / "attempts.csv"}
    for candidate in candidates:
        sources[f"legacy-{candidate['doc_id']}.bin"] = source_path(candidate["asset_path"])
    return plan, provenance, sources
