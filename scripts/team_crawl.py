"""Plan independent crawl ranges and merge their verified raw checkpoints.

This lives outside the pipeline package so preparing a team run does not change
the extractor's code fingerprint or invalidate an already reviewed candidate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq

from vietmedbridge.artifacts import (atomic_json, digest_json, publish_file, read_json,
                                     sha256_file, utc_now, verify_file)
from vietmedbridge.crawl import (completed_parts, iter_raw_records,
                                 read_completed_crawl)
from vietmedbridge.dataset import iter_link_shards


def team_plan(links_path: str | Path, *, workers: int = 3, shard_size: int = 512,
              run_prefix: str = "full-team-v1") -> dict:
    """Split physical Parquet rows at shard boundaries, never by official ID."""
    if workers < 2 or shard_size < 1:
        raise ValueError("Expected at least two workers and a positive shard size.")
    if not run_prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in run_prefix):
        raise ValueError("Use a simple run prefix without spaces or slashes.")
    links_path = Path(links_path)
    rows = pq.ParquetFile(links_path).metadata.num_rows
    full_shards, final_rows = divmod(rows, shard_size)
    total_shards = full_shards + bool(final_rows)
    if total_shards < workers:
        raise ValueError("Fewer shards than workers.")
    per_worker, extra = divmod(total_shards, workers)
    parts = []
    first_shard = 0
    for index in range(workers):
        shard_count = per_worker + (index < extra)
        next_shard = first_shard + shard_count
        start = min(first_shard * shard_size, rows)
        stop = min(next_shard * shard_size, rows)
        parts.append({"member": index + 1, "start_row": start, "stop_row": stop,
                      "rows": stop - start, "shards": shard_count,
                      "run_name": f"{run_prefix}-part-{index + 1:02d}-of-{workers:02d}"})
        first_shard = next_shard
    plan = {"format": "team-crawl-plan-v1", "links_sha256": sha256_file(links_path),
            "total_rows": rows, "shard_size": shard_size, "workers": workers,
            "run_prefix": run_prefix, "parts": parts,
            "merged_run_name": f"{run_prefix}-merged"}
    plan["plan_sha256"] = digest_json(plan)
    return plan


def merge_team_crawls(plan: dict, *, links_path: str | Path,
                      origin_corpus_sha256: str, source_dirs: list[str | Path],
                      output_root: str | Path) -> dict:
    """Verify each complete independent run, then publish one build-compatible run.

    Sources must all be accessible from this machine. The merge copies raw bytes;
    it never edits or removes a teammate's checkpoint.
    """
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if digest_json(plan_body) != plan.get("plan_sha256"):
        raise ValueError("Team plan was changed after it was generated.")
    links_path = Path(links_path)
    if team_plan(links_path, workers=plan["workers"], shard_size=plan["shard_size"],
                 run_prefix=plan["run_prefix"]) != plan:
        raise ValueError("Team plan does not match this official Parquet file.")
    if len(source_dirs) != plan["workers"]:
        raise ValueError("Provide exactly one source run directory per team member, in member order.")
    source_dirs = [Path(path) for path in source_dirs]
    source_runs = []
    source_parts = []
    common = None
    for expected, source in zip(plan["parts"], source_dirs, strict=True):
        if source.name != expected["run_name"]:
            raise ValueError(f"Member {expected['member']} must provide {expected['run_name']}, got {source.name}.")
        run = read_json(source / "run.json")
        if (run.get("requested_range") != {"start": expected["start_row"], "stop": expected["stop_row"]}
                or run.get("requested_input_records") != expected["rows"]
                or run.get("links_sha256") != plan["links_sha256"]
                or run.get("origin_corpus_sha256") != origin_corpus_sha256
                or run.get("config", {}).get("shard_size") != plan["shard_size"]):
            raise ValueError(f"Member {expected['member']} has the wrong input, range or shard size.")
        compatibility = {key: run.get(key) for key in
                         ("config", "crawler_version", "fetch_engine", "enhancer", "code_sha256")}
        if common is None:
            common = compatibility
        elif compatibility != common:
            raise ValueError(f"Member {expected['member']} used different crawl code or configuration.")
        read_completed_crawl(source, links_path=links_path,
                             origin_corpus_sha256=origin_corpus_sha256,
                             start=expected["start_row"], stop=expected["stop_row"])
        parts = completed_parts(source, verify=False)
        source_runs.append({"member": expected["member"], "run_name": source.name,
                            "signature": run["signature"], "shards": len(parts)})
        source_parts.extend((source, part) for part in parts)
    source_parts.sort(key=lambda entry: entry[1]["first_input_row"])
    cursor = 0
    seen_files = set()
    for source, part in source_parts:
        if part["first_input_row"] != cursor:
            raise ValueError(f"Missing or overlapping physical rows at {cursor}.")
        cursor += part["records"]
        relative = Path(part["raw_file"])
        if relative.is_absolute() or ".." in relative.parts or relative in seen_files:
            raise ValueError(f"Unsafe or duplicated raw file path: {relative}")
        seen_files.add(relative)
        verify_file(source / relative, part["raw_sha256"])
    if cursor != plan["total_rows"]:
        raise ValueError(f"Combined runs cover {cursor} of {plan['total_rows']} rows.")
    # Manifests prove the intended inputs. Also inspect the actual raw records so
    # a stale or incorrectly published capture cannot silently change an ID/URL.
    expected_shards = iter_link_shards(links_path, start=0,
                                      stop=plan["total_rows"],
                                      shard_size=plan["shard_size"])
    for (source, part), (first, expected_rows) in zip(source_parts, expected_shards, strict=True):
        if first != part["first_input_row"] or len(expected_rows) != part["records"]:
            raise ValueError(f"Shard boundary mismatch at row {first}.")
        actual = iter_raw_records(source / part["raw_file"])
        for expected in expected_rows:
            record = next(actual, None)
            if record is None or record.get("doc_id") != expected["id"] or record.get("url") != expected["url"]:
                raise ValueError(f"Raw ID/URL mismatch in {source / part['raw_file']}.")
        if next(actual, None) is not None:
            raise ValueError(f"Too many raw records in {source / part['raw_file']}.")

    output_root = Path(output_root)
    target = output_root / plan["merged_run_name"]
    staging = output_root / f".{plan['merged_run_name']}.staging"
    identity = {"links_sha256": plan["links_sha256"],
                "origin_corpus_sha256": origin_corpus_sha256,
                "row_range": {"start": 0, "stop": plan["total_rows"]},
                "requested_range": {"start": 0, "stop": plan["total_rows"]},
                "requested_input_records": plan["total_rows"],
                "config": common["config"], "crawler_version": "merged-independent-v1",
                "fetch_engine": common["fetch_engine"], "enhancer": common["enhancer"],
                "code_sha256": common["code_sha256"],
                "team_plan_sha256": plan["plan_sha256"], "source_runs": source_runs}
    signature = digest_json(identity)
    if target.exists():
        if read_json(target / "run.json").get("signature") != signature:
            raise ValueError("Existing merged run belongs to another plan or configuration.")
        summary = read_completed_crawl(target, links_path=links_path,
                                       origin_corpus_sha256=origin_corpus_sha256,
                                       start=0, stop=plan["total_rows"])
        return {**summary, "run_dir": str(target), "source_runs": source_runs}
    if staging.exists():
        if read_json(staging / "run.json").get("signature") != signature:
            raise ValueError("Staging directory belongs to another plan or configuration.")
    else:
        staging.mkdir(parents=True)
        atomic_json(staging / "run.json", {**identity, "signature": signature,
                                           "created_at": utc_now()})
    for source, part in source_parts:
        relative = Path(part["raw_file"])
        destination = staging / relative
        pointer = destination.parent / f"{part['part']}.done.json"
        if pointer.exists():
            saved = read_json(pointer)
            if (saved.get("run_signature") != signature
                    or saved.get("raw_sha256") != part["raw_sha256"]
                    or saved.get("first_input_row") != part["first_input_row"]):
                raise ValueError(f"Staging checkpoint differs from source: {pointer}")
            verify_file(destination, part["raw_sha256"])
            continue
        copied_sha = publish_file(source / relative, destination)
        if copied_sha != part["raw_sha256"]:
            raise ValueError(f"Raw copy changed in transit: {destination}")
        atomic_json(pointer, {**part, "run_signature": signature})
    summary = read_completed_crawl(staging, links_path=links_path,
                                   origin_corpus_sha256=origin_corpus_sha256,
                                   start=0, stop=plan["total_rows"])
    summary["run_name"] = target.name
    atomic_json(staging / "summary.json", summary)
    os.replace(staging, target)
    return {**summary, "run_name": target.name, "run_dir": str(target),
            "source_runs": source_runs}
