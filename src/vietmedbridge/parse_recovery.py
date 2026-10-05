"""Re-fetch only saved Lao Động cookie pages and rebuild a full raw crawl snapshot."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .artifacts import (
    atomic_json, code_fingerprint, digest_json, local_workspace, publish_file,
    read_json, runtime_versions, utc_now, verify_file,
)
from .crawl import CrawlConfig, Fetcher, completed_parts, iter_raw_records, range_complete
from .source_challenges import laodong_cookie_challenge
from .text import EXPECTED_PARSE_ERRORS, extract_source


def _saved_challenge(record: dict) -> bool:
    if record["status"] != "ok":
        return False
    body = base64.b64decode(record["body_base64"], validate=True)
    if len(body) != record["body_bytes"] or hashlib.sha256(body).hexdigest() != record["body_sha256"]:
        raise ValueError("Saved raw body checksum mismatch.")
    return laodong_cookie_challenge(body, record.get("final_url") or record["url"]) is not None


async def recover_laodong_challenges(
    source_crawl_dir: str | Path, output_root: str | Path, *,
    run_name: str = "stage-a-laodong-recovered-v1", config: CrawlConfig | None = None,
    work_dir: str | Path | None = None, max_shards: int | None = None, transport=None,
) -> dict:
    """Preserve every original ID and replace only validated Lao Động challenge captures.

    Source crawl checkpoints remain immutable. New raw shards retain all old
    records except those whose refreshed HTTP body passes the article adapter.
    A failed refresh leaves the old record, so the new build can still report
    the original parse failure. The run can resume by completed shard.
    """
    source_crawl_dir, output_root = Path(source_crawl_dir), Path(output_root)
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("Use a simple run name.")
    config = config or CrawlConfig(concurrency=2, per_host_delay=2.0, attempts=2)
    config.validate()
    source_run = read_json(source_crawl_dir / "run.json")
    source_parts = completed_parts(source_crawl_dir)
    if not range_complete(source_run, source_parts):
        raise ValueError("Source crawl is incomplete.")
    identity = {
        **{key: source_run[key] for key in (
            "links_sha256", "origin_corpus_sha256", "requested_range", "requested_input_records",
        )},
        "config": source_run["config"],  # The original shard size and input mapping.
        "source_crawl_signature": source_run["signature"],
        "source_part_hashes": [part["raw_sha256"] for part in source_parts],
        "recovery_config": asdict(config),
        "crawler_version": "laodong-parse-recovery-v1",
        "code_sha256": code_fingerprint(), "runtime": runtime_versions(),
    }
    signature = digest_json(identity)
    target = output_root / run_name
    if target.resolve() == source_crawl_dir.resolve():
        raise ValueError("Recovery must use a new crawl run name.")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "run.json"
    if manifest_path.exists() and read_json(manifest_path)["signature"] != signature:
        raise ValueError("Recovery input/config changed. Use a new run_name.")
    if not manifest_path.exists():
        atomic_json(manifest_path, {**identity, "signature": signature, "created_at": utc_now()})

    fetcher = Fetcher(config, transport=transport)
    skipped, written = 0, 0
    try:
        for part in source_parts:
            relative = Path(part["raw_file"])
            bucket = relative.parent
            pointer = target / bucket / f"{part['part']}.done.json"
            if pointer.exists():
                saved = read_json(pointer)
                if saved["run_signature"] != signature or saved["source_raw_sha256"] != part["raw_sha256"]:
                    raise ValueError("Recovery checkpoint belongs to another source shard.")
                verify_file(target / saved["raw_file"], saved["raw_sha256"])
                skipped += 1
                continue
            if max_shards is not None and written >= max_shards:
                break
            originals = list(iter_raw_records(source_crawl_dir / part["raw_file"]))
            if len(originals) != part["records"]:
                raise ValueError("Source shard record count mismatch.")
            pairs = [{"id": row["doc_id"], "url": row["url"]} for row in originals]
            if digest_json(sorted(pairs, key=lambda row: row["id"])) != part["input_pairs_sha256"]:
                raise ValueError("Source shard ID/URL mapping mismatch.")
            candidates = [row for row in originals if _saved_challenge(row)]
            replacements, failures = {}, []
            for row in candidates:
                fresh = await fetcher.fetch({"id": row["doc_id"], "url": row["url"]})
                if fresh["status"] == "ok":
                    try:
                        body = base64.b64decode(fresh["body_base64"], validate=True)
                        article = extract_source(body, fresh.get("content_type", ""),
                                                 source_url=fresh.get("final_url") or fresh["url"])
                        if article["parser"] != "laodong-article-dom-v1" or len(article["source_text"]) < 300:
                            raise ValueError("refreshed_page_is_not_a_full_laodong_article")
                    except EXPECTED_PARSE_ERRORS as exc:
                        failures.append({"doc_id": row["doc_id"], "reason": str(exc)[:200]})
                        continue
                    fresh["recovery_original_body_sha256"] = row["body_sha256"]
                    fresh["recovery_source_run"] = source_crawl_dir.name
                    replacements[row["doc_id"]] = fresh
                else:
                    failures.append({"doc_id": row["doc_id"], "reason": fresh.get("error", "fetch_failed")})
            records = [replacements.get(row["doc_id"], row) for row in originals]
            statuses = Counter(row["status"] if row["status"] == "ok" else row.get("error", "unknown_error")
                               for row in records)
            capture_kinds = Counter(row.get("capture_kind", "unknown_capture")
                                    for row in records if row["status"] == "ok")
            raw_relative = (bucket / f"{part['part']}-recovered.raw.jsonl.gz").as_posix()
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / "recovered.raw.jsonl.gz"
                with local.open("wb") as stream:
                    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                        for row in records:
                            compressed.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                raw_sha = publish_file(local, target / raw_relative)
            checkpoint = {
                "part": part["part"], "first_input_row": part["first_input_row"],
                "records": part["records"], "run_signature": signature,
                "input_sha256": part["input_sha256"],
                "input_pairs_sha256": part["input_pairs_sha256"],
                "raw_file": raw_relative, "raw_sha256": raw_sha,
                "source_raw_sha256": part["raw_sha256"],
                "shard_attempt": 0,
                "failed_ids": [row["doc_id"] for row in records if row["status"] != "ok"],
                "statuses": dict(statuses), "capture_kinds": dict(capture_kinds),
                "attempted_count": len(candidates), "recovered_count": len(replacements),
                "recovery_failures": failures, "completed_at": utc_now(),
            }
            atomic_json(pointer, checkpoint)
            written += 1
        parts = completed_parts(target)
        summary = {
            "run_name": run_name, "source_run": source_crawl_dir.name,
            "requested_input_records": source_run["requested_input_records"],
            "recorded_documents": sum(part["records"] for part in parts),
            "range_complete": range_complete({**identity, "signature": signature}, parts),
            "target_challenges": sum(part["attempted_count"] for part in parts),
            "recovered_articles": sum(part["recovered_count"] for part in parts),
            "remaining_challenges": sum(part["attempted_count"] - part["recovered_count"] for part in parts),
            "complete_shards": len(parts), "available_source_shards": len(source_parts),
            "skipped_shards_this_call": skipped, "updated_at": utc_now(),
        }
        atomic_json(target / "summary.json", summary)
        return summary
    finally:
        await fetcher.client.aclose()
