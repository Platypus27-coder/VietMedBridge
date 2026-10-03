"""Bounded HTTP fetching with immutable shard attempts and Drive checkpoints."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .artifacts import (
    atomic_json, code_fingerprint, digest_json, local_workspace, publish_file, read_json,
    runtime_versions, sha256_file, utc_now, verify_file,
)
from .dataset import iter_link_shards, url_host
from .robots import RobotsDecision, RobotsResolver


@dataclass(frozen=True)
class CrawlConfig:
    concurrency: int = 4
    shard_size: int = 128
    per_host_delay: float = 1.0
    timeout_seconds: float = 30.0
    max_bytes: int = 16 * 1024 * 1024
    attempts: int = 3
    respect_robots: bool = True
    user_agent: str = "VietMedBridge/0.2 (+https://github.com/Platypus27-coder/VietMedBridge)"
    max_shard_bytes: int = 512 * 1024 * 1024

    def validate(self) -> None:
        if min(self.concurrency, self.shard_size, self.max_bytes, self.attempts) < 1:
            raise ValueError("Concurrency, shard size, byte limit and attempts must be positive.")
        if self.per_host_delay < 0 or self.timeout_seconds <= 0:
            raise ValueError("Invalid host delay or HTTP timeout.")
        if self.max_shard_bytes < self.max_bytes:
            raise ValueError("Shard byte budget must be at least the per-response byte limit.")


class SourceFailure(Exception):
    def __init__(self, code: str, metadata: dict | None = None):
        self.code = code
        self.metadata = metadata or {}
        super().__init__(code)


class Fetcher:
    def __init__(self, config: CrawlConfig, *, transport=None, enhancer=None):
        config.validate()
        self.config = config
        self.enhancer = enhancer
        self.client = httpx.AsyncClient(
            timeout=config.timeout_seconds, follow_redirects=False,
            headers={"User-Agent": config.user_agent},
            limits=httpx.Limits(max_connections=config.concurrency),
            transport=transport,
        )
        self.host_locks: dict[str, asyncio.Lock] = {}
        self.host_times: dict[str, float] = {}
        self.robots = RobotsResolver(self.client, self.pace, user_agent=config.user_agent)

    async def pace(self, url: str, minimum_delay: float = 0.0) -> None:
        host = url_host(url)
        lock = self.host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            delay = self.host_times.get(host, 0.0) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.host_times[host] = time.monotonic() + max(self.config.per_host_delay, minimum_delay)

    async def robots_decision(self, url: str) -> RobotsDecision:
        if not self.config.respect_robots:
            return RobotsDecision("ROBOTS_DISABLED", True, "", utc_now())
        return await self.robots.decision(url)

    async def allowed(self, url: str) -> bool:
        return (await self.robots_decision(url)).allowed

    async def request(self, url: str) -> tuple[dict, bytes]:
        current = url
        robots_checks = []
        for _ in range(9):
            url_host(current)
            decision = await self.robots_decision(current)
            robots_checks.append(decision.record())
            if not decision.allowed:
                error = ("robots_blocked" if decision.state == "ROBOTS_OK_DISALLOWED"
                         else decision.state.lower())
                raise SourceFailure(error, {**decision.record(), "robots_checks": robots_checks})
            await self.pace(current, max(
                decision.crawl_delay_seconds or 0.0,
                decision.request_rate_gap_seconds or 0.0,
            ))
            async with self.client.stream("GET", current) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise SourceFailure("redirect_without_location", {
                            **decision.record(), "robots_checks": robots_checks,
                        })
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response,
                    )
                    error.robots_metadata = {**decision.record(), "robots_checks": robots_checks}
                    raise error
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.config.max_bytes:
                        raise SourceFailure("body_too_large", {
                            **decision.record(), "robots_checks": robots_checks,
                        })
                if not body:
                    raise SourceFailure("empty_body", {
                        **decision.record(), "robots_checks": robots_checks,
                    })
                engine = getattr(self.client._transport, "engine_name", "httpx-async")
                capture_kind = "scrapling_http_entity" if engine.startswith("scrapling") else "http_entity"
                return {
                    "final_url": str(response.url),
                    "http_status": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "encoding": response.encoding,
                    "fetch_engine": engine,
                    "capture_kind": capture_kind,
                    **decision.record(),
                    "robots_checks": robots_checks,
                }, bytes(body)
        raise SourceFailure("too_many_redirects", {"robots_checks": robots_checks})

    async def fetch(self, row: dict) -> dict:
        started = time.monotonic()
        result = {"doc_id": int(row["id"]), "url": row["url"],
                  "fetched_at": utc_now(), "status": "error"}
        for attempt in range(self.config.attempts):
            try:
                metadata, body = await self.request(row["url"])
                if self.enhancer is not None:
                    try:
                        enhanced = await self.enhancer.enhance(
                            row["url"], metadata, body, guard=self,
                        )
                    except Exception as exc:
                        enhanced = {"body": None, "metadata": {"render_attempt": {
                            "engine": "crawl4ai", "state": "enhancer_error",
                            "error_type": type(exc).__name__, "error": str(exc)[:300],
                        }}}
                    if enhanced:
                        metadata.update(enhanced.get("metadata") or {})
                        selected_body = enhanced.get("body")
                        if isinstance(selected_body, (bytes, bytearray)) and selected_body:
                            selected_body = bytes(selected_body)
                            if selected_body != body:
                                metadata.update({
                                    "http_response_sha256": hashlib.sha256(body).hexdigest(),
                                    "http_response_bytes": len(body),
                                    "http_response_encoding": "base64-decoded-http-entity",
                                    "http_response_base64": base64.b64encode(body).decode("ascii"),
                                })
                                body = selected_body
                return {
                    **result, **metadata, "status": "ok", "attempts": attempt + 1,
                    "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
                    "body_encoding": (
                        "base64-rendered-dom-utf8"
                        if metadata.get("capture_kind") == "crawl4ai_rendered_dom"
                        else "base64-decoded-http-entity"
                    ),
                    "body_base64": base64.b64encode(body).decode("ascii"),
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                }
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                result.update(error=f"http_{code}", http_status=code)
                result.update(getattr(exc, "robots_metadata", {}))
                if code != 429 and code < 500:
                    break
                retry_after = exc.response.headers.get("retry-after", "")
                try:
                    wait = float(retry_after)
                except ValueError:
                    try:
                        wait = parsedate_to_datetime(retry_after).timestamp() - time.time()
                    except (ValueError, TypeError):
                        wait = 2.0 ** attempt
                wait = min(120.0, max(0.0, wait))
                self.host_times[url_host(row["url"])] = max(
                    self.host_times.get(url_host(row["url"]), 0.0), time.monotonic() + wait,
                )
                if attempt + 1 < self.config.attempts:
                    await asyncio.sleep(wait)
            except SourceFailure as exc:
                result["error"] = exc.code
                result.update(exc.metadata)
                break  # A new crawl window can re-probe after the robots cache expires.
            except (httpx.RequestError, ValueError) as exc:
                result["error"] = type(exc).__name__
                if attempt + 1 < self.config.attempts:
                    await asyncio.sleep(2.0 ** attempt)
        result["attempts"] = attempt + 1
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return result


def completed_parts(run_dir: str | Path, verify: bool = True) -> list[dict]:
    run_dir = Path(run_dir)
    parts = []
    for pointer in sorted(run_dir.rglob("part-*.done.json")):
        manifest = read_json(pointer)
        if verify:
            verify_file(run_dir / manifest["raw_file"], manifest["raw_sha256"])
        parts.append(manifest)
    return parts


def iter_raw_records(path: str | Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def range_complete(run: dict, parts: list[dict]) -> bool:
    """All requested physical rows need outcomes, including failures, without gaps."""
    requested = run["requested_range"]
    cursor = requested["start"]
    for part in sorted(parts, key=lambda row: row["first_input_row"]):
        if part["run_signature"] != run["signature"] or part["first_input_row"] != cursor:
            return False
        cursor += part["records"]
    return cursor == requested["stop"]


def read_completed_crawl(crawl_dir: str | Path, *, links_path: str | Path,
                         origin_corpus_sha256: str, start: int = 0,
                         stop: int | None = None) -> dict:
    """Verify and reuse a complete raw capture without resuming its old code.

    Code/runtime fingerprints remain on the original run. They may differ from
    the current extractor; this function only reads and verifies saved artifacts.
    """
    crawl_dir, links_path = Path(crawl_dir), Path(links_path)
    run = read_json(crawl_dir / "run.json")
    rows = pq.ParquetFile(links_path).metadata.num_rows
    if start < 0 or (stop is not None and stop < start):
        raise ValueError("Invalid requested row range.")
    effective_stop = min(rows, rows if stop is None else stop)
    if effective_stop <= start:
        raise ValueError("Requested row range is empty.")
    if run.get("links_sha256") != sha256_file(links_path):
        raise ValueError("Cached crawl belongs to another input file.")
    if run.get("origin_corpus_sha256") != origin_corpus_sha256:
        raise ValueError("Cached crawl belongs to another official corpus snapshot.")
    if (run.get("requested_range") != {"start": start, "stop": effective_stop}
            or run.get("requested_input_records") != effective_stop - start):
        raise ValueError("Cached crawl does not cover the requested range.")
    parts = completed_parts(crawl_dir)
    if not range_complete(run, parts):
        raise ValueError("Cached crawl is incomplete. Resume with its original pinned code or use a new run_name.")
    expected = iter_link_shards(
        links_path, start=start, stop=stop, shard_size=run["config"]["shard_size"],
    )
    counts, captures = Counter(), Counter()
    for part, (first, inputs) in zip(sorted(parts, key=lambda p: p["first_input_row"]), expected, strict=True):
        if (part["first_input_row"] != first or part["records"] != len(inputs)
                or part["input_sha256"] != digest_json(inputs)
                or part["input_pairs_sha256"] != digest_json(sorted(inputs, key=lambda r: r["id"]))):
            raise ValueError("Cached shard does not match the requested official ID/URL pairs.")
        if sum(part["statuses"].values()) != part["records"]:
            raise ValueError("Cached shard status counts do not match its records.")
        counts.update(part["statuses"])
        captures.update(part.get("capture_kinds", {}))
    return {
        "run_name": crawl_dir.name, "run_signature": run["signature"],
        "complete_shards": len(parts), "recorded_documents": sum(p["records"] for p in parts),
        "requested_input_records": run["requested_input_records"], "range_complete": True,
        "statuses": dict(counts), "capture_kinds": dict(captures),
        "reused_existing_run": True,
        "call": {"written_shards": 0, "requested_this_call": 0, "ok_this_call": 0,
                 "skipped_shards": len(parts)},
    }


async def crawl_links(
    links_path: str | Path, output_root: str | Path, *, run_name: str = "smoke-v1",
    config: CrawlConfig | None = None, start: int = 0, stop: int | None = None,
    worker_index: int = 0, worker_count: int = 1, retry_failed: bool = False,
    max_shards: int | None = None, work_dir: str | Path | None = None,
    origin_corpus_sha256: str | None = None, transport=None, enhancer=None,
) -> dict:
    """Resume complete shards; incomplete temporary files are never treated as results."""
    config = config or CrawlConfig()
    config.validate()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("Use a simple run name without slashes.")
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid worker partition.")
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be positive or None.")
    links_path, output_root = Path(links_path), Path(output_root)
    input_rows = pq.ParquetFile(links_path).metadata.num_rows
    if start < 0 or (stop is not None and stop < start):
        raise ValueError("Invalid requested row range.")
    effective_stop = min(input_rows, input_rows if stop is None else stop)
    if effective_stop <= start:
        raise ValueError("Requested row range is empty.")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "links_sha256": sha256_file(links_path),
        "origin_corpus_sha256": origin_corpus_sha256,
        "row_range": {"start": start, "stop": stop},
        "requested_range": {"start": start, "stop": effective_stop},
        "requested_input_records": effective_stop - start,
        "config": asdict(config), "crawler_version": (
            "hybrid-shards-v3" if enhancer is not None
            or getattr(transport, "engine_name", "httpx-async") != "httpx-async"
            else "http-shards-v3"
        ),
        "fetch_engine": getattr(transport, "engine_name", "httpx-async"),
        "enhancer": enhancer.identity if enhancer is not None else None,
        "code_sha256": code_fingerprint(),
        "runtime": runtime_versions(),
    }
    signature = digest_json(identity)
    run_manifest = run_dir / "run.json"
    if run_manifest.exists() and read_json(run_manifest)["signature"] != signature:
        raise ValueError("Input/config changed. Use a new run_name; do not mix checkpoints.")
    if not run_manifest.exists():
        atomic_json(run_manifest, {**identity, "signature": signature,
                                   "created_at": utc_now(), "runtime": runtime_versions()})
    totals = Counter()
    fetcher = Fetcher(config, transport=transport, enhancer=enhancer)
    try:
        shards = iter_link_shards(links_path, start=start, stop=stop, shard_size=config.shard_size)
        for index, (first, rows) in enumerate(tqdm(shards, desc="Crawl shards", unit="shard")):
            if index % worker_count != worker_index:
                continue
            part_name = f"part-{first:010d}-{len(rows):05d}"
            bucket = Path("parts") / f"{first // (config.shard_size * 128):06d}"
            pointer = run_dir / bucket / f"{part_name}.done.json"
            input_sha = digest_json(rows)
            old = read_json(pointer) if pointer.exists() else None
            if old:
                if old["input_sha256"] != input_sha or old["run_signature"] != signature:
                    raise ValueError(f"Checkpoint input mismatch: {pointer}")
                verify_file(run_dir / old["raw_file"], old["raw_sha256"])
                if not retry_failed or not old["failed_ids"]:
                    totals["skipped_shards"] += 1
                    continue
            if max_shards is not None and totals["written_shards"] >= max_shards:
                break
            attempt = old["shard_attempt"] + 1 if old else 0
            pending_ids = set(old["failed_ids"]) if old else {int(row["id"]) for row in rows}
            pending = [row for row in rows if int(row["id"]) in pending_ids]
            filename = (bucket / f"{part_name}-a{attempt:03d}.raw.jsonl.gz").as_posix()
            statuses, capture_kinds, failed, count, successful_this_call = Counter(), Counter(), [], 0, 0
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / Path(filename).name
                shard_bytes = 0
                with local.open("wb") as raw_file:
                    with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed:
                        def write(record):
                            nonlocal count, shard_bytes
                            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                            shard_bytes += len(encoded)
                            if shard_bytes > config.max_shard_bytes:
                                atomic_json(run_dir / bucket / f"{part_name}.failed.json", {
                                    "part": part_name, "reason": "SHARD_BYTE_BUDGET_EXCEEDED",
                                    "run_signature": signature, "recorded_before_failure": count,
                                    "status": "FAILED", "completed_at": utc_now(),
                                })
                                raise ValueError("Shard byte budget exceeded. Use a smaller shard_size in a new run; completed shards remain intact.")
                            compressed.write(encoded)
                            statuses[record["status"] if record["status"] == "ok"
                                     else record.get("error", "unknown_error")] += 1
                            if record["status"] == "ok":
                                capture_kinds[record.get("capture_kind", "unknown_capture")] += 1
                            if record["status"] != "ok":
                                failed.append(record["doc_id"])
                            count += 1
                        if old:
                            for record in iter_raw_records(run_dir / old["raw_file"]):
                                if record["status"] == "ok":
                                    write(record)
                        for lower in range(0, len(pending), config.concurrency):
                            results = await asyncio.gather(*(
                                fetcher.fetch(row)
                                for row in pending[lower:lower + config.concurrency]
                            ))
                            for result in results:
                                write(result)
                                successful_this_call += result["status"] == "ok"
                if count != len(rows):
                    raise ValueError("Incomplete shard: every input ID must produce a record.")
                checksum = publish_file(local, run_dir / filename)
            manifest = {
                "part": part_name, "first_input_row": first, "records": count,
                "run_signature": signature, "input_sha256": input_sha,
                "input_pairs_sha256": digest_json(sorted(rows, key=lambda row: row["id"])),
                "shard_attempt": attempt, "raw_file": filename, "raw_sha256": checksum,
                "failed_ids": sorted(failed), "statuses": dict(statuses),
                "capture_kinds": dict(capture_kinds), "completed_at": utc_now(),
            }
            # Attempt manifests remain immutable; only the latest-complete pointer changes.
            atomic_json(run_dir / bucket / f"{part_name}-a{attempt:03d}.manifest.json", manifest)
            atomic_json(pointer, manifest)
            totals["written_shards"] += 1
            totals["requested_this_call"] += len(pending)
            totals["ok_this_call"] += successful_this_call
        parts = completed_parts(run_dir, verify=False)
        counts = Counter()
        capture_counts = Counter()
        for part in parts:
            counts.update(part["statuses"])
            capture_counts.update(part.get("capture_kinds", {}))
        summary = {
            "run_name": run_name, "run_signature": signature, "complete_shards": len(parts),
            "recorded_documents": sum(item["records"] for item in parts),
            "requested_input_records": identity["requested_input_records"],
            "range_complete": range_complete({**identity, "signature": signature}, parts),
            "statuses": dict(counts), "capture_kinds": dict(capture_counts),
            "call": dict(totals), "updated_at": utc_now(),
        }
        atomic_json(run_dir / "summary.json", summary)
        return summary
    finally:
        await fetcher.client.aclose()
