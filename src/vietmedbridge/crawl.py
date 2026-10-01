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
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from tqdm.auto import tqdm

from .artifacts import (
    atomic_json, code_fingerprint, digest_json, local_workspace, publish_file, read_json,
    runtime_versions, sha256_file, utc_now, verify_file,
)
from .dataset import iter_link_shards, url_host


@dataclass(frozen=True)
class CrawlConfig:
    concurrency: int = 4
    shard_size: int = 128
    per_host_delay: float = 1.0
    timeout_seconds: float = 30.0
    max_bytes: int = 16 * 1024 * 1024
    attempts: int = 3
    respect_robots: bool = True
    user_agent: str = "VietMedBridge/0.1 (+https://github.com/Platypus27-coder/VietMedBridge)"

    def validate(self) -> None:
        if min(self.concurrency, self.shard_size, self.max_bytes, self.attempts) < 1:
            raise ValueError("Concurrency, shard size, byte limit and attempts must be positive.")
        if self.per_host_delay < 0 or self.timeout_seconds <= 0:
            raise ValueError("Invalid host delay or HTTP timeout.")


class SourceFailure(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Fetcher:
    def __init__(self, config: CrawlConfig, *, transport=None):
        config.validate()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_seconds, follow_redirects=False,
            headers={"User-Agent": config.user_agent},
            limits=httpx.Limits(max_connections=config.concurrency),
            transport=transport,
        )
        self.host_locks: dict[str, asyncio.Lock] = {}
        self.host_times: dict[str, float] = {}
        self.robot_locks: dict[str, asyncio.Lock] = {}
        self.robots: dict[str, RobotFileParser | bool] = {}

    async def pace(self, url: str) -> None:
        host = url_host(url)
        lock = self.host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            delay = self.host_times.get(host, 0.0) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.host_times[host] = time.monotonic() + self.config.per_host_delay

    async def allowed(self, url: str) -> bool:
        if not self.config.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self.robot_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self.robots:
                robots_url = origin + "/robots.txt"
                for redirect in range(9):
                    url_host(robots_url)
                    await self.pace(robots_url)
                    async with self.client.stream("GET", robots_url) as response:
                        if response.status_code in (301, 302, 303, 307, 308):
                            location = response.headers.get("location")
                            if not location:
                                raise SourceFailure("robots_unavailable")
                            robots_url = urljoin(robots_url, location)
                            continue
                        if response.status_code in (404, 410):
                            self.robots[origin] = True
                        elif response.status_code in (401, 403):
                            self.robots[origin] = False
                        elif response.status_code != 200:
                            raise SourceFailure("robots_unavailable")
                        else:
                            body = bytearray()
                            async for chunk in response.aiter_bytes():
                                body.extend(chunk)
                                if len(body) > 512 * 1024:
                                    raise SourceFailure("robots_too_large")
                            parser = RobotFileParser(robots_url)
                            parser.parse(body.decode("utf-8", errors="replace").splitlines())
                            self.robots[origin] = parser
                        break
                else:
                    raise SourceFailure("robots_too_many_redirects")
        policy = self.robots[origin]
        return policy if isinstance(policy, bool) else policy.can_fetch("VietMedBridge", url)

    async def request(self, url: str) -> tuple[dict, bytes]:
        current = url
        for _ in range(9):
            url_host(current)
            if not await self.allowed(current):
                raise SourceFailure("robots_blocked")
            await self.pace(current)
            async with self.client.stream("GET", current) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise SourceFailure("redirect_without_location")
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response,
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.config.max_bytes:
                        raise SourceFailure("body_too_large")
                if not body:
                    raise SourceFailure("empty_body")
                return {
                    "final_url": str(response.url),
                    "http_status": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "encoding": response.encoding,
                }, bytes(body)
        raise SourceFailure("too_many_redirects")

    async def fetch(self, row: dict) -> dict:
        result = {"doc_id": int(row["id"]), "url": row["url"],
                  "fetched_at": utc_now(), "status": "error"}
        for attempt in range(self.config.attempts):
            try:
                metadata, body = await self.request(row["url"])
                return {
                    **result, **metadata, "status": "ok", "attempts": attempt + 1,
                    "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
                    "body_encoding": "base64-decoded-http-entity",
                    "body_base64": base64.b64encode(body).decode("ascii"),
                }
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                result.update(error=f"http_{code}", http_status=code)
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
                if exc.code != "robots_unavailable":
                    break
                if attempt + 1 < self.config.attempts:
                    await asyncio.sleep(2.0 ** attempt)
            except (httpx.RequestError, ValueError) as exc:
                result["error"] = type(exc).__name__
                if attempt + 1 < self.config.attempts:
                    await asyncio.sleep(2.0 ** attempt)
        result["attempts"] = attempt + 1
        return result


def completed_parts(run_dir: str | Path, verify: bool = True) -> list[dict]:
    run_dir = Path(run_dir)
    parts = []
    for pointer in sorted(run_dir.glob("part-*.done.json")):
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


async def crawl_links(
    links_path: str | Path, output_root: str | Path, *, run_name: str = "smoke-v1",
    config: CrawlConfig | None = None, start: int = 0, stop: int | None = None,
    worker_index: int = 0, worker_count: int = 1, retry_failed: bool = False,
    max_shards: int | None = None, work_dir: str | Path | None = None,
    origin_corpus_sha256: str | None = None, transport=None,
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
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "links_sha256": sha256_file(links_path),
        "origin_corpus_sha256": origin_corpus_sha256,
        "row_range": {"start": start, "stop": stop},
        "config": asdict(config), "crawler_version": "http-shards-v1",
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
    fetcher = Fetcher(config, transport=transport)
    try:
        shards = iter_link_shards(links_path, start=start, stop=stop, shard_size=config.shard_size)
        for index, (first, rows) in enumerate(tqdm(shards, desc="Crawl shards", unit="shard")):
            if index % worker_count != worker_index:
                continue
            part_name = f"part-{first:010d}-{len(rows):05d}"
            pointer = run_dir / f"{part_name}.done.json"
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
            filename = f"{part_name}-a{attempt:03d}.raw.jsonl.gz"
            statuses, failed, count, successful_this_call = Counter(), [], 0, 0
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / filename
                with local.open("wb") as raw_file:
                    with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed:
                        def write(record):
                            nonlocal count
                            compressed.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                            statuses[record["status"] if record["status"] == "ok"
                                     else record.get("error", "unknown_error")] += 1
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
                "shard_attempt": attempt, "raw_file": filename, "raw_sha256": checksum,
                "failed_ids": sorted(failed), "statuses": dict(statuses), "completed_at": utc_now(),
            }
            # Attempt manifests remain immutable; only the latest-complete pointer changes.
            atomic_json(run_dir / f"{part_name}-a{attempt:03d}.manifest.json", manifest)
            atomic_json(pointer, manifest)
            totals["written_shards"] += 1
            totals["requested_this_call"] += len(pending)
            totals["ok_this_call"] += successful_this_call
        parts = completed_parts(run_dir, verify=False)
        counts = Counter()
        for part in parts:
            counts.update(part["statuses"])
        summary = {
            "run_name": run_name, "run_signature": signature, "complete_shards": len(parts),
            "recorded_documents": sum(item["records"] for item in parts),
            "statuses": dict(counts), "call": dict(totals), "updated_at": utc_now(),
        }
        atomic_json(run_dir / "summary.json", summary)
        return summary
    finally:
        await fetcher.client.aclose()
