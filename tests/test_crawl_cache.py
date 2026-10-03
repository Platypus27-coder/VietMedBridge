"""Reuse immutable completed captures without mixing old and new crawl signatures."""

import asyncio

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vietmedbridge.artifacts import atomic_json, read_json
from vietmedbridge.crawl import CrawlConfig, completed_parts, crawl_links, read_completed_crawl


def capture(tmp_path, *, max_shards=None):
    links = tmp_path / "links.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": 7, "url": "https://example.test/article"},
        {"id": 41, "url": "https://example.test/forbidden"},
    ]), links)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path == "/forbidden":
            return httpx.Response(403)
        return httpx.Response(200, content=b"Saved article", headers={"content-type": "text/plain"})

    asyncio.run(crawl_links(
        links, tmp_path / "crawl", run_name="stage-a-v2",
        config=CrawlConfig(shard_size=1, attempts=1, per_host_delay=0),
        transport=httpx.MockTransport(handler), work_dir=tmp_path,
        origin_corpus_sha256="official-snapshot", max_shards=max_shards,
    ))
    return links, tmp_path / "crawl/stage-a-v2", calls


def test_complete_old_capture_is_read_only_after_code_changes(tmp_path, monkeypatch):
    links, root, calls = capture(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    requests_before = list(calls)
    monkeypatch.setattr("vietmedbridge.crawl.code_fingerprint", lambda: "new-extractor-code")
    summary = read_completed_crawl(root, links_path=links, origin_corpus_sha256="official-snapshot")
    assert summary["reused_existing_run"] and summary["range_complete"]
    assert summary["statuses"] == {"ok": 1, "http_403": 1}
    assert summary["recorded_documents"] == summary["requested_input_records"] == 2
    assert summary["run_signature"] == read_json(root / "run.json")["signature"]
    assert summary["call"]["requested_this_call"] == 0
    assert calls == requests_before
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_incomplete_capture_cannot_be_reported_as_complete(tmp_path):
    links, root, _ = capture(tmp_path, max_shards=1)
    with pytest.raises(ValueError, match="incomplete"):
        read_completed_crawl(root, links_path=links, origin_corpus_sha256="official-snapshot")


@pytest.mark.parametrize("change, message", [
    ("corpus", "another official corpus"),
    ("input", "another input file"),
    ("range", "requested range"),
    ("pairs", "official ID/URL pairs"),
    ("raw", "Missing or changed artifact"),
])
def test_reuse_rejects_changed_provenance_or_damaged_artifacts(tmp_path, change, message):
    links, root, _ = capture(tmp_path)
    kwargs = {"links_path": links, "origin_corpus_sha256": "official-snapshot"}
    if change == "corpus":
        kwargs["origin_corpus_sha256"] = "other-snapshot"
    elif change == "input":
        other = tmp_path / "other.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": 8, "url": "https://example.test/other"}]), other)
        kwargs["links_path"] = other
    elif change == "range":
        kwargs["start"] = 1
    else:
        part = completed_parts(root)[0]
        if change == "pairs":
            part["input_pairs_sha256"] = "changed-pair"
            atomic_json(next(root.rglob("part-0000000000-*.done.json")), part)
        else:
            raw = root / part["raw_file"]
            raw.write_bytes(raw.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match=message):
        read_completed_crawl(root, **kwargs)
