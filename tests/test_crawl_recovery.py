import asyncio
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vietmedbridge.crawl import CrawlConfig, crawl_links
from scripts.crawl_recovery import prepare_crawl_recovery_source, recover_crawl_failures


CORPUS_SHA = "a" * 64
ARTICLE = ("<html lang='vi'><head><title>Nghiên cứu y sinh</title></head><body><article>"
           "<h1>Nghiên cứu y sinh</h1><p>" +
           "Nội dung về chẩn đoán, theo dõi và điều trị trong nghiên cứu lâm sàng. " * 10 +
           "</p></article></body></html>").encode()


class FakeAdvancedRuntime:
    identity = {"engine": "test-double"}

    def configure_guard(self, _guard):
        pass

    async def read_robots(self, _url, **_kwargs):
        return None


def response(url, status, body, content_type="text/html"):
    return SimpleNamespace(url=url, status=status, body=body,
                           headers={"content-type": content_type})


async def no_sleep(_seconds):
    pass


def make_baseline(tmp_path, *, disallow=False):
    rows = [
        {"id": 583, "url": "https://source.test/ok"},
        {"id": 1009, "url": "https://source.test/fail-one"},
        {"id": 1000000, "url": "https://source.test/fail-two"},
    ]
    if disallow:
        rows.append({"id": 2000000, "url": "https://source.test/deny/article"})
    links = tmp_path / "links.parquet"
    pq.write_table(pa.Table.from_pylist(rows), links)

    def handler(request):
        if request.url.path == "/robots.txt":
            robots = "User-agent: *\nDisallow: /deny\nAllow: /\n" if disallow else "User-agent: *\nAllow: /\n"
            return httpx.Response(200, text=robots)
        if request.url.path == "/ok":
            return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})
        return httpx.Response(404, content=b"missing", headers={"content-type": "text/plain"})

    summary = asyncio.run(crawl_links(
        links, tmp_path / "data/crawl", run_name="range-v1",
        config=CrawlConfig(concurrency=2, shard_size=2, per_host_delay=0, attempts=1),
        start=0, stop=len(rows), origin_corpus_sha256=CORPUS_SHA,
        transport=httpx.MockTransport(handler), work_dir=tmp_path,
    ))
    return links, rows, summary


def test_preflight_indexes_only_failures_after_proving_every_id_url(tmp_path):
    links, rows, _summary = make_baseline(tmp_path, disallow=True)
    source = prepare_crawl_recovery_source(tmp_path / "data", links, "range-v1",
        expected_origin_corpus_sha256=CORPUS_SHA, work_dir=tmp_path)
    assert source["range_complete"] is True
    assert source["official_ids"] == len(rows)
    assert source["baseline_successes"] == 1
    assert source["failed_ids"] == 3
    assert source["pending_recovery_ids"] == 2
    assert source["robots_policy_review_ids"] == 1
    with gzip.open(source["failures_path"], "rt", encoding="utf-8") as stream:
        failures = [json.loads(line) for line in stream]
    assert {row["doc_id"] for row in failures} == {1009, 1000000, 2000000}
    denied = next(row for row in failures if row["doc_id"] == 2000000)
    assert denied["state"] == "ROBOTS_POLICY_REVIEW"
    assert denied["prior_robots_state"] == "ROBOTS_OK_DISALLOWED"

    changed = tmp_path / "changed.parquet"
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["url"] += "?changed=1"
    pq.write_table(pa.Table.from_pylist(changed_rows), changed)
    with pytest.raises(ValueError, match="differs from the input"):
        prepare_crawl_recovery_source(tmp_path / "data", changed, "range-v1",
            expected_origin_corpus_sha256=CORPUS_SHA, work_dir=tmp_path)


def test_recovery_resumes_completed_batches_without_refetching(tmp_path):
    links, _rows, _summary = make_baseline(tmp_path)
    source = prepare_crawl_recovery_source(tmp_path / "data", links, "range-v1",
        expected_origin_corpus_sha256=CORPUS_SHA, work_dir=tmp_path)
    article_calls = Counter()

    def get(url, **_kwargs):
        if url.endswith("/robots.txt"):
            return response(url, 200, b"User-agent: *\nAllow: /\n", "text/plain")
        article_calls[url] += 1
        return response(url, 200, ARTICLE)

    runtime = FakeAdvancedRuntime()
    kwargs = dict(advanced_runtime=runtime, versions={"test": "1"}, batch_size=1,
                  max_batches_this_session=1, sleep=no_sleep)
    first = asyncio.run(recover_crawl_failures(
        source, tmp_path / "data/reports/recovery", get, **kwargs))
    assert first["recovery_complete"] is False
    assert first["evaluated_failed_ids"] == 1
    assert first["pending_or_incomplete_failed_ids"] == 1
    assert sum(article_calls.values()) == 1

    second = asyncio.run(recover_crawl_failures(
        source, tmp_path / "data/reports/recovery", get, **kwargs))
    assert second["recovery_complete"] is True
    assert second["evaluated_failed_ids"] == 2
    assert second["baseline_http_captures"] == 1
    assert second["failure_states"] == {"ARTICLE_CANDIDATE_REVIEW": 2}
    assert len(article_calls) == 2 and set(article_calls.values()) == {1}
    with Path(second["failed_id_coverage_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["doc_id"]) for row in rows} == {1009, 1000000}
    assert {row["recovery_state"] for row in rows} == {"ARTICLE_CANDIDATE_REVIEW"}
    assert Path(second["coverage_by_domain_csv"]).is_file()
    assert Path(second["candidate_review_csv"]).is_file()
    assert len(list(csv.DictReader(Path(second["candidate_review_csv"]).open(encoding="utf-8-sig")))) == 2
    summary = json.loads((Path(second["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert "records" not in summary and "coverage_ledger" not in summary


def test_empty_failures_need_no_browser_runtime(tmp_path):
    links = tmp_path / "links.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": 9, "url": "https://source.test/ok"},
    ]), links)

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})

    asyncio.run(crawl_links(links, tmp_path / "data/crawl", run_name="all-ok",
        config=CrawlConfig(shard_size=1, per_host_delay=0), origin_corpus_sha256=CORPUS_SHA,
        transport=httpx.MockTransport(handler), work_dir=tmp_path))
    source = prepare_crawl_recovery_source(tmp_path / "data", links, "all-ok",
        expected_origin_corpus_sha256=CORPUS_SHA, work_dir=tmp_path)
    result = asyncio.run(recover_crawl_failures(source, tmp_path / "data/reports/recovery",
        lambda *_args, **_kwargs: pytest.fail("no request expected"), advanced_runtime=None))
    assert result["recovery_complete"] is True
    assert result["failed_id_coverage_sha256"]
    assert result["baseline_http_captures"] == 1
    assert result["failure_states"] == {}
