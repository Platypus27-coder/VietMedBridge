"""A saved HTTP 200 cookie page must be recovered as the right source article."""

import asyncio
import base64
import gzip
import hashlib
import json
from dataclasses import asdict

import httpx
import pytest

from vietmedbridge.artifacts import atomic_json, digest_json, sha256_file
from vietmedbridge.crawl import CrawlConfig, Fetcher, completed_parts, iter_raw_records
from vietmedbridge.parse_recovery import recover_laodong_challenges
from vietmedbridge.source_challenges import laodong_cookie_challenge
from vietmedbridge.text import extract_source


URL = "https://laodong.vn/suc-khoe/bai-viet-y-khoa-12345.ldo"
CHALLENGE = (b'<html><body><script>document.cookie="D1N=0123456789abcdef0123456789abcdef"+'
             b'"; expires=Fri, 31 Dec 2099 23:59:59 GMT; path=/";'
             b'window.location.reload(true);</script></body></html>')
ARTICLE = f'''<!doctype html><html lang="vi"><head><title>Nghiên cứu điều trị</title></head>
<body><nav>Đăng nhập và quảng cáo</nav><div itemprop="articleBody" articleid="12345">
<article class="article-detail"><div class="wrapper-01"><h1 class="title">Nghiên cứu điều trị</h1></div>
<div class="chappeau">Các bác sĩ theo dõi triệu chứng ở người bệnh.</div>
<div class="art-body"><p>Liều 1<span>.5</span>mg được theo dõi trong nghiên cứu.
Người bệnh được đánh giá thường xuyên bằng các chỉ số lâm sàng. Dữ liệu điều trị
được ghi nhận theo từng giai đoạn và bác sĩ quyết định dựa trên kết quả khám.</p>
<h2>Kết quả</h2><p>Bệnh nhân cần tiếp tục theo dõi sau khi điều trị. Đây là thông tin
thuộc chính bài báo và phải được giữ trong văn bản nguồn đã trích xuất.</p></div>
<div class="related"><p>Bài viết liên quan không được trích.</p></div></article></div>
<article><p>Tin khác không được trích.</p></article></body></html>'''.encode("utf-8")


def _old_record(doc_id, url, body):
    return {"doc_id": doc_id, "url": url, "final_url": url, "status": "ok",
            "http_status": 200, "content_type": "text/html", "body_bytes": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_base64": base64.b64encode(body).decode("ascii"),
            "fetched_at": "2026-10-01T00:00:00+00:00"}


def test_laodong_adapter_excludes_ui_and_rejects_cookie_page():
    assert laodong_cookie_challenge(CHALLENGE, URL) == "D1N=0123456789abcdef0123456789abcdef"
    assert laodong_cookie_challenge(CHALLENGE, "https://elsewhere.test/a") is None
    with pytest.raises(ValueError, match="requires_recrawl"):
        extract_source(CHALLENGE, "text/html", source_url=URL)
    result = extract_source(ARTICLE, "text/html", source_url=URL)
    assert result["parser"] == "laodong-article-dom-v1"
    assert "Liều 1.5mg" in result["source_text"]
    assert "Các bác sĩ theo dõi" in result["source_text"]
    assert "Kết quả" in result["source_text"]
    assert all(x not in result["source_text"] for x in (
        "Đăng nhập và quảng cáo", "Bài viết liên quan", "Tin khác",
    ))
    with pytest.raises(ValueError, match="article_id_or_layout_changed"):
        extract_source(ARTICLE, "text/html", source_url=URL.replace("12345", "54321"))


def test_fetcher_retries_only_observed_cookie_interstitial():
    calls = []

    def handler(request):
        calls.append((request.url.path, request.headers.get("cookie")))
        if request.url.path == "/robots.txt":
            if not request.headers.get("cookie"):
                return httpx.Response(200, content=CHALLENGE, headers={"content-type": "text/html"})
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.headers.get("cookie") == "D1N=0123456789abcdef0123456789abcdef":
            return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})
        return httpx.Response(200, content=CHALLENGE, headers={"content-type": "text/html"})

    async def run():
        fetcher = Fetcher(CrawlConfig(per_host_delay=0, attempts=1), transport=httpx.MockTransport(handler))
        try:
            return await fetcher.fetch({"id": 12345, "url": URL})
        finally:
            await fetcher.client.aclose()

    record = asyncio.run(run())
    assert record["status"] == "ok"
    assert record["challenge_kind"] == "laodong-d1n-cookie-v1"
    assert record["challenge_response_sha256"] == hashlib.sha256(CHALLENGE).hexdigest()
    assert base64.b64decode(record["body_base64"]) == ARTICLE
    assert [cookie for path, cookie in calls if path != "/robots.txt"] == [
        None, "D1N=0123456789abcdef0123456789abcdef",
    ]
    assert [cookie for path, cookie in calls if path == "/robots.txt"] == [
        None, "D1N=0123456789abcdef0123456789abcdef",
    ]


def test_recovery_keeps_old_ids_and_replaces_only_verified_article(tmp_path):
    source = tmp_path / "crawl" / "stage-a-v2"
    bucket = source / "parts" / "000000"
    bucket.mkdir(parents=True)
    rows = [{"id": 12345, "url": URL}, {"id": 9, "url": "https://example.org/article"}]
    other = _old_record(9, rows[1]["url"], b"<html><body><article>Unchanged.</article></body></html>")
    records = [_old_record(12345, URL, CHALLENGE), other]
    raw = bucket / "part-0000000000-00002-a000.raw.jsonl.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    config = CrawlConfig(shard_size=2, per_host_delay=0, attempts=1)
    atomic_json(source / "run.json", {
        "signature": "old-signature", "links_sha256": "links-hash",
        "origin_corpus_sha256": "corpus-hash", "requested_range": {"start": 0, "stop": 2},
        "requested_input_records": 2, "config": asdict(config),
    })
    atomic_json(bucket / "part-0000000000-00002.done.json", {
        "part": "part-0000000000-00002", "first_input_row": 0, "records": 2,
        "run_signature": "old-signature", "input_sha256": digest_json(rows),
        "input_pairs_sha256": digest_json(sorted(rows, key=lambda row: row["id"])),
        "raw_file": raw.relative_to(source).as_posix(), "raw_sha256": sha256_file(raw),
        "failed_ids": [], "statuses": {"ok": 2}, "shard_attempt": 0,
    })

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.headers.get("cookie"):
            return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})
        return httpx.Response(200, content=CHALLENGE, headers={"content-type": "text/html"})

    kwargs = {"run_name": "repaired", "config": config, "work_dir": tmp_path,
              "transport": httpx.MockTransport(handler)}
    summary = asyncio.run(recover_laodong_challenges(source, source.parent, **kwargs))
    assert summary["range_complete"] and summary["recorded_documents"] == 2
    assert summary["target_challenges"] == summary["recovered_articles"] == 1
    part = completed_parts(source.parent / "repaired")[0]
    output = list(iter_raw_records(source.parent / "repaired" / part["raw_file"]))
    assert [record["doc_id"] for record in output] == [12345, 9]
    assert output[1] == other
    assert base64.b64decode(output[0]["body_base64"]) == ARTICLE
    again = asyncio.run(recover_laodong_challenges(source, source.parent, **kwargs))
    assert again["skipped_shards_this_call"] == 1

    def unresolved(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=CHALLENGE, headers={"content-type": "text/html"})

    failed = asyncio.run(recover_laodong_challenges(
        source, source.parent, run_name="unresolved", config=config,
        work_dir=tmp_path, transport=httpx.MockTransport(unresolved),
    ))
    assert failed["target_challenges"] == 1 and failed["recovered_articles"] == 0
    failed_part = completed_parts(source.parent / "unresolved")[0]
    failed_rows = list(iter_raw_records(source.parent / "unresolved" / failed_part["raw_file"]))
    assert failed_rows == records
