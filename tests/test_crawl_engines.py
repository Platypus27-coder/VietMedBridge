"""Unit coverage for the Scrapling transport and selective Crawl4AI path."""

import asyncio
import base64
import gzip
import hashlib
import json

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from vietmedbridge.crawl import CrawlConfig, Fetcher, crawl_links
from vietmedbridge.crawl4ai_enhancer import Crawl4AIEnhancer, looks_like_javascript_shell
from vietmedbridge.scrapling_transport import ScraplingTransport


def test_scrapling_transport_adapts_response_and_closes_session():
    class Response:
        status = 200
        url = "https://example.test/article"
        headers = {"content-type": "text/html", "content-encoding": "gzip"}
        body = b"<html>decoded</html>"

    class Session:
        calls = []

        async def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    class Manager:
        entered = False
        exited = False

        async def __aenter__(self):
            self.entered = True
            return Session()

        async def __aexit__(self, *_args):
            self.exited = True

    manager = Manager()
    transport = ScraplingTransport(
        user_agent="VietMedBridge-test", session_factory=lambda: manager,
    )

    async def run():
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            response = await client.get("https://example.test/article#section")
            assert response.status_code == 200
            assert response.content == b"<html>decoded</html>"
            assert "content-encoding" not in response.headers
        assert manager.entered and manager.exited

    asyncio.run(run())
    url, kwargs = Session.calls[0]
    assert url == "https://example.test/article"
    assert kwargs["follow_redirects"] is False and kwargs["retries"] == 0
    assert kwargs["stealthy_headers"] is False


def test_javascript_shell_selector_excludes_normal_and_challenge_pages():
    shell = b'<html><head><title>News</title></head><body><div id="root"></div><script src="/app.js"></script></body></html>'
    article = b'<html><body><div id="root"><p>' + (b"A full server rendered article. " * 20) + b'</p></div><script src="/app.js"></script></body></html>'
    challenge = b'<html><head><title>Access Denied</title></head><body><div id="root"></div><script></script></body></html>'
    assert looks_like_javascript_shell(shell, "text/html")
    assert not looks_like_javascript_shell(article, "text/html")
    assert not looks_like_javascript_shell(challenge, "text/html")
    assert not looks_like_javascript_shell(shell, "application/pdf")


def test_crawl4ai_enhancer_only_selects_improved_safe_render(monkeypatch):
    calls = []
    rendered = (
        "<html><head><title>Rendered article</title></head><body><article><h1>Rendered article</h1>"
        f"<p>{'Rendered medical article with useful source text. ' * 30}</p></article></body></html>"
    ).encode()

    async def fake_probe(url, guard, **kwargs):
        calls.append((url, kwargs))
        return ({"http_status": 200, "rendered_http_status": 200,
                 "rendered_url": url, "guard_errors": [], "held_hosts": [],
                 "request_log": [], "response_evidence": {}},
                {"rendered.html": rendered})

    monkeypatch.setattr("vietmedbridge.crawl4ai_probe.crawl4ai_browser_probe", fake_probe)
    enhancer = Crawl4AIEnhancer(max_render_pages_per_call=1, min_rendered_chars=200)
    shell = b'<html><head><title>Medical article</title></head><body><div id="root"></div><script src="/app.js"></script></body></html>'

    async def run():
        guard = Fetcher(CrawlConfig(per_host_delay=0))
        try:
            selected = await enhancer.enhance(
                "https://example.test/article",
                {"http_status": 200, "content_type": "text/html"}, shell, guard,
            )
            blocked = await enhancer.enhance(
                "https://example.test/blocked",
                {"http_status": 403, "content_type": "text/html"}, shell, guard,
            )
            return selected, blocked
        finally:
            await guard.client.aclose()

    selected, blocked = asyncio.run(run())
    assert selected["body"] == rendered
    assert selected["metadata"]["capture_kind"] == "crawl4ai_rendered_dom"
    assert selected["metadata"]["render_attempt"]["state"] == "render_selected"
    assert blocked is None and len(calls) == 1
    assert calls[0][1]["user_agent"].startswith("VietMedBridge/")


def test_checkpointed_crawl_keeps_original_and_selected_rendered_body(tmp_path):
    original = b'<html><head><title>Thin</title></head><body><div id="root"></div><script></script></body></html>'
    selected = b"<html><title>Rendered</title><body><article>Readable article</article></body></html>"

    class Renderer:
        identity = {"engine": "test-renderer-v1"}

        async def enhance(self, _url, _metadata, _body, *, guard):
            return {"body": selected, "metadata": {
                "capture_kind": "crawl4ai_rendered_dom", "render_attempt": {"state": "render_selected"},
            }}

    def response(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=original, headers={"content-type": "text/html"})

    links = tmp_path / "links.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": 7, "url": "https://example.test/article"}]), links)
    summary = asyncio.run(crawl_links(
        links, tmp_path / "crawl", run_name="engine-test-v1",
        config=CrawlConfig(attempts=1, per_host_delay=0, shard_size=1),
        transport=httpx.MockTransport(response), enhancer=Renderer(), work_dir=tmp_path,
    ))
    assert summary["capture_kinds"] == {"crawl4ai_rendered_dom": 1}
    run_dir = tmp_path / "crawl" / "engine-test-v1"
    manifest = json.loads(next(run_dir.rglob("part-*.done.json")).read_text())
    record, = (json.loads(line) for line in gzip.open(run_dir / manifest["raw_file"], "rt"))
    assert base64.b64decode(record["body_base64"]) == selected
    assert record["body_sha256"] == hashlib.sha256(selected).hexdigest()
    assert record["body_encoding"] == "base64-rendered-dom-utf8"
    assert base64.b64decode(record["http_response_base64"]) == original
    assert record["http_response_sha256"] == hashlib.sha256(original).hexdigest()
    run_info = json.loads((run_dir / "run.json").read_text())
    assert run_info["fetch_engine"] == "httpx-async"
    assert run_info["enhancer"] == Renderer.identity
