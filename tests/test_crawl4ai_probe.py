import asyncio
import base64
import importlib.util
import os
from types import SimpleNamespace

import httpx
import pytest

from vietmedbridge.browser_probe import BrowserProbeConfig
from vietmedbridge.crawl import CrawlConfig, Fetcher
from vietmedbridge.crawl4ai_probe import crawl4ai_browser_probe


URL = "https://crawl4ai-probe.test/start"


def test_crawl4ai_setup_failure_cannot_navigate():
    class CDP:
        async def send(self, name, *_args):
            if name == "Page.getFrameTree":
                return {"frameTree": {"frame": {"id": "main"}}}
            if name == "Fetch.enable":
                raise RuntimeError("Cannot install interception")
        def on(self, *_args):
            pass
        async def detach(self):
            pass
    class Page:
        closed = False
        async def add_init_script(self, *_args):
            pass
        async def evaluate(self, *_args):
            return "Mozilla/5.0"
        def is_closed(self):
            return self.closed
        async def close(self):
            self.closed = True
    class Context:
        async def new_cdp_session(self, _page):
            return CDP()
    class Crawler:
        navigated = False
        hooks = {}
        page = Page()
        def __init__(self):
            self.crawler_strategy = self
        def set_hook(self, name, hook):
            self.hooks[name] = hook
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            pass
        async def arun(self, url, config):
            await self.hooks["before_goto"](self.page, Context(), url)
            self.navigated = True
    crawler = Crawler()
    async def run():
        guard = Fetcher(CrawlConfig())
        try:
            return await crawl4ai_browser_probe(URL, guard, crawler_factory=lambda: (crawler, None))
        finally:
            await guard.client.aclose()
    record, assets = asyncio.run(run())
    assert not crawler.navigated and crawler.page.closed
    assert "request_guard_setup_incomplete" in record["guard_errors"]
    assert not assets


def test_crawl4ai_engine_success_without_guard_is_not_a_capture():
    class Crawler:
        def __init__(self):
            self.crawler_strategy = self
        def set_hook(self, *_args):
            pass  # Simulate a library path that forgets hooks entirely.
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            pass
        async def arun(self, **_kwargs):
            return SimpleNamespace(success=True, status_code=200, error_message="")
    async def run():
        guard = Fetcher(CrawlConfig())
        try:
            return await crawl4ai_browser_probe(URL, guard, crawler_factory=lambda: (Crawler(), None))
        finally:
            await guard.client.aclose()
    record, assets = asyncio.run(run())
    assert record["engine_success"] and not record["article_candidate"]
    assert record["guard_errors"] and not assets


@pytest.mark.skipif(not importlib.util.find_spec("crawl4ai") or not os.environ.get("VMB_TEST_BROWSER_EXE"),
                    reason="Optional Crawl4AI/Edge integration")
@pytest.mark.parametrize("disallow", [True, False])
def test_real_crawl4ai_redirect_gate_and_http_error_bytes(disallow):
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    class SyntheticCDP:
        def __init__(self, cdp):
            self.cdp, self.urls = cdp, {}
        def on(self, name, callback):
            def wrapped(event):
                self.urls[event["requestId"]] = event["request"]["url"]
                callback(event)
            self.cdp.on(name, wrapped)
        async def send(self, name, params=None):
            if name == "Fetch.continueRequest":
                if self.urls[params["requestId"]].endswith("/start"):
                    return await self.cdp.send("Fetch.fulfillRequest", {
                        "requestId": params["requestId"], "responseCode": 302,
                        "responseHeaders": [{"name": "location", "value": "/private"}],
                    })
                return await self.cdp.send("Fetch.fulfillRequest", {
                    "requestId": params["requestId"], "responseCode": 403,
                    "responseHeaders": [{"name": "content-type", "value": "text/html"}],
                    "body": base64.b64encode(b"<html><title>Access Denied</title><body>Forbidden</body></html>").decode(),
                })
            return await self.cdp.send(name, params)
        async def detach(self):
            await self.cdp.detach()

    class Context:
        def __init__(self, context):
            self.context = context
        async def new_cdp_session(self, page):
            return SyntheticCDP(await self.context.new_cdp_session(page))

    def factory():
        crawler = AsyncWebCrawler(config=BrowserConfig(chrome_channel="msedge", channel="msedge", headless=True,
                                  enable_stealth=False, ignore_https_errors=False, verbose=False))
        original = crawler.crawler_strategy.set_hook
        def set_hook(name, callback):
            if name == "before_goto":
                async def wrapped(page, context, url, **kwargs):
                    return await callback(page, Context(context), url, **kwargs)
                original(name, wrapped)
            else:
                original(name, callback)
        crawler.crawler_strategy.set_hook = set_hook
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, check_robots_txt=False,
            max_retries=0, fallback_fetch_function=None, wait_until="domcontentloaded",
            page_timeout=10000, delay_before_return_html=0, verbose=False)
        return crawler, config

    async def run():
        def robots(_request):
            return httpx.Response(200, text="User-agent: *\n" + (
                "Disallow: /private\n" if disallow else "Allow: /\n"))
        guard = Fetcher(CrawlConfig(per_host_delay=0), transport=httpx.MockTransport(robots))
        try:
            return await crawl4ai_browser_probe(URL, guard, config=BrowserProbeConfig(
                allowed_hosts=("crawl4ai-probe.test",), document_path_prefix="/",
                document_delay_seconds=0, render_wait_seconds=0), crawler_factory=factory)
        finally:
            await guard.client.aclose()
    record, assets = asyncio.run(run())
    assert "request_log" in record, record
    documents = [r for r in record["request_log"] if r["resource_type"] == "Document"]
    assert len(documents) == 2 and documents[1]["redirected_request_id"]
    assert not record["guard_errors"]
    if disallow:
        assert documents[1]["decision"] == "blocked" and not record["article_candidate"]
    else:
        assert documents[1]["decision"] == "allowed"
        assert record["http_status"] == record["rendered_http_status"] == 403
        assert record["response_evidence"]["page_title"] == "Access Denied"
        assert b"Access Denied" in assets["response.html"] and b"Access Denied" in assets["rendered.html"]
