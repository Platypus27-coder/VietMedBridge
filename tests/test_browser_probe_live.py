"""Optional real Chromium protocol checks, with synthetic responses and no website traffic.

Set VMB_TEST_BROWSER_EXE to an installed Chromium/Edge executable to run these.
The HTTP responses are fulfilled inside CDP; DNS/network is never needed.
"""

import asyncio
import base64
import os

import httpx
import pytest

from vietmedbridge.browser_probe import BrowserProbeConfig, guarded_browser_probe
from vietmedbridge.crawl import CrawlConfig, Fetcher


@pytest.mark.skipif(not os.environ.get("VMB_TEST_BROWSER_EXE"), reason="Optional Chromium integration")
@pytest.mark.parametrize("disallow", [True, False])
def test_real_cdp_redirect_is_checked_and_error_page_is_archived(disallow):
    from scrapling.fetchers import AsyncDynamicSession

    class SyntheticCDP:
        def __init__(self, cdp):
            self.cdp, self.urls = cdp, {}

        def on(self, name, callback):
            def wrapped(event):
                self.urls[event["requestId"]] = event["request"]["url"]
                callback(event)
            self.cdp.on(name, wrapped)

        async def send(self, method, params=None):
            if method == "Fetch.continueRequest":
                if self.urls[params["requestId"]].endswith("/start"):
                    return await self.cdp.send("Fetch.fulfillRequest", {
                        "requestId": params["requestId"], "responseCode": 302,
                        "responseHeaders": [{"name": "location", "value": "/bai-viet/private"}],
                    })
                return await self.cdp.send("Fetch.fulfillRequest", {
                    "requestId": params["requestId"], "responseCode": 403,
                    "responseHeaders": [{"name": "content-type", "value": "text/html"}],
                    "body": base64.b64encode(b"<html><title>Access Denied</title><body>Forbidden</body></html>").decode(),
                })
            return await self.cdp.send(method, params)

        async def detach(self):
            await self.cdp.detach()

    class Context:
        def __init__(self, context):
            self.context = context
        async def new_page(self):
            return await self.context.new_page()
        async def new_cdp_session(self, page):
            return SyntheticCDP(await self.context.new_cdp_session(page))

    async def run():
        def robots(_request):
            return httpx.Response(200, text="User-agent: *\n" + (
                "Disallow: /bai-viet/private\n" if disallow else "Allow: /\n"))
        guard = Fetcher(CrawlConfig(per_host_delay=0), transport=httpx.MockTransport(robots))
        try:
            async with AsyncDynamicSession(headless=True, google_search=False, retries=1,
                    executable_path=os.environ["VMB_TEST_BROWSER_EXE"],
                    additional_args={"service_workers": "block"}) as session:
                return await guarded_browser_probe(Context(session.context), "https://probe.test/bai-viet/start",
                    guard, config=BrowserProbeConfig(allowed_hosts=("probe.test",),
                        render_wait_seconds=0, document_delay_seconds=0, navigation_timeout_ms=10000))
        finally:
            await guard.client.aclose()

    result, assets = asyncio.run(run())
    documents = [r for r in result["request_log"] if r["resource_type"] == "Document"]
    assert len(documents) == 2
    assert documents[1]["redirected_request_id"]
    assert not result["guard_errors"]
    if disallow:
        assert documents[1]["decision"] == "blocked"
        assert documents[1]["robots_state"] == "ROBOTS_OK_DISALLOWED"
        assert result["outcome"] == "browser_navigation_error"
    else:
        assert documents[1]["decision"] == "allowed"
        assert result["http_status"] == 403
        assert result["rendered_http_status"] == 403
        assert result["response_evidence"]["page_title"] == "Access Denied"
        assert assets["response.html"].startswith(b"<html>")
        assert b"Access Denied" in assets["rendered.html"]
        assert "screenshot.png" in assets
        assert result["article_candidate"] is False
