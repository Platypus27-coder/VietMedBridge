"""Verify browser gates without installing or opening a browser in unit tests."""

import asyncio

import httpx
import pytest

from vietmedbridge.browser_probe import (
    BrowserProbeConfig, _RequestGate, guarded_browser_probe, response_evidence,
)
from vietmedbridge.crawl import CrawlConfig, Fetcher
from vietmedbridge.robots import RobotsResolver


class CDP:
    def __init__(self, fail_setup=False):
        self.sent = []
        self.fail_setup = fail_setup

    async def send(self, method, params=None):
        self.sent.append((method, params))
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "main"}}}
        if method == "Fetch.enable" and self.fail_setup:
            raise RuntimeError("Cannot install interception")
        return {}

    def on(self, *_args):
        pass

    async def detach(self):
        pass


def paused(url, *, kind="Document", method="GET", frame="main", redirect=None):
    return {"requestId": url, "request": {"url": url, "method": method},
            "resourceType": kind, "frameId": frame, "redirectedRequestId": redirect}


def exercise(events, *, robots_text="User-agent: *\nAllow: /\n", robots_status=200,
             settings=None, broken_robots=False):
    async def run():
        calls = []
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(robots_status, text=robots_text)
        guard = Fetcher(CrawlConfig(per_host_delay=0), transport=httpx.MockTransport(handler))
        async def no_pace(*_args):
            pass
        guard.pace = no_pace
        if broken_robots:
            async def fail(_url):
                raise RuntimeError("Robots callback failed")
            guard.robots_decision = fail
        cdp = CDP()
        browser_robots = RobotsResolver(guard.client, guard.pace, user_agent="Mozilla/5.0")
        gate = _RequestGate(cdp, guard, browser_robots, "main", settings or BrowserProbeConfig())
        try:
            for event in events:
                await gate.handle(event)
            return cdp, gate, calls
        finally:
            await guard.client.aclose()
    return asyncio.run(run())


def test_redirect_destination_has_its_own_robots_check():
    first = "https://nhathuoclongchau.com.vn/bai-viet/allowed.html"
    target = "https://nhathuoclongchau.com.vn/bai-viet/private.html"
    cdp, gate, _ = exercise([paused(first), paused(target, redirect=first)],
                           robots_text="User-agent: *\nDisallow: /bai-viet/private.html\n")
    assert [m for m, _ in cdp.sent] == ["Fetch.continueRequest", "Fetch.failRequest"]
    assert gate.log[-1]["robots_state"] == "ROBOTS_OK_DISALLOWED"
    assert gate.log[-1]["redirected_request_id"] == first


def test_scope_blocks_other_hosts_post_and_iframe_before_network():
    events = [paused("https://outside.test/bai-viet/article"),
              paused("https://nhathuoclongchau.com.vn/api", kind="XHR", method="POST"),
              paused("https://nhathuoclongchau.com.vn/bai-viet/frame", frame="iframe"),
              paused("http://nhathuoclongchau.com.vn/bai-viet/article")]
    cdp, gate, robots_calls = exercise(events)
    assert len(cdp.sent) == 4
    assert all(m == "Fetch.failRequest" for m, _ in cdp.sent)
    assert [r["reason"] for r in gate.log] == ["host_outside_probe", "method_outside_probe",
                                             "document_outside_probe", "https_only"]
    assert robots_calls == []


@pytest.mark.parametrize("status,broken", [(403, False), (200, True)])
def test_robots_hold_or_callback_error_cannot_release_request(status, broken):
    cdp, gate, _ = exercise([paused("https://nhathuoclongchau.com.vn/bai-viet/article")],
                           robots_status=status, broken_robots=broken)
    assert cdp.sent[0][0] == "Fetch.failRequest"
    assert gate.log[0]["decision"] == "blocked"
    assert bool(gate.errors) == broken


def test_browser_and_crawler_product_rules_are_both_enforced():
    cdp, gate, _ = exercise([paused("https://nhathuoclongchau.com.vn/bai-viet/article")],
                           robots_text="User-agent: Mozilla\nDisallow: /\n\nUser-agent: *\nAllow: /\n")
    assert cdp.sent[0][0] == "Fetch.failRequest"
    assert gate.log[0]["robots_state"] == "ROBOTS_OK_ALLOWED"
    assert gate.log[0]["browser_robots_state"] == "ROBOTS_OK_DISALLOWED"


def test_volume_budget_prevents_more_documents():
    cdp, gate, _ = exercise(
        [paused(f"https://nhathuoclongchau.com.vn/bai-viet/{i}") for i in range(3)],
        settings=BrowserProbeConfig(max_documents=2),
    )
    assert [m for m, _ in cdp.sent] == ["Fetch.continueRequest", "Fetch.continueRequest", "Fetch.failRequest"]
    assert gate.log[-1]["reason"] == "document_budget"


def test_failed_guard_setup_never_navigates():
    class Page:
        navigated = False
        closed = False
        async def add_init_script(self, *_args):
            pass
        async def evaluate(self, *_args):
            return "Mozilla/5.0"
        async def goto(self, *_args, **_kwargs):
            self.navigated = True
        async def close(self):
            self.closed = True
    class Context:
        page = Page()
        async def new_page(self):
            return self.page
        async def new_cdp_session(self, _page):
            return CDP(fail_setup=True)
    context = Context()
    async def run():
        guard = Fetcher(CrawlConfig())
        try:
            await guarded_browser_probe(context, "https://nhathuoclongchau.com.vn/bai-viet/article", guard)
        finally:
            await guard.client.aclose()
    with pytest.raises(RuntimeError, match="Cannot install"):
        asyncio.run(run())
    assert context.page.closed
    assert not context.page.navigated


def test_evidence_keeps_error_html_hash_without_cookie_or_auth_headers():
    evidence = response_evidence(b"<html><title>Access Denied</title><body>Forbidden</body></html>",
                                 {"server": "example", "set-cookie": "private",
                                  "authorization": "secret"})
    assert evidence["page_title"] == "Access Denied"
    assert evidence["block_page_signals"] == ["access denied", "forbidden"]
    assert len(evidence["body_sha256"]) == 64
    assert evidence["response_headers"] == {"server": "example"}
