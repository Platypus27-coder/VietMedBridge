import asyncio
import base64
import importlib.util
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from vietmedbridge.advanced_recovery import AdvancedRecoveryRuntime, article_capture_ready
from vietmedbridge.chrome_transport import ChromeRobotsTransport
from vietmedbridge.crawl import CrawlConfig, Fetcher
from vietmedbridge.recovery_browser_gate import RecoveryBrowserConfig, RecoveryRequestGate
from vietmedbridge.robots import RobotsResolver, valid_robots_bytes


HOST = "recovery-fixture.test"
URL = f"https://{HOST}/article"
ROBOTS = f"https://{HOST}/robots.txt"
ARTICLE = "Nội dung y sinh từ bài viết phải giữ nguyên và đủ các đoạn văn bản. " * 15


async def no_pace(*_args):
    pass


@pytest.mark.parametrize("body,allowed", [
    (b"User-agent: *\nDisallow: /article\n", False),
    (b"User-agent: *\nAllow: /\n", True),
    (b"<html><pre>User-agent: *\nAllow: /</pre></html>", False),
    (b"Access Denied", False),
])
def test_browser_robots_requires_raw_policy_not_a_rendered_allow_all(body, allowed):
    events = []
    def get(url, **kwargs):
        return SimpleNamespace(url=url, status=403, body=b"Forbidden", headers={})
    async def browser(_url):
        return {"status": 200, "body": body, "evidence": {"test": "raw response"}}
    async def run():
        async with httpx.AsyncClient(transport=ChromeRobotsTransport(get, browser_read=browser,
            observer=lambda e, b: events.append((e, b))), headers={"User-Agent": "VietMedBridge/0.2"}) as client:
            return await RobotsResolver(client, no_pace, user_agent="VietMedBridge/0.2").decision(URL)
    decision = asyncio.run(run())
    assert decision.allowed is allowed
    if b"Disallow" in body:
        assert decision.state == "ROBOTS_OK_DISALLOWED"
    assert events[-1][1] == body


@pytest.mark.parametrize("status,headers", [(401, {}), (429, {}), (503, {"retry-after": "3600"})])
def test_browser_fallback_does_not_ignore_auth_or_retry_after(status, headers):
    def get(url, **kwargs):
        return SimpleNamespace(url=url, status=status, body=b"Wait", headers=headers)
    async def browser(_url):
        pytest.fail("Cannot escalate a current rate/authentication hold to browser")
    async def run():
        async with httpx.AsyncClient(transport=ChromeRobotsTransport(get, browser_read=browser),
                                    headers={"User-Agent": "VietMedBridge/0.2"}) as client:
            return await client.get(ROBOTS)
    assert asyncio.run(run()).status_code == status


def test_cancelled_browser_does_not_send_queued_robots_requests():
    from threading import Event
    started, release, calls = Event(), Event(), []
    def get(url, **kwargs):
        calls.append(url)
        started.set()
        assert release.wait(5)
        return SimpleNamespace(url=url, status=200, body=b"User-agent: *\nAllow: /\n", headers={})
    async def run():
        transport = ChromeRobotsTransport(get)
        def request(url):
            return httpx.Request("GET", url, headers={"User-Agent": "VietMedBridge/0.2"})
        first = asyncio.create_task(transport.handle_async_request(request(ROBOTS)))
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(transport.handle_async_request(request("https://cdn.test/robots.txt")))
        await asyncio.sleep(0)
        first.cancel()
        second.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
    asyncio.run(run())
    assert calls == [ROBOTS]


def test_bootstrap_only_navigates_robots_and_http_redirects():
    class CDP:
        async def send(self, *_args):
            pass
    async def run():
        guard = Fetcher(CrawlConfig(per_host_delay=0), transport=httpx.MockTransport(
            lambda _r: pytest.fail("Robots bootstrap must not recursively request robots")))
        gate = RecoveryRequestGate(CDP(), guard, guard.robots, "main",
            RecoveryBrowserConfig(allowed_hosts=(HOST,), robots_bootstrap_url=ROBOTS, document_delay_seconds=0))
        urls = [(ROBOTS, None), (URL, None), ("https://redirect.test/policy.txt", "previous"),
                ("https://redirect.test/article", None), ("https://127.0.0.1/robots.txt", "previous")]
        try:
            for url, redirect in urls:
                await gate.handle({"requestId": url, "request": {"url": url, "method": "GET"},
                    "resourceType": "Document", "frameId": "main", "redirectedRequestId": redirect})
            return gate.log
        finally:
            await guard.client.aclose()
    assert [r["decision"] for r in asyncio.run(run())] == ["allowed", "blocked", "allowed", "blocked", "blocked"]


def test_fallback_keeps_both_engines_evidence_without_stale_selected_bytes(tmp_path, monkeypatch):
    runtime = AdvancedRecoveryRuntime(tmp_path)
    shell = b"<html><title>Article</title><body>Empty source shell</body></html>"
    rendered = f"<html><title>Article</title><article><p>{ARTICLE}</p></article></html>".encode()
    async def attempt(engine, url, settings):
        record = {"method": engine, "http_status": 200, "rendered_http_status": 200,
                  "guard_errors": [], "final_url": url}
        return record, {"response.html": shell} if engine == "crawl4ai_stealth" else {"rendered.html": rendered}
    monkeypatch.setattr(runtime, "_attempt", attempt)
    record, assets = asyncio.run(runtime.browser_probe(URL, None))
    assert article_capture_ready(record, assets)
    assert record["method"] == "scrapling_stealth" and len(record["engine_attempts"]) == 2
    assert "response.html" not in assets and "crawl4ai_stealth/response.html" in assets
    assert runtime.preferred[HOST] == "scrapling_stealth"


def test_parallel_cdn_robots_bootstraps_are_serial_and_cached(tmp_path, monkeypatch):
    runtime = AdvancedRecoveryRuntime(tmp_path)
    active, peak, calls = 0, 0, []
    async def attempt(engine, url, settings):
        nonlocal active, peak
        calls.append(url)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"http_status": 200, "guard_errors": []}, {"response.html": b"User-agent: *\nAllow: /\n"}
    monkeypatch.setattr(runtime, "_attempt", attempt)
    async def run():
        await asyncio.gather(runtime.read_robots(ROBOTS), runtime.read_robots("https://cdn.test/robots.txt"),
                             runtime.read_robots(ROBOTS))
    asyncio.run(run())
    assert peak == 1 and len(calls) == 2
    assert runtime.preferred[HOST] == "scrapling_stealth"


@pytest.mark.skipif(not importlib.util.find_spec("crawl4ai"), reason="Optional recovery runtime")
def test_full_recovery_reopens_robots_403_and_resumes_dual_engine_results(tmp_path, monkeypatch):
    from test_stage_a_recovery import plan_for_ids, BODY, response
    from vietmedbridge.stage_a_recovery import recover_remaining
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("article-5"):
            raise ConnectionError("Injected HTTP connection failure")
        return response(url, 403, b"Forbidden")
    async def run():
        async with AdvancedRecoveryRuntime(tmp_path / "work") as runtime:
            async def attempt(engine, url, settings):
                calls.append((engine, url))
                record = {"method": engine, "http_status": 200, "rendered_http_status": 200,
                          "guard_errors": [], "final_url": url}
                if settings.robots_bootstrap_url:
                    return record, {"response.html": b"User-agent: *\nDisallow: /article-4\n"}
                if engine == "crawl4ai_stealth":
                    record.update(http_status=403, rendered_http_status=403)
                    return record, {"response.html": b"Forbidden"}
                return record, {"response.html": BODY, "rendered.html": BODY}
            monkeypatch.setattr(runtime, "_attempt", attempt)
            kwargs = dict(advanced_runtime=runtime, browser_probe=runtime.browser_probe, sleep=no_pace)
            result = await recover_remaining(plan_for_ids(), tmp_path / "output", get, **kwargs)
            before = list(calls)
            resumed = await recover_remaining(plan_for_ids(), tmp_path / "output", get, **kwargs)
            assert calls == before
            assert result["coverage_ledger"] == resumed["coverage_ledger"]
            return result
    result = asyncio.run(run())
    assert result["official_ids"] == 6 and result["states"]["ARTICLE_CANDIDATE_REVIEW"] == 2
    assert result["states"]["ROBOTS_POLICY_REVIEW"] == 1
    assert "https://source.test/article-4" not in calls
    assert not any(isinstance(c, tuple) and c[1].endswith("article-4") for c in calls)
    for row in result["records"]:
        if row["state"] == "ARTICLE_CANDIDATE_REVIEW":
            assert row["source_method"] == "scrapling_stealth"
            assert any("scrapling_stealth/response.html" == name for name in row["assets"])


@pytest.mark.skipif(not importlib.util.find_spec("crawl4ai") or not os.environ.get("VMB_TEST_BROWSER_EXE"),
                    reason="Optional real dual-engine browser integration")
@pytest.mark.parametrize("engine", ["crawl4ai_stealth", "scrapling_stealth"])
@pytest.mark.parametrize("purpose", ["article", "robots"])
def test_real_engines_load_cdn_content_reuse_cookie_and_capture_raw_robots(tmp_path, engine, purpose):
    network = []
    class CDP:
        def __init__(self, cdp):
            self.cdp, self.events = cdp, {}
        def on(self, name, callback):
            if name != "Fetch.requestPaused":
                return self.cdp.on(name, callback)
            def wrapped(event):
                self.events[event["requestId"]] = event
                network.append(event["request"]["url"])
                callback(event)
            self.cdp.on(name, wrapped)
        async def send(self, method, params=None):
            if method != "Fetch.continueRequest":
                return await self.cdp.send(method, params)
            url = self.events[params["requestId"]]["request"]["url"]
            if url.endswith("robots.txt"):
                body, kind = b"User-agent: *\nDisallow: /private\n", "text/plain"
            elif url.endswith("article.js"):
                js = "setTimeout(() => { document.querySelector('article').textContent = " + json.dumps(ARTICLE) + "; "
                js += "document.querySelector('article').textContent += ' Seen session: ' + document.cookie; "
                js += "document.cookie = 'visited=yes; path=/'; }, 350);"
                body, kind = js.encode(), "application/javascript"
            else:
                body = b'<html><title>Source article</title><body><article></article><script src="https://cdn-fixture.test/article.js"></script></body></html>'
                kind = "text/html"
            return await self.cdp.send("Fetch.fulfillRequest", {"requestId": params["requestId"],
                "responseCode": 200, "responseHeaders": [{"name": "content-type", "value": kind}],
                "body": base64.b64encode(body).decode()})
        async def detach(self):
            await self.cdp.detach()
    class Context:
        def __init__(self, real):
            self.real = real
        async def new_cdp_session(self, page):
            return CDP(await self.real.new_cdp_session(page))
        async def new_page(self):
            return await self.real.new_page()
    async def run():
        async with AdvancedRecoveryRuntime(tmp_path, browser_executable=os.environ["VMB_TEST_BROWSER_EXE"],
            profiles={HOST: {"readiness_timeout_ms": 5000, "scroll_steps": 0,
                             "challenge_timeout_seconds": 0}}) as runtime:
            guard = Fetcher(CrawlConfig(per_host_delay=0), transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, text="User-agent: *\nAllow: /\n")))
            runtime.guard = guard
            original = runtime._new
            async def new(engine_name, host):
                manager, resource = await original(engine_name, host)
                if engine_name == "crawl4ai_stealth":
                    original_set = resource.crawler_strategy.set_hook
                    def set_hook(name, callback):
                        if name == "before_goto":
                            async def wrapped(page, context, url, **kwargs):
                                return await callback(page, Context(context), url, **kwargs)
                            original_set(name, wrapped)
                        else:
                            original_set(name, callback)
                    resource.crawler_strategy.set_hook = set_hook
                else:
                    resource = SimpleNamespace(context=Context(resource.context), _cloudflare_solver=resource._cloudflare_solver)
                return manager, resource
            runtime._new = new
            try:
                url = ROBOTS if purpose == "robots" else URL
                settings = replace(runtime.settings(url, robots=purpose == "robots"), document_delay_seconds=0)
                first = await runtime._attempt(engine, url, settings)
                if purpose == "robots":
                    assert valid_robots_bytes(first[1].get("response.html")), json.dumps(first[0], ensure_ascii=False)
                    assert b"Disallow: /private" in first[1]["response.html"]
                else:
                    assert article_capture_ready(*first), json.dumps(first[0], ensure_ascii=False)
                    second = await runtime._attempt(engine, URL + "-second", settings)
                    assert article_capture_ready(*second), second[0]
                    assert b"visited=yes" not in first[1]["rendered.html"]
                    assert b"visited=yes" in second[1]["rendered.html"]
                    assert "https://cdn-fixture.test/article.js" in network
                assert not first[0]["guard_errors"]
            finally:
                await guard.client.aclose()
    asyncio.run(run())
