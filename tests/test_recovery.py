"""Recovery requests must obey robots and pacing at every redirect/retry."""

import asyncio
from dataclasses import dataclass, field

import httpx

from vietmedbridge.crawl import CrawlConfig, Fetcher
from vietmedbridge.recovery import RecoveryConfig, guarded_http_fetch


@dataclass
class Response:
    url: str
    status: int
    body: bytes = b""
    headers: dict = field(default_factory=dict)


def run_recovery(robots_handler, get, *, url="https://source.test/article",
                 config=None, pace_probe=None, held_hosts=None):
    async def run():
        guard = Fetcher(CrawlConfig(attempts=1, per_host_delay=0),
                        transport=httpx.MockTransport(robots_handler))
        if pace_probe is not None:
            async def record_pace(current, minimum_delay=0):
                pace_probe.append((current, minimum_delay))
            guard.pace = record_pace
        try:
            async def no_sleep(_seconds):
                return None
            return await guarded_http_fetch(
                url, guard, get, user_agent=guard.config.user_agent,
                config=config, held_hosts=held_hosts, sleep=no_sleep,
            )
        finally:
            await guard.client.aclose()
    return asyncio.run(run())


def test_robots_denial_never_calls_scrapling():
    def robots(request):
        return httpx.Response(403)
    def get(*_args, **_kwargs):
        raise AssertionError("Forbidden page fetched")

    result, body = run_recovery(robots, get)
    assert result["outcome"] == "robots_hold"
    assert result["robots_state"] == "ROBOTS_403"
    assert body is None


def test_redirect_is_rechecked_and_crawl_delay_is_passed_to_pacer():
    called = []
    paced = []

    def robots(request):
        return httpx.Response(200, text=(
            "User-agent: VietMedBridge\nCrawl-delay: 4\n"
            "Disallow: /private\nAllow: /\n"
        ))
    def get(url, **kwargs):
        called.append((url, kwargs))
        return Response(url, 302, headers={"location": "/private"})

    result, body = run_recovery(robots, get, pace_probe=paced)
    assert result["outcome"] == "robots_hold"
    assert [row[0] for row in called] == ["https://source.test/article"]
    assert paced == [("https://source.test/article", 4.0)]
    assert called[0][1]["follow_redirects"] is False
    assert called[0][1]["stealthy_headers"] is False
    assert called[0][1]["impersonate"] is None
    assert body is None


def test_retry_after_above_inline_limit_holds_without_retry():
    called = []

    def robots(_request):
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    def get(url, **_kwargs):
        called.append(url)
        return Response(url, 429, headers={"retry-after": "300"})

    held_hosts = set()
    result, body = run_recovery(robots, get, held_hosts=held_hosts)
    assert result["outcome"] == "rate_limited_retry_later"
    assert result["retry_after_seconds"] == 300
    assert len(called) == 1
    assert held_hosts == {"source.test"}
    assert body is None

    second, body = run_recovery(robots, get, held_hosts=held_hosts)
    assert second["outcome"] == "host_rate_limit_hold"
    assert len(called) == 1
    assert body is None


def test_server_error_retries_then_returns_bytes():
    called = []

    def robots(_request):
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    def get(url, **_kwargs):
        called.append(url)
        if len(called) == 1:
            return Response(url, 503)
        return Response(url, 200, b"article", {"content-type": "text/plain"})

    result, body = run_recovery(robots, get)
    assert result["outcome"] == "http_200"
    assert result["attempts"] == 2
    assert len(result["request_log"]) == 2
    assert body == b"article"


def test_repeated_429_without_header_holds_host():
    called = []
    held_hosts = set()

    def robots(_request):
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    def get(url, **_kwargs):
        called.append(url)
        return Response(url, 429)

    result, body = run_recovery(robots, get, held_hosts=held_hosts)
    assert result["outcome"] == "rate_limited_retry_later"
    assert result["retry_after_seconds"] == 60
    assert len(called) == 2
    assert held_hosts == {"source.test"}
    assert body is None


def test_hidden_redirect_is_not_accepted():
    def robots(_request):
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    def get(_url, **_kwargs):
        return Response("https://other.test/article", 200, b"article")

    result, body = run_recovery(robots, get)
    assert result["outcome"] == "unguarded_redirect_detected"
    assert body is None


def test_private_redirect_target_is_rejected_before_request():
    called = []

    def robots(_request):
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    def get(url, **_kwargs):
        called.append(url)
        return Response(url, 302, headers={"location": "http://127.0.0.1/private"})

    result, body = run_recovery(robots, get)
    assert result["outcome"] == "invalid_redirect_target"
    assert len(called) == 1
    assert body is None
