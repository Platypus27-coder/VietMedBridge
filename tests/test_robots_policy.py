"""Robots decisions must be auditable and must never turn failures into allow-all."""

import asyncio
import socket
from collections import Counter

import httpx
import pytest

from vietmedbridge.crawl import CrawlConfig, Fetcher
from vietmedbridge.robots import RobotsResolver


def fetch_one(handler, url="https://example.org/article"):
    async def run():
        fetcher = Fetcher(CrawlConfig(attempts=1, per_host_delay=0),
                          transport=httpx.MockTransport(handler))
        try:
            return await fetcher.fetch({"id": 42, "url": url})
        finally:
            await fetcher.client.aclose()
    return asyncio.run(run())


@pytest.mark.parametrize("status,expected,allowed", [
    (404, "ROBOTS_404", True),
    (410, "ROBOTS_410", True),
    (401, "ROBOTS_401", False),
    (403, "ROBOTS_403", False),
    (429, "ROBOTS_429", False),
    (503, "ROBOTS_5XX", False),
])
def test_robots_http_statuses_are_distinct(status, expected, allowed):
    requests = Counter()

    def handler(request):
        requests[request.url.path] += 1
        if request.url.path == "/robots.txt":
            return httpx.Response(status, headers={"retry-after": "120"})
        return httpx.Response(200, content=b"<html><article>Valid article</article></html>")

    result = fetch_one(handler)
    assert result["robots_state"] == expected
    assert result["robots_http_status"] == status
    assert result["status"] == ("ok" if allowed else "error")
    assert requests["/article"] == int(allowed)
    if status == 429:
        assert result["robots_retry_after_seconds"] == 120


def test_robots_timeout_is_not_allow_all():
    def handler(request):
        if request.url.path == "/robots.txt":
            raise httpx.ConnectTimeout("timeout", request=request)
        pytest.fail("Page must not be fetched when robots is unreachable")

    result = fetch_one(handler)
    assert result["status"] == "error"
    assert result["error"] == "robots_connect_timeout"
    assert result["robots_state"] == "ROBOTS_CONNECT_TIMEOUT"


def test_robots_dns_error_is_distinct():
    def handler(request):
        if request.url.path == "/robots.txt":
            try:
                raise socket.gaierror("DNS failure")
            except socket.gaierror as exc:
                raise httpx.ConnectError("connect failure", request=request) from exc
        pytest.fail("Page must not be fetched after robots DNS failure")

    result = fetch_one(handler)
    assert result["error"] == "robots_dns_error"


def test_robots_rate_directives_are_exposed():
    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text="User-agent: VietMedBridge\nCrawl-delay: 3\nRequest-rate: 2/10\nAllow: /\n"
        )))
        async def no_delay(_url):
            return None
        try:
            return await RobotsResolver(client, no_delay, user_agent="VietMedBridge/0.2").decision(
                "https://example.org/article"
            )
        finally:
            await client.aclose()

    decision = asyncio.run(run())
    assert decision.allowed
    assert decision.crawl_delay_seconds == 3
    assert decision.request_rate_gap_seconds == 5


def test_robots_specific_agent_and_redirect_are_respected():
    calls = Counter()

    def handler(request):
        calls[str(request.url)] += 1
        if request.url.scheme == "http" and request.url.path == "/robots.txt":
            return httpx.Response(301, headers={"location": "https://example.org/robots.txt"})
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=(
                "User-agent: VietMedBridge\nDisallow: /private\n"
                "User-agent: *\nAllow: /\n"
            ))
        pytest.fail("Disallowed page must not be fetched")

    result = fetch_one(handler, "http://example.org/private")
    assert result["error"] == "robots_blocked"
    assert result["robots_state"] == "ROBOTS_OK_DISALLOWED"
    assert result["robots_url"] == "https://example.org/robots.txt"
    assert calls["http://example.org/robots.txt"] == 1
    assert calls["https://example.org/robots.txt"] == 1


def test_robots_result_cached_for_same_origin():
    calls = Counter()

    def handler(request):
        calls[request.url.path] += 1
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"<html><article>Valid article</article></html>")

    async def run():
        fetcher = Fetcher(CrawlConfig(attempts=1, per_host_delay=0),
                          transport=httpx.MockTransport(handler))
        try:
            return await asyncio.gather(*(
                fetcher.fetch({"id": i, "url": f"https://example.org/{i}"})
                for i in range(3)
            ))
        finally:
            await fetcher.client.aclose()

    results = asyncio.run(run())
    assert all(result["status"] == "ok" for result in results)
    assert calls["/robots.txt"] == 1


def test_page_403_preserves_robots_evidence():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(403, text="Forbidden")

    result = fetch_one(handler)
    assert result["error"] == "http_403"
    assert result["robots_state"] == "ROBOTS_OK_ALLOWED"
    assert result["robots_http_status"] == 200
    assert len(result["robots_checks"]) == 1
