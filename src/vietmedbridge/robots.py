"""Auditable, conservative robots.txt decisions for new crawl runs.

RFC 9309 permits access after a robots 4xx, but does not require it. This
project allows 404/410 and holds other 4xx for source review. A robots 429
is treated as a rate-limit signal, never as blanket crawl permission.
"""

from __future__ import annotations

import asyncio
import hashlib
import socket
import ssl
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .artifacts import utc_now
from .dataset import url_host


@dataclass(frozen=True)
class RobotsDecision:
    state: str
    allowed: bool
    robots_url: str
    fetched_at: str
    http_status: int | None = None
    content_sha256: str | None = None
    retry_after_seconds: float | None = None
    crawl_delay_seconds: float | None = None
    request_rate_gap_seconds: float | None = None

    def record(self) -> dict:
        return {
            "robots_state": self.state,
            "robots_allowed": self.allowed,
            "robots_url": self.robots_url,
            "robots_fetched_at": self.fetched_at,
            "robots_http_status": self.http_status,
            "robots_content_sha256": self.content_sha256,
            "robots_retry_after_seconds": self.retry_after_seconds,
            "robots_crawl_delay_seconds": self.crawl_delay_seconds,
            "robots_request_rate_gap_seconds": self.request_rate_gap_seconds,
        }


@dataclass
class _CacheEntry:
    state: str
    robots_url: str
    fetched_at: str
    http_status: int | None
    content_sha256: str | None
    retry_after_seconds: float | None
    parser: RobotFileParser | None
    expires_at: float


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, min(seconds, 3600.0))


def _connect_error_state(exc: BaseException) -> str:
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return "ROBOTS_SSL_ERROR"
        if isinstance(current, socket.gaierror):
            return "ROBOTS_DNS_ERROR"
        current = current.__cause__ or current.__context__
    return "ROBOTS_CONNECTION_ERROR"


class RobotsResolver:
    """Cache robots responses by origin while preserving each decision's reason."""

    def __init__(self, client: httpx.AsyncClient, pace, *, user_agent: str,
                 normal_ttl_seconds: float = 86400, error_ttl_seconds: float = 300):
        self.client = client
        self.pace = pace
        self.user_agent = user_agent.split("/", 1)[0]
        self.normal_ttl_seconds = normal_ttl_seconds
        self.error_ttl_seconds = error_ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def decision(self, url: str) -> RobotsDecision:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self._locks.setdefault(origin, asyncio.Lock()):
            entry = self._cache.get(origin)
            if entry is None or entry.expires_at <= time.monotonic():
                entry = await self._fetch(origin)
                self._cache[origin] = entry
        if entry.parser is not None:
            allowed = entry.parser.can_fetch(self.user_agent, url)
            state = "ROBOTS_OK_ALLOWED" if allowed else "ROBOTS_OK_DISALLOWED"
            crawl_delay = entry.parser.crawl_delay(self.user_agent)
            request_rate = entry.parser.request_rate(self.user_agent)
            rate_gap = (request_rate.seconds / request_rate.requests
                        if request_rate and request_rate.requests > 0 else None)
        else:
            allowed = entry.state in {"ROBOTS_404", "ROBOTS_410"}
            state = entry.state
            crawl_delay = rate_gap = None
        return RobotsDecision(
            state, allowed, entry.robots_url, entry.fetched_at,
            entry.http_status, entry.content_sha256, entry.retry_after_seconds,
            float(crawl_delay) if crawl_delay is not None else None, rate_gap,
        )

    async def _fetch(self, origin: str) -> _CacheEntry:
        robots_url = origin + "/robots.txt"
        fetched_at = utc_now()
        status = None
        parser = None
        digest = None
        retry_after = None
        state = "ROBOTS_REDIRECT_LIMIT"
        try:
            # RFC 9309 asks crawlers to follow at least five redirects.
            for _ in range(6):
                url_host(robots_url)
                await self.pace(robots_url)
                async with self.client.stream("GET", robots_url) as response:
                    status = response.status_code
                    if status in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            state = "ROBOTS_REDIRECT_LIMIT"
                            break
                        robots_url = urljoin(robots_url, location)
                        continue
                    if status == 200:
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > 512 * 1024:
                                state = "ROBOTS_TOO_LARGE"
                                break
                        else:
                            digest = hashlib.sha256(body).hexdigest()
                            prefix = bytes(body[:8192]).lower()
                            if any(tag in prefix for tag in (b"<html", b"<!doctype html", b"<head", b"<body")):
                                state = "ROBOTS_INVALID_CONTENT"
                            else:
                                parser = RobotFileParser(robots_url)
                                parser.parse(body.decode("utf-8-sig", errors="replace").splitlines())
                                state = "ROBOTS_OK"
                    elif status in (404, 410):
                        state = f"ROBOTS_{status}"
                    elif status == 429:
                        state = "ROBOTS_429"
                        retry_after = _retry_after(response.headers.get("retry-after"))
                    elif status in (401, 403):
                        state = f"ROBOTS_{status}"
                    elif 400 <= status < 500:
                        state = "ROBOTS_4XX_OTHER"
                    elif 500 <= status < 600:
                        state = "ROBOTS_5XX"
                    else:
                        state = "ROBOTS_UNEXPECTED_STATUS"
                    break
        except httpx.ConnectTimeout:
            state = "ROBOTS_CONNECT_TIMEOUT"
        except httpx.ReadTimeout:
            state = "ROBOTS_READ_TIMEOUT"
        except httpx.ConnectError as exc:
            state = _connect_error_state(exc)
        except httpx.RequestError:
            state = "ROBOTS_CONNECTION_ERROR"
        ttl = self.normal_ttl_seconds if state in {"ROBOTS_OK", "ROBOTS_404", "ROBOTS_410"} else self.error_ttl_seconds
        if state == "ROBOTS_429" and retry_after is not None:
            ttl = max(ttl, retry_after)
        return _CacheEntry(state, robots_url, fetched_at, status, digest,
                           retry_after, parser, time.monotonic() + ttl)
