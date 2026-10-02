"""Guarded HTTP recovery for URLs that a prior crawl could not fetch.

The caller supplies a synchronous HTTP fetch function (Scrapling in Colab).
Every request, including each redirect hop and retry, first gets a fresh
robots decision and shares the normal crawler's per-host pacing.
"""

from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

from .crawl import Fetcher
from .dataset import url_host
from .robots import _retry_after


@dataclass(frozen=True)
class RecoveryConfig:
    attempts: int = 2
    max_redirects: int = 5
    max_body_bytes: int = 16 * 1024 * 1024
    timeout_seconds: float = 45.0
    max_inline_wait_seconds: float = 120.0
    impersonate: str | None = None

    def validate(self) -> None:
        if min(self.attempts, self.max_redirects, self.max_body_bytes) < 1:
            raise ValueError("Recovery attempts, redirects and byte limit must be positive.")
        if self.timeout_seconds <= 0 or self.max_inline_wait_seconds < 0:
            raise ValueError("Invalid recovery timeout or wait limit.")


def _public_url(url: str) -> str:
    host = url_host(url)
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError("Local source URL is not allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if not address.is_global:
        raise ValueError("Non-public source IP is not allowed.")
    return host


def _same_target(requested: str, reported: str) -> bool:
    left, right = urlsplit(requested), urlsplit(reported)
    return (left.scheme.lower(), left.netloc.lower(), left.path or "/", left.query) == (
        right.scheme.lower(), right.netloc.lower(), right.path or "/", right.query,
    )


async def guarded_http_fetch(
    url: str,
    guard: Fetcher,
    get: Callable[..., Any],
    *,
    user_agent: str,
    config: RecoveryConfig | None = None,
    held_hosts: set[str] | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> tuple[dict, bytes | None]:
    """Fetch with explicit robots checks; never follow unexamined redirects.

    `get` must accept Scrapling Fetcher.get-style arguments. A result of
    `outcome == 'http_200'` is only a fetched body, not a verified article.
    """
    settings = config or RecoveryConfig()
    settings.validate()
    if not guard.config.respect_robots:
        raise ValueError("Guarded recovery requires respect_robots=True.")
    _public_url(url)
    held_hosts = held_hosts if held_hosts is not None else set()
    checks: list[dict] = []
    requests: list[dict] = []
    last: dict = {}
    for attempt in range(1, settings.attempts + 1):
        current = url
        redirects: list[dict] = []
        for hop in range(settings.max_redirects + 1):
            try:
                host = _public_url(current)
            except ValueError as exc:
                return ({"outcome": "invalid_redirect_target", "error_message": str(exc),
                         "attempts": attempt, "request_log": requests,
                         "robots_checks": checks, "redirect_chain": redirects}, None)
            if host in held_hosts:
                return ({"outcome": "host_rate_limit_hold", "final_url": current,
                         "attempts": attempt, "request_log": requests,
                         "robots_checks": checks, "redirect_chain": redirects}, None)
            decision = await guard.robots_decision(current)
            checks.append({"url": current, **decision.record()})
            if not decision.allowed:
                return ({"outcome": "robots_hold", "robots_state": decision.state,
                         "attempts": attempt, "request_log": requests,
                         "robots_checks": checks, "redirect_chain": redirects}, None)
            minimum = max(decision.crawl_delay_seconds or 0.0,
                          decision.request_rate_gap_seconds or 0.0)
            await guard.pace(current, minimum)
            try:
                response = get(
                    current, timeout=settings.timeout_seconds, retries=0,
                    follow_redirects=False, stealthy_headers=False,
                    impersonate=settings.impersonate,
                    headers={"User-Agent": user_agent},
                )
            except Exception as exc:
                last = {"outcome": "fetch_error", "error_type": type(exc).__name__,
                        "error_message": str(exc)[:300], "final_url": current}
                requests.append({"url": current, "attempt": attempt, "hop": hop,
                                 "error_type": last["error_type"]})
                break
            reported_url = str(response.url)
            status = int(response.status)
            requests.append({"url": current, "attempt": attempt, "hop": hop,
                             "http_status": status})
            if not _same_target(current, reported_url):
                return ({"outcome": "unguarded_redirect_detected", "attempts": attempt,
                         "final_url": reported_url, "request_log": requests,
                         "robots_checks": checks, "redirect_chain": redirects}, None)
            if status in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    return ({"outcome": "redirect_without_location", "attempts": attempt,
                             "final_url": current, "request_log": requests,
                             "robots_checks": checks, "redirect_chain": redirects}, None)
                target = urljoin(current, location)
                redirects.append({"from": current, "to": target, "http_status": status})
                if hop == settings.max_redirects:
                    return ({"outcome": "redirect_limit", "attempts": attempt,
                             "final_url": current, "request_log": requests,
                             "robots_checks": checks, "redirect_chain": redirects}, None)
                current = target
                continue
            last = {"http_status": status, "final_url": current,
                    "content_type": response.headers.get("content-type", ""),
                    "attempts": attempt}
            if status == 200:
                body = bytes(response.body)
                if len(body) > settings.max_body_bytes:
                    last["outcome"] = "body_too_large"
                elif not body:
                    last["outcome"] = "empty_body"
                else:
                    return ({**last, "outcome": "http_200", "body_bytes": len(body),
                             "request_log": requests, "robots_checks": checks,
                             "redirect_chain": redirects}, body)
                break
            last["outcome"] = f"http_{status}"
            if status == 429 or status >= 500:
                retry_after = _retry_after(response.headers.get("retry-after"))
                if status == 429 and retry_after is None:
                    retry_after = 60.0
                if retry_after is not None:
                    last["retry_after_seconds"] = retry_after
                if status == 429 and attempt == settings.attempts:
                    held_hosts.add(host)
                    last["outcome"] = "rate_limited_retry_later"
                    return ({**last, "request_log": requests, "robots_checks": checks,
                             "redirect_chain": redirects}, None)
                if retry_after is not None and retry_after > settings.max_inline_wait_seconds:
                    held_hosts.add(host)
                    last["outcome"] = "rate_limited_retry_later" if status == 429 else "server_retry_later"
                    return ({**last, "request_log": requests, "robots_checks": checks,
                             "redirect_chain": redirects}, None)
            break
        else:  # pragma: no cover - loop always returns at its redirect limit
            raise AssertionError("Unreachable redirect loop exit")
        if attempt < settings.attempts and (last["outcome"] == "fetch_error" or
                                             last["outcome"] == "http_429" or
                                             last.get("http_status", 0) >= 500):
            delay = last.get("retry_after_seconds", min(2.0 ** attempt, 30.0))
            await sleep(delay)
            continue
        return ({**last, "request_log": requests, "robots_checks": checks,
                 "redirect_chain": redirects}, None)
    raise AssertionError("Unreachable recovery attempt loop")
