"""Read robots through the same Chrome TLS transport as recovery HTTP.

This changes the HTTP client, not robots access policy. 403, unreachable robots,
rate limits and explicit Disallow still produce a hold in RobotsResolver.
"""

from __future__ import annotations

import asyncio
from typing import Callable

import httpx

from .recovery import _public_url
from .robots import _retry_after, is_robots_html, valid_robots_bytes


class ChromeRobotsTransport(httpx.AsyncBaseTransport):
    def __init__(self, get: Callable, *, attempts: int = 2, observer=None,
                 sleep=asyncio.sleep, max_wait_seconds: float = 30, browser_read=None):
        self.get, self.attempts, self.observer = get, attempts, observer
        self.sleep, self.max_wait_seconds = sleep, max_wait_seconds
        self.browser_read = browser_read
        self._request_lock = asyncio.Lock()
        if attempts < 1 or max_wait_seconds < 0:
            raise ValueError("Invalid robots transport retry limits.")

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _public_url(str(request.url))
        if request.method != "GET":
            raise ValueError("Robots transport only accepts GET.")
        for attempt in range(1, self.attempts + 1):
            try:
                response = await self._get(request)
            except Exception as exc:
                if self.observer:
                    self.observer({"url": str(request.url), "attempt": attempt,
                                   "error_type": type(exc).__name__, "error": str(exc)[:300]}, None)
                if attempt < self.attempts:
                    await self.sleep(min(3 * attempt, self.max_wait_seconds))
                    continue
                recovered = await self._browser_response(request)
                if recovered is not None:
                    return recovered
                if "timeout" in type(exc).__name__.lower() or getattr(exc, "code", None) == 28:
                    raise httpx.ReadTimeout(str(exc), request=request) from exc
                raise httpx.ConnectError(str(exc), request=request) from exc
            if str(response.url) != str(request.url):
                raise httpx.RequestError("Robots transport followed an unguarded redirect.", request=request)
            status, body = int(response.status), bytes(response.body)
            headers = {str(k).lower(): str(v) for k, v in response.headers.items()
                       if str(k).lower() not in {"content-encoding", "content-length", "transfer-encoding"}}
            if self.observer:
                self.observer({"url": str(request.url), "attempt": attempt, "http_status": status,
                               "response_headers": {k: v for k, v in headers.items()
                                                    if k.lower() in {"content-type", "retry-after", "cf-mitigated", "location"}}}, body)
            if 500 <= status < 600 and attempt < self.attempts:
                wait = _retry_after(headers.get("retry-after"))
                wait = max(3 * attempt, wait or 0)
                if wait <= self.max_wait_seconds:
                    await self.sleep(wait)
                    continue
            if status in {403, 408} or status >= 500 or (status == 200 and is_robots_html(body)):
                wait = _retry_after(headers.get("retry-after")) or 0
                if wait <= self.max_wait_seconds:
                    if wait:
                        await self.sleep(wait)
                    recovered = await self._browser_response(request)
                    if recovered is not None:
                        return recovered
            # Authentication/rate-limit responses are never released by a fallback.
            return httpx.Response(status, content=body, headers=headers, request=request)
        raise AssertionError("Unreachable robots transport loop")

    async def _get(self, request):
        # Do not dispatch many worker threads that will issue requests after
        # their browser page has closed. Drain the single in-flight request
        # before closing the shared HTTP session; queued requests can cancel.
        async with self._request_lock:
            worker = asyncio.create_task(asyncio.to_thread(
                self.get, str(request.url), timeout=45, retries=0,
                follow_redirects=False, stealthy_headers=False,
                impersonate="chrome", headers={"User-Agent": request.headers["user-agent"]},
            ))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                try:
                    await worker
                except Exception:
                    pass
                raise

    async def _browser_response(self, request):
        if self.browser_read is None:
            return None
        result = await self.browser_read(str(request.url))
        if result is None:
            return None
        if self.observer:
            self.observer({"url": str(request.url), "method": "browser_robots",
                           **result["evidence"]}, result.get("body"))
            for name, body in result.get("assets", {}).items():
                self.observer({"url": str(request.url), "method": "browser_robots_asset", "name": name}, body)
        body, status = result.get("body"), result.get("status")
        # A rendered <pre> or an engine success flag cannot authorize article access.
        if status == 200 and valid_robots_bytes(body):
            return httpx.Response(200, content=body, headers={"content-type": "text/plain"}, request=request)
        if status == 429:
            return httpx.Response(429, headers=result.get("headers", {}), request=request)
        return None
