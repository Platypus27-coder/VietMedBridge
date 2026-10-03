"""Scrapling HTTP transport for the checkpointed VietMedBridge crawler.

Scrapling supplies a reusable curl_cffi session and Chrome TLS fingerprinting.
Redirects remain disabled here so ``Fetcher`` can check robots.txt at every hop.
"""

from __future__ import annotations

import asyncio

import httpx


class ScraplingTransport(httpx.AsyncBaseTransport):
    """Adapt Scrapling's async ``FetcherSession`` to httpx's transport API."""

    engine_name = "scrapling-fetcher-session"

    def __init__(self, *, user_agent: str, timeout_seconds: float = 30.0,
                 session_factory=None):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.session_factory = session_factory
        self._manager = None
        self._session = None
        self._open_lock = asyncio.Lock()

    async def _get_session(self):
        if self._session is not None:
            return self._session
        async with self._open_lock:
            if self._session is None:
                if self.session_factory is None:
                    try:
                        from scrapling.fetchers import FetcherSession
                    except ImportError as exc:
                        raise RuntimeError(
                            "Scrapling is required for this crawl. Install the pinned "
                            "crawl_engines extra before starting Notebook 01."
                        ) from exc
                    self._manager = FetcherSession(
                        impersonate="chrome", retries=0, follow_redirects=False,
                        stealthy_headers=False,
                        headers={"User-Agent": self.user_agent},
                    )
                else:
                    self._manager = self.session_factory()
                self._session = await self._manager.__aenter__()
        return self._session

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET":
            raise httpx.UnsupportedProtocol("VietMedBridge only fetches source URLs with GET.")
        session = await self._get_session()
        request_url = request.url.copy_with(fragment=None)
        try:
            response = await session.get(
                str(request_url), timeout=self.timeout_seconds,
                retries=0, follow_redirects=False, stealthy_headers=False,
                headers={str(k): str(v) for k, v in request.headers.items()},
            )
        except Exception as exc:
            if isinstance(exc, httpx.RequestError):
                raise
            raise httpx.ConnectError(
                f"Scrapling request failed: {type(exc).__name__}: {str(exc)[:300]}",
                request=request,
            ) from exc

        reported_url = httpx.URL(str(response.url)).copy_with(fragment=None)
        if reported_url != request_url:
            raise httpx.RequestError(
                "Scrapling followed a redirect before the crawler could recheck robots.txt.",
                request=request,
            )
        headers = {
            str(key).lower(): str(value)
            for key, value in response.headers.items()
            if str(key).lower() not in {
                "content-encoding", "content-length", "transfer-encoding",
            }
        }
        return httpx.Response(
            int(response.status), content=bytes(response.body), headers=headers,
            request=request,
        )

    async def aclose(self) -> None:
        if self._manager is not None:
            manager, self._manager = self._manager, None
            self._session = None
            await manager.__aexit__(None, None, None)
