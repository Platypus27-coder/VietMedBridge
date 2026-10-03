"""Use Crawl4AI only for sparse JavaScript application shells.

The ordinary HTTP response remains the first capture. Crawl4AI's browser is
opened only for a small, bounded set of HTTPS pages that look like client-side
shells. Its CDP request guard rechecks VietMedBridge robots policy on every
document/resource request. Access-denied, CAPTCHA, robots-blocked and rate-limit
responses are never sent to this renderer.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from .browser_probe import BrowserProbeConfig
from .recovery import _public_url
from .text import EXPECTED_PARSE_ERRORS, extract_source

_APP_MARKERS = (
    b"__next_data__", b"id=\"__next\"", b"id='__next'", b"id=\"root\"",
    b"id='root'", b"id=\"app\"", b"id='app'", b"data-v-app",
    b"data-reactroot", b"ng-version=", b"__nuxt__",
    b"enable javascript", b"javascript is required",
)
_BLOCK_MARKERS = (
    "access denied", "forbidden", "just a moment", "attention required",
    "request blocked", "unusual traffic", "rate limit", "temporarily blocked",
    "captcha", "verify you are human", "sign in to continue",
)


def looks_like_javascript_shell(body: bytes, content_type: str) -> bool:
    """Cheaply select thin app shells; do not browser-render ordinary HTML."""
    if "html" not in (content_type or "").lower() or len(body) > 2 * 1024 * 1024:
        return False
    sample = body[:512 * 1024]
    lowered = sample.lower()
    if not any(marker in lowered for marker in _APP_MARKERS):
        return False
    try:
        soup = BeautifulSoup(sample, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        has_script = bool(soup.find("script"))
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        visible = soup.get_text(" ", strip=True)
    except Exception:
        return False
    lower_visible = (title + " " + visible[:1500]).lower()
    if any(signal in lower_visible for signal in _BLOCK_MARKERS):
        return False
    # Normal pages with an app framework still have enough server-rendered text.
    return len(visible) < 180 and has_script


class Crawl4AIEnhancer:
    """Bounded renderer attached to the baseline crawler's per-ID fetch path."""

    def __init__(self, *, max_render_pages_per_call: int = 12,
                 render_wait_seconds: float = 2.0, min_rendered_chars: int = 300,
                 max_requests: int = 120, navigation_timeout_ms: int = 25000):
        if max_render_pages_per_call < 0 or not 0 <= render_wait_seconds <= 10:
            raise ValueError("Invalid Crawl4AI page or wait limit.")
        if min_rendered_chars < 80 or max_requests < 1 or navigation_timeout_ms < 1000:
            raise ValueError("Invalid Crawl4AI capture limits.")
        self.max_render_pages_per_call = max_render_pages_per_call
        self.render_wait_seconds = render_wait_seconds
        self.min_rendered_chars = min_rendered_chars
        self.max_requests = max_requests
        self.navigation_timeout_ms = navigation_timeout_ms
        self._rendered = 0
        self._claim_lock = asyncio.Lock()
        self._browser_semaphore = asyncio.Semaphore(1)

    @property
    def identity(self) -> dict:
        return {
            "engine": "crawl4ai-browser-render-v1",
            "max_render_pages_per_call": self.max_render_pages_per_call,
            "render_wait_seconds": self.render_wait_seconds,
            "min_rendered_chars": self.min_rendered_chars,
            "max_requests": self.max_requests,
            "navigation_timeout_ms": self.navigation_timeout_ms,
            "uses_stealth": False,
            "uses_robots_request_guard": True,
        }

    async def _claim_page(self) -> bool:
        async with self._claim_lock:
            if self._rendered >= self.max_render_pages_per_call:
                return False
            self._rendered += 1
            return True

    async def enhance(self, url: str, metadata: dict, body: bytes, guard) -> dict | None:
        if metadata.get("http_status") != 200:
            return None
        if not looks_like_javascript_shell(body, metadata.get("content_type", "")):
            return None
        if not guard.config.respect_robots:
            return {"body": None, "metadata": {"render_attempt": {
                "engine": "crawl4ai", "state": "robots_enforcement_required",
            }}}
        render_url = metadata.get("final_url") or url
        host = _public_url(render_url)
        if urlsplit(render_url).scheme != "https":
            return {"body": None, "metadata": {"render_attempt": {
                "engine": "crawl4ai", "state": "https_required",
            }}}
        if not await self._claim_page():
            return {"body": None, "metadata": {"render_attempt": {
                "engine": "crawl4ai", "state": "session_budget_exhausted",
            }}}

        static_chars = 0
        try:
            static = extract_source(body, metadata.get("content_type", "text/html"), source_url=render_url)
            static_chars = len(static["source_text"])
        except EXPECTED_PARSE_ERRORS:
            pass

        settings = BrowserProbeConfig(
            allowed_hosts=(host,), document_path_prefix="/",
            max_requests=self.max_requests, max_documents=1,
            navigation_timeout_ms=self.navigation_timeout_ms,
            render_wait_seconds=self.render_wait_seconds,
            document_delay_seconds=max(guard.config.per_host_delay, 1.0),
            max_body_bytes=guard.config.max_bytes,
        )
        async with self._browser_semaphore:
            try:
                from .crawl4ai_probe import crawl4ai_browser_probe

                record, assets = await crawl4ai_browser_probe(
                    render_url, guard, config=settings,
                    user_agent=guard.config.user_agent,
                )
            except Exception as exc:
                return {"body": None, "metadata": {"render_attempt": {
                    "engine": "crawl4ai", "state": "browser_error",
                    "error_type": type(exc).__name__, "error": str(exc)[:300],
                }}}

        rendered = assets.get("rendered.html")
        evidence = {
            "engine": "crawl4ai", "state": "rendered",
            "http_status": record.get("http_status"),
            "rendered_http_status": record.get("rendered_http_status"),
            "final_url": record.get("final_url"),
            "rendered_url": record.get("rendered_url"),
            "guard_errors": record.get("guard_errors", []),
            "held_hosts": record.get("held_hosts", []),
            "request_log": record.get("request_log", []),
            "response_evidence": record.get("response_evidence", {}),
            "engine_error": record.get("engine_error", ""),
        }
        if (not rendered or record.get("http_status") != 200 or
                record.get("rendered_http_status") != 200 or
                record.get("guard_errors") or record.get("held_hosts")):
            evidence["state"] = "render_not_accepted"
            return {"body": None, "metadata": {"render_attempt": evidence}}
        try:
            document = extract_source(
                rendered, "text/html", source_url=record.get("rendered_url") or render_url,
            )
        except EXPECTED_PARSE_ERRORS as exc:
            evidence.update(state="render_extract_rejected", error=f"{type(exc).__name__}: {str(exc)[:250]}")
            return {"body": None, "metadata": {"render_attempt": evidence}}
        rendered_chars = len(document["source_text"])
        if (rendered_chars < self.min_rendered_chars or
                rendered_chars <= static_chars or not document["title"].strip()):
            evidence.update(state="render_not_better", static_chars=static_chars,
                            rendered_chars=rendered_chars)
            return {"body": None, "metadata": {"render_attempt": evidence}}

        evidence.update(state="render_selected", static_chars=static_chars,
                        rendered_chars=rendered_chars)
        return {
            "body": rendered,
            "metadata": {
                "capture_kind": "crawl4ai_rendered_dom",
                "final_url": record.get("rendered_url") or record.get("final_url") or render_url,
                "render_attempt": evidence,
            },
        }
