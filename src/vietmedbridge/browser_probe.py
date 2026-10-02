"""Small browser diagnostic for allowed public articles, separate from the corpus.

Use Scrapling's AsyncDynamicSession.context with a fresh profile. Install CDP
interception before navigating: unlike page.route, it reports redirect requests
as well. This probe intentionally restricts hosts, methods, frames and volume.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from .crawl import Fetcher
from .recovery import _public_url
from .robots import RobotsResolver


def response_evidence(body: bytes, headers: Any) -> dict:
    """Diagnostic hints are observations, not proof of a particular blocker."""
    selected = {str(k).lower(): str(v)[:1000] for k, v in headers.items()
                if str(k).lower() in {
                    "content-type", "server", "retry-after", "location",
                    "cf-ray", "cf-mitigated", "x-cache", "x-request-id",
                }}
    soup = BeautifulSoup(body[:512 * 1024], "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for node in soup(["script", "style"]):
        node.decompose()
    visible = soup.get_text(" ", strip=True)
    lower = (title + " " + visible[:2000]).lower()
    signals = [phrase for phrase in (
        "access denied", "forbidden", "just a moment", "attention required",
        "captcha", "verify you are human", "enable javascript", "sign in",
    ) if phrase in lower]
    return {"body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
            "response_headers": selected, "page_title": title,
            "visible_preview": visible[:1000], "block_page_signals": signals}


@dataclass(frozen=True)
class BrowserProbeConfig:
    allowed_hosts: tuple[str, ...] = ("nhathuoclongchau.com.vn",)
    document_path_prefix: str = "/bai-viet/"
    max_requests: int = 160
    max_documents: int = 6
    navigation_timeout_ms: int = 60000
    render_wait_seconds: float = 5.0
    document_delay_seconds: float = 3.0
    max_body_bytes: int = 16 * 1024 * 1024

    def validate(self) -> None:
        if not self.allowed_hosts or min(self.max_requests, self.max_documents,
                                        self.navigation_timeout_ms, self.max_body_bytes) <= 0:
            raise ValueError("Invalid browser probe limits.")
        if not 0 <= self.render_wait_seconds <= 30 or self.document_delay_seconds < 0:
            raise ValueError("Invalid browser probe timing.")


class _RequestGate:
    def __init__(self, cdp, guard: Fetcher, browser_robots: RobotsResolver,
                 frame_id: str, settings: BrowserProbeConfig):
        self.cdp, self.guard, self.browser_robots = cdp, guard, browser_robots
        self.frame_id, self.settings = frame_id, settings
        self.log: list[dict] = []
        self.tasks: set[asyncio.Task] = set()
        self.errors: list[str] = []
        self.documents = 0
        self.held_hosts: set[str] = set()

    def paused(self, event: dict) -> None:
        task = asyncio.create_task(self.handle(event))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def handle(self, event: dict) -> None:
        request = event["request"]
        url, method = request["url"], request["method"]
        kind = event["resourceType"]
        row = {"url": url, "method": method, "resource_type": kind,
               "redirected_request_id": event.get("redirectedRequestId"),
               "decision": "blocked"}
        self.log.append(row)
        try:
            host = _public_url(url)
            parts = urlsplit(url)
            if parts.scheme != "https" or parts.port not in (None, 443):
                row["reason"] = "https_only"
            elif host not in self.settings.allowed_hosts:
                row["reason"] = "host_outside_probe"
            elif host in self.held_hosts:
                row["reason"] = "rate_limit_hold"
            elif method != "GET":
                row["reason"] = "method_outside_probe"
            elif len(self.log) > self.settings.max_requests:
                row["reason"] = "request_budget"
            elif kind not in {"Document", "Script", "Stylesheet", "XHR", "Fetch", "Image", "Font"}:
                row["reason"] = "resource_outside_probe"
            elif kind == "Document" and (
                event["frameId"] != self.frame_id or
                not parts.path.startswith(self.settings.document_path_prefix)
            ):
                row["reason"] = "document_outside_probe"
            else:
                if kind == "Document":
                    self.documents += 1
                if self.documents > self.settings.max_documents:
                    row["reason"] = "document_budget"
                else:
                    decision = await self.guard.robots_decision(url)
                    browser_decision = await self.browser_robots.decision(url)
                    row.update(decision.record())
                    row["browser_robots_state"] = browser_decision.state
                    if not decision.allowed or not browser_decision.allowed:
                        row["reason"] = "robots_hold"
                    else:
                        gap = max(
                            decision.crawl_delay_seconds or 0,
                            decision.request_rate_gap_seconds or 0,
                            browser_decision.crawl_delay_seconds or 0,
                            browser_decision.request_rate_gap_seconds or 0,
                            self.settings.document_delay_seconds if kind == "Document" else 0,
                        )
                        await self.guard.pace(url, gap)
                        # Recheck a rate-limit event that may have arrived while pacing.
                        if host in self.held_hosts:
                            row["reason"] = "rate_limit_hold"
                        else:
                            await self.cdp.send("Fetch.continueRequest", {"requestId": event["requestId"]})
                            row.update(decision="allowed", reason="robots_allowed")
                            return
        except Exception as exc:
            row["reason"] = "guard_error"
            self.errors.append(f"{type(exc).__name__}: {str(exc)[:250]}")
        try:
            await self.cdp.send("Fetch.failRequest", {
                "requestId": event["requestId"], "errorReason": "BlockedByClient",
            })
        except Exception as exc:
            self.errors.append(f"Abort failed: {type(exc).__name__}: {str(exc)[:250]}")


async def guarded_browser_probe(context, url: str, guard: Fetcher, *,
                                config: BrowserProbeConfig | None = None) -> tuple[dict, dict[str, bytes]]:
    """Return evidence even for HTTP 403; never promote it to an article here.

    The caller must create a fresh context with service_workers='block'. Setup
    exceptions propagate before goto. No page_setup callback, automatic browser
    retries, authentication, challenge solver or proxy rotation is used.
    """
    settings = config or BrowserProbeConfig()
    settings.validate()
    if not guard.config.respect_robots:
        raise ValueError("Browser probe requires robots enforcement.")
    host = _public_url(url)
    if host not in settings.allowed_hosts or not urlsplit(url).path.startswith(settings.document_path_prefix):
        raise ValueError("URL outside browser probe scope.")
    page = await context.new_page()
    cdp = None
    gate = None
    assets: dict[str, bytes] = {}
    record: dict = {"method": "scrapling_dynamic_cdp_probe", "url": url,
                    "article_candidate": False}
    response_log: list[dict] = []
    try:
        # Disable alternate execution contexts in this single-article diagnostic.
        await page.add_init_script("""
            window.open = () => null;
            window.Worker = window.SharedWorker = class {
                constructor() { throw new Error('Workers disabled in article probe'); }
            };
        """)
        cdp = await context.new_cdp_session(page)
        frame_id = (await cdp.send("Page.getFrameTree"))["frameTree"]["frame"]["id"]
        user_agent = await page.evaluate("navigator.userAgent")
        record["browser_user_agent"] = user_agent
        record["robots_product"] = guard.robots.user_agent
        browser_robots = RobotsResolver(guard.client, guard.pace, user_agent=user_agent)
        gate = _RequestGate(cdp, guard, browser_robots, frame_id, settings)
        cdp.on("Fetch.requestPaused", gate.paused)
        await cdp.send("Fetch.enable", {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]})
        # Fetch interception does not cover WebSockets.
        async def deny_socket(socket):
            await socket.close()
        await page.route_web_socket("**/*", deny_socket)

        def observed(response):
            response_log.append({"url": response.url, "http_status": response.status,
                                 "resource_type": response.request.resource_type})
            if response.status == 429:
                gate.held_hosts.add(urlsplit(response.url).hostname)
        page.on("response", observed)
        response = None
        try:
            response = await page.goto(url, wait_until="domcontentloaded",
                                       timeout=settings.navigation_timeout_ms)
            if response is not None:
                record.update(http_status=response.status, final_url=response.url,
                              content_type=response.headers.get("content-type", ""))
            if response is not None and response.status == 200:
                await asyncio.sleep(settings.render_wait_seconds)
            record["outcome"] = f"browser_http_{response.status}" if response else "browser_no_response"
        except Exception as exc:
            record.update(outcome="browser_navigation_error", error_type=type(exc).__name__,
                          error_message=str(exc)[:500])
        # Capture the response bytes and rendered DOM separately, including error pages.
        for name, read in (
            ("response.html", response.body if response is not None else None),
            ("rendered.html", page.content),
        ):
            if read is None:
                continue
            try:
                body = await read()
                body = body.encode("utf-8") if isinstance(body, str) else body
                if len(body) <= settings.max_body_bytes:
                    assets[name] = body
                    if name == "response.html":
                        record["response_evidence"] = response_evidence(body, response.headers)
                else:
                    record.setdefault("capture_errors", []).append(name + ": body_too_large")
            except Exception as exc:
                record.setdefault("capture_errors", []).append(name + ": " + type(exc).__name__)
        try:
            assets["screenshot.png"] = await page.screenshot(full_page=False, timeout=10000)
            record["page_title"] = await page.title()
            record["rendered_url"] = page.url
            document_responses = [r for r in response_log if r["resource_type"] == "document"
                                  and r["url"] == page.url]
            if document_responses:
                record["rendered_http_status"] = document_responses[-1]["http_status"]
        except Exception as exc:
            record.setdefault("capture_errors", []).append("screenshot: " + type(exc).__name__)
    finally:
        # Close the page before cancelling handlers: cancelling must not release
        # paused requests as unguarded traffic during teardown.
        await page.close()
        if gate is not None:
            tasks = list(gate.tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            record.update(request_log=gate.log, guard_errors=gate.errors,
                          held_hosts=sorted(gate.held_hosts))
        record["response_log"] = response_log
        if cdp is not None:
            try:
                await cdp.detach()
            except Exception:
                pass  # The page was already closed; no requests can resume.
    return record, assets
