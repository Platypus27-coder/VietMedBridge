"""Bounded, stateful recovery using both Crawl4AI and Scrapling browser engines."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from .browser_probe import response_evidence
from .crawl4ai_probe import Crawl4AIProbe
from .recovery import _public_url
from .recovery_browser_gate import RecoveryBrowserConfig, RecoveryRequestGate
from .robots import valid_robots_bytes
from .text import EXPECTED_PARSE_ERRORS, extract_source


def article_capture_ready(record, assets):
    if record.get("http_status") != 200 or record.get("rendered_http_status") != 200 or record.get("guard_errors"):
        return False
    for name in ("response.html", "rendered.html"):
        if name not in assets:
            continue
        try:
            doc = extract_source(assets[name], record.get("content_type") or "text/html",
                                 source_url=record.get("final_url"))
            if len(doc["source_text"]) >= 300 and doc["title"].strip():
                return True
        except EXPECTED_PARSE_ERRORS:
            pass
    return False


class RecoveryProbe(Crawl4AIProbe):
    def __init__(self, guard, settings, *, solver=None):
        super().__init__(guard, settings, gate_factory=RecoveryRequestGate)
        self.solver = solver
        self.storage_state = None
        self.network_documents = []

    async def before_goto(self, page, context, url, **kwargs):
        await super().before_goto(page, context, url, **kwargs)
        def observed(event):
            response = event["response"]
            if int(response["status"]) == 429:
                self.gate.held_hosts.add(urlsplit(response["url"]).hostname)
            if event.get("type") == "Document" and event.get("frameId") == self.gate.frame_id:
                self.network_documents.append(event)
        self.cdp.on("Network.responseReceived", observed)
        # Patchright can suppress Playwright response events. Capture protocol
        # evidence explicitly rather than infer status/body from rendered DOM.
        await self.cdp.send("Network.enable", {"maxResourceBufferSize": self.settings.max_body_bytes,
                                              "maxTotalBufferSize": 2 * self.settings.max_body_bytes})
        return page

    async def capture(self, page, html, **kwargs):
        if not self.installed:
            raise RuntimeError("Recovery guard was not installed.")
        current = await page.content()
        evidence = response_evidence(current.encode(), {})
        challenged = any(s in evidence["block_page_signals"] for s in (
            "just a moment", "captcha", "verify you are human", "attention required"))
        if challenged and self.settings.challenge_timeout_seconds:
            try:
                if self.solver:
                    await asyncio.wait_for(self.solver(page), self.settings.challenge_timeout_seconds)
                    self.record["challenge_action"] = "scrapling_solver_finished"
                else:
                    await page.wait_for_function("""() => !/just a moment|verify you are human|attention required/i.test(
                        document.title + ' ' + (document.body?.innerText || '').slice(0, 1000))""",
                        timeout=int(self.settings.challenge_timeout_seconds * 1000))
                    self.record["challenge_action"] = "browser_wait_finished"
            except Exception as exc:
                self.record["challenge_action"] = f"unresolved:{type(exc).__name__}"
        if not self.settings.robots_bootstrap_url:
            await self.wait_for_article(page)
        await super().capture(page, await page.content(), **kwargs)
        documents = [e for e in self.network_documents if e["response"]["url"] == page.url]
        if documents:
            event = documents[-1]
            response = event["response"]
            headers = {str(k).lower(): str(v) for k, v in response["headers"].items()}
            self.record.update(http_status=int(response["status"]), rendered_http_status=int(response["status"]),
                               content_type=headers.get("content-type", response.get("mimeType", "")))
            try:
                captured = await self.cdp.send("Network.getResponseBody", {"requestId": event["requestId"]})
                body = base64.b64decode(captured["body"]) if captured["base64Encoded"] else captured["body"].encode()
                if len(body) <= self.settings.max_body_bytes:
                    self.assets["response.html"] = body
                    self.record["response_evidence"] = response_evidence(body, headers)
                    self.record["response_capture_backend"] = "cdp_network"
            except Exception as exc:
                self.record["cdp_body_error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
        self.storage_state = await page.context.storage_state()
        if self.record.get("http_status") != 200:
            try:
                self.assets["screenshot.png"] = await page.screenshot(full_page=False, timeout=5000)
            except Exception as exc:
                self.record["screenshot_error"] = type(exc).__name__
        return page

    async def wait_for_article(self, page):
        selector = self.settings.article_selector
        try:
            if self.settings.readiness_timeout_ms:
                await page.wait_for_function("""selector => {
                    const e = document.querySelector(selector);
                    return !!e && (e.innerText || '').trim().length >= 300;
                }""", arg=selector, timeout=self.settings.readiness_timeout_ms)
            self.record["readiness"] = "article_container_ready"
        except Exception as exc:
            self.record["readiness"] = f"container_wait:{type(exc).__name__}"
        # Bounded scrolling also triggers lazy article paragraphs. Stop when
        # the selected source text is stable on two observations at the bottom.
        previous = None
        for step in range(self.settings.scroll_steps + 1):
            state = await page.evaluate("""selector => {
                const e = document.querySelector(selector) || document.body;
                return {text: (e?.innerText || '').slice(0, 200000),
                    bottom: window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2};
            }""", selector)
            digest = hashlib.sha256(state["text"].encode()).hexdigest()
            if previous == digest and state["bottom"]:
                self.record["content_stable"] = True
                break
            previous = digest
            if step < self.settings.scroll_steps:
                await page.evaluate("window.scrollBy(0, Math.max(window.innerHeight, 800))")
                await page.wait_for_timeout(750)
        self.record["scroll_observations"] = step + 1


class AdvancedRecoveryRuntime:
    """Reuse per-domain state with a small LRU browser pool on Colab CPU.

    Browser profiles stay in local WORK_DIR and are never included in evidence
    ZIPs. Only this run's fresh profiles are used, never a user's signed-in browser.
    """
    def __init__(self, work_dir, *, profiles=None, session_limit=2, browser_executable=None):
        self.work_dir = Path(work_dir)
        self.profiles = profiles or {}
        self.session_limit = session_limit
        if not 1 <= session_limit <= 4:
            raise ValueError("Use 1–4 active browser sessions.")
        self.browser_executable = browser_executable
        self.sessions, self.active, self.states = OrderedDict(), set(), {}
        self.resource_hosts, self.preferred, self.robots_cache = {}, {}, {}
        self.http_manager = self.http = self.guard = self.rate_limiter = None
        self.http_lock = RLock()
        self.resource_locks, self.resource_times = {}, {}
        self.robots_browser_lock = asyncio.Lock()

    @property
    def identity(self):
        return {"strategy": "dual-engine-recovery-v1", "profiles": self.profiles,
                "session_limit": self.session_limit,
                "engines": ["crawl4ai_stealth", "scrapling_stealth"],
                "robots_engines": ["scrapling_stealth", "crawl4ai_stealth"]}

    async def __aenter__(self):
        from crawl4ai import RateLimiter
        from scrapling.fetchers import FetcherSession
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = RateLimiter(base_delay=(1, 2), max_delay=30, max_retries=2)
        self.http_manager = FetcherSession(impersonate="chrome", retries=0, follow_redirects=False,
                                           stealthy_headers=False)
        self.http = self.http_manager.__enter__()
        return self

    async def __aexit__(self, *_args):
        for key in list(self.sessions):
            await self._close(key)
        if self.http_manager:
            self.http_manager.__exit__(None, None, None)

    def configure_guard(self, guard):
        self.guard = guard
        original = guard.pace
        async def pace(url, minimum_delay=0):
            await original(url, minimum_delay)
            await self.rate_limiter.wait_if_needed(url)
        guard.pace = pace
        guard.robots.pace = pace
        async def resource_pace(url, _minimum=0):
            # Browser subresources may load in a bounded burst; explicit robots
            # delays still use guard.pace in the gate, like document navigation.
            host = _public_url(url)
            async with self.resource_locks.setdefault(host, asyncio.Lock()):
                wait = self.resource_times.get(host, 0) - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self.resource_times[host] = time.monotonic() + 0.1
        guard.browser_resource_pace = resource_pace

    def get(self, url, **kwargs):
        with self.http_lock:
            result = self.http.get(url, **kwargs)
            domain = self.rate_limiter.get_domain(url)
            if domain in self.rate_limiter.domains:
                self.rate_limiter.update_delay(url, int(result.status))
            if "html" in result.headers.get("content-type", "").lower():
                self._discover_resources(str(result.url), bytes(result.body))
            return result

    def _discover_resources(self, url, body):
        hosts = self.resource_hosts.setdefault(_public_url(url), set())
        soup = BeautifulSoup(body[:512 * 1024], "lxml")
        for tag in soup.select("script[src], link[href]"):
            target = urljoin(url, tag.get("src") or tag.get("href"))
            try:
                if urlsplit(target).scheme == "https" and len(hosts) < 32:
                    hosts.add(_public_url(target))
            except ValueError:
                continue

    def settings(self, url, *, robots=False):
        host = _public_url(url)
        overrides = dict(self.profiles.get(host, {}))
        permitted = {"article_selector", "readiness_timeout_ms", "scroll_steps",
                     "challenge_timeout_seconds", "resource_hosts", "block_ads"}
        if set(overrides) - permitted:
            raise ValueError(f"Unsupported recovery profile fields: {set(overrides) - permitted}")
        hosts = set(overrides.pop("resource_hosts", ())) | self.resource_hosts.get(host, set())
        return RecoveryBrowserConfig(allowed_hosts=(host,), document_path_prefix="/",
            max_requests=240, max_documents=6, render_wait_seconds=0,
            resource_hosts=tuple(sorted(hosts)), robots_bootstrap_url=url if robots else None, **overrides)

    async def _new(self, engine, host):
        key = (engine, host)
        if engine == "crawl4ai_stealth":
            from crawl4ai import AsyncWebCrawler, BrowserConfig
            channel = "msedge" if self.browser_executable else "chromium"
            manager = AsyncWebCrawler(config=BrowserConfig(headless=True,
                chrome_channel=channel, channel=channel, enable_stealth=True,
                ignore_https_errors=False, accept_downloads=False,
                storage_state=self.states.get(key), verbose=False), base_directory=str(self.work_dir))
        else:
            from scrapling.fetchers import AsyncStealthySession
            profile = self.work_dir / "sessions" / hashlib.sha256(host.encode()).hexdigest()[:16]
            manager = AsyncStealthySession(headless=True, user_data_dir=str(profile),
                executable_path=self.browser_executable, google_search=False,
                additional_args={"ignore_https_errors": False, "service_workers": "block"})
        resource = await manager.__aenter__()
        return manager, resource

    async def _close(self, key):
        manager, _resource = self.sessions.pop(key)
        await manager.__aexit__(None, None, None)

    @asynccontextmanager
    async def _lease(self, engine, host):
        key = (engine, host)
        if key in self.active:
            raise RuntimeError("Nested use of the same browser session is not allowed.")
        if key not in self.sessions:
            while len(self.sessions) >= self.session_limit:
                idle = next((k for k in self.sessions if k not in self.active), None)
                if idle is None:
                    break  # A robots bootstrap may temporarily coexist with its paused article.
                await self._close(idle)
            self.sessions[key] = await self._new(engine, host)
        self.sessions.move_to_end(key)
        self.active.add(key)
        try:
            yield self.sessions[key][1]
        finally:
            self.active.discard(key)

    async def _attempt(self, engine, url, settings):
        host = _public_url(url)
        probe = RecoveryProbe(self.guard, settings)
        probe.record.update(method=engine, purpose="robots" if settings.robots_bootstrap_url else "article",
                            settings=asdict(settings))
        try:
            async with self._lease(engine, host) as resource:
                async def run():
                    if engine == "crawl4ai_stealth":
                        from crawl4ai import CacheMode, CrawlerRunConfig
                        resource.crawler_strategy.set_hook("before_goto", probe.before_goto)
                        resource.crawler_strategy.set_hook("before_return_html", probe.capture)
                        result = await resource.arun(url=url, config=CrawlerRunConfig(
                            cache_mode=CacheMode.BYPASS, check_robots_txt=False, max_retries=0,
                            fallback_fetch_function=None, page_timeout=settings.navigation_timeout_ms,
                            wait_until="domcontentloaded", delay_before_return_html=0,
                            remove_overlay_elements=False, process_iframes=False, verbose=False))
                        probe.record.update(engine_success=bool(result.success), engine_error=str(result.error_message or "")[:500])
                    else:
                        page = await resource.context.new_page()
                        # Scrapling page_setup swallows exceptions; install directly
                        # before navigation and use its pinned solver only after setup.
                        await probe.before_goto(page, resource.context, url)
                        probe.solver = resource._cloudflare_solver
                        try:
                            await page.goto(url, wait_until="domcontentloaded", timeout=settings.navigation_timeout_ms)
                        except Exception as exc:
                            probe.record["navigation_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                        await probe.capture(page, await page.content())
                try:
                    await asyncio.wait_for(run(), timeout=120)
                finally:
                    await probe.finish()
                if probe.storage_state is not None:
                    self.states[(engine, host)] = probe.storage_state
        except Exception as exc:
            probe.record.update(error_type=type(exc).__name__, error_message=str(exc)[:500])
            if not probe.record.get("guard_errors") and not probe.installed:
                probe.record["guard_errors"] = ["browser_start_or_guard_failed"]
        probe.record["outcome"] = f"browser_http_{probe.record.get('http_status', 'unknown')}"
        return probe.record, probe.assets

    async def read_robots(self, url):
        # Many CDN requests can pause simultaneously. Keep their bootstraps
        # serial so they cannot each launch another browser on a small runtime.
        async with self.robots_browser_lock:
            return await self._read_robots(url)

    async def _read_robots(self, url):
        cached = self.robots_cache.get(url)
        if cached and cached[0] > time.monotonic():
            return {**cached[1], "evidence": {**cached[1]["evidence"], "cache_reused": True}}
        attempts, chosen_body, chosen_status, headers, evidence_assets = [], None, None, {}, {}
        for engine in ("scrapling_stealth", "crawl4ai_stealth"):
            print("robots browser:", engine, url, flush=True)
            record, assets = await self._attempt(engine, url, self.settings(url, robots=True))
            evidence_assets.update({f"{engine}/{name}": body for name, body in assets.items()})
            body = assets.get("response.html")
            attempts.append(record)
            status = record.get("http_status")
            if record.get("guard_errors"):
                continue
            if status == 429 or urlsplit(url).hostname in record.get("held_hosts", []):
                chosen_status = 429
                headers = record.get("response_evidence", {}).get("response_headers", {})
                break
            if status == 200 and valid_robots_bytes(body):
                chosen_body, chosen_status = body, status
                self.preferred[_public_url(url)] = engine
                break
        result = {"body": chosen_body, "status": chosen_status, "headers": headers, "assets": evidence_assets,
                  "evidence": {"browser_attempts": attempts, "browser_final_status": chosen_status}}
        self.robots_cache[url] = (time.monotonic() + 300, result)
        return result

    async def browser_probe(self, url, guard, *, config=None):
        host = _public_url(url)
        engines = ["crawl4ai_stealth", "scrapling_stealth"]
        if host in self.preferred:
            engines.sort(key=lambda name: name != self.preferred[host])
        attempts, all_assets, chosen, chosen_assets = [], {}, {}, {}
        for engine in engines:
            print("article browser:", engine, url, flush=True)
            record, assets = await self._attempt(engine, url, self.settings(url))
            attempts.append(record)
            all_assets.update({f"{engine}/{name}": body for name, body in assets.items()})
            chosen = record
            chosen_assets = assets
            if article_capture_ready(record, assets):
                self.preferred[host] = engine
                break
            if record.get("http_status") in {401, 429} or host in record.get("held_hosts", []):
                break
        return {**chosen, "engine_attempts": attempts}, {**all_assets, **chosen_assets}
