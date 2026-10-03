"""Guard Crawl4AI browser requests and preserve raw response/DOM separately.

The upstream robots helper allows requests after network failures; use the
project's own resolver at every intercepted request instead. Hooks propagate
setup exceptions before navigation in the audited Crawl4AI source commit.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from .browser_probe import BrowserProbeConfig, _RequestGate, response_evidence
from .recovery import _public_url
from .robots import RobotsResolver


class Crawl4AIProbe:
    """Install on one fresh crawler instance/context, never shared profiles."""
    def __init__(self, guard, settings: BrowserProbeConfig):
        settings.validate()
        if not guard.config.respect_robots:
            raise ValueError("Crawl4AI probe requires robots enforcement.")
        self.guard, self.settings = guard, settings
        self.page = self.cdp = self.gate = None
        self.installed = False
        self.responses, self.assets = [], {}
        self.record = {"method": "crawl4ai_cdp_probe", "article_candidate": False}

    async def before_goto(self, page, context, url, **_kwargs):
        if self.gate is not None:
            raise RuntimeError("Use a fresh crawler for each browser attempt.")
        if _public_url(url) not in self.settings.allowed_hosts or urlsplit(url).scheme != "https":
            raise ValueError("Crawl4AI URL outside probe scope.")
        self.page = page
        await page.add_init_script("""
            window.open = () => null;
            window.Worker = window.SharedWorker = class {
                constructor() { throw new Error('Workers disabled in article probe'); }
            };
            if (navigator.serviceWorker) navigator.serviceWorker.register = async () => {
                throw new Error('Service workers disabled in article probe');
            };
        """)
        self.cdp = await context.new_cdp_session(page)
        frame_id = (await self.cdp.send("Page.getFrameTree"))["frameTree"]["frame"]["id"]
        agent = await page.evaluate("navigator.userAgent")
        browser_robots = RobotsResolver(self.guard.client, self.guard.pace, user_agent=agent)
        self.gate = _RequestGate(self.cdp, self.guard, browser_robots, frame_id, self.settings)
        self.cdp.on("Fetch.requestPaused", self.gate.paused)
        await self.cdp.send("Fetch.enable", {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]})
        async def deny_socket(socket):
            await socket.close()
        await page.route_web_socket("**/*", deny_socket)
        def observed(response):
            self.responses.append(response)
            if response.status == 429:
                self.gate.held_hosts.add(urlsplit(response.url).hostname)
        page.on("response", observed)
        self.record.update(url=url, browser_user_agent=agent, robots_product=self.guard.robots.user_agent)
        self.installed = True
        return page

    async def capture(self, page, html, **_kwargs):
        if not self.installed:
            raise RuntimeError("Crawl4AI returned content without installing request guard.")
        self.record.update(final_url=page.url, rendered_url=page.url, page_title=await page.title())
        body = html.encode("utf-8")
        if len(body) <= self.settings.max_body_bytes:
            self.assets["rendered.html"] = body
        documents = [r for r in self.responses if r.url == page.url and r.request.resource_type == "document"]
        if documents:
            response = documents[-1]
            self.record.update(http_status=response.status, rendered_http_status=response.status,
                               content_type=response.headers.get("content-type", ""))
            try:
                body = await response.body()
                if len(body) <= self.settings.max_body_bytes:
                    self.assets["response.html"] = body
                    self.record["response_evidence"] = response_evidence(body, response.headers)
            except Exception as exc:
                self.record["response_body_error"] = type(exc).__name__
        return page

    async def finish(self):
        if self.page is not None and not self.page.is_closed():
            await self.page.close()
        if self.gate is not None:
            tasks = list(self.gate.tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.record.update(request_log=self.gate.log, guard_errors=self.gate.errors,
                               held_hosts=sorted(self.gate.held_hosts))
        else:
            self.record["guard_errors"] = ["request_guard_not_installed"]
        if not self.installed:
            self.record.setdefault("guard_errors", []).append("request_guard_setup_incomplete")
        if self.cdp is not None:
            try:
                await self.cdp.detach()
            except Exception:
                pass  # Page already closed; no intercepted requests can resume.
        return self.record, self.assets


async def crawl4ai_browser_probe(url, guard, *, config=None, crawler_factory=None):
    settings = config or BrowserProbeConfig(allowed_hosts=(_public_url(url),), document_path_prefix="/")
    probe = Crawl4AIProbe(guard, settings)
    if crawler_factory is None:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
        crawler = AsyncWebCrawler(config=BrowserConfig(headless=True, browser_mode="dedicated",
                                  use_persistent_context=False, accept_downloads=False,
                                  enable_stealth=False, ignore_https_errors=False, verbose=False))
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, check_robots_txt=False,
            max_retries=0, fallback_fetch_function=None, page_timeout=settings.navigation_timeout_ms,
            wait_until="domcontentloaded", delay_before_return_html=settings.render_wait_seconds,
            remove_overlay_elements=False, process_iframes=False, verbose=False)
    else:
        crawler, run_config = crawler_factory()
    crawler.crawler_strategy.set_hook("before_goto", probe.before_goto)
    crawler.crawler_strategy.set_hook("before_return_html", probe.capture)
    try:
        async with crawler:
            result = await crawler.arun(url=url, config=run_config)
            probe.record.update(engine_success=bool(result.success), engine_status=result.status_code,
                                engine_error=str(result.error_message or "")[:500])
            probe.record["outcome"] = f"browser_http_{probe.record.get('http_status', 'unknown')}"
    except Exception as exc:
        probe.record.update(outcome="browser_error", error_type=type(exc).__name__, error_message=str(exc)[:500])
    finally:
        record, assets = await probe.finish()
    return record, assets
