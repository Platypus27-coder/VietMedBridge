"""Request policy for article recovery and bounded robots-file bootstrap."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit

from .browser_probe import BrowserProbeConfig, _RequestGate
from .recovery import _public_url, _same_target


@lru_cache(maxsize=1)
def _ad_domains():
    from scrapling.engines.toolbelt.ad_domains import AD_DOMAINS
    return frozenset(AD_DOMAINS)


def _is_ad_domain(host):
    pieces = host.split(".")
    return any(".".join(pieces[i:]) in _ad_domains() for i in range(len(pieces) - 1))


@dataclass(frozen=True)
class RecoveryBrowserConfig(BrowserProbeConfig):
    resource_hosts: tuple[str, ...] = ()
    allow_public_static_resources: bool = True
    block_ads: bool = True
    article_selector: str = "article, main, [role='main']"
    readiness_timeout_ms: int = 15000
    scroll_steps: int = 3
    challenge_timeout_seconds: float = 30
    robots_bootstrap_url: str | None = None

    def validate(self):
        super().validate()
        if not 0 <= self.scroll_steps <= 10 or not 0 <= self.readiness_timeout_ms <= 60000:
            raise ValueError("Invalid content readiness limits.")
        if not 0 <= self.challenge_timeout_seconds <= 60:
            raise ValueError("Invalid challenge time budget.")


class RecoveryRequestGate(_RequestGate):
    """Keep navigations separate from the static resources needed to render them.

    Reading robots itself cannot depend on first reading robots. Bootstrap
    allows only that document/reloads/HTTP redirects, plus bounded resources;
    it cannot navigate to an unrelated article or synthesize policy from DOM.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.document_hosts = set(self.settings.allowed_hosts)
        self.document_urls = {self.settings.robots_bootstrap_url} if self.settings.robots_bootstrap_url else set()

    async def handle(self, event):
        request = event["request"]
        url, method, kind = request["url"], request["method"], event["resourceType"]
        row = {"url": url, "method": method, "resource_type": kind,
               "redirected_request_id": event.get("redirectedRequestId"),
               "decision": "blocked", "reason": "awaiting_guard"}
        self.log.append(row)
        try:
            host, parts = _public_url(url), urlsplit(url)
            bootstrap = self.settings.robots_bootstrap_url is not None
            challenge = (parts.path.startswith("/cdn-cgi/challenge-platform/") and
                         (host in self.document_hosts or host == "challenges.cloudflare.com"))
            main_document = kind == "Document" and event["frameId"] == self.frame_id
            static = kind in {"Script", "Stylesheet", "Image", "Font"}
            known_resource = host in self.document_hosts or host in self.settings.resource_hosts
            if parts.scheme != "https" or parts.port not in (None, 443):
                reason = "https_only"
            elif host in self.held_hosts:
                reason = "rate_limit_hold"
            elif not main_document and host not in self.document_hosts and self.settings.block_ads and _is_ad_domain(host):
                reason = "scrapling_ad_domain"
            elif len(self.log) > self.settings.max_requests:
                reason = "request_budget"
            elif method != "GET" and not (method == "POST" and challenge):
                reason = "method_outside_recovery"
            elif main_document:
                self.documents += 1
                allowed_target = (any(_same_target(url, target) for target in self.document_urls)
                                  if bootstrap else host in self.document_hosts)
                if self.documents > self.settings.max_documents:
                    reason = "document_budget"
                elif not allowed_target and not event.get("redirectedRequestId"):
                    reason = "document_outside_recovery"
                else:
                    self.document_hosts.add(host)
                    self.document_urls.add(url)
                    reason = None
            elif kind == "Document":
                reason = None if challenge else "frame_outside_recovery"
            elif kind not in {"Script", "Stylesheet", "Image", "Font", "XHR", "Fetch"}:
                reason = "resource_outside_recovery"
            elif bootstrap and kind in {"XHR", "Fetch"} and not challenge:
                reason = "robots_bootstrap_xhr_outside_challenge"
            elif not (known_resource or challenge or (static and self.settings.allow_public_static_resources)):
                reason = "resource_host_outside_recovery"
            else:
                reason = None
            if reason is None:
                minimum = self.settings.document_delay_seconds if main_document else 0
                if not bootstrap:
                    decision = await self.guard.robots_decision(url)
                    browser_decision = await self.browser_robots.decision(url)
                    row.update(decision.record(), browser_robots_state=browser_decision.state)
                    if not decision.allowed or not browser_decision.allowed:
                        reason = "robots_hold"
                    minimum = max(minimum, decision.crawl_delay_seconds or 0,
                                  decision.request_rate_gap_seconds or 0,
                                  browser_decision.crawl_delay_seconds or 0,
                                  browser_decision.request_rate_gap_seconds or 0)
                if reason is None:
                    pace = self.guard.pace
                    if not main_document and minimum == 0:
                        pace = getattr(self.guard, "browser_resource_pace", pace)
                    await pace(url, minimum)
                    if host in self.held_hosts:
                        reason = "rate_limit_hold"
                    else:
                        await self.cdp.send("Fetch.continueRequest", {"requestId": event["requestId"]})
                        row.update(decision="allowed", reason="robots_bootstrap" if bootstrap else "robots_allowed")
                        return
            row["reason"] = reason
        except asyncio.CancelledError:
            row.update(decision="cancelled", reason="page_closed_before_request")
            raise
        except Exception as exc:
            row["reason"] = "guard_error"
            self.errors.append(f"{type(exc).__name__}: {str(exc)[:250]}")
        try:
            await self.cdp.send("Fetch.failRequest", {"requestId": event["requestId"], "errorReason": "BlockedByClient"})
        except Exception as exc:
            self.errors.append(f"Abort failed: {type(exc).__name__}: {str(exc)[:250]}")
