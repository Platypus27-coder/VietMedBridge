from __future__ import annotations

import ipaddress
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from scrapy import signals
from scrapy.exceptions import IgnoreRequest, StopDownload
from scrapy.downloadermiddlewares.redirect import RedirectMiddleware, MetaRefreshMiddleware
from twisted.internet import defer
from w3lib.url import canonicalize_url


def public_ip(address: str) -> bool:
    return ipaddress.ip_address(address).is_global


class UnsafeDestination(IgnoreRequest):
    pass


class RedirectLoopGuard:
    """Turn a redirect cycle into an errback before the dupefilter drops it."""

    def _redirect(self, redirected, request, spider, reason):
        visited = [request.url, *request.meta.get('redirect_urls', [])]
        if request.meta.get('crawl_url_id') and canonicalize_url(redirected.url) in {canonicalize_url(url) for url in visited}:
            request.meta['_redirect_loop'] = redirected.url
            request.meta['_redirect_loop_status'] = reason if isinstance(reason, int) else None
            raise IgnoreRequest(f'Redirect loop to {redirected.url}')
        return super()._redirect(redirected, request, spider, reason)


class FrontierRedirectMiddleware(RedirectLoopGuard, RedirectMiddleware):
    pass


class FrontierMetaRefreshMiddleware(RedirectLoopGuard, MetaRefreshMiddleware):
    pass


class SafetyMiddleware:
    """Check original/redirect/robots hosts and pin the checked IPv4 resolution."""

    def process_request(self, request, spider):
        host = urlsplit(request.url).hostname
        if not host or urlsplit(request.url).scheme not in ('http', 'https'):
            raise UnsafeDestination('Unsupported destination.')
        # Test harness injects exact loopback origins directly; no CLI/config bypass exists.
        allowed = getattr(spider, '_test_origins', frozenset())
        origin = (urlsplit(request.url).scheme, host, urlsplit(request.url).port)
        if origin in allowed:
            return None
        if host.lower() in ('localhost', 'localhost.localdomain') or host.lower().endswith('.localhost'):
            raise UnsafeDestination('Private/local destination blocked.')
        try:
            if not public_ip(host):
                raise UnsafeDestination('Non-public IP destination blocked.')
            return None
        except ValueError:
            from twisted.internet import reactor
            # Use Scrapy's installed caching/threaded resolver and DNS deadline.
            deferred = reactor.resolve(host)
            def checked(address):
                if not public_ip(address):
                    raise UnsafeDestination('DNS resolved to a non-public IP.')
                # The downloader must use this checked resolution as well; prevent a second lookup.
                from scrapy.resolver import dnscache
                dnscache[host] = address
                return None
            deferred.addCallback(checked)
            return deferred


def retry_after_seconds(header: str | None, now: float, fallback: float) -> float:
    if header:
        try:
            return max(0.0, float(int(header.strip())))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(header).timestamp() - now)
            except (ValueError, TypeError, OverflowError):
                return fallback
    return fallback


class CooldownMiddleware:
    def process_request(self, request, spider):
        if request.meta.get('crawl_url_id'):
            delay = spider.crawl_state.cooldown_until(urlsplit(request.url).hostname) - time.time()
            if delay > 0:
                from twisted.internet import reactor
                from twisted.internet.task import deferLater
                return deferLater(reactor, delay, self.process_request, request, spider)
        return None

    def process_response(self, request, response, spider):
        if not request.meta.get('crawl_url_id'):
            return response
        retry_after = response.headers.get(b'Retry-After')
        if response.status == 429 or (response.status == 503 and retry_after):
            policy = spider.config['crawler']['rate_limit']
            value = retry_after.decode('ascii', 'replace') if retry_after else None
            delay = retry_after_seconds(value, time.time(), policy['default_cooldown_seconds'] * 2 ** (request.meta['attempt_no']-1))
            spider.crawl_state.cooldown(urlsplit(request.url).hostname, time.time() + delay)
            request.meta['retry_not_before'] = time.time() + delay
            request.meta['dont_retry'] = True
        return response


class SizeGuard:
    @classmethod
    def from_crawler(cls, crawler):
        instance = cls(crawler.settings.getint('DOWNLOAD_MAXSIZE'))
        crawler.signals.connect(instance.headers, signal=signals.headers_received)
        crawler.signals.connect(instance.body, signal=signals.bytes_received)
        return instance

    def __init__(self, maximum):
        self.maximum = maximum

    def headers(self, headers, body_length, request, spider):
        request.meta['_received_bytes'] = 0
        # Twisted uses a string UNKNOWN_LENGTH sentinel for chunked bodies.
        # Streaming bytes below enforce the limit when no length is known.
        if isinstance(body_length, int) and body_length > self.maximum:
            request.meta['_size_exceeded'] = True
            raise StopDownload()

    def body(self, data, request, spider):
        request.meta['_received_bytes'] = request.meta.get('_received_bytes', 0) + len(data)
        if request.meta['_received_bytes'] > self.maximum:
            request.meta['_size_exceeded'] = True
            raise StopDownload()
