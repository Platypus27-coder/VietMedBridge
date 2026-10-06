import hashlib

from scrapy.dupefilters import RFPDupeFilter


class FrontierDupeFilter(RFPDupeFilter):
    """Redirects of different BTC frontier URLs must each produce an outcome."""
    def request_fingerprint(self, request):
        original = super().request_fingerprint(request)
        return hashlib.sha256((original + ':' + request.meta.get('crawl_url_id', '')).encode()).hexdigest()
