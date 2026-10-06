from email.utils import formatdate

import pytest
from scrapy import Request
from scrapy.exceptions import StopDownload

from src.crawler.middlewares import SizeGuard, public_ip, retry_after_seconds


def test_unknown_length_stream_is_capped_and_retry_counter_resets():
    guard, request = SizeGuard(10), Request('https://example.com/')
    guard.headers({}, 'twisted.web.iweb.UNKNOWN_LENGTH', request, None)
    guard.body(b'123456', request, None)
    with pytest.raises(StopDownload):
        guard.body(b'123456', request, None)
    assert request.meta['_size_exceeded']
    guard.headers({}, 8, request, None)
    assert request.meta['_received_bytes'] == 0


def test_retry_after_date_and_public_destinations():
    now = 1700000000
    assert retry_after_seconds(formatdate(now+60, usegmt=True), now, 10) == 60
    assert retry_after_seconds('bad', now, 10) == 10
    assert public_ip('8.8.8.8')
    assert not public_ip('127.0.0.1') and not public_ip('169.254.169.254')
