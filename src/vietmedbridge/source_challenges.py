"""Recognize small HTTP interstitials without treating them as article text."""

from __future__ import annotations

import re
from urllib.parse import urlsplit


_LAODONG_COOKIE = re.compile(
    rb'document\.cookie\s*=\s*"D1N=([0-9a-f]{32})"\s*\+\s*'
    rb'";\s*expires=[^"<>]{1,100};\s*path=/"\s*;\s*'
    rb'window\.location\.reload\(true\)\s*;', re.I,
)


def laodong_cookie_challenge(body: bytes, url: str) -> str | None:
    """Return the page's D1N cookie only for the observed Lao Động interstitial."""
    host = (urlsplit(url).hostname or "").lower()
    if host != "laodong.vn" and not host.endswith(".laodong.vn"):
        return None
    if len(body) > 2048 or b"<script" not in body.lower():
        return None
    match = _LAODONG_COOKIE.search(body)
    if match is None or b"<article" in body.lower():
        return None
    return "D1N=" + match.group(1).decode("ascii")
