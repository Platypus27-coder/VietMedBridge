from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit


def normalize_url(value, remove_fragments: bool = True) -> tuple[str | None, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, 'EMPTY_URL'
    if not isinstance(value, str):
        return None, 'INVALID_URL'
    value = value.strip()
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in ('http', 'https'):
            return None, 'UNSUPPORTED_SCHEME' if parts.scheme else 'INVALID_URL'
        hostname = parts.hostname
        if not hostname or parts.username is not None or parts.password is not None:
            return None, 'INVALID_URL'
        if any(char.isspace() or ord(char) < 32 for char in value):
            return None, 'INVALID_URL'
        port = parts.port
        host = f'[{hostname.lower()}]' if ':' in hostname else hostname.lower()
        netloc = host + (f':{port}' if port is not None else '')
        return urlunsplit((parts.scheme.lower(), netloc, parts.path or '/', parts.query,
                          '' if remove_fragments else parts.fragment)), 'VALID_HTTP_URL'
    except (ValueError, UnicodeError):
        return None, 'INVALID_URL'


def crawl_url_id(normalized_url: str) -> str:
    return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()
