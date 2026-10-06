"""Resolve HTML encoding from immutable bytes, then pass UTF-8 bytes to parsers."""
from __future__ import annotations

import codecs
import re

from charset_normalizer import from_bytes

META_CHARSET = re.compile(rb'<meta\b[^>]*charset\s*=\s*["\']?\s*([a-zA-Z0-9_.-]+)', re.I)


def normalize_charset(value):
    try:
        name = codecs.lookup(value.strip()).name if value else None
    except (LookupError, ValueError):
        return None
    return 'gb18030' if name in ('gb2312', 'gbk') else name


def html_bytes(body: bytes, declared: str | None = None):
    if not isinstance(body, bytes):
        raise TypeError('HTML extraction requires raw bytes.')
    match = META_CHARSET.search(body[:16384])
    meta = normalize_charset(match.group(1).decode('ascii')) if match else None
    header = normalize_charset(declared)
    choices = []
    for marker, charset in ((codecs.BOM_UTF32_LE, 'utf-32'), (codecs.BOM_UTF32_BE, 'utf-32'),
                            (codecs.BOM_UTF8, 'utf-8-sig'), (codecs.BOM_UTF16_LE, 'utf-16'),
                            (codecs.BOM_UTF16_BE, 'utf-16')):
        if body.startswith(marker):
            choices.append((charset, 'bom'))
            break
    choices += [('utf-8', 'strict_utf8'), (meta, 'meta_charset'), (header, 'http_charset')]
    for charset, method in choices:
        if charset is None:
            continue
        try:
            text = body.decode(charset, errors='strict')
        except UnicodeError:
            continue
        return text.encode('utf-8'), {'extraction_charset': charset, 'encoding_method': method,
            'encoding_conflict': bool(meta and header and meta != header) or
                                 bool(charset == 'utf-8' and (meta or header) not in (None, 'utf-8'))}
    detected = from_bytes(body).best()
    if detected is None:
        raise UnicodeError('No usable HTML encoding.')
    return str(detected).encode('utf-8'), {'extraction_charset': detected.encoding,
        'encoding_method': 'charset_normalizer_fallback', 'encoding_conflict': True}
