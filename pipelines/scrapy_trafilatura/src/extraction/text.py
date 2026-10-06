from __future__ import annotations

from charset_normalizer import from_bytes


def decode_text(body: bytes, declared_charset: str | None = None) -> tuple[str, str, str]:
    for marker, charset in [(b'\xff\xfe\x00\x00', 'utf-32'), (b'\x00\x00\xfe\xff', 'utf-32'),
                            (b'\xef\xbb\xbf', 'utf-8-sig'), (b'\xff\xfe', 'utf-16'), (b'\xfe\xff', 'utf-16')]:
        if body.startswith(marker):
            return body.decode(charset), charset, 'bom'
    for charset in filter(None, (declared_charset, 'utf-8')):
        try:
            return body.decode(charset, errors='strict'), charset, 'declared_or_utf8'
        except (UnicodeDecodeError, LookupError):
            continue
    result = from_bytes(body).best()
    if result is not None:
        return str(result), result.encoding, 'charset_normalizer'
    return body.decode('utf-8', errors='replace'), 'utf-8', 'replacement_fallback'
