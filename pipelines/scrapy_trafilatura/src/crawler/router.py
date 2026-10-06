from __future__ import annotations

import re


def sniff_content(body: bytes, declared: str | None) -> dict:
    declared = declared or ''
    mime = declared.split(';')[0].strip().lower()
    charset = re.search(r'charset\s*=\s*["\']?([^;\s"\']+)', declared, re.I)
    prefix = body[:8192]
    probe = prefix.lstrip(b'\xef\xbb\xbf \t\r\n').lower()
    if prefix.startswith((b'\xff\xfe', b'\xfe\xff')):
        probe = prefix.decode('utf-16', errors='replace').strip().lower().encode('utf-8')
    if b'%pdf-' in prefix[:1024].lower():
        effective, reason = 'application/pdf', 'pdf_magic'
    elif re.search(rb'<(?:!doctype\s+html|html\b|head\b|body\b|article\b)', probe):
        effective, reason = 'text/html', 'html_markup'
    elif b'\x00' not in prefix and mime == 'text/plain':
        effective, reason = 'text/plain', 'declared_text'
    elif b'\x00' not in prefix and mime in ('text/html', 'application/xhtml+xml'):
        effective, reason = 'text/html', 'declared_html'
    else:
        effective, reason = None, 'unsupported_or_unknown'
    return {'declared_content_type': declared, 'effective_content_type': effective,
            'content_type': effective, 'content_type_mismatch': bool(effective and effective != mime),
            'declared_charset': charset.group(1) if charset else None,
            'sniff_reason': reason, 'detected_charset': None, 'charset_detection_method': None}
