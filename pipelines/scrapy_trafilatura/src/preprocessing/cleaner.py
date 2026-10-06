from __future__ import annotations

import hashlib
import re
import unicodedata


def clean_text(text: str, *, markdown: bool = False) -> str:
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    if not markdown:
        text = '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n'))
    # Do not collapse whitespace inside fenced code or indented list content.
    if not markdown:
        text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip('\n') if markdown else text.strip()


def content_hash(text: str) -> str:
    view = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()
    return hashlib.sha256(view.encode('utf-8')).hexdigest()
