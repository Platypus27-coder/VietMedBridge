from __future__ import annotations

import re

ERROR_PATTERNS = ('404 not found', 'page not found', 'trang không tồn tại', 'access denied',
                  'verify you are human', 'checking your browser', 'enable javascript and cookies',
                  'cloudflare ray id', 'just a moment...')


def assess_quality(text: str, markdown: str, title: str | None, config: dict) -> dict:
    count = len(text)
    replacements = text.count('\ufffd')/max(count, 1)
    lowered = text.lower()
    signals = [p for p in ERROR_PATTERNS if p in lowered]
    if not text.strip():
        tier, reason = 'QUARANTINE', 'empty_text'
    elif replacements > config['quality']['max_replacement_char_ratio']:
        tier, reason = 'QUARANTINE', 'severe_decode_failure'
    elif signals and (count < 3000 or len(signals) >= 2):
        tier, reason = 'QUARANTINE', 'error_page:' + ','.join(signals)
    elif count < config['quality']['min_chars']:
        tier, reason = 'LOW', 'short_valid_text'
    elif count < config['quality']['high_min_chars']:
        tier, reason = 'MEDIUM', 'length_band'
    else:
        tier, reason = 'HIGH', 'length_band'
    return {'char_count': count, 'paragraph_count': len([p for p in re.split(r'\n\s*\n', text) if p.strip()]),
            'replacement_char_ratio': replacements, 'heading_count': len(re.findall(r'^#{1,6}\s', markdown, re.M)),
            'list_item_count': len(re.findall(r'^\s*(?:[-*+] |\d+\. )', markdown, re.M)),
            'table_count': len(re.findall(r'^\|?\s*:?-{3,}:?\s*\|', markdown, re.M)),
            'quality_tier': tier, 'quality_reason': reason}
