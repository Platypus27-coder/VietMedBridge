from __future__ import annotations

import pymupdf


def extract_pdf(body: bytes, config: dict) -> dict:
    with pymupdf.open(stream=body, filetype='pdf') as document:
        pages = [page.get_text(sort=True) for page in document]
        text = '\n\n'.join(pages)
        count = len(pages)
        scanned = bool(count and len(text.strip()) < count * config['extraction']['pdf']['min_chars_per_page'])
        return {'title': document.metadata.get('title') or None, 'text': text, 'text_markdown': text,
                'text_format': 'plain', 'structure_source': 'pdf_pages', 'page_count': count,
                'extract_status': 'LIKELY_SCANNED_PDF' if scanned else 'EXTRACT_SUCCESS'}
