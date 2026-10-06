import time

import pymupdf
import pytest

from src.extraction.html import extract_html
from src.extraction.pdf import extract_pdf
from src.extraction.text import decode_text
from src.extraction.worker import supervised_extract
from src.preprocessing.cleaner import clean_text, content_hash
from src.preprocessing.language import detect_language
from src.preprocessing.quality import assess_quality


@pytest.mark.parametrize('language,body', [
    ('vi', 'Người bệnh cần điều trị và theo dõi với bác sĩ. Bệnh nhân dùng 0.5 mg/kg, HbA1c và SpO₂.'),
    ('en', 'The patient with the disease needs treatment and monitoring for the condition. H. pylori BRCA1 HER2.'),
    ('zh', '患者需要治疗和监测。医生根据患者症状安排治疗并进行检查。这个医学文档讨论疾病和治疗。')])
def test_html_structure_and_language(config, language, body):
    html = f'<html><head><title>Medical article</title></head><body><article><h1>Medical title</h1><p>{body*8}</p><h2>Section</h2><p>{body*8}</p><ul><li>{body}</li><li>{body}</li></ul><table><tr><th>Dose</th><th>Unit</th></tr><tr><td>0.5</td><td>mg/kg</td></tr></table></article></body></html>'.encode()
    result = extract_html(html, 'https://example.org/article', config)
    assert result is not None
    assert '# Medical title' in result['text_markdown']
    assert '## Section' in result['text_markdown']
    assert any(line.lstrip().startswith(('- ', '* ')) for line in result['text_markdown'].splitlines())
    import re
    assert re.search(r'\|[^\n]*Dose[^\n]*\|[^\n]*Unit[^\n]*\|', result['text_markdown'])
    assert re.search(r'\|[^\n]*0\.5[^\n]*\|[^\n]*mg/kg[^\n]*\|', result['text_markdown'])
    assert 'mg/kg' in result['text_markdown']
    assert detect_language(result['text'], config)['language'] == language


def test_pdf_and_decode(config):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72,72), 'Biomedical content HbA1c 0.5 mg/kg')
    body = document.tobytes()
    document.close()
    assert 'HbA1c' in extract_pdf(body, config)['text']
    assert decode_text('治疗'.encode('gb18030'), 'gb18030')[0] == '治疗'


def test_cleaning_and_quality(config):
    text = 'H. pylori HbA1c BRCA1 HER2 SARS-CoV-2 SpO₂ ICD-10 0.5 mg mg/kg Na+ K+'
    assert clean_text(text) == text
    markdown = '# Heading\n\n  - item\n\n| A | B |\n|---|---|'
    assert clean_text(markdown, markdown=True) == markdown
    assert content_hash('A\n B') == content_hash('A B')
    assert content_hash('A') != content_hash('a')
    assert assess_quality('404 not found', '', None, config)['quality_tier'] == 'QUARANTINE'


def hanging_task(row, config):
    if row['crawl_url_id'] == 'hang':
        time.sleep(5)
    return {'crawl_url_id': row['crawl_url_id'], 'extract_status': 'EXTRACT_SUCCESS'}


def test_supervisor_kills_hang_and_continues(config):
    config['extraction']['workers'] = 1
    config['extraction']['html']['timeout_seconds'] = 1.5
    tasks = [({'crawl_url_id': key, 'raw_sha256': key, 'effective_content_type': 'text/html'}, 1) for key in ('hang', 'ok')]
    result = list(supervised_extract(tasks, config, hanging_task))
    assert result[0][0]['extract_status'] == 'EXTRACTION_TIMEOUT'
    assert result[1][0]['extract_status'] == 'EXTRACT_SUCCESS'
