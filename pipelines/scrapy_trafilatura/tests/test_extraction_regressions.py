"""Regressions reproduced from native 10k source shapes, without network fixtures."""
import os

from lxml import html

from src.extraction.encoding import html_bytes
from src.extraction.html import extract_html
from src.extraction.structure import serialize
from src.extraction.worker import supervised_extract
from src.preprocessing.language import detect_language


def test_strict_charset_and_lying_meta():
    source = '<html><meta charset="gb2312"><p>患者需要治疗。直肠癌口服化疗。</p></html>'
    body, evidence = html_bytes(source.encode('gb18030'), 'utf-8,gbk')
    assert body.decode('utf-8') == source
    assert evidence['extraction_charset'] == 'gb18030'
    body, evidence = html_bytes(source.encode('utf-8'), 'gb2312')
    assert body.decode('utf-8') == source
    assert evidence['encoding_conflict'] is True


def test_nested_headings_lists_table_spans_and_link_text():
    node = html.fromstring('''<article><h2><strong>Điều trị <span>viêm họng</span></strong></h2>
    <ul><li>Giới hạn<ul><li><a href="/MRSA">MRSA</a> kháng thuốc</li></ul></li></ul>
    <table><caption>Phân nhóm</caption><tr><th>Thế hệ</th><th>Thuốc</th></tr>
    <tr><td rowspan="2"><b>Thứ nhất</b></td><td>0.5 mg/kg</td></tr>
    <tr><td>H. pylori</td></tr><tfoot><tr><td colspan="2">MRSA = 金黄色葡萄球菌</td></tr></tfoot></table></article>''')
    output = serialize([node])
    assert '## Điều trị viêm họng' in output['text_markdown']
    assert '- MRSA kháng thuốc' in output['text_markdown']
    for field in ('text', 'text_markdown'):
        assert 'MRSA = 金黄色葡萄球菌' in output[field]
        assert 'H. pylori' in output[field] and '0.5 mg/kg' in output[field]
        assert output[field].count('Thứ nhất') == 2


def test_qa_avoids_shared_menu_and_keeps_short_answer(config):
    source = '<html><title>缺钙如何治疗？_问答</title><body><nav>药品库 疾病用药 用药安全</nav><div id="d_msCon"><p>健康咨询描述：缺钙如何治疗？</p></div><div id="reply12" class="crazy_new"><p>建议检查微量元素。</p></div></body></html>'
    row = extract_html(source.encode(), 'https://www.120ask.com/question/12.htm', config)
    assert '检查微量元素' in row['text'] and '缺钙如何治疗' in row['text']
    assert '药品库' not in row['text']
    assert row['source_issue'] is None


def test_source_gates_preserve_explicit_reason(config):
    body = b'<html><body><script>document.cookie="a=1";window.location.reload();</script></body></html>'
    assert extract_html(body, 'https://example.org/article', config)['extract_status'] == 'JAVASCRIPT_CHALLENGE'
    body = b'<html><title>Home</title><article><p>Home news and general navigation.</p></article></html>'
    row = extract_html(body, 'https://example.org/', config, fetch_url='https://example.org/missing-article')
    assert row['quality_tier'] == 'QUARANTINE' and row['extract_status'] == 'SOURCE_REDIRECT_HOME'
    body = b'<html><article><p>{title} {publish} {head}</p></article></html>'
    assert extract_html(body, 'https://example.org/article', config)['extract_status'] == 'UNRESOLVED_TEMPLATE'


def test_chinese_punctuation_is_not_japanese(config):
    text = '患者需要治疗和检查。头孢菌素・抗菌药ー治疗疾病。' * 4
    assert detect_language(text, config)['language'] == 'zh'
    assert detect_language(text + 'これは日本語', config)['language'] == 'other'


def test_msd_chinese_domain_keeps_table_group_and_qualifications(config):
    source = '''<html><title>头孢菌素</title><div class="TopicMainContent_content__test">
    <h2>适应证</h2><p>这些头孢菌素对以下细菌具有极好的活性</p>
    <ul><li>革兰阳性菌</li><li>头孢菌素类有下列局限性：MRSA</li></ul>
    <table><tr><th>药物治疗</th><th>途径</th></tr><tr><td colspan="2"><b>第一代</b></td></tr>
    <tr><td>头孢唑林</td><td>肠外给药</td></tr><tfoot><tr><td colspan="2">MRSA = 甲氧西林耐药 Staphylococcus aureus （金黄色葡萄球菌）。</td></tr></tfoot></table>
    </div><div class="TopicTableView_mediaprintfooter__test">duplicate print footer</div></html>'''
    row = extract_html(source.encode(), 'https://www.msdmanuals.cn/professional/cephalosporins', config)
    assert row['structure_source'] == 'source_dom:msdmanuals.cn'
    for token in ('第一代', '头孢菌素类有下列局限性', '甲氧西林耐药', '金黄色葡萄球菌'):
        assert token in row['text'] and token in row['text_markdown']
    assert 'duplicate print footer' not in row['text']


def test_an_giang_keeps_lead_and_body(config):
    source = '<html><title>Vaccine</title><div class="detail-desc"><p>Triển khai tiêm vaccine.</p></div><div id="newscontents"><p>247 triệu liều vaccine.</p></div></html>'
    row = extract_html(source.encode(), 'https://baoangiang.com.vn/vaccine.html', config)
    assert 'Triển khai tiêm vaccine.' in row['text'] and '247 triệu liều vaccine.' in row['text']


def test_audited_standalone_bold_section_is_recorded(config):
    source = '<html><title>Article</title><div itemprop="articleBody"><div><strong>Dầu ô liu với sữa chua</strong></div><p>Nội dung bài.</p></div></html>'
    row = extract_html(source.encode(), 'https://thanhnien.vn/article.htm', config)
    assert '### Dầu ô liu với sữa chua' in row['text_markdown']
    assert row['inferred_heading_count'] == 1


def test_comment_only_dom_container_does_not_hide_disease_article(config):
    body = '體表血絡擴張而形成的一種瘤。疾病描述和治疗方法。' * 20
    source = f'<html><title>血瘤</title><div class="art-picbox"><!-- hidden obsolete content --></div><article><h1>血瘤</h1><p>{body}</p></article></html>'
    row = extract_html(source.encode(), 'http://www.wujue.com/jbdq/article.html', config)
    assert '體表血絡擴張' in row['text'] and 'hidden obsolete content' not in row['text']
    assert row['fallback_used'] is True


def dying_task(row, config):
    if row['crawl_url_id'] == 'die':
        os._exit(23)
    return {'crawl_url_id': row['crawl_url_id'], 'extract_status': 'EXTRACT_SUCCESS'}


def test_supervisor_native_death_then_recovers(config):
    config['extraction']['html']['timeout_seconds'] = 10
    tasks = [({'crawl_url_id': key, 'raw_sha256': key, 'effective_content_type': 'text/html'}, 1) for key in ('die', 'ok')]
    rows = list(supervised_extract(tasks, config, dying_task))
    assert [r['extract_status'] for r, _ in rows] == ['WORKER_CRASHED', 'EXTRACT_SUCCESS']
