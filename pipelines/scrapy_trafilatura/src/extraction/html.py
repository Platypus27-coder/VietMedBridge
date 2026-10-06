from __future__ import annotations

import re
from copy import deepcopy
from urllib.parse import urlsplit

from lxml import html
from trafilatura import bare_extraction
from trafilatura.settings import DEFAULT_CONFIG
from .encoding import html_bytes
from .structure import serialize

IMPLEMENTATION_VERSION = 'html-structure-encoding-v2'

# These containers were checked against cached native articles in the 10k audit.
ADAPTERS = {
    'baocantho.com.vn': '//*[@id="newscontents"]//*[contains(concat(" ",normalize-space(@class)," ")," content-fck ")]',
    'baoangiang.com.vn': '//*[contains(concat(" ",normalize-space(@class)," ")," detail-desc ")]|//*[@id="newscontents"]',
    'cnkang.com': '//*[contains(concat(" ",normalize-space(@class)," ")," detailc ")]',
    'wujue.com': '//*[contains(concat(" ",normalize-space(@class)," ")," art-picbox ")]',
    'familydoctor.com.cn': '//*[@id="viewContent"]',
    'medlatec.vn': '//*[contains(@class,"block-posts-single")]//*[contains(concat(" ",normalize-space(@class)," ")," description ")]',
    'msdmanuals.com': '//*[starts-with(@class,"TopicMainContent_content__")]',
    'msdmanuals.cn': '//*[starts-with(@class,"TopicMainContent_content__")]',
}


def domain_matches(host, domain):
    return host == domain or host.endswith('.' + domain)


def source_containers(tree, host):
    if domain_matches(host, '120ask.com'):
        return tree.xpath('//*[@id="d_msCon"]|//div[starts-with(@id,"reply") and contains(concat(" ",normalize-space(@class)," ")," crazy_new ")]'), '120ask_question_answers'
    for domain, selector in ADAPTERS.items():
        if domain_matches(host, domain):
            nodes = tree.xpath(selector)
            return [n for n in nodes if not any(a in nodes for a in n.iterancestors())], domain
    nodes = tree.xpath('//*[@itemprop="articleBody"]')
    if len(nodes) == 1:
        return nodes, 'schema_article_body'
    nodes = tree.xpath('//article')
    return (nodes, 'single_article') if len(nodes) == 1 else ([], None)


def source_title(tree, nodes):
    if nodes:
        headings = nodes[0].xpath('.//h1')
        if headings:
            return ' '.join(''.join(headings[0].itertext()).split())
    values = tree.xpath('//meta[@property="og:title"]/@content|//meta[@name="twitter:title"]/@content')
    if values:
        return values[0].strip()
    values = tree.xpath('//title/text()')
    return re.split(r'[_|]', values[0], maxsplit=1)[0].strip() if values else None


def promote_bold_sections(nodes, host):
    """Two audited publishers encode section titles as wholly bold short paragraphs."""
    if not any(domain_matches(host, domain) for domain in ('thanhnien.vn', 'familydoctor.com.cn')):
        return 0
    count = 0
    for root in nodes:
        for node in root.xpath('.//p|.//div'):
            if node.xpath('.//p|.//div|.//table|.//ul|.//ol|.//h1|.//h2|.//h3'):
                continue
            visible = ''.join(node.itertext()).strip()
            bold = node.xpath('./strong|./b')
            if len(bold) == 1 and 4 <= len(visible) <= 160 and visible == ''.join(bold[0].itertext()).strip():
                node.tag = 'h3'
                count += 1
    return count


def candidate(body: bytes, url: str, config: dict, fast: bool):
    options = config['extraction']['html']
    settings = deepcopy(DEFAULT_CONFIG)
    settings.set('DEFAULT', 'MIN_EXTRACTED_SIZE', str(options['min_extracted_size']))
    settings.set('DEFAULT', 'MIN_OUTPUT_SIZE', str(options['min_output_size']))
    doc = bare_extraction(body, url=url, fast=fast, output_format=options['output_format'],
                          include_formatting=options['include_formatting'], include_comments=options['include_comments'],
                          include_links=options['include_links'], include_tables=options['include_tables'],
                          with_metadata=True, config=settings)
    if doc is None:
        return None
    return dict(serialize([doc.body]), title=doc.title, text_format='markdown',
                structure_source='trafilatura_main_body', page_count=None)


def extract_html(body: bytes, url: str, config: dict, declared_charset=None, fetch_url=None) -> dict | None:
    normalized, encoding = html_bytes(body, declared_charset)
    tree = html.fromstring(normalized, parser=html.HTMLParser(encoding='utf-8', recover=True))
    host = (urlsplit(url).hostname or '').lower()
    script = '\n'.join(tree.xpath('//script/text()'))
    visible = ' '.join(tree.xpath('//body//text()[not(ancestor::script) and not(ancestor::style)]'))
    issue = None
    if len(visible.strip()) < 200 and 'document.cookie' in script and re.search(r'(?:window\.)?location\.reload', script):
        issue = 'JAVASCRIPT_CHALLENGE'
    requested, final = urlsplit(fetch_url or url), urlsplit(url)
    if (requested.hostname == final.hostname and requested.path.strip('/') and
            not final.path.strip('/') and not final.query):
        issue = 'SOURCE_REDIRECT_HOME'
    for node in tree.xpath('//script|//style|//noscript|//form|//button|//iframe|//*[contains(@class,"mediaprintfooter") or contains(@class,"article__social") or contains(@class,"article__share") or contains(@class,"article-relate")]'):
        if node.getparent() is not None:
            node.drop_tree()
    nodes, adapter = source_containers(tree, host)
    title = source_title(tree, nodes)
    inferred = promote_bold_sections(nodes, host)
    used_dom = False
    if issue == 'JAVASCRIPT_CHALLENGE':
        chosen = dict(text='', text_markdown='', title=title, structure_source='source_gate')
    else:
        chosen = dict(serialize(nodes), title=title, structure_source='source_dom:' + adapter) if nodes else None
        used_dom = bool(chosen and chosen['text'].strip())
        # Normal fallback is needed even when the fast result is a long menu.
        if not used_dom:
            chosen = candidate(normalized, url, config, False)
            if chosen is None or not chosen['text'].strip():
                # Readability can reject short dictionary pages; a nominal DOM
                # container can contain only comments. Try the primary path too.
                chosen = candidate(normalized, url, config, True)
                if chosen:
                    chosen['structure_source'] = 'trafilatura_fast_recovery'
        if chosen and title:
            chosen['title'] = title
    if chosen is None:
        if not issue:
            return None
        chosen = dict(text='', text_markdown='', title=title, structure_source='source_gate')
    placeholders = re.findall(r'\{(?:title|publish|head|content|description)\}', chosen['text'], re.I)
    if len(set(placeholders)) >= 2:
        issue = 'UNRESOLVED_TEMPLATE'
    if domain_matches(host, '120ask.com') and not nodes:
        menu = ('药品库', '疾病用药', '药品心得', '用药安全', '药品资讯', '用药方案', '品牌药企', '明星药品', '子女教育', '心理健康', '医院在线')
        if sum(token in chosen['text'] for token in menu) >= 9 and len(chosen['text']) <= 350:
            issue = 'NAVIGATION_ONLY'
    chosen.update(encoding, source_issue=issue, implementation_version=IMPLEMENTATION_VERSION,
                  fast_pass_success=False, fallback_used=not used_dom, text_format='markdown', page_count=None,
                  inferred_heading_count=inferred)
    if issue:
        chosen.update(extract_status=issue, quality_tier='QUARANTINE', quality_reason=issue.lower())
    return chosen
