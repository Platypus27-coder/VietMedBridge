"""Offline quality audit. Writes separate evidence; never changes ingestion or approval state."""
from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
from lxml import html

from src.crawler.raw_store import RawStore
from src.preprocessing.cleaner import content_hash
from src.storage.io import atomic_json, parquet_rows, sha256_file
from src.utils.config import load_config
from src.extraction.encoding import html_bytes

AUDIT_VERSION = 'offline-quality-v2'
META_CHARSET = re.compile(rb'<meta\b[^>]*charset\s*=\s*[\"\']?\s*([a-zA-Z0-9_.-]+)', re.I)
PLACEHOLDER = re.compile(r'\{(?:title|publish|head|content|description)\}', re.I)
MOJIBAKE = re.compile(r'(?:Ã[\x80-\xff]|Â[\x80-\xbf]|á[º»][\x80-\xff])')
HEADINGS = re.compile(r'^#{1,6}\s+(.+)$', re.M)
NAV_120ASK = ('药品库', '疾病用药', '药品心得', '用药安全', '药品资讯', '用药方案',
              '品牌药企', '明星药品', '子女教育', '心理健康', '医院在线')
BIO_TOKEN = re.compile(r'\b(?:IL-\d+|CD\d+|HbA1c|SARS-CoV-\d|COVID-\d+|DNA|RNA|HIV|HPV|MRSA)\b|'
                       r'\d+(?:[.,]\d+)?\s*(?:mmol/L|mmHg|mg|mcg|μg|µg|kg|mL|ml|%|°C)', re.I)


def compact(text):
    return re.sub(r'\s+', '', text or '').strip()


def visible_text(node):
    return ' '.join(' '.join(node.itertext()).split())


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def features(row, raw):
    text, markdown = row.get('text') or '', row.get('text_markdown') or ''
    letters = sum(c.isalpha() for c in text)
    cjk = sum('\u4e00' <= c <= '\u9fff' for c in text)
    cyrillic = sum('\u0400' <= c <= '\u04ff' for c in text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    flags = []
    if row['extract_status'] != 'EXTRACT_SUCCESS':
        flags.append(row['extract_status'].lower())
    if text and row['quality_tier'] in ('LOW', 'QUARANTINE'):
        flags.append(row['quality_tier'].lower())
    if cyrillic / max(letters, 1) > .2:
        flags.append('unexpected_cyrillic')
    if '\ufffd' in text:
        flags.append('replacement_characters')
    if MOJIBAKE.search(text):
        flags.append('possible_latin_mojibake')
    if PLACEHOLDER.search(text):
        flags.append('unresolved_template_placeholder')
    requested, final = urlsplit(raw['fetch_url']), urlsplit(raw.get('final_url') or raw['fetch_url'])
    if (requested.hostname == final.hostname and requested.path.strip('/')
            and not final.path.strip('/') and not final.query):
        flags.append('article_redirected_to_home')
    if sum(token in text for token in NAV_120ASK) >= 9:
        flags.append('120ask_navigation_present')
        if len(text) <= 350:
            flags.append('120ask_navigation_dominates')
    if text and not row.get('title'):
        flags.append('missing_title')
    japanese = {c for c in text[:10000] if '\u3040' <= c <= '\u30ff'}
    if (row.get('language') == 'other' and cjk/max(letters, 1) > .4 and japanese
            and japanese <= {'\u30fb', '\u30fc'} and not any('\uac00' <= c <= '\ud7af' for c in text[:10000])):
        flags.append('possible_lid_punctuation_false_positive')
    if text and len(lines) >= 8 and len(set(lines))/len(lines) < .65:
        flags.append('repeated_lines')
    if text and (text != unicodedata.normalize('NFC', text) or '\x00' in text):
        flags.append('unicode_normalization_or_null')
    if text and content_hash(text) != row['content_hash']:
        flags.append('content_hash_mismatch')
    return {
        'crawl_url_id': row['crawl_url_id'], 'domain': raw['domain'], 'url': raw['fetch_url'],
        'final_url': raw['final_url'], 'title': row.get('title'), 'language': row.get('language'),
        'extract_status': row['extract_status'], 'quality_tier': row.get('quality_tier'),
        'char_count': len(text), 'heading_count': row.get('heading_count') or 0,
        'list_item_count': row.get('list_item_count') or 0, 'table_count': row.get('table_count') or 0,
        'pipe_rows': sum(line.lstrip().startswith('|') for line in markdown.splitlines()),
        'replacement_count': text.count('\ufffd'), 'cyrillic_ratio': cyrillic/max(letters, 1),
        'cjk_ratio': cjk/max(letters, 1), 'flags': flags, 'content_hash': row.get('content_hash'),
        'fast_pass_success': row.get('fast_pass_success'), 'fallback_used': row.get('fallback_used'),
        'preview': text[:600],
    }


def source_evidence(body, raw, extracted):
    match = META_CHARSET.search(body[:16384])
    meta = match.group(1).decode('ascii') if match else None
    encoding = meta or raw.get('declared_charset')
    try:
        if encoding:
            codecs.lookup(encoding)
    except LookupError:
        encoding = None
    normalized, encoding_evidence = html_bytes(body, raw.get('declared_charset'))
    parser = html.HTMLParser(encoding='utf-8', recover=True)
    tree = html.fromstring(normalized, parser=parser)
    titles = tree.xpath('//title')
    title = visible_text(titles[0]) if titles else None
    # A DOM charset comparison is evidence, not a new extraction or a gold main-body reference.
    for node in tree.xpath('//script|//style|//noscript|//nav|//header|//footer'):
        node.drop_tree()
    text, markdown = extracted.get('text') or '', extracted.get('text_markdown') or ''
    text_view = compact(text)
    output_headings = [compact(h.strip('*_ ')) for h in HEADINGS.findall(markdown)]
    source_headings = [visible_text(node) for node in tree.xpath('//h1|//h2|//h3|//h4|//h5|//h6')]
    # A title in metadata is valid even if it is omitted from the article body. Require
    # a full text line, rather than a coincidental substring inside another paragraph.
    plain_lines = {compact(line) for line in text.splitlines()}
    title_view = compact(extracted.get('title'))
    retained = [h for h in source_headings if 6 <= len(h) <= 180
                and compact(h) in plain_lines and compact(h) != title_view]
    demoted = [h for h in retained if compact(h) not in output_headings]
    candidate_xpath = ('//article|//main|//*[@itemprop="articleBody"]|'
                       '//*[@id="artibody" or @id="article-content" or @id="content" or @id="article"]|'
                       '//*[contains(@class,"article-content") or contains(@class,"article_body") or '
                       'contains(@class,"article-body") or contains(@class,"entry-content") or '
                       'contains(@class,"detail-content") or contains(@class,"content-detail") or '
                       'contains(@class,"content_detail")]')
    candidates = tree.xpath(candidate_xpath)
    candidate = max(candidates, key=lambda n: len(visible_text(n))) if candidates else tree
    source_text = visible_text(candidate)
    tokens = sorted(set(BIO_TOKEN.findall(source_text)))
    absent_tokens = [token for token in tokens if compact(token) not in text_view]
    letters = sum(c.isalpha() for c in source_text)
    return {
        'raw_sha256': raw['raw_sha256'], 'raw_checksum_verified': True,
        'declared_charset': raw.get('declared_charset'), 'meta_charset': meta,
        'recorded_detected_charset': raw.get('detected_charset'), 'dom_encoding_override': encoding,
        'dom_encoding_resolution': encoding_evidence,
        'source_title': title, 'source_headings': source_headings[:40],
        'source_headings_present_in_output_text': retained[:40],
        'present_headings_without_markdown_heading': demoted[:40],
        'source_main_candidate_tag': candidate.tag, 'source_main_candidate_chars': len(source_text),
        'source_main_candidate_cjk_ratio': sum('\u4e00' <= c <= '\u9fff' for c in source_text)/max(letters, 1),
        'source_main_candidate_preview': source_text[:4000],
        'source_main_candidate_tail': source_text[-1000:],
        'source_visible_chars': len(visible_text(tree)),
        'source_tables': len(candidate.xpath('.//table')),
        'source_list_items': len(candidate.xpath('.//li')),
        'source_biomedical_tokens': tokens[:60],
        'biomedical_tokens_absent_in_output_candidate_only': absent_tokens[:60],
        'candidate_note': 'DOM selection is heuristic; missing tokens/headings need inspection, not automatic failure.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/stage_b_windows_10k'))
    parser.add_argument('--baseline-root', type=Path, help='Include the exact previous raw review sample for paired comparison.')
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(output_dir=root)
    destination = root/'reports/stage_b/quality_audit'
    destination.mkdir(parents=True, exist_ok=True)
    crawl_path = root/'data/manifests/crawl_manifest.parquet'
    extract_path = root/'data/manifests/extraction_manifest.parquet'
    document_path = root/'data/processed/documents/documents.parquet'
    crawl = {row['crawl_url_id']: row for row in parquet_rows(crawl_path)}
    records, groups = {}, defaultdict(list)
    counts, domains = defaultdict(Counter), defaultdict(Counter)
    for row in parquet_rows(extract_path, batch_size=128):
        item = features(row, crawl[row['crawl_url_id']])
        records[row['crawl_url_id']] = item
        for flag in item['flags']:
            counts['audit_flags'][flag] += 1
        counts['extract_status'][item['extract_status']] += 1
        counts['all_extraction_language'][item['language']] += 1
        domains[item['domain']][item['extract_status']] += 1
        if item['content_hash']:
            groups[item['content_hash']].append(item['crawl_url_id'])
        if item['extract_status'] == 'EXTRACT_SUCCESS':
            counts['extracted_quality'][item['quality_tier']] += 1
            counts['successful_extraction_language'][item['language']] += 1
            domains[item['domain']][item['quality_tier']] += 1
            for flag in item['flags']:
                domains[item['domain']][flag] += 1
            for name in ('heading_count', 'list_item_count', 'table_count', 'pipe_rows'):
                counts['successful_structure'][name + '_nonzero'] += int(bool(item[name]))
            counts['successful_metadata']['title_present'] += int(bool(item['title']))

    canonical_ids = set()
    for row in parquet_rows(document_path, batch_size=128):
        canonical_ids.add(row['representative_crawl_url_id'])
        counts['canonical_quality'][row['quality_tier']] += 1
        counts['canonical_language'][row['language']] += 1
        for flag in records[row['representative_crawl_url_id']]['flags']:
            counts['canonical_audit_flags'][flag] += 1
    for record in records.values():
        record['is_canonical_representative'] = record['crawl_url_id'] in canonical_ids

    duplicates = []
    for digest, ids in groups.items():
        if len(ids) < 2:
            continue
        items = [records[url_id] for url_id in ids]
        distinct_titles = sorted(set(item['title'] or '' for item in items))
        suspected = len(ids) >= 3 and len(distinct_titles) >= 3
        if suspected:
            for item in items:
                item['flags'].append('same_body_different_titles_review')
        duplicates.append({'content_hash': digest, 'url_count': len(ids),
                           'domain_count': len(set(item['domain'] for item in items)),
                           'distinct_title_count': len(distinct_titles), 'titles': distinct_titles[:8],
                           'domain': items[0]['domain'], 'char_count': items[0]['char_count'],
                           'representative_crawl_url_id': next((i for i in ids if i in canonical_ids), None),
                           'suspected_template': suspected, 'preview': items[0]['preview']})
    duplicates.sort(key=lambda item: (-item['url_count'], item['content_hash']))

    with (root/'reports/stage_b/manual_review.csv').open(encoding='utf-8-sig', newline='') as handle:
        original_sample = [row['crawl_url_id'] for row in csv.DictReader(handle)]
    reasons = defaultdict(list)
    if args.baseline_root:
        baseline_path = args.baseline_root/'reports/stage_b/quality_audit/sample_evidence.jsonl'
        for line in baseline_path.read_text(encoding='utf-8').splitlines():
            key = json.loads(line)['crawl_url_id']
            if key in records:
                reasons[key].append('paired_baseline_raw_sample')
    for url_id in original_sample:
        reasons[url_id].append('existing_100_doc_review_sample')
    # Inspect all unexpected scripts plus deterministic targeted examples of other risks/structure.
    for url_id, item in sorted(records.items()):
        if 'unexpected_cyrillic' in item['flags'] or 'replacement_characters' in item['flags']:
            reasons[url_id].append('encoding_risk')
    for label, predicate in (
        ('empty_extraction', lambda r: r['extract_status'] != 'EXTRACT_SUCCESS'),
        ('low_vi', lambda r: r['quality_tier'] == 'LOW' and r['language'] == 'vi'),
        ('low_zh', lambda r: r['quality_tier'] == 'LOW' and r['language'] == 'zh'),
        ('template_placeholder', lambda r: 'unresolved_template_placeholder' in r['flags']),
        ('table', lambda r: bool(r['table_count'])),
        ('heading_vi', lambda r: r['language'] == 'vi' and r['heading_count'] >= 3),
        ('heading_zh', lambda r: r['language'] == 'zh' and r['heading_count'] >= 3),
        ('missing_title', lambda r: 'missing_title' in r['flags']),
    ):
        for item in sorted((r for r in records.values() if predicate(r)), key=lambda r: r['crawl_url_id'])[:5]:
            reasons[item['crawl_url_id']].append(label)
    for group in duplicates[:12]:
        reasons[group['representative_crawl_url_id']].append('top_duplicate_group')
    selected = set(reasons)
    samples = []
    store = RawStore(config)
    for row in parquet_rows(extract_path, batch_size=128):
        url_id = row['crawl_url_id']
        if url_id not in selected:
            continue
        item = dict(records[url_id], sample_reasons=reasons[url_id],
                    text=row.get('text') or '', text_markdown=row.get('text_markdown') or '')
        try:
            body = store.read(crawl[url_id], config['crawler']['max_response_bytes'])
            item['source'] = source_evidence(body, crawl[url_id], row)
        except Exception as error:
            item['source_error'] = f'{type(error).__name__}: {error}'
        samples.append(item)
        if len(samples) % 25 == 0:
            print(f'Raw evidence collected: {len(samples)}/{len(selected)}', flush=True)
    samples.sort(key=lambda row: row['crawl_url_id'])
    sample_path = destination/'sample_evidence.jsonl'
    with sample_path.open('w', encoding='utf-8') as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    flagged = [dict(item, flags=';'.join(item['flags'])) for item in records.values() if item['flags']]
    write_csv(destination/'flagged_documents.csv', flagged,
              ['crawl_url_id','url','final_url','domain','title','language','extract_status','quality_tier',
               'char_count','is_canonical_representative','content_hash','flags','preview'])
    domain_rows = [{'domain': name, **values} for name, values in sorted(domains.items())]
    fields = ['domain'] + sorted(set(key for row in domain_rows for key in row if key != 'domain'))
    write_csv(destination/'domain_quality.csv', domain_rows, fields)
    all_flags = Counter(flag for item in records.values() for flag in item['flags'])
    report = {
        'audit_version': AUDIT_VERSION, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Offline audit of the completed 10k pilot; no network or pipeline mutations.',
        'inputs': {str(p.relative_to(root)): {'sha256': sha256_file(p), 'rows': pq.ParquetFile(p).metadata.num_rows}
                   for p in (crawl_path, extract_path, document_path)},
        'counts': {name: dict(value) for name, value in counts.items()},
        'all_flags_including_duplicate_review': dict(all_flags),
        'dedup': {'duplicate_groups': len(duplicates), 'collapsed_url_rows': sum(d['url_count']-1 for d in duplicates),
                  'top_groups': duplicates[:30]},
        'raw_comparison': {
            'sample_size': len(samples), 'includes_existing_review_sample': len(original_sample),
            'raw_checksum_verified': sum('source' in sample for sample in samples),
            'source_errors': [{'crawl_url_id':s['crawl_url_id'], 'error':s['source_error']} for s in samples if 'source_error' in s],
            'unexpected_script_samples_with_source_cjk': sum('unexpected_cyrillic' in s['flags'] and
                    s.get('source',{}).get('source_main_candidate_cjk_ratio',0) > .4 for s in samples),
            'samples_with_present_heading_demoted': sum(bool(s.get('source',{}).get('present_headings_without_markdown_heading')) for s in samples),
            'note': 'Risk-enriched source comparison, not a random quality pass-rate estimate.',
        },
        'review_status': 'automated_evidence_complete_agent_review_pending',
        'human_review_complete': False, 'gold_relevance_evaluable': False, 'scale_authorized': False,
        'reference_manual_review_sha256': sha256_file(root/'reports/stage_b/manual_review.csv'),
        'limitations': ['Length tiers are proxies, not semantic quality scores.',
                       'Raw DOM main candidate is heuristic and cannot establish article completeness alone.',
                       'No qrels: relevance, Recall/MRR/nDCG and gold LOW impact cannot be measured.',
                       'This stratified 10k sample is not a full-corpus quality or language distribution estimate.'],
    }
    atomic_json(destination/'audit.json', report)
    print(json.dumps({'audit': str(destination/'audit.json'), 'samples':len(samples),
                     'counts':report['counts'], 'flags':dict(all_flags),
                     'raw_comparison':report['raw_comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
