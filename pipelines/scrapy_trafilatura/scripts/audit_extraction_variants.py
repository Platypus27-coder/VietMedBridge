"""Supervised offline experiments; original manifests/documents remain immutable."""
from __future__ import annotations

import codecs
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crawler.raw_store import RawStore
from src.extraction.html import extract_html
from src.extraction.worker import supervised_extract
from src.preprocessing.cleaner import clean_text
from src.preprocessing.language import detect_language
from src.preprocessing.quality import assess_quality
from src.storage.io import atomic_json, parquet_rows
from src.utils.config import load_config


def trial_candidate(row, config):
    body = RawStore(config).read(row, config['extraction']['html']['max_input_bytes'])
    evidence = {'crawl_url_id': row['crawl_url_id'], 'variant': row['audit_variant']}
    if row['audit_variant'] == 'slow_charset_aware_bytes':
        match = re.search(rb'<meta\b[^>]*charset\s*=\s*[\"\']?\s*([a-zA-Z0-9_.-]+)', body[:16384], re.I)
        charset = match.group(1).decode('ascii') if match else row.get('declared_charset')
        try:
            codec = codecs.lookup(charset).name if charset else 'utf-8'
        except LookupError:
            codec = 'utf-8'
        if codec in ('gb2312', 'gbk'):
            codec = 'gb18030'
        # Diagnostic only: an explicit source charset is decoded strictly, and the
        # extractor still receives bytes. The immutable raw is never rewritten.
        body = body.decode(codec, errors='strict').encode('utf-8')
        evidence['explicit_encoding'] = codec
    extracted = extract_html(body, row['final_url'], config)
    if not extracted:
        return dict(evidence, extract_status='EMPTY_MAIN_TEXT', text='', text_markdown='')
    extracted['text'] = clean_text(extracted['text'])
    extracted['text_markdown'] = clean_text(extracted['text_markdown'], markdown=True)
    extracted.update(assess_quality(extracted['text'], extracted['text_markdown'], extracted['title'], config))
    extracted.update(detect_language(extracted['text'], config))
    return dict(extracted, **evidence, extract_status='EXTRACT_SUCCESS')


def trial_one(row, config):
    try:
        return trial_candidate(row, config)
    except Exception as error:
        return {'crawl_url_id':row['crawl_url_id'], 'variant':row['audit_variant'],
                'extract_status':'TRIAL_FAILED', 'error':f'{type(error).__name__}: {error}',
                'text':'', 'text_markdown':''}


def main():
    root = Path('outputs/stage_b_windows_10k').resolve()
    destination = root/'reports/stage_b/quality_audit'
    samples = [json.loads(line) for line in (destination/'sample_evidence.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = {}
    report = json.loads((destination/'audit.json').read_text(encoding='utf-8'))
    for group in report['dedup']['top_groups'][:5]:
        selected[group['representative_crawl_url_id']] = 'top_duplicate_group'
    for label, count in [('encoding_risk', 4), ('empty_extraction', 2), ('heading_zh', 2), ('table', 2)]:
        for sample in [s for s in samples if label in s['sample_reasons']][:count]:
            selected[sample['crawl_url_id']] = label
    extra = next(s for s in samples if s['domain']=='thanhnien.vn' and 'tin liên quan' in s['text'])
    selected[extra['crawl_url_id']] = 'related_article_noise'
    config = load_config(output_dir=root)
    config['extraction']['html']['fast_first'] = False
    crawl = {r['crawl_url_id']:r for r in parquet_rows(root/'data/manifests/crawl_manifest.parquet')}
    tasks = []
    for url_id in selected:
        for variant in ('slow_bytes', 'slow_charset_aware_bytes'):
            tasks.append((dict(crawl[url_id], audit_variant=variant), 1))
    results = []
    for result, _ in supervised_extract(tasks, config, task_function=trial_one):
        results.append(result)
        print(f'Offline experiment {len(results)}/{len(tasks)}: {result.get("variant")} '
              f'{result.get("extract_status")} chars={result.get("char_count",0)}', flush=True)
    with (destination/'extraction_variant_evidence.jsonl').open('w', encoding='utf-8') as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False)+'\n')
    original = {s['crawl_url_id']:s for s in samples}
    summaries = []
    for result in results:
        before = original[result['crawl_url_id']]
        text = result.get('text') or ''
        summaries.append({'crawl_url_id':result['crawl_url_id'], 'domain':before['domain'],
                          'reason':selected[result['crawl_url_id']], 'variant':result.get('variant'),
                          'original_chars':before['char_count'], 'new_chars':len(text),
                          'original_title':before['title'], 'new_title':result.get('title'),
                          'original_language':before['language'], 'new_language':result.get('language'),
                          'status':result['extract_status'], 'original_headings':before['heading_count'],
                          'new_headings':result.get('heading_count'),
                          'warning':'More text or headings alone does not establish correct main-body extraction.'})
    atomic_json(destination/'extraction_variant_summary.json', {'cases':len(selected), 'variants':2,
                'supervised_timeout_seconds':30, 'original_data_changed':False, 'results':summaries})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
