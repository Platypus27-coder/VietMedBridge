"""Compare repaired extraction against the fixed native source sample, offline."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.worker import supervised_extract
from src.storage.io import atomic_json, parquet_rows
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/stage_b_windows_10k'))
    parser.add_argument('--destination', type=Path, default=Path('outputs/extraction_v2_regression'))
    args = parser.parse_args()
    audit = args.root/'reports/stage_b/quality_audit'
    samples = {r['crawl_url_id']: r for r in map(json.loads, (audit/'sample_evidence.jsonl').read_text(encoding='utf-8').splitlines())}
    selected = json.loads((audit/'agent_review_selection.json').read_text(encoding='utf-8'))
    crawl = {r['crawl_url_id']: r for r in parquet_rows(args.root/'data/manifests/crawl_manifest.parquet')}
    config = load_config(output_dir=args.root)
    args.destination.mkdir(parents=True, exist_ok=True)
    rows, statuses = {}, Counter()
    with (args.destination/'results.jsonl').open('w', encoding='utf-8') as handle:
        tasks = ((crawl[key], 1) for key in sorted(samples))
        for row, _ in supervised_extract(tasks, config):
            key = row['crawl_url_id']
            rows[key] = row
            statuses[row['extract_status']] += 1
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            handle.flush()
            if len(rows) % 25 == 0:
                print(f'Native raw regressions: {len(rows)}/{len(samples)}', flush=True)
    report = {'sample_size': len(rows), 'statuses': dict(statuses),
              'unexpected_cyrillic_before': sum('unexpected_cyrillic' in r['flags'] for r in samples.values()),
              'unexpected_cyrillic_after': sum(sum('\u0400' <= c <= '\u04ff' for c in r.get('text',''))/max(sum(c.isalpha() for c in r.get('text','')),1) > .2 for r in rows.values()),
              'replacement_after': sum('\ufffd' in r.get('text','') for r in rows.values()), 'selected': []}
    for number, item in enumerate(selected, 1):
        key = item['crawl_url_id']
        old, new = samples[key], rows[key]
        report['selected'].append({'number': number, 'crawl_url_id':key, 'url':old['url'],
            'before_chars':len(old['text']), 'after_chars':len(new.get('text','')), 'title':new.get('title'),
            'status':new['extract_status'], 'structure_source':new.get('structure_source'),
            'headings_before':old['heading_count'], 'headings_after':new.get('heading_count'),
            'preview':new.get('text','')[:240]})
    atomic_json(args.destination/'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
