"""Paired offline comparison of all URL rows and fixed native raw regressions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_quality import features
from src.storage.io import atomic_json, parquet_rows, sha256_file

EXTRA = {'96ffab2dcacef7f04f97c7b15b90af58c13a451fb7adebf31f231336b4251a35',
         'e1c7ba82a0d58b4b63b19cc992a09dac6010d414db28c85c51b31e02c217e1b8'}


def collect(root, selected):
    crawl={r['crawl_url_id']:r for r in parquet_rows(root/'data/manifests/crawl_manifest.parquet')}
    records,full,counts={},{},{'status':Counter(),'flags':Counter(),'canonical_tiers':Counter(),'extraction_runs':Counter()}
    for row in parquet_rows(root/'data/manifests/extraction_manifest.parquet',batch_size=128):
        key=row['crawl_url_id'];record=features(row,crawl[key]);record['raw_sha256']=row['raw_sha256']
        records[key]=record;counts['status'][row['extract_status']]+=1;counts['flags'].update(record['flags'])
        counts['extraction_runs'][row['extraction_run_id']]+=1
        if key in selected:full[key]=row
    for row in parquet_rows(root/'data/processed/documents/documents.parquet',batch_size=128):
        counts['canonical_tiers'][row['quality_tier']]+=1
    return records,full,{key:dict(values) for key,values in counts.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,default=Path('outputs/stage_b_windows_10k'))
    parser.add_argument('--after',type=Path,default=Path('outputs/stage_b_windows_10k_reextract_v2'))
    args=parser.parse_args();audit=args.before/'reports/stage_b/quality_audit'
    samples={r['crawl_url_id']:r for r in map(json.loads,(audit/'sample_evidence.jsonl').read_text(encoding='utf-8').splitlines())}
    selection=json.loads((audit/'agent_review_selection.json').read_text(encoding='utf-8'))
    old,old_full,before=collect(args.before,set(samples)|EXTRA)
    new,new_full,after=collect(args.after,set(samples)|EXTRA)
    same_ids=old.keys()==new.keys()
    same_raw=same_ids and all(old[k]['raw_sha256']==new[k]['raw_sha256'] for k in old)
    if not same_raw:raise ValueError('Comparison requires identical URL IDs and raw SHA256.')
    target=args.after/'reports/stage_b/quality_comparison';target.mkdir(parents=True,exist_ok=True)
    delta=[]
    for key in sorted(old):
        a,b=old[key],new[key]
        delta.append({'crawl_url_id':key,'url':a['url'],'domain':a['domain'],
            'status_before':a['extract_status'],'status_after':b['extract_status'],
            'tier_before':a['quality_tier'],'tier_after':b['quality_tier'],
            'chars_before':a['char_count'],'chars_after':b['char_count'],
            'headings_before':a['heading_count'],'headings_after':b['heading_count'],
            'flags_before':';'.join(a['flags']),'flags_after':';'.join(b['flags'])})
    with (target/'url_deltas.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(delta[0]));writer.writeheader();writer.writerows(delta)
    checks=[]
    def selected(number):return selection[number-1]['crawl_url_id']
    def check(name,key,tokens=(),markdown_tokens=(),status='EXTRACT_SUCCESS'):
        row=new_full[key]
        checks.append({'name':name,'crawl_url_id':key,'url':new[key]['url'],
            'passed':row['extract_status']==status and all(t in row.get('text','') for t in tokens) and
                     all(t in row.get('text_markdown','') for t in markdown_tokens),
            'status':row['extract_status'],'required_text':list(tokens),'required_markdown':list(markdown_tokens)})
    check('SuperShoes article, not recommendations',selected(6),('SuperShoes','Dhairya'))
    check('120ask question and doctor answer',selected(16),('我的孩子5岁','最好查查微量元素'))
    check('120ask former shared menu now contains answer',selected(30),('肝动脉栓塞','根据你所咨询'))
    check('Cnkang colorectal chemotherapy body',selected(27),('卡培他滨','5×5'))
    checks.append({'name':'Cnkang title matches colorectal article','crawl_url_id':selected(27),
                   'passed':'直肠癌' in (new_full[selected(27)]['title'] or '')})
    check('GB2312 Wujue decoded Chinese',selected(31),('有些妈妈生下小宝宝','乳头错觉'))
    check('MSD table generations, antimicrobial limits and footnote',selected(34),
          ('第一代','第二代','头孢菌素类有下列局限性','甲氧西林耐药','金黄色葡萄球菌'),('## 头孢菌素的药代动力学',))
    check('An Giang lead and 247 million vaccine doses',selected(38),('Việt Nam là nước triển khai','247 triệu'))
    check('Laodong HTTP200 cookie gate deferred',selected(37),status='JAVASCRIPT_CHALLENGE')
    check('Phu Tho template quarantined',selected(26),status='UNRESOLVED_TEMPLATE')
    for prefix in ('005bdbff','4c39c3e1'):
        key=next(k for k in samples if k.startswith(prefix))
        checks.append({'name':'Medlatec nested H2 preserved','crawl_url_id':key,
                       'passed':any(line.startswith('## ') for line in new_full[key]['text_markdown'].splitlines()),
                       'headings_before':old[key]['heading_count'],'headings_after':new[key]['heading_count']})
    checks.append({'name':'71 formerly Cyrillic source samples recovered',
        'passed':all('unexpected_cyrillic' not in new[k]['flags'] and new_full[k]['extract_status']=='EXTRACT_SUCCESS'
                     for k in samples if 'unexpected_cyrillic' in samples[k]['flags']),
        'sample_size':sum('unexpected_cyrillic' in r['flags'] for r in samples.values())})
    checks.append({'name':'120ask shared navigation removed from all outputs',
                   'passed':not after['flags'].get('120ask_navigation_present',0),
                   'urls_before':before['flags'].get('120ask_navigation_present',0),
                   'urls_after':after['flags'].get('120ask_navigation_present',0)})
    check('Cnkang short tag introduction retained',next(k for k in EXTRA if k.startswith('96ff')),('葡萄籽简介','葡萄籽功效'))
    check('Wujue short disease entry retained',next(k for k in EXTRA if k.startswith('e1c7')),('血瘤','體表血絡'))
    checks.append({'name':'No previously nonempty URL became empty',
                   'passed':not any(a['extract_status']=='EXTRACT_SUCCESS' and new[k]['extract_status']=='EMPTY_MAIN_TEXT' for k,a in old.items())})
    checks.append({'name':'All output rows belong to one extraction implementation run',
                   'passed':len(after['extraction_runs'])==1})
    evidence=[]
    for number,item in enumerate(selection,1):
        key=item['crawl_url_id'];row=new_full[key]
        evidence.append({'number':number,'crawl_url_id':key,'url':new[key]['url'],'title':row.get('title'),
            'status':row['extract_status'],'tier':row.get('quality_tier'),'structure_source':row.get('structure_source'),
            'text':row.get('text'),'text_markdown':row.get('text_markdown')})
    with (target/'fixed_40_evidence.jsonl').open('w',encoding='utf-8') as handle:
        for row in evidence:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'evaluated_at':datetime.now(timezone.utc).isoformat(),'scope':'Paired offline comparison of identical cached pilot raw',
        'same_url_ids':same_ids,'same_raw_checksums':same_raw,'url_rows':len(old),'fixed_raw_sample':len(samples),
        'additional_regression_samples':len(EXTRA),'before':before,'after':after,'checks':checks,
        'native_regressions_passed':all(c['passed'] for c in checks),'network_requests':0,
        'human_review_complete':False,'qrels_available':False,'scale_authorized':False,
        'baseline_manual_review_sha256':sha256_file(args.before/'reports/stage_b/manual_review.csv'),
        'limitations':['Length tiers are not fidelity or retrieval scores.','Targeted samples are not a population pass-rate.',
                       'Source medical accuracy was not validated.','No qrels: gold coverage and Recall/MRR/nDCG unavailable.']}
    atomic_json(target/'comparison.json',report)
    print(json.dumps({k:report[k] for k in ('url_rows','before','after','native_regressions_passed','checks')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
