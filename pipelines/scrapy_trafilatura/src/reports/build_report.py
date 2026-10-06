from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from importlib.metadata import version

import pyarrow as pa

from src.storage.index import ManifestIndex
from src.storage.io import atomic_json, parquet_rows, sha256_file, write_parquet
from src.utils.config import output_path
from src.utils.environment import environment_info


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def build_report(config: dict, stage='stage_a') -> dict:
    target = output_path(config, f'reports/{stage}')
    target.mkdir(parents=True, exist_ok=True)
    inventory_state = json.loads(output_path(config, 'checkpoints/inventory_state.json').read_text())
    sample = json.loads(output_path(config, 'checkpoints/sample.json').read_text()) if output_path(config, 'checkpoints/sample.json').exists() else None
    crawls = ManifestIndex(output_path(config, 'checkpoints/crawl_index.sqlite'), output_path(config, 'data/manifests/crawl_manifest.parquet'))
    extracts = ManifestIndex(output_path(config, 'checkpoints/extraction_index.sqlite'), output_path(config, 'data/manifests/extraction_manifest.parquet'))
    availability_path = output_path(config, 'checkpoints/gold_availability.json')
    gold_available = availability_path.exists() and json.loads(availability_path.read_text()).get('available', False)
    outcomes = (ManifestIndex(output_path(config, 'checkpoints/outcomes_index.sqlite'), output_path(config, 'data/mappings/doc_outcomes.parquet'), key='source_row_no')
                if gold_available else None)
    inventory = sqlite3.connect(output_path(config, 'data/inventory/inventory.sqlite'))
    statuses = Counter()
    languages = Counter()
    quality = Counter()
    extract_statuses = Counter()
    source_issues = Counter()
    structure_sources = Counter()
    mimes = Counter()
    domain_counts = defaultdict(Counter)
    elapsed = defaultdict(list)
    raw_bytes = compressed_bytes = mismatches = fallback = fast = extraction_success = 0
    try:
        for row in crawls:
            statuses[row['status']] += 1
            domain_counts[row['domain']][row['status']] += 1
            elapsed[row['domain']].append(row['elapsed_ms']/1000)
            mismatches += bool(row['content_type_mismatch'])
            if row['status'] == 'SUCCESS_RAW':
                raw_bytes += row['raw_size_bytes']
                compressed_bytes += row['compressed_size_bytes']
                mimes[row['effective_content_type']] += 1
        for row in extracts:
            extract_statuses[row['extract_status']] += 1
            if row.get('source_issue'):
                source_issues[row['source_issue']] += 1
            structure_sources[row.get('structure_source') or 'unknown'] += 1
            languages[row['language'] or 'unknown'] += 1
            extraction_success += row['extract_status'] == 'EXTRACT_SUCCESS'
            fast += bool(row['fast_pass_success'])
            fallback += bool(row['fallback_used'])
        # Count document rows for accounting; canonical quality separately avoids duplicate bias.
        for row in parquet_rows(output_path(config, 'data/processed/documents/documents.parquet')):
            quality[row['quality_tier']] += 1
        gold = {}
        queries = defaultdict(set)
        for row in parquet_rows(output_path(config, 'data/inventory/gold_doc_map.parquet')):
            key = json.dumps(row['gold_doc_id'], ensure_ascii=False)
            gold[key] = row['gold_doc_id']
            queries[row['query_id']].add(key)
        audited = {}
        coverage_rows = []
        missing = invalid = pending = 0
        for key, doc_id in gold.items():
            source = inventory.execute('SELECT row_no,url_id,status FROM source_rows WHERE doc_key=? LIMIT 1', (key,)).fetchone()
            if not source:
                outcome = {'crawl_status': 'MISSING_FROM_INVENTORY', 'extract_status': None, 'quality_tier': None, 'handoff_eligible': False}
                missing += 1
            else:
                outcome = outcomes.get(source[0])
                invalid += source[1] is None
            is_pending = outcome['crawl_status'] == 'PENDING' or (outcome['crawl_status'] == 'SUCCESS_RAW' and outcome['extract_status'] is None)
            pending += is_pending
            if not is_pending:
                audited[key] = outcome
            coverage_rows.append({'gold_doc_id': doc_id, 'crawl_url_id': source[1] if source else None,
                                  'crawl_status': outcome['crawl_status'], 'extract_status': outcome['extract_status'],
                                  'quality_tier': outcome['quality_tier'], 'audited': not is_pending,
                                  'handoff_eligible': outcome['handoff_eligible']})
        def usable(row):
            return row['extract_status'] == 'EXTRACT_SUCCESS' and row['quality_tier'] not in (None, 'QUARANTINE')
        raw_gold = sum(row['crawl_status'] == 'SUCCESS_RAW' for row in audited.values())
        usable_gold = sum(usable(row) for row in audited.values())
        eligible_gold = sum(row['handoff_eligible'] for row in audited.values())
        complete_queries = [values for values in queries.values() if values and values <= audited.keys()]
        query_fractions = [sum(usable(audited[key]) for key in values)/len(values) for values in complete_queries]
        id_type = __import__('pyarrow.parquet', fromlist=['ParquetFile']).ParquetFile(output_path(config, 'data/inventory/gold_doc_map.parquet')).schema_arrow.field('gold_doc_id').type
        coverage_schema = pa.schema([('gold_doc_id', id_type), ('crawl_url_id', pa.string()), ('crawl_status', pa.string()),
                                    ('extract_status', pa.string()), ('quality_tier', pa.string()), ('audited', pa.bool_()), ('handoff_eligible', pa.bool_())])
        write_parquet(target/'gold_coverage.parquet', coverage_rows, coverage_schema)
        domain_rows = []
        domain_decisions = []
        for domain, count in domain_counts.items():
            total = sum(count.values())
            mean = statistics.mean(elapsed[domain])
            concurrency = config['crawler']['download_slots'].get(domain, {}).get('concurrency', config['crawler']['concurrent_requests_per_domain'])
            rate = concurrency/mean if mean else None
            projected = inventory.execute('SELECT COUNT(*) FROM urls WHERE domain=?', (domain,)).fetchone()[0]
            blocked = ratio(count['ROBOTS_DENIED'] + count['HTTP_403'], total)
            domain_rows.append({'domain': domain, 'sampled_urls': total, 'unique_url_count': projected,
                                'mean_request_seconds': mean, 'nominal_urls_per_second': rate,
                                'estimated_download_hours': projected/rate/3600 if rate else None, 'blocked_rate': blocked})
            if blocked > config['coverage']['blocked_review_threshold']:
                domain_decisions.append({'domain': domain, 'blocked_rate': blocked, 'decision': 'pending_review', 'options': ['btc_cache', 'approved_archive', 'accept_missing', 'defer_for_later'], 'decided_by': None})
        write_parquet(target/'domain_estimates.parquet', domain_rows, pa.schema([
            ('domain', pa.string()), ('sampled_urls', pa.int64()), ('unique_url_count', pa.int64()),
            ('mean_request_seconds', pa.float64()), ('nominal_urls_per_second', pa.float64()),
            ('estimated_download_hours', pa.float64()), ('blocked_rate', pa.float64())]))
        decisions_path = target/'coverage_decisions.json'
        existing = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
        retained = {d['domain']: d for d in existing.get('domains', [])}
        for decision in domain_decisions:
            if decision['domain'] in retained:
                decision.update({k: retained[decision['domain']][k] for k in ('decision', 'decided_by', 'decided_at', 'notes') if k in retained[decision['domain']]})
        # Keep earlier decisions even when a later sample has no URLs for that domain.
        domain_decisions.extend(d for domain, d in retained.items() if domain not in {r['domain'] for r in domain_decisions})
        atomic_json(decisions_path, {'domains': domain_decisions})
        review_path = target/'manual_review.csv'
        existing_reviews = []
        if review_path.exists():
            with review_path.open(encoding='utf-8', newline='') as handle:
                existing_reviews = list(csv.DictReader(handle))
        if len(existing_reviews) < config['storage']['manual_review_size']:
            existing_ids = {row['crawl_url_id'] for row in existing_reviews}
            candidate_rows = []
            for row in extracts:
                if row['crawl_url_id'] in existing_ids:
                    continue
                raw = crawls.get(row['crawl_url_id'])
                score = hashlib.sha256((str(config['dataset']['sample_seed'])+row['crawl_url_id']).encode()).hexdigest()
                candidate_rows.append((score, row, raw))
                if len(candidate_rows) > config['storage']['manual_review_size']*2:
                    candidate_rows.sort(key=lambda r: r[0])
                    del candidate_rows[config['storage']['manual_review_size']:]
            candidate_rows.sort(key=lambda r: r[0])
            fields = ['crawl_url_id', 'url', 'language', 'title', 'preview', 'content_ok', 'title_ok', 'encoding_ok',
                      'heading_ok', 'list_ok', 'table_ok', 'biomedical_tokens_ok', 'boilerplate_level', 'notes']
            with review_path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(existing_reviews)
                available = config['storage']['manual_review_size'] - len(existing_reviews)
                for _, row, raw in candidate_rows[:available]:
                    writer.writerow({'crawl_url_id': row['crawl_url_id'], 'url': raw['final_url'], 'language': row['language'],
                                     'title': row['title'], 'preview': (row['text'] or '')[:500]})
        completed = sum(statuses.values())
        blocked_rate = ratio(statuses['ROBOTS_DENIED']+statuses['HTTP_403'], completed)
        report = {'report_status': 'measured', 'snapshot_id': inventory_state['snapshot_id'], 'config_sha256': config['_config_sha256'],
                  'versions': {'scrapy': version('scrapy'), 'trafilatura': version('trafilatura')},
                  'environment': environment_info(output_path(config, '.')), 'sample_seed': config['dataset']['sample_seed'],
                  'handoff_mode': config['quality']['handoff_mode'], 'input_docs': inventory_state['input_rows'],
                  'inventory_unique_urls': inventory_state['unique_urls'], 'unique_urls': sample['sample_size'] if sample else inventory_state['unique_urls'],
                  'invalid_urls': inventory_state['invalid_rows'], 'crawl': dict(statuses), 'crawl_rates': {k: ratio(v, completed) for k, v in statuses.items()},
                  'blocked_rate': blocked_rate, 'storage': {'raw_bytes': raw_bytes, 'compressed_bytes': compressed_bytes, 'compression_ratio': ratio(raw_bytes, compressed_bytes)},
                  'content_types': dict(mimes), 'content_type_mismatches': mismatches,
                  'extraction': {'success': extraction_success, 'total_raw_documents': sum(extract_statuses.values()),
                                 'statuses': dict(extract_statuses), 'source_issues': dict(source_issues),
                                 'structure_sources':dict(structure_sources), 'fast_pass_success': fast, 'fallback_used': fallback},
                  'quality': dict(quality), 'languages': dict(languages),
                  'gold_coverage': {'distinct_gold_doc_ids_total': len(gold), 'missing_from_inventory': missing, 'invalid_url_doc_ids': invalid,
                                    'audited_gold_doc_ids': len(audited), 'pending_gold_doc_ids': pending,
                                    'raw_coverage_on_audited_set': ratio(raw_gold, len(audited)), 'usable_coverage_on_audited_set': ratio(usable_gold, len(audited)),
                                    'handoff_coverage_on_audited_set': ratio(eligible_gold, len(audited)),
                                    'low_doc_ids': sum(r['quality_tier'] == 'LOW' for r in audited.values()),
                                    'quarantine_doc_ids': sum(r['quality_tier'] == 'QUARANTINE' for r in audited.values()),
                                    'fully_evaluated_queries': len(complete_queries), 'pending_queries': len(queries)-len(complete_queries),
                                    'query_any_gold_usable_rate': ratio(sum(f > 0 for f in query_fractions), len(query_fractions)),
                                    'query_all_gold_usable_rate': ratio(sum(f == 1 for f in query_fractions), len(query_fractions))},
                  'gates': {'blocked_domains_decision_complete': all(d['decision'] != 'pending_review' for d in domain_decisions),
                            'manual_review_complete': False, 'stage_b_ready': False, 'low_handoff_policy': config['quality']['low_handoff_policy']}}
        lock = output_path(config, 'requirements-lock.txt')
        if not lock.exists():
            from pathlib import Path
            lock = Path(config['_project_root'])/'requirements-lock.txt'
        report['lockfile_sha256'] = sha256_file(lock) if lock.exists() else None
        availability = output_path(config, 'checkpoints/gold_availability.json')
        if availability.exists():
            report['gold_coverage'].update(json.loads(availability.read_text()))
            report['gates']['gold_coverage_available'] = report['gold_coverage']['available']
        review_path = target/'review_decision.json'
        review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
        integrity_path = output_path(config, 'reports/integrity.json')
        integrity = json.loads(integrity_path.read_text(encoding='utf-8')) if integrity_path.exists() else {}
        report['review_decision'] = review or None
        report['gates'].update(review_gates(review, domain_decisions, gold_available, integrity.get('passed', False)))
        atomic_json(target/'report.json', report)
        return report
    finally:
        crawls.close()
        extracts.close()
        if outcomes is not None:
            outcomes.close()
        inventory.close()


def review_gates(review, domains, gold_available, integrity_passed):
    """An explicit overall user review does not invent per-document CSV ratings."""
    content_ok = review.get('content_review', {}).get('accepted') is True
    gold_ok = gold_available or review.get('gold_coverage', {}).get('decision') == 'proceed_without_qrels'
    allowed = {'btc_cache', 'approved_archive', 'accept_missing', 'defer_for_later'}
    domains_ok = all(d.get('decision') in allowed and d.get('decided_by') for d in domains)
    pilot = review.get('stage_b_pilot', {})
    authorized = pilot.get('authorized') is True and pilot.get('max_urls', 0) == 10000
    deployment_ok = review.get('deployment', {}).get('target') in {'windows_local', 'linux_vps', 'wsl', 'kaggle'}
    return {'manual_review_complete': content_ok, 'gold_coverage_decision_complete': gold_ok,
            'blocked_domains_decision_complete': domains_ok, 'integrity_passed': integrity_passed,
            'deployment_decision_complete': deployment_ok, 'stage_b_pilot_authorized': authorized,
            'stage_b_ready': bool(content_ok and gold_ok and domains_ok and integrity_passed and deployment_ok and authorized)}
