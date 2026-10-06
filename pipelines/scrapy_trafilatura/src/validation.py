from __future__ import annotations

import sqlite3
from collections import Counter

from src.crawler.raw_store import RawStore
from src.storage.index import ManifestIndex
from src.storage.io import atomic_json, parquet_rows
from src.utils.config import output_path


def validate_ingestion(config: dict, require_complete=True) -> dict:
    errors = []
    database = sqlite3.connect(output_path(config, 'data/inventory/inventory.sqlite'))
    crawl = ManifestIndex(output_path(config, 'checkpoints/crawl_index.sqlite'), output_path(config, 'data/manifests/crawl_manifest.parquet'))
    extract = ManifestIndex(output_path(config, 'checkpoints/extraction_index.sqlite'), output_path(config, 'data/manifests/extraction_manifest.parquet'))
    documents = ManifestIndex(output_path(config, 'checkpoints/canonical_index.sqlite'), output_path(config, 'data/processed/documents/documents.parquet'), key='canonical_content_id')
    try:
        original_rows = database.execute('SELECT COUNT(*) FROM source_rows').fetchone()[0]
        originals = iter(database.execute('SELECT row_no,doc_key,url_id FROM source_rows ORDER BY row_no'))
        counted = 0
        for row in parquet_rows(output_path(config, 'data/mappings/doc_outcomes.parquet')):
            counted += 1
            original = next(originals, None)
            from src.inventory.build_inventory import id_key
            if not original or original != (row['source_row_no'], id_key(row['doc_id']), row['crawl_url_id']):
                errors.append(f'Source identity lost at row {row["source_row_no"]}')
            if row['canonical_content_id'] and not documents.get(row['canonical_content_id']):
                errors.append(f'Orphan canonical ID for source row {row["source_row_no"]}')
        if counted != original_rows:
            errors.append(f'Source/accounted row mismatch: {original_rows}/{counted}')
        mapping_count = 0
        for row in parquet_rows(output_path(config, 'data/mappings/content_doc_map.parquet')):
            mapping_count += 1
            if not documents.get(row['canonical_content_id']) or not extract.get(row['crawl_url_id']):
                errors.append(f'Orphan content mapping: {row["crawl_url_id"]}')
            if not database.execute('SELECT 1 FROM source_rows WHERE doc_key=? AND url_id=?', (id_key(row['doc_id']), row['crawl_url_id'])).fetchone():
                errors.append('Content mapping changed a BTC ID or URL.')
        sample = output_path(config, 'data/inventory/stage_b_sample.parquet')
        if not sample.exists():
            sample = output_path(config, 'data/inventory/stage_a_sample.parquet')
        frontier = sample if sample.exists() else output_path(config, 'data/inventory/unique_urls.parquet')
        for row in parquet_rows(frontier):
            if require_complete and crawl.get(row['crawl_url_id']) is None:
                errors.append(f'Frontier URL has no terminal record: {row["crawl_url_id"]}')
        checked = success = 0
        for row in crawl:
            if not database.execute('SELECT 1 FROM urls WHERE url_id=?', (row['crawl_url_id'],)).fetchone():
                errors.append(f'Orphan crawl URL: {row["crawl_url_id"]}')
            if row['status'] == 'SUCCESS_RAW':
                success += 1
                if not output_path(config, row['raw_path']).is_file():
                    errors.append(f'Success raw missing: {row["crawl_url_id"]}')
                elif checked < config['storage']['checksum_sample_size']:
                    try:
                        RawStore(config).read(row, config['crawler']['max_response_bytes'])
                    except (OSError, ValueError, OverflowError) as error:
                        errors.append(f'Raw integrity failure {row["crawl_url_id"]}: {error}')
                    checked += 1
        for row in extract:
            if row['extract_status'] == 'EXTRACT_SUCCESS':
                raw = crawl.get(row['crawl_url_id'])
                if not raw or raw['status'] != 'SUCCESS_RAW' or raw['raw_sha256'] != row['raw_sha256']:
                    errors.append(f'Extraction/raw reference mismatch: {row["crawl_url_id"]}')
        result = {'passed': not errors, 'source_rows': original_rows, 'outcome_rows': counted,
                  'content_mapping_rows': mapping_count, 'success_raw': success, 'raw_checksums_checked': checked, 'errors': errors}
        atomic_json(output_path(config, 'reports/integrity.json'), result)
        return result
    finally:
        database.close()
        crawl.close()
        extract.close()
        documents.close()
