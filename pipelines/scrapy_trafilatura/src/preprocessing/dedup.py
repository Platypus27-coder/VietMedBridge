from __future__ import annotations

import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from src.storage.index import ManifestIndex
from src.storage.io import atomic_json, parquet_rows, write_parquet
from src.storage.schemas import DOCUMENT_SCHEMA
from src.utils.config import output_path


def build_documents(config: dict) -> dict:
    crawl = ManifestIndex(output_path(config, 'checkpoints/crawl_index.sqlite'), output_path(config, 'data/manifests/crawl_manifest.parquet'))
    inventory = sqlite3.connect(output_path(config, 'data/inventory/inventory.sqlite'))
    db = sqlite3.connect(output_path(config, 'data/processed/documents/build.sqlite'))
    try:
        db.executescript('DROP TABLE IF EXISTS records; DROP TABLE IF EXISTS canonical; CREATE TABLE records(url_id TEXT PRIMARY KEY, hash TEXT, domain TEXT, rank INTEGER, replacement REAL, title_present INTEGER, min_doc, payload TEXT); CREATE INDEX by_hash ON records(hash); CREATE TABLE canonical(hash TEXT PRIMARY KEY, url_id TEXT, payload TEXT);')
        with db:
            for row in parquet_rows(output_path(config, 'data/manifests/extraction_manifest.parquet')):
                raw = crawl.get(row['crawl_url_id'])
                if not raw or raw['status'] != 'SUCCESS_RAW' or raw['raw_sha256'] != row['raw_sha256'] or not row['content_hash']:
                    continue
                minimum = inventory.execute('SELECT MIN(json_extract(doc_key,\'$\')) FROM source_rows WHERE url_id=?', (row['crawl_url_id'],)).fetchone()[0]
                payload = dict(row, canonical_content_id=row['content_hash'], representative_crawl_url_id=row['crawl_url_id'],
                               url=raw['fetch_url'], final_url=raw['final_url'], domain=raw['domain'], content_type=raw['effective_content_type'])
                rank = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2, 'QUARANTINE': 3}[row['quality_tier']]
                db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?)', (row['crawl_url_id'], row['content_hash'], raw['domain'], rank,
                           row['replacement_char_ratio'], int(bool(row['title'])), minimum, json.dumps(payload, ensure_ascii=False)))
        suspicious = 0
        quarantined = 0
        groups = db.execute('SELECT hash,COUNT(*),COUNT(DISTINCT domain) FROM records GROUP BY hash')
        for digest, urls, domains in groups:
            is_widespread = urls >= config['dedup']['widespread_duplicate_min_urls'] and domains >= config['dedup']['widespread_duplicate_min_domains']
            suspicious += int(is_widespread)
            representative = db.execute('SELECT url_id,payload FROM records WHERE hash=? ORDER BY rank,replacement,title_present DESC,min_doc,url_id LIMIT 1', (digest,)).fetchone()
            url_id, serialized = representative
            payload = json.loads(serialized)
            confirmed = is_widespread and db.execute('SELECT 1 FROM records WHERE hash=? AND rank=3 LIMIT 1', (digest,)).fetchone() is not None
            if confirmed:
                payload['quality_tier'], payload['quality_reason'] = 'QUARANTINE', 'widespread_error_boilerplate'
                quarantined += 1
            db.execute('INSERT INTO canonical VALUES (?,?,?)', (digest, url_id, json.dumps(payload, ensure_ascii=False)))
        db.commit()
        documents = write_parquet(output_path(config, 'data/processed/documents/documents.parquet'),
                                  (json.loads(r[0]) for r in db.execute('SELECT payload FROM canonical ORDER BY hash')), DOCUMENT_SCHEMA)
        id_type = pq.ParquetFile(output_path(config, 'data/inventory/url_doc_map.parquet')).schema_arrow.field('doc_id').type
        map_schema = pa.schema([('canonical_content_id', pa.string()), ('crawl_url_id', pa.string()), ('doc_id', id_type)])
        inventory.execute('ATTACH DATABASE ? AS built', (str(output_path(config, 'data/processed/documents/build.sqlite')),))
        def mappings():
            for key, url_id, digest in inventory.execute('SELECT s.doc_key,s.url_id,r.hash FROM source_rows s JOIN built.records r ON r.url_id=s.url_id ORDER BY s.row_no'):
                yield {'canonical_content_id': digest, 'crawl_url_id': url_id, 'doc_id': json.loads(key)}
        mapped = write_parquet(output_path(config, 'data/mappings/content_doc_map.parquet'), mappings(), map_schema)
        extracted = ManifestIndex(output_path(config, 'checkpoints/extraction_index.sqlite'), output_path(config, 'data/manifests/extraction_manifest.parquet'))
        inventory.execute('ATTACH DATABASE ? AS crawled', (str(output_path(config, 'checkpoints/crawl_index.sqlite')),))
        inventory.execute('ATTACH DATABASE ? AS extracted', (str(output_path(config, 'checkpoints/extraction_index.sqlite')),))
        outcome_schema = pa.schema([('doc_id', id_type), ('source_row_no', pa.int64()), ('crawl_url_id', pa.string()),
                                   ('canonical_content_id', pa.string()), ('crawl_status', pa.string()), ('extract_status', pa.string()),
                                   ('quality_tier', pa.string()), ('quality_reason', pa.string()), ('handoff_eligible', pa.bool_()),
                                   ('handoff_mode', pa.string())])
        def outcomes():
            rows = inventory.execute('''SELECT s.row_no,s.doc_key,s.url_id,s.status,c.payload,e.payload,r.hash,k.payload
                FROM source_rows s LEFT JOIN crawled.records c ON c.id=s.url_id
                LEFT JOIN extracted.records e ON e.id=s.url_id LEFT JOIN built.records r ON r.url_id=s.url_id
                LEFT JOIN built.canonical k ON k.hash=r.hash ORDER BY s.row_no''')
            for n, key, url_id, status, raw_json, text_json, digest, canonical_json in rows:
                raw = json.loads(raw_json) if raw_json else None
                text = json.loads(text_json) if text_json else None
                canonical = json.loads(canonical_json) if canonical_json else None
                tier = canonical['quality_tier'] if canonical else (text or {}).get('quality_tier')
                eligible = tier in ('HIGH', 'MEDIUM') or (tier == 'LOW' and config['quality']['handoff_mode'] == 'benchmark')
                yield {'doc_id': json.loads(key), 'source_row_no': n, 'crawl_url_id': url_id,
                       'canonical_content_id': digest,
                       'crawl_status': (raw or {}).get('status', 'PENDING' if url_id else status),
                       'extract_status': (text or {}).get('extract_status'), 'quality_tier': tier,
                       'quality_reason': (canonical or text or {}).get('quality_reason'),
                       'handoff_eligible': bool(canonical and eligible and (text or {}).get('extract_status') == 'EXTRACT_SUCCESS'),
                       'handoff_mode': config['quality']['handoff_mode']}
        try:
            outcome_count = write_parquet(output_path(config, 'data/mappings/doc_outcomes.parquet'), outcomes(), outcome_schema)
        finally:
            extracted.close()
        result = {'canonical_documents': documents, 'mapped_doc_rows': mapped, 'accounted_source_rows': outcome_count,
                  'widespread_groups': suspicious, 'confirmed_quarantine_groups': quarantined}
        atomic_json(output_path(config, 'checkpoints/documents_state.json'), result)
        return result
    finally:
        db.close()
        inventory.close()
        crawl.close()
