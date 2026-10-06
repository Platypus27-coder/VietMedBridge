from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from src.storage.io import atomic_json, sha256_file, write_parquet
from src.utils.config import input_path, output_path, prepare_directories
from .url_normalizer import crawl_url_id, normalize_url


def snapshot_dataset(config: dict) -> dict:
    sources = []
    for key in ('links_path', 'query_path'):
        path = input_path(config, key)
        if not path.is_file():
            if key == 'query_path' and not config['dataset']['query_required']:
                continue
            raise FileNotFoundError(f'{path} missing. Place the dataset or run scripts/download_dataset.py.')
        checksum = sha256_file(path)
        destination = output_path(config, f'data/source/snapshots/{checksum}/{path.name}')
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            shutil.copyfile(path, temporary)
            if sha256_file(temporary) != checksum:
                raise ValueError(f'Source changed during snapshot: {path}')
            temporary.replace(destination)
        if sha256_file(destination) != checksum:
            raise ValueError(f'Snapshot checksum mismatch: {destination}')
        parquet = pq.ParquetFile(destination)
        sources.append({'kind': key, 'original_path': str(path), 'path': str(destination),
                        'sha256': checksum, 'size_bytes': destination.stat().st_size,
                        'rows': parquet.metadata.num_rows, 'schema': str(parquet.schema_arrow)})
    identity = hashlib.sha256(json.dumps([(s['kind'], s['sha256']) for s in sources]).encode()).hexdigest()
    manifest = {'snapshot_id': identity, 'created_at': datetime.now(timezone.utc).isoformat(),
                'adapter_version': 1, 'sources': sources}
    existing_path = output_path(config, 'data/source/dataset_manifest.json')
    if existing_path.exists() and json.loads(existing_path.read_text())['snapshot_id'] != identity:
        raise ValueError('Output directory already belongs to a different dataset snapshot. Select a new output-dir.')
    atomic_json(existing_path, manifest)
    return manifest


def id_key(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def build_inventory(config: dict) -> dict:
    prepare_directories(config)
    manifest = snapshot_dataset(config)
    source = Path(next(s['path'] for s in manifest['sources'] if s['kind'] == 'links_path'))
    parquet = pq.ParquetFile(source)
    id_column = config['dataset']['id_column']
    if id_column not in parquet.schema_arrow.names and 'doc_id' in parquet.schema_arrow.names:
        id_column = 'doc_id'
    url_column = config['dataset']['url_column']
    if id_column not in parquet.schema_arrow.names or url_column not in parquet.schema_arrow.names:
        raise ValueError(f'Expected id/url columns; actual schema: {parquet.schema_arrow}')
    id_type = parquet.schema_arrow.field(id_column).type
    if not (pa.types.is_integer(id_type) or pa.types.is_string(id_type) or pa.types.is_large_string(id_type)):
        raise ValueError(f'doc_id must be an immutable integer or string, found {id_type}')
    db = sqlite3.connect(output_path(config, 'data/inventory/inventory.sqlite'))
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''CREATE TABLE IF NOT EXISTS source_rows(row_no INTEGER PRIMARY KEY, doc_key TEXT, original_url TEXT, url_id TEXT, status TEXT);
            CREATE INDEX IF NOT EXISTS row_url_id ON source_rows(url_id);
            CREATE INDEX IF NOT EXISTS row_doc_key ON source_rows(doc_key);
            CREATE TABLE IF NOT EXISTS urls(url_id TEXT PRIMARY KEY, fetch_url TEXT, domain TEXT);
            CREATE INDEX IF NOT EXISTS urls_domain ON urls(domain);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);''')
        previous = db.execute("SELECT value FROM meta WHERE key='snapshot_id'").fetchone()
        if previous and previous[0] != manifest['snapshot_id']:
            raise ValueError('Inventory checkpoint belongs to another snapshot.')
        with db:
            db.execute("INSERT OR REPLACE INTO meta VALUES ('snapshot_id',?)", (manifest['snapshot_id'],))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('id_schema',?)", (base64.b64encode(pa.schema([('doc_id', id_type)]).serialize()).decode(),))
        committed = db.execute('SELECT COALESCE(MAX(row_no),-1) FROM source_rows').fetchone()[0]
        row_no = -1
        progress_at = time.monotonic()
        for batch in parquet.iter_batches(batch_size=config['inventory']['batch_rows'], columns=[id_column, url_column]):
            records = []
            urls = []
            for row in batch.to_pylist():
                row_no += 1
                if row_no <= committed:
                    continue
                if row[id_column] is None:
                    raise ValueError(f'Null BTC doc_id at row {row_no}')
                normalized, status = normalize_url(row[url_column], config['inventory']['remove_fragments'])
                url_id = crawl_url_id(normalized) if normalized else None
                records.append((row_no, id_key(row[id_column]), row[url_column], url_id, status))
                if normalized:
                    urls.append((url_id, normalized, urlsplit(normalized).hostname))
            with db:
                db.executemany('INSERT INTO source_rows VALUES (?,?,?,?,?)', records)
                db.executemany('INSERT OR IGNORE INTO urls VALUES (?,?,?)', urls)
            if records and time.monotonic() - progress_at >= 30:
                print(f'Inventory committed {row_no+1:,}/{parquet.metadata.num_rows:,} source rows', flush=True)
                progress_at = time.monotonic()
        conflicting = db.execute('SELECT doc_key FROM source_rows GROUP BY doc_key HAVING COUNT(DISTINCT COALESCE(url_id,status))>1 LIMIT 1').fetchone()
        if conflicting:
            raise ValueError(f'BTC doc_id maps to conflicting URLs/statuses: {conflicting[0]}')
        unique_schema = pa.schema([('crawl_url_id', pa.string()), ('original_url_example', pa.string()),
                                  ('fetch_url', pa.string()), ('domain', pa.string()), ('doc_count', pa.int64())])
        write_parquet(output_path(config, 'data/inventory/unique_urls.parquet'),
                      ({'crawl_url_id': u, 'original_url_example': original, 'fetch_url': url, 'domain': domain, 'doc_count': count}
                       for u, url, domain, original, count in db.execute('SELECT u.url_id,u.fetch_url,u.domain,MIN(s.original_url),COUNT(*) FROM urls u JOIN source_rows s USING(url_id) GROUP BY u.url_id ORDER BY u.url_id')), unique_schema)
        mapping_schema = pa.schema([('crawl_url_id', pa.string()), ('doc_id', id_type), ('original_url', pa.string()), ('source_row_no', pa.int64())])
        write_parquet(output_path(config, 'data/inventory/url_doc_map.parquet'),
                      ({'crawl_url_id': u, 'doc_id': json.loads(doc), 'original_url': original, 'source_row_no': n}
                       for n, doc, original, u in db.execute('SELECT row_no,doc_key,original_url,url_id FROM source_rows WHERE url_id IS NOT NULL ORDER BY row_no')), mapping_schema)
        invalid_schema = pa.schema([('doc_id', id_type), ('original_url', pa.string()), ('status', pa.string()), ('source_row_no', pa.int64())])
        write_parquet(output_path(config, 'data/inventory/invalid_urls.parquet'),
                      ({'doc_id': json.loads(doc), 'original_url': original, 'status': status, 'source_row_no': n}
                       for n, doc, original, status in db.execute('SELECT row_no,doc_key,original_url,status FROM source_rows WHERE url_id IS NULL ORDER BY row_no')), invalid_schema)
        total = db.execute('SELECT COUNT(*) FROM source_rows').fetchone()[0]
        unique = db.execute('SELECT COUNT(*) FROM urls').fetchone()[0]
        domain_schema = pa.schema([('domain', pa.string()), ('unique_url_count', pa.int64()), ('doc_id_count', pa.int64()), ('percentage', pa.float64())])
        write_parquet(output_path(config, 'data/inventory/domain_stats.parquet'),
                      ({'domain': domain, 'unique_url_count': n, 'doc_id_count': docs, 'percentage': n / unique if unique else 0}
                       for domain, n, docs in db.execute('SELECT domain,COUNT(*),(SELECT COUNT(*) FROM source_rows s JOIN urls v ON v.url_id=s.url_id WHERE v.domain=u.domain) FROM urls u GROUP BY domain ORDER BY COUNT(*) DESC')), domain_schema)
        gold = build_gold_map(config, manifest, id_type)
        result = {'snapshot_id': manifest['snapshot_id'], 'input_rows': total, 'unique_urls': unique,
                  'invalid_rows': db.execute('SELECT COUNT(*) FROM source_rows WHERE url_id IS NULL').fetchone()[0], 'gold_relations': gold}
        atomic_json(output_path(config, 'checkpoints/inventory_state.json'), result)
        return result
    finally:
        db.close()


def build_gold_map(config: dict, manifest: dict, id_type: pa.DataType) -> int:
    query = next((s for s in manifest['sources'] if s['kind'] == 'query_path'), None)
    schema = pa.schema([('query_id', pa.string()), ('gold_doc_id', id_type)])
    if query is None:
        return write_parquet(output_path(config, 'data/inventory/gold_doc_map.parquet'), [], schema)
    parquet = pq.ParquetFile(query['path'])
    names = parquet.schema_arrow.names
    query_column = config['dataset']['query_id_column']
    gold_column = config['dataset']['gold_doc_ids_column']
    if gold_column is None:
        atomic_json(output_path(config, 'checkpoints/gold_availability.json'), {
            'available': False, 'reason': 'No gold_doc_ids column configured; native ViBioMIR query.parquet contains id/query only.',
            'query_rows': parquet.metadata.num_rows, 'actual_columns': names})
        return write_parquet(output_path(config, 'data/inventory/gold_doc_map.parquet'), [], schema)
    if not query_column:
        raise ValueError(f'Configure dataset.query_id_column and gold_doc_ids_column from the actual schema, not a guess: {parquet.schema_arrow}')
    if query_column not in names or gold_column not in names:
        raise ValueError(f'Query adapter columns absent: {parquet.schema_arrow}')
    atomic_json(output_path(config, 'checkpoints/gold_availability.json'), {'available': True, 'query_rows': parquet.metadata.num_rows, 'actual_columns': names})
    def rows():
        for batch in parquet.iter_batches(columns=[query_column, gold_column], batch_size=config['inventory']['batch_rows']):
            for row in batch.to_pylist():
                if row[query_column] is None or row[gold_column] is None:
                    raise ValueError('Query/gold IDs must not be null.')
                values = row[gold_column] if isinstance(row[gold_column], list) else [row[gold_column]]
                for value in values:
                    if not isinstance(value, str if pa.types.is_string(id_type) or pa.types.is_large_string(id_type) else int):
                        raise ValueError(f'Gold ID type differs from BTC corpus ID type: {value!r}')
                    yield {'query_id': id_key(row[query_column]), 'gold_doc_id': value}
    return write_parquet(output_path(config, 'data/inventory/gold_doc_map.parquet'), rows(), schema)
