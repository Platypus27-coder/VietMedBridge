from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

from src.storage.io import atomic_json, parquet_rows, write_parquet
from src.utils.config import output_path


def build_shards(config: dict, source: Path | None = None) -> list[Path]:
    source = source or output_path(config, 'data/inventory/unique_urls.parquet')
    parquet = pq.ParquetFile(source)
    count = max(1, math.ceil(parquet.metadata.num_rows / config['sharding']['urls_per_shard']))
    database = sqlite3.connect(output_path(config, 'data/crawl_shards/shards.sqlite'))
    try:
        database.execute('DROP TABLE IF EXISTS urls')
        database.execute('CREATE TABLE urls(id TEXT PRIMARY KEY, shard INTEGER, payload TEXT)')
        with database:
            for row in parquet_rows(source):
                database.execute('INSERT INTO urls VALUES (?,?,?)', (row['crawl_url_id'], int(row['crawl_url_id'], 16) % count, json.dumps(row)))
        paths = []
        for shard in range(count):
            path = output_path(config, f'data/crawl_shards/shard_{shard:05d}.parquet')
            write_parquet(path, (json.loads(r[0]) for r in database.execute('SELECT payload FROM urls WHERE shard=? ORDER BY id', (shard,))), parquet.schema_arrow)
            paths.append(path)
        atomic_json(output_path(config, 'checkpoints/shards.json'), {'shard_count': count, 'urls': parquet.metadata.num_rows, 'function': 'int(sha256,16)%N', 'execution': 'sequential'})
        return paths
    finally:
        database.close()


def sample_inventory(config: dict, size: int, seed: int, *, stage='stage_a') -> Path:
    """Deterministic round-robin across top/medium/tail and predicted MIME strata."""
    if size <= 0:
        raise ValueError('sample-size must be positive.')
    db = sqlite3.connect(output_path(config, 'data/inventory/inventory.sqlite'))
    db.row_factory = sqlite3.Row
    try:
        domains = [r[0] for r in db.execute('SELECT domain FROM urls GROUP BY domain ORDER BY COUNT(*) DESC,domain')]
        rank = {d: i for i, d in enumerate(domains)}
        buckets = {}
        for row in db.execute('SELECT url_id,fetch_url,domain FROM urls'):
            n = rank[row['domain']]
            group = 'top' if n < min(10, len(domains)) else 'medium' if n < max(11, len(domains)//2) else 'tail'
            mime = 'pdf' if row['fetch_url'].lower().split('?')[0].endswith('.pdf') else 'html'
            score = hashlib.sha256(f'{seed}:{row["url_id"]}'.encode()).hexdigest()
            bucket = buckets.setdefault((group, mime), [])
            bucket.append((score, dict(row)))
            if len(bucket) > size * 2:
                bucket.sort(key=lambda r: r[0])
                del bucket[size:]
        for bucket in buckets.values():
            bucket.sort(key=lambda r: r[0])
            del bucket[size:]
        selected = {}
        gold_path = output_path(config, 'data/inventory/gold_doc_map.parquet')
        budget = min(config['coverage']['stage_a_gold_url_budget'], size//5)
        for gold in parquet_rows(gold_path):
            row = db.execute('SELECT u.url_id,u.fetch_url,u.domain FROM urls u JOIN source_rows s ON s.url_id=u.url_id WHERE s.doc_key=? LIMIT 1', (json.dumps(gold['gold_doc_id'], ensure_ascii=False),)).fetchone()
            if row:
                selected[row['url_id']] = dict(row)
            if len(selected) >= budget:
                break
        queues = sorted(buckets)
        position = 0
        while len(selected) < size and any(buckets[k] for k in queues):
            for key in queues:
                if buckets[key] and len(selected) < size:
                    _, row = buckets[key].pop(0)
                    selected[row['url_id']] = row
            position += 1
        source = output_path(config, 'data/inventory/unique_urls.parquet')
        def rows():
            for row in parquet_rows(source):
                if row['crawl_url_id'] in selected:
                    yield row
        if stage not in ('stage_a', 'stage_b'):
            raise ValueError('Unsupported sampling stage.')
        path = output_path(config, f'data/inventory/{stage}_sample.parquet')
        write_parquet(path, rows(), pq.ParquetFile(source).schema_arrow)
        atomic_json(output_path(config, 'checkpoints/sample.json'), {'stage': stage, 'sample_size': len(selected), 'seed': seed, 'strategy': 'stratified_mime_domain_with_gold', 'crawl_url_ids': sorted(selected)})
        return path
    finally:
        db.close()
