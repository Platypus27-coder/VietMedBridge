import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.inventory.build_inventory import build_inventory
from src.inventory.build_shards import build_shards, sample_inventory
from src.inventory.url_normalizer import crawl_url_id, normalize_url
from src.storage.io import parquet_rows
from src.utils.config import output_path


@pytest.mark.parametrize('original,expected,status', [
    (' HTTPS://EXAMPLE.COM/a?x=1&lang=vi#section ', 'https://example.com/a?x=1&lang=vi', 'VALID_HTTP_URL'),
    ('https://example.com', 'https://example.com/', 'VALID_HTTP_URL'),
    ('ftp://example.com/a', None, 'UNSUPPORTED_SCHEME'), ('', None, 'EMPTY_URL'),
    ('https://example.com:bad/a', None, 'INVALID_URL'), ('https://user:password@example.com', None, 'INVALID_URL')])
def test_normalize(original, expected, status):
    assert normalize_url(original) == (expected, status)


def test_inventory_preserves_native_ids_and_resume(config, tmp_path):
    links = tmp_path/'links.parquet'
    query = tmp_path/'query.parquet'
    pq.write_table(pa.table({'id': pa.array([101, 102, 103, 104], type=pa.int64()),
                            'url': ['https://Example.com/a#x', 'https://example.com/a', '', 'https://other.org/report.pdf']}), links)
    pq.write_table(pa.table({'query_id': ['q1'], 'gold_doc_ids': [[101, 103, 999]]}), query)
    config['dataset'].update(links_path=str(links), query_path=str(query))
    result = build_inventory(config)
    assert result['input_rows'] == 4 and result['unique_urls'] == 2 and result['invalid_rows'] == 1
    mapping = list(parquet_rows(output_path(config, 'data/inventory/url_doc_map.parquet')))
    assert mapping[0]['crawl_url_id'] == mapping[1]['crawl_url_id']
    assert [r['doc_id'] for r in mapping] == [101, 102, 104]
    assert build_inventory(config) == result
    sample = sample_inventory(config, 2, 42)
    before = sample.read_bytes()
    assert sample_inventory(config, 2, 42).read_bytes() == before
    config['sharding']['urls_per_shard'] = 1
    shards = build_shards(config, sample)
    ids = [r['crawl_url_id'] for shard in shards for r in parquet_rows(shard)]
    assert len(ids) == len(set(ids)) == 2
    assert pq.ParquetFile(output_path(config, 'data/inventory/url_doc_map.parquet')).schema_arrow.field('doc_id').type == pa.int64()


def test_stable_id():
    assert crawl_url_id('https://example.com/a') == crawl_url_id('https://example.com/a')
    assert len(crawl_url_id('https://example.com/a')) == 64


def test_native_query_ids_are_not_gold_ids(config, tmp_path):
    links, query = tmp_path/'links.parquet', tmp_path/'query.parquet'
    pq.write_table(pa.table({'id': [1, 2], 'url': ['https://example.com/a', 'https://example.com/b']}), links)
    pq.write_table(pa.table({'id': [1], 'query': ['treatment']}), query)
    config['dataset'].update(links_path=str(links), query_path=str(query), query_id_column='id', gold_doc_ids_column=None)
    assert build_inventory(config)['gold_relations'] == 0
    assert not json.loads(output_path(config, 'checkpoints/gold_availability.json').read_text())['available']
    assert list(parquet_rows(output_path(config, 'data/inventory/gold_doc_map.parquet'))) == []
