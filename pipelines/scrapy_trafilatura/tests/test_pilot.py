import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.crawler.run import run_crawl
from src.inventory.build_inventory import build_inventory
from src.inventory.build_shards import build_shards, sample_inventory
from src.pilot import require_pilot_review, reuse_inventory, save_deferred_urls
from src.reports.build_report import review_gates
from src.storage.io import atomic_json, parquet_rows
from src.utils.config import output_path


def approved_review():
    return {'content_review': {'accepted': True, 'decided_by': 'user'},
            'gold_coverage': {'decision': 'proceed_without_qrels', 'decided_by': 'user'},
            'deployment': {'target': 'windows_local'},
            'stage_b_pilot': {'authorized': True, 'max_urls': 10000}}


def test_review_accepts_missing_qrels_without_fabricating_coverage():
    domains = [{'domain': 'blocked.test', 'decision': 'defer_for_later', 'decided_by': 'user'}]
    gates = review_gates(approved_review(), domains, False, True)
    assert gates['stage_b_ready'] and gates['gold_coverage_decision_complete']
    assert not review_gates(approved_review(), domains, False, False)['stage_b_ready']
    domains[0]['decision'] = 'invalid'
    assert not review_gates(approved_review(), domains, False, True)['stage_b_ready']
    review = approved_review()
    del review['gold_coverage']
    assert not review_gates(review, [], False, True)['stage_b_ready']


def make_inventory(config, tmp_path):
    links, query = tmp_path/'links.parquet', tmp_path/'query.parquet'
    pq.write_table(pa.table({'id': [71, 72], 'url': ['https://blocked.test/a', 'https://blocked.test/b']}), links)
    pq.write_table(pa.table({'id': [1], 'query': ['treatment']}), query)
    config['dataset'].update(links_path=str(links), query_path=str(query), query_id_column='id', gold_doc_ids_column=None)
    return build_inventory(config)


def test_deferred_domain_has_terminal_accounting_and_no_network(config, tmp_path):
    make_inventory(config, tmp_path)
    config['coverage']['deferred_domains'] = ['blocked.test']
    sample = sample_inventory(config, 2, 42, stage='stage_b')
    assert sample.name == 'stage_b_sample.parquet'
    shard = build_shards(config, sample)[0]
    assert run_crawl(config, shard)['pending'] == 0
    records = list(parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet')))
    assert len(records) == 2
    assert all(r['status'] == 'DEFERRED_DOMAIN' and r['attempt_count'] == 0 and r['raw_path'] is None for r in records)
    assert run_crawl(config, shard)['pending'] == 0
    assert len(list(parquet_rows(output_path(config, 'data/manifests/crawl_manifest.parquet')))) == 2
    assert save_deferred_urls(config) == 2


def test_pilot_rejects_large_unapproved_or_same_root(config, tmp_path):
    root = tmp_path/'stage_a'
    for size in (0, 10001, 100000):
        with pytest.raises(ValueError):
            require_pilot_review(config, root, size)
    with pytest.raises(ValueError, match='separate'):
        require_pilot_review(config, Path(config['_output_root']), 10000)
    reports = root/'reports/stage_a'
    atomic_json(reports/'review_decision.json', approved_review())
    atomic_json(reports/'coverage_decisions.json', {'domains': []})
    atomic_json(reports/'report.json', {'snapshot_id': 'fixture', 'gold_coverage': {'available': False}})
    atomic_json(root/'reports/integrity.json', {'passed': True})
    config['environment']['deployment'].update(target='windows_local', stage_b_authorized=False)
    with pytest.raises(ValueError, match='authorization'):
        require_pilot_review(config, root, 10000)
    config['environment']['deployment']['stage_b_authorized'] = True
    assert require_pilot_review(config, root, 10000)['gates']['stage_b_ready']
    config['environment']['deployment']['target'] = 'linux_vps'
    with pytest.raises(ValueError, match='target'):
        require_pilot_review(config, root, 10000)


def test_inventory_reuse_is_independent_and_preserves_native_ids(config, tmp_path):
    import copy
    import sqlite3
    import yaml
    source_root = Path(config['_output_root'])
    result = make_inventory(config, tmp_path)
    output_path(config, 'checkpoints/effective_config.yaml').write_text(
        yaml.safe_dump({k: v for k, v in config.items() if not k.startswith('_')}), encoding='utf-8')
    target = copy.deepcopy(config)
    target['_output_root'] = str(tmp_path/'pilot')
    target['environment']['output_dir'] = target['_output_root']
    from src.utils.config import prepare_directories
    prepare_directories(target)
    reuse_inventory(target, source_root, result['snapshot_id'])
    rows = list(parquet_rows(output_path(target, 'data/inventory/url_doc_map.parquet')))
    assert [r['doc_id'] for r in rows] == [71, 72]
    with sqlite3.connect(output_path(target, 'data/inventory/inventory.sqlite')) as db:
        db.execute('DELETE FROM urls')
    with sqlite3.connect(source_root/'data/inventory/inventory.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM urls').fetchone()[0] == 2
    with pytest.raises(ValueError, match='snapshot'):
        reuse_inventory(target, source_root, 'changed')
