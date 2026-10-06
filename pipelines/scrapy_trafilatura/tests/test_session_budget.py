import copy
import time
from pathlib import Path

import pytest

from scripts.prepare_kaggle_campaign import prepare_campaign
from src.crawler.settings import scrapy_settings
from src.kaggle_batch import run_kaggle_batch
from src.utils.session_budget import stage_time_limit
from tests.integration.test_pipeline import make_source


def deadline_config():
    return {'kaggle_batch': {'session_deadline_unix': 1100, 'extract': True,
        'extraction_reserve_seconds': 100, 'export_reserve_seconds': 50}}


def test_session_deadline_deducts_elapsed_setup_and_preserves_reserves():
    config = deadline_config()
    assert stage_time_limit(config, 'crawl', now=100) == 850
    assert stage_time_limit(config, 'crawl', now=300) == 650
    assert stage_time_limit(config, 'extraction', now=950) == 100
    assert stage_time_limit(config, 'crawl', configured_seconds=60, now=300) == 60
    config['kaggle_batch']['extract'] = False
    assert stage_time_limit(config, 'crawl', now=300) == 750


def test_expired_budget_never_becomes_scrapy_unlimited(config, monkeypatch):
    config['kaggle_batch'] = deadline_config()['kaggle_batch']
    config['crawler']['session_time_limit_seconds'] = 0
    monkeypatch.setattr('src.utils.session_budget.time.time', lambda: 2000)
    assert stage_time_limit(config, 'crawl') == 0
    assert stage_time_limit(config, 'extraction') == 0
    assert scrapy_settings(config)['CLOSESPIDER_TIMEOUT'] == 0.001
    config.pop('kaggle_batch')
    assert scrapy_settings(config)['CLOSESPIDER_TIMEOUT'] == 0
    assert stage_time_limit(config, 'extraction') is None


@pytest.mark.parametrize('field,value', [
    ('session_deadline_unix', float('nan')), ('export_reserve_seconds', -1)])
def test_invalid_deadlines_fail_closed(field, value):
    config = deadline_config()
    config['kaggle_batch'][field] = value
    with pytest.raises(ValueError):
        stage_time_limit(config, 'crawl')


def test_expired_session_exports_pending_without_crawling_or_extracting(config, tmp_path, monkeypatch):
    make_source(config, tmp_path, ['https://example.com/doc-1', 'https://example.com/doc-2'])
    campaign_root = tmp_path/'campaign'
    prepare_campaign(Path(config['_output_root']), campaign_root)
    runtime = copy.deepcopy(config)
    runtime['_output_root'] = str(tmp_path/'session')
    runtime['crawler']['session_time_limit_seconds'] = 0
    runtime['kaggle_batch'] = {'extract': True, 'session_deadline_unix': time.time() - 1,
        'extraction_reserve_seconds': 60, 'export_reserve_seconds': 30}
    def unexpected(*args, **kwargs):
        pytest.fail('Expired session must preserve pending work without starting new workers.')
    monkeypatch.setattr('src.crawler.run.subprocess.run', unexpected)
    monkeypatch.setattr('src.kaggle_batch.extract_raw', unexpected)
    report = run_kaggle_batch(runtime, campaign_root)
    assert report['status'] == 'PARTIAL' and report['crawl_remaining'] == 2
    assert report['successful_raw_urls'] == 0 and report['crawler_finish_reason'] == 'session_deadline'
    assert report['extraction_run']['stopped_by_time_budget']
    assert report['integrity_passed'] and report['bundle']['verified']
