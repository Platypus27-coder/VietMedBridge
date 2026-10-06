from __future__ import annotations

import argparse


def crawl_entry(config_path, shard, stop_after=None, test_origins=frozenset()):
    from scrapy.crawler import CrawlerProcess
    from src.utils.config import load_config, output_path
    from .settings import scrapy_settings
    from .spiders.corpus_spider import CorpusSpider
    config = load_config(config_path)
    settings = scrapy_settings(config)
    settings['SCHEDULER'] = 'src.crawler.scheduler.CooldownScheduler'
    settings['JOBDIR'] = str(output_path(config, f'crawl_jobs/{shard.stem}'))
    settings['LOG_FILE'] = str(output_path(config, f'logs/{shard.stem}.log'))
    if stop_after:
        settings['CLOSESPIDER_ITEMCOUNT'] = stop_after
    process = CrawlerProcess(settings)
    crawler = process.create_crawler(CorpusSpider)
    failures = []
    result = process.crawl(crawler, shard=str(shard), config=str(config_path), _test_origins=test_origins)
    result.addErrback(lambda failure: failures.append(failure))
    process.start()
    from datetime import datetime, timedelta
    from src.storage.io import atomic_json
    def serialize_stat(value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, timedelta):
            return value.total_seconds()
        return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
    atomic_json(output_path(config, f'reports/crawl_stats/{shard.stem}.json'), {
        key: serialize_stat(value) for key, value in crawler.stats.get_stats().items()})
    if failures:
        failures[0].raiseException()


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--shard', type=Path, required=True)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    crawl_entry(args.config, args.shard, args.stop_after)
