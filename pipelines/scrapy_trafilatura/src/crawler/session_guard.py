"""Stop a bounded notebook session while durable manifests remain resumable."""
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

from src.storage.io import atomic_json

from scrapy import signals
from twisted.internet.task import LoopingCall


class SessionGuard:
    def __init__(self, crawler):
        self.crawler = crawler
        self.loop = None
        self.stopping = False
        self.raw_bytes = 0
        self.last_heartbeat = float('-inf')

    @classmethod
    def from_crawler(cls, crawler):
        extension = cls(crawler)
        crawler.signals.connect(extension.opened, signal=signals.spider_opened)
        crawler.signals.connect(extension.scraped, signal=signals.item_scraped)
        crawler.signals.connect(extension.closed, signal=signals.spider_closed)
        return extension

    def opened(self, spider):
        self.spider = spider
        self.root = Path(spider.config['_output_root'])
        self.limits = spider.config.get('kaggle_batch', {})
        self.raw_bytes = sum(p.stat().st_size for p in (self.root/'data/raw').rglob('*') if p.is_file())
        self.loop = LoopingCall(self.check)
        self.loop.start(10, now=True)

    def scraped(self, item, response, spider):
        if item.get('status') == 'SUCCESS_RAW':
            # Conservative: duplicate content is counted again for this session.
            self.raw_bytes += item.get('compressed_size_bytes', 0)
        self.check()

    def check(self):
        if time.monotonic()-self.last_heartbeat >= 10:
            stats = self.crawler.stats
            scheduler = getattr(self.crawler.engine, '_slot', None)
            scheduler = getattr(scheduler, 'scheduler', None)
            held_requests, until = 0, 0
            held = getattr(scheduler, 'held', None)
            if held is not None:
                for host, count in held.execute('SELECT host,COUNT(*) FROM requests GROUP BY host'):
                    held_requests += count
                    cooldown = self.spider.crawl_state.cooldown_until(host)
                    if cooldown > time.time():
                        until = min(until or cooldown, cooldown)
            atomic_json(self.root/'checkpoints/crawl_heartbeat.json', {
                'last_updated_at': datetime.now(timezone.utc).isoformat(),
                'requests': stats.get_value('downloader/request_count', 0),
                'responses': stats.get_value('downloader/response_count', 0),
                'retries': stats.get_value('retry/count', 0), 'raw_bytes': self.raw_bytes,
                'held_requests': held_requests, 'next_cooldown_until': until})
            self.last_heartbeat = time.monotonic()
        if self.stopping:
            return
        reason = None
        if self.raw_bytes >= self.limits.get('max_raw_bytes', float('inf')):
            reason = 'raw_storage_budget'
        elif shutil.disk_usage(self.root).free < self.limits.get('min_free_disk_bytes', 0):
            reason = 'free_disk_budget'
        if reason:
            self.stopping = True
            self.crawler.stats.set_value('session_guard/raw_bytes', self.raw_bytes)
            self.crawler.engine.close_spider(self.spider, reason=reason)

    def closed(self, spider, reason):
        if self.loop is not None and self.loop.running:
            self.loop.stop()
