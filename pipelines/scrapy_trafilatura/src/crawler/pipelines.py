from __future__ import annotations

from datetime import datetime, timezone
from collections import Counter
import json

from src.storage.io import atomic_json
from src.storage.manifests import EventWriter
from src.utils.config import output_path
from .raw_store import RawStore


class RawCachePipeline:
    def process_item(self, item, spider):
        if item['status'] == 'SUCCESS_RAW':
            try:
                item.update(RawStore(spider.config).write(item.pop('body'), item['effective_content_type']))
            except (OSError, ValueError, MemoryError) as error:
                item['status'] = 'RAW_WRITE_FAILED'
                item['error_type'] = type(error).__name__
                item['error_message'] = str(error)
                spider.logger.exception('Raw write failed for %s', item['crawl_url_id'])
        item.pop('body', None)
        return item


class CrawlManifestPipeline:
    def open_spider(self, spider):
        self.writer = EventWriter(output_path(spider.config, 'data/manifests/crawl'), spider.config['storage']['jsonl_records_per_part'])
        self.completed = 0
        self.counts = Counter()
        path = output_path(spider.config, 'checkpoints/run_state.json')
        self.scheduled = json.loads(path.read_text(encoding='utf-8')).get('scheduled_this_run') if path.exists() else None

    def process_item(self, item, spider):
        self.writer.append(dict(item))
        self.completed += 1
        self.counts[item['status']] += 1
        atomic_json(output_path(spider.config, 'checkpoints/run_state.json'), {
            'stage': 'crawl', 'shard': spider.shard_id, 'run_id': spider.run_id,
            'last_updated_at': datetime.now(timezone.utc).isoformat(), 'completed_this_run': self.completed,
            'counts_this_run': dict(self.counts), 'scheduled_this_run': self.scheduled, 'status': 'RUNNING'})
        return item

    def close_spider(self, spider):
        self.writer.close()
