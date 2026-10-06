from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import scrapy
from scrapy.exceptions import IgnoreRequest
from twisted.internet.error import DNSLookupError, TCPTimedOutError, TimeoutError

from src.storage.io import parquet_rows
from src.storage.state import StateStore
from src.utils.config import load_config, output_path
from src.crawler.router import sniff_content


class CorpusSpider(scrapy.Spider):
    name = 'corpus'

    def __init__(self, shard, config=None, run_id=None, _test_origins=frozenset(), **kwargs):
        super().__init__(**kwargs)
        self.config = load_config(config)
        self.shard = Path(shard)
        self.shard_id = self.shard.stem
        self.run_id = run_id or uuid.uuid4().hex
        self._test_origins = _test_origins
        self.crawl_state = StateStore(output_path(self.config, 'checkpoints/crawl_state.sqlite'))
        snapshot = output_path(self.config, 'data/source/dataset_manifest.json')
        self.snapshot_id = json.loads(snapshot.read_text())['snapshot_id'] if snapshot.exists() else 'test_fixture'

    async def start(self):
        for row in parquet_rows(self.shard):
            attempt = row.get('attempt_no') or self.crawl_state.allocate_attempt(row['crawl_url_id'])
            yield scrapy.Request(row['fetch_url'], callback=self.parse, errback=self.errback, dont_filter=False,
                                 meta={'crawl_url_id': row['crawl_url_id'], 'fetch_url': row['fetch_url'],
                                       'domain': row['domain'], 'shard_id': self.shard_id, 'attempt_no': attempt,
                                       'started_at': time.monotonic()})

    def record(self, request):
        meta = request.meta
        return {'crawl_url_id': meta['crawl_url_id'], 'event_id': f'{meta["crawl_url_id"]}:{meta["attempt_no"]}:1',
                'event_seq': 1, 'attempt_no': meta['attempt_no'], 'run_id': self.run_id,
                'snapshot_id': self.snapshot_id, 'config_sha256': self.config['_config_sha256'],
                'shard_id': self.shard_id, 'fetch_url': meta['fetch_url'], 'final_url': request.url,
                'domain': meta['domain'], 'http_status': None, 'attempt_count': meta.get('retry_times', 0) + 1,
                'retry_exhausted': False,
                'redirect_count': len(meta.get('redirect_urls', [])), 'elapsed_ms': (time.monotonic()-meta['started_at'])*1000,
                'error_type': None, 'error_message': None, 'crawl_timestamp': datetime.now(timezone.utc).isoformat(),
                'retry_not_before': meta.get('retry_not_before'), 'source_provenance': 'live_http'}

    def parse(self, response):
        item = self.record(response.request)
        item['http_status'] = response.status
        header = response.headers.get(b'Content-Type', b'').decode('latin-1')
        item.update(sniff_content(response.body, header))
        retry_after = response.headers.get(b'Retry-After')
        item['retry_after'] = retry_after.decode('ascii', 'replace') if retry_after else None
        if 200 <= response.status < 300:
            item['status'] = 'SUCCESS_RAW' if item['effective_content_type'] else 'UNSUPPORTED_CONTENT_TYPE'
            if not response.body:
                item['status'] = 'EMPTY_RESPONSE'
            elif len(response.body) > self.config['crawler']['max_response_bytes']:
                item['status'] = 'RESPONSE_TOO_LARGE'
            if item['status'] == 'SUCCESS_RAW':
                item['body'] = response.body
        else:
            item['status'] = {404: 'HTTP_404', 403: 'HTTP_403', 429: 'HTTP_429'}.get(response.status, 'HTTP_5XX' if response.status >= 500 else 'HTTP_OTHER')
            item['retry_exhausted'] = bool(response.status in self.config['crawler']['retry_http_codes']
                                           and response.meta.get('retry_times', 0) >= self.config['crawler']['retry_times'])
        return item

    def errback(self, failure):
        request = failure.request
        item = self.record(request)
        item['error_type'] = type(failure.value).__name__
        item['error_message'] = str(failure.value)
        if request.meta.get('_redirect_loop'):
            status = 'REDIRECT_LOOP'
            item['http_status'] = request.meta.get('_redirect_loop_status')
        elif request.meta.get('_size_exceeded'):
            status = 'RESPONSE_TOO_LARGE'
        elif failure.check(IgnoreRequest):
            status = 'ROBOTS_DENIED' if str(failure.value) == 'Forbidden by robots.txt' else 'REQUEST_IGNORED'
            item['attempt_count'] = 0
        elif failure.check(DNSLookupError):
            status = 'DNS_ERROR'
        elif failure.check(TimeoutError, TCPTimedOutError) or 'Timeout' in type(failure.value).__name__:
            status = 'TIMEOUT'
        elif any(word in type(failure.value).__name__.lower() for word in ('ssl', 'tls', 'certificate')):
            status = 'SSL_ERROR'
        else:
            status = 'CONNECTION_ERROR'
        item['status'] = status
        item['retry_exhausted'] = bool(status in ('TIMEOUT', 'DNS_ERROR', 'CONNECTION_ERROR')
                                       and request.meta.get('retry_times', 0) >= self.config['crawler']['retry_times'])
        return item

    def closed(self, reason):
        self.crawl_state.close()
