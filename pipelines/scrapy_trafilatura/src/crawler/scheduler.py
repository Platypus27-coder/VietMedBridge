from __future__ import annotations

import pickle
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit

from scrapy.core.scheduler import Scheduler
from scrapy.utils.request import request_from_dict


class CooldownScheduler(Scheduler):
    """Keep cooled requests outside downloader slots, persisted beside JOBDIR."""

    def open(self, spider):
        result = super().open(spider)
        self.spider = spider
        path = Path(self.dqdir) / 'cooldown.sqlite'
        self.held = sqlite3.connect(path)
        self.held.execute('CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY AUTOINCREMENT, host TEXT, payload BLOB)')
        return result

    def next_request(self):
        for row_id, host, payload in self.held.execute('SELECT id,host,payload FROM requests ORDER BY id'):
            if self.spider.crawl_state.cooldown_until(host) <= time.time():
                request = request_from_dict(pickle.loads(payload), spider=self.spider)
                with self.held:
                    self.held.execute('DELETE FROM requests WHERE id=?', (row_id,))
                return request
        for _ in range(1000):
            request = super().next_request()
            if request is None:
                return None
            host = urlsplit(request.url).hostname
            if self.spider.crawl_state.cooldown_until(host) <= time.time():
                return request
            with self.held:
                self.held.execute('INSERT INTO requests(host,payload) VALUES (?,?)', (host, pickle.dumps(request.to_dict(spider=self.spider))))
        return None

    def has_pending_requests(self):
        return super().has_pending_requests() or bool(self.held.execute('SELECT 1 FROM requests LIMIT 1').fetchone())

    def close(self, reason):
        self.held.close()
        return super().close(reason)
