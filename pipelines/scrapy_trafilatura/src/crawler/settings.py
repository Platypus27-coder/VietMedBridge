BOT_NAME = 'vibiomir'
SPIDER_MODULES = ['src.crawler.spiders']
NEWSPIDER_MODULE = 'src.crawler.spiders'
ROBOTSTXT_OBEY = True
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'


def scrapy_settings(config: dict) -> dict:
    from src.utils.session_budget import stage_time_limit
    c = config['crawler']
    a = c['autothrottle']
    timeout = stage_time_limit(config, 'crawl', c.get('session_time_limit_seconds', 0))
    return {
        'BOT_NAME': BOT_NAME, 'SPIDER_MODULES': SPIDER_MODULES, 'ROBOTSTXT_OBEY': c['robots_obey'],
        'TWISTED_REACTOR': TWISTED_REACTOR,
        'CONCURRENT_REQUESTS': c['concurrent_requests'], 'CONCURRENT_REQUESTS_PER_DOMAIN': c['concurrent_requests_per_domain'],
        'CONCURRENT_REQUESTS_PER_IP': 0, 'DOWNLOAD_TIMEOUT': c['download_timeout'],
        'DOWNLOAD_MAXSIZE': c['max_response_bytes'], 'DOWNLOAD_WARNSIZE': c['warn_response_bytes'],
        'DOWNLOAD_SLOTS': c['download_slots'], 'RETRY_TIMES': c['retry_times'], 'RETRY_HTTP_CODES': c['retry_http_codes'],
        'AUTOTHROTTLE_ENABLED': a['enabled'], 'AUTOTHROTTLE_START_DELAY': a['start_delay'],
        'AUTOTHROTTLE_MAX_DELAY': a['max_delay'], 'AUTOTHROTTLE_TARGET_CONCURRENCY': a['target_concurrency'],
        'REACTOR_THREADPOOL_MAXSIZE': c['reactor_threadpool_maxsize'], 'DNSCACHE_ENABLED': c['dnscache_enabled'],
        'DNSCACHE_SIZE': c['dnscache_size'], 'USER_AGENT': c['user_agent'], 'COOKIES_ENABLED': False,
        'DNS_TIMEOUT': c.get('dns_timeout', 10),
        'SCHEDULER_DEBUG': True, 'LOG_LEVEL': 'INFO',
        'HTTPERROR_ALLOW_ALL': True,
        'TELNETCONSOLE_ENABLED': False,
        'DUPEFILTER_CLASS': 'src.crawler.dupefilter.FrontierDupeFilter',
        'DOWNLOADER_MIDDLEWARES': {'src.crawler.middlewares.SafetyMiddleware': 50,
                                 'src.crawler.middlewares.CooldownMiddleware': 560,
                                 'scrapy.downloadermiddlewares.redirect.RedirectMiddleware': None,
                                 'scrapy.downloadermiddlewares.redirect.MetaRefreshMiddleware': None,
                                 'src.crawler.middlewares.FrontierRedirectMiddleware': 600,
                                 'src.crawler.middlewares.FrontierMetaRefreshMiddleware': 580},
        'ITEM_PIPELINES': {'src.crawler.pipelines.RawCachePipeline': 100,
                          'src.crawler.pipelines.CrawlManifestPipeline': 200},
        'EXTENSIONS': {'src.crawler.middlewares.SizeGuard': 500,
                       'src.crawler.session_guard.SessionGuard': 510 if config.get('kaggle_batch') else None},
        'CLOSESPIDER_TIMEOUT': 0 if timeout is None else max(0.001, timeout),
    }
