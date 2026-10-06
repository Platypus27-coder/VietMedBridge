"""Share a saved session deadline across crawl, extraction and output export."""
import math
import time


def stage_time_limit(config, stage, configured_seconds=0, *, now=None):
    """Return seconds left, or None for an unlimited legacy stage.

    Zero remaining is a stop, never Scrapy's zero-means-unlimited setting.
    The absolute deadline includes notebook setup, restore and preparation.
    """
    if stage not in ('crawl', 'extraction'):
        raise ValueError('Unknown session stage.')
    fixed = float(configured_seconds or 0)
    if not math.isfinite(fixed) or fixed < 0:
        raise ValueError('Stage time limit must be finite and nonnegative.')
    limits = config.get('kaggle_batch', {})
    deadline = limits.get('session_deadline_unix')
    if deadline is None:
        return fixed or None
    deadline = float(deadline)
    export = float(limits.get('export_reserve_seconds', 0))
    extraction = float(limits.get('extraction_reserve_seconds', 0))
    if not all(math.isfinite(v) for v in (deadline, export, extraction)) or min(export, extraction) < 0:
        raise ValueError('Invalid session deadline or reserve.')
    reserve = export
    if stage == 'crawl' and limits.get('extract', True):
        reserve += extraction
    remaining = max(0.0, deadline - (time.time() if now is None else now) - reserve)
    return min(fixed, remaining) if fixed else remaining
