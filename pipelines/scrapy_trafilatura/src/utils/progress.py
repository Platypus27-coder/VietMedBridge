"""Read-only live telemetry: a heartbeat alone does not establish useful work."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import sys

from tqdm import tqdm

from src.storage.io import atomic_json
from src.utils.config import output_path


def activity(config, stage, **metrics):
    if config.get('kaggle_batch'):
        atomic_json(output_path(config, 'checkpoints/kaggle_activity.json'), {
            'stage': stage, 'last_updated_at': datetime.now(timezone.utc).isoformat(), **metrics})


def progress_callback(config, stage, interval_seconds=5):
    """Throttle durable progress writes during loops over many small files."""
    last = float('-inf')

    def update(**metrics):
        nonlocal last
        now = time.monotonic()
        if now-last >= interval_seconds:
            activity(config, stage, **metrics)
            last = now
    return update


class TqdmProgress:
    """Render real outcome counters, keeping newline JSON telemetry alongside."""
    counters = {
        'crawl_prepare': ('inspected_urls', 'frontier_urls', 'Prepare frontier', 'url'),
        'crawl': ('completed_this_run', 'scheduled_this_run', 'Crawl remaining', 'url'),
        'extraction': ('committed_this_run', None, 'Extract new raw', 'doc'),
        'validation': ('validated_urls', 'frontier_urls', 'Validate shard', 'url'),
        'bundling': ('packed_files', None, 'Pack checkpoint', 'file'),
        'bundle_verify': ('verified_files', None, 'Verify checkpoint', 'file'),
    }

    def __init__(self, *, file=None):
        self.file = file or sys.stdout
        self.bar = None
        self.stage = None

    def __call__(self, row):
        stage = row['stage']
        counter = self.counters.get(stage)
        if self.bar is not None and (stage != self.stage or not counter):
            self.bar.close()
            self.bar = None
        if counter:
            current_key, total_key, label, unit = counter
            current = row.get(current_key, 0)
            total = row.get(total_key) if total_key else None
            if self.bar is not None and current < self.bar.n:
                self.bar.close()
                self.bar = None
            if self.bar is None:
                self.bar = tqdm(total=total, desc=label, unit=unit, file=self.file,
                                ascii=True, ncols=110, mininterval=0, leave=True)
            self.bar.total = total
            self.bar.update(current-self.bar.n)
            self.bar.set_postfix(health=row['health'], idle=f"{row.get('idle_seconds', 0):.0f}s", refresh=False)
            self.bar.refresh()
        self.stage = stage
        tqdm.write('[ViBioMIR progress] '+json.dumps(row), file=self.file)
        self.file.flush()


class BatchProgressMonitor:
    def __init__(self, root, shard_id, *, interval_seconds=30, stale_seconds=300,
                 emit=None, clock=time.monotonic):
        if interval_seconds <= 0 or stale_seconds <= 0:
            raise ValueError('Progress intervals must be positive.')
        self.root = Path(root)
        self.shard_id = shard_id
        self.interval = interval_seconds
        self.stale = stale_seconds
        self.emit = emit or (lambda row: print('[ViBioMIR progress] '+json.dumps(row), flush=True))
        self.clock = clock
        self.started = self.changed = self.stage_started = clock()
        self.signature = self.stage = None
        self.stop = threading.Event()
        self.thread = None

    def _read(self, name):
        try:
            value = json.loads((self.root/'checkpoints'/name).read_text(encoding='utf-8'))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}

    def snapshot(self):
        now = self.clock()
        phase = self._read('kaggle_activity.json')
        stage = phase.get('stage', 'starting')
        metrics = {k: v for k, v in phase.items() if k not in ('stage', 'last_updated_at')}
        cooldown = False
        if stage == 'crawl':
            crawl = self._read('run_state.json')
            heartbeat = self._read('crawl_heartbeat.json')
            metrics.update({k: crawl[k] for k in ('completed_this_run', 'counts_this_run', 'scheduled_this_run') if k in crawl})
            metrics.update({k: heartbeat[k] for k in ('requests', 'responses', 'retries', 'raw_bytes') if k in heartbeat})
            until = heartbeat.get('next_cooldown_until', 0)
            cooldown = bool(heartbeat.get('held_requests') and until and until > time.time())
            if cooldown:
                metrics['cooldown_until'] = until
        elif stage == 'extraction':
            extraction = self._read('extraction_state.json')
            metrics.update({k: extraction[k] for k in ('committed_this_run', 'counts_this_run') if k in extraction})
        elif stage == 'bundling':
            temporary = self.root.parent/f'vibiomir_shard_{self.shard_id:05d}.tar.tmp'
            try:
                metrics['archive_bytes'] = temporary.stat().st_size
            except OSError:
                pass
        # Do not treat timestamps, heartbeat rewrites or cooldown changes as work.
        signature = (stage, json.dumps({k: v for k, v in metrics.items() if k != 'cooldown_until'}, sort_keys=True))
        changed = signature != self.signature
        if changed:
            self.signature, self.changed = signature, now
        if stage != self.stage:
            self.stage, self.stage_started = stage, now
        idle = now-self.changed
        health = 'PROGRESS' if changed else 'ALIVE_NO_PROGRESS'
        if idle >= self.stale:
            health = 'WAITING_COOLDOWN' if cooldown else 'POSSIBLE_STALL'
        return {'at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                'shard_id': self.shard_id, 'stage': stage, 'health': health,
                'elapsed_seconds': round(now-self.started, 1),
                'stage_elapsed_seconds': round(now-self.stage_started, 1),
                'idle_seconds': round(idle, 1), **metrics}

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.emit(self.snapshot())

    def __enter__(self):
        self.emit(self.snapshot())
        self.thread = threading.Thread(target=self._loop, name='batch-progress', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=5)
        self.emit({'at': datetime.now(timezone.utc).isoformat(), 'shard_id': self.shard_id,
                   'stage': 'runner_exit', 'health': 'FAILED' if kind else 'FINISHED',
                   'elapsed_seconds': round(self.clock()-self.started, 1)})
