import json
import threading
import time
from io import StringIO

import pytest

from src.storage.io import atomic_json
from src.utils.progress import BatchProgressMonitor, TqdmProgress, activity


def test_unchanged_heartbeat_is_not_work_and_new_work_clears_stall(config):
    config['kaggle_batch'] = {'extract': True}
    clock = [0.0]
    monitor = BatchProgressMonitor(config['_output_root'], 0, stale_seconds=300,
                                   clock=lambda: clock[0])
    activity(config, 'crawl', scheduled_this_run=100)
    checkpoints = monitor.root/'checkpoints'
    atomic_json(checkpoints/'run_state.json', {'completed_this_run': 7,
                                              'counts_this_run': {'SUCCESS_RAW': 5, 'HTTP_403': 2}})
    atomic_json(checkpoints/'crawl_heartbeat.json', {'requests': 10, 'responses': 8, 'last_updated_at': 'old'})
    first = monitor.snapshot()
    assert first['health'] == 'PROGRESS' and first['completed_this_run'] == 7
    clock[0] = 301
    atomic_json(checkpoints/'crawl_heartbeat.json', {'requests': 10, 'responses': 8, 'last_updated_at': 'new'})
    stalled = monitor.snapshot()
    assert stalled['health'] == 'POSSIBLE_STALL' and stalled['idle_seconds'] == 301
    # A real network counter change clears the alarm; time rewrites do not.
    atomic_json(checkpoints/'crawl_heartbeat.json', {'requests': 11, 'responses': 9})
    assert monitor.snapshot()['health'] == 'PROGRESS'
    clock[0] += 301
    atomic_json(checkpoints/'crawl_heartbeat.json', {'requests': 11, 'responses': 9,
        'held_requests': 1, 'next_cooldown_until': time.time()+600})
    assert monitor.snapshot()['health'] == 'WAITING_COOLDOWN'
    activity(config, 'extraction')
    atomic_json(checkpoints/'extraction_state.json', {'committed_this_run': 3})
    extraction = monitor.snapshot()
    assert extraction['stage'] == 'extraction' and extraction['committed_this_run'] == 3
    assert 'completed_this_run' not in extraction and extraction['health'] == 'PROGRESS'


def test_monitor_emits_during_blocking_main_work_and_preserves_failure(tmp_path):
    received = threading.Event()
    messages = []
    def emit(row):
        messages.append(row)
        if row.get('health') == 'POSSIBLE_STALL':
            received.set()
    monitor = BatchProgressMonitor(tmp_path/'uncreated', 2, interval_seconds=0.01,
                                   stale_seconds=0.02, emit=emit)
    with pytest.raises(RuntimeError, match='original runner failure'):
        with monitor:
            assert received.wait(timeout=3)
            raise RuntimeError('original runner failure')
    assert messages[-1]['health'] == 'FAILED'
    assert not monitor.thread.is_alive() and not monitor.root.exists()


def test_monitor_reads_bundle_size_and_tolerates_bad_checkpoint(tmp_path):
    root = tmp_path/'session'
    (root/'checkpoints').mkdir(parents=True)
    path = root/'checkpoints/kaggle_activity.json'
    path.write_text('{"stage":')
    clock = [0]
    monitor = BatchProgressMonitor(root, 3, clock=lambda: clock[0])
    assert monitor.snapshot()['stage'] == 'starting'
    atomic_json(path, {'stage': 'bundling'})
    archive = tmp_path/'vibiomir_shard_00003.tar.tmp'
    archive.write_bytes(b'payload')
    first = monitor.snapshot()
    assert first['archive_bytes'] == 7
    clock[0] = 10
    with archive.open('ab') as handle:
        handle.write(b'more')
    assert monitor.snapshot()['health'] == 'PROGRESS'


def test_tqdm_tracks_committed_outcomes_and_closes_on_stage_change():
    output = StringIO()
    progress = TqdmProgress(file=output)
    for count in (4, 8):
        progress({'stage': 'crawl', 'health': 'PROGRESS', 'idle_seconds': 0,
                  'completed_this_run': count, 'scheduled_this_run': 10})
    assert progress.bar.n == 8 and progress.bar.total == 10
    progress({'stage': 'extraction', 'health': 'PROGRESS', 'committed_this_run': 2})
    assert progress.bar.n == 2 and progress.bar.total is None
    progress({'stage': 'runner_exit', 'health': 'FINISHED'})
    assert progress.bar is None
    text = output.getvalue()
    assert 'Crawl remaining' in text and '8/10' in text and 'Extract new raw' in text
    assert '[ViBioMIR progress]' in text
