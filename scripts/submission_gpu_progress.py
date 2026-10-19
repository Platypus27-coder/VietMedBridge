"""Bounded parallel map checks and visible GPU startup; producer files stay pinned."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import time

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, utc_now, verify_file
from vietmedbridge.content_embeddings import checked, seal
from vietmedbridge.portable_embeddings import resources_relative


def read_bge_maps_parallel(data_root, candidate, inputs, config, *, progress=print):
    """Same BGE identity/map/payload checks as imported_content_manifest, in frozen order."""
    view = Path(data_root) / resources_relative(candidate, config) / 'bge'
    path = view / 'config.json'
    if not path.is_file():
        return view, None
    identity = read_json(path)
    if (identity.get('input_manifest_sha256') != inputs['manifest_sha256']
        or identity.get('input_count') != inputs['input_count']
        or identity.get('format') != 'content-reference-v1'
        or identity.get('dimension') != 1024
        or any(identity.get('encoder', {}).get(k) != v for k, v in config['dense'].items())):
        raise ValueError('Imported BGE view differs from the active frozen corpus/model.')
    signature = digest_json(identity)
    sources = inputs['parts']
    progress(f'BGE mapping check: 0/{len(sources):,} | up to 8 readers | SHA256 preserved')

    def check(source):
        marker = view / f"map-{source['start']:012d}.done.json"
        if not marker.is_file():
            return None
        saved = checked(read_json(marker))
        expected = {'signature': signature, 'start': source['start'], 'rows': source['rows'],
            'input_sha256': source['sha256'], 'path': f"map-{source['start']:012d}.parquet"}
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Imported BGE mapping differs from the frozen input order.')
        verify_file(view / saved['path'], saved['sha256'])
        return saved

    ordered = [None] * len(sources)
    found = 0
    pool = ThreadPoolExecutor(max_workers=min(8, max(1, len(sources))))
    try:
        futures = {pool.submit(check, source): i for i, source in enumerate(sources)}
        for count, future in enumerate(as_completed(futures), 1):
            saved = future.result()
            ordered[futures[future]] = saved
            found += saved is not None
            if count % 32 == 0 or count == len(sources):
                progress(f'BGE mapping check: {count:,}/{len(sources):,} | valid maps {found:,}')
    except BaseException:
        # Do not trap the notebook in executor shutdown if Drive readers stall.
        # In-flight work only reads files; it never publishes or changes checkpoints.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    parts = [p for p in ordered if p is not None]
    return view, seal(identity | {'signature': signature, 'parts': parts,
        'state': 'COMPLETE' if len(parts) == len(sources) else 'IN_PROGRESS'})


class SubmissionProgress:
    def __init__(self, data_root, work_dir, label, *, interval=15):
        self.root, self.work, self.label = Path(data_root), Path(work_dir), label
        self.interval, self.main_id = interval, threading.get_ident()
        self.started, self.last_saved = time.monotonic(), 0
        self.phase = 'Entering submission workflow'
        self.binding = {}
        self.stop = threading.Event()

    def location(self):
        frame = sys._current_frames().get(self.main_id)
        location = None
        while frame is not None:
            if frame.f_code.co_name in {'read_json', 'sha256_file'}:
                value = frame.f_locals.get('path')
                if isinstance(value, (str, Path)):
                    location = str(value)
                    break
            if 'vietmedbridge' in frame.f_code.co_filename:
                location = Path(frame.f_code.co_filename).name + ':' + str(frame.f_lineno)
            frame = frame.f_back
        return location

    def snapshot(self, state='RUNNING'):
        return {'state': state, 'variant': self.label, 'phase': self.phase,
            'elapsed_seconds': round(time.monotonic() - self.started, 1),
            'waiting_at': self.location(), 'updated_at': utc_now(),
            **self.binding,
            'scope': 'Live stage diagnostic; not an embedding completion or submission receipt'}

    def report(self, message):
        self.phase = message
        print(message, flush=True)
        if time.monotonic() - self.last_saved >= 15:
            try:
                atomic_json(self.root / 'retrieval/submission_progress.json', self.snapshot())
                self.last_saved = time.monotonic()
            except OSError:
                pass

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            status = self.snapshot()
            print(f"SUBMISSION heartbeat | {status['elapsed_seconds']:.0f}s | {self.phase}"
                f" | waiting at {status['waiting_at']}", flush=True)
            try:
                atomic_json(self.work / 'submission_progress.json', status)
            except OSError:
                pass  # Diagnostics cannot replace or invalidate native checkpoint errors.


@contextmanager
def submission_gpu_progress(data_root, work_dir, *, label='baseline'):
    """Single-writer notebook adapter; restores every patched binding on exit."""
    from vietmedbridge import scale_baseline, submission_workflow
    progress = SubmissionProgress(data_root, work_dir, label)
    original_maps = scale_baseline.imported_bge_manifest
    names = ('_submission_context', '_saved_submission', '_release_catalog_staging', '_ensure_primary')
    originals = {name: getattr(submission_workflow, name) for name in names}

    def wrap(name, function):
        def observed(*args, **kwargs):
            progress.report('SUBMISSION stage: ' + name)
            result = function(*args, **kwargs)
            if name == '_submission_context':
                progress.binding = {'build_run': result[3].name,
                    'candidate_manifest_sha256': result[4]['candidate_manifest_sha256'],
                    'input_count': result[5]['input_count']}
            progress.report('SUBMISSION stage complete: ' + name)
            return result
        return observed

    scale_baseline.imported_bge_manifest = lambda *a, **k: read_bge_maps_parallel(*a, **k, progress=progress.report)
    for name, function in originals.items():
        setattr(submission_workflow, name, wrap(name, function))
    thread = threading.Thread(target=progress.heartbeat, name='submission-progress', daemon=True)
    progress.report('SUBMISSION startup | live progress every 15 seconds | same candidate and checkpoints')
    thread.start()
    state = 'STOPPED'
    try:
        yield progress
        state = 'CALL_FINISHED'
    finally:
        progress.stop.set()
        thread.join(timeout=1)
        scale_baseline.imported_bge_manifest = original_maps
        for name, function in originals.items():
            setattr(submission_workflow, name, function)
        status = progress.snapshot(state)
        try:
            atomic_json(Path(work_dir) / 'submission_progress.json', status)
        except OSError:
            pass
        try:
            atomic_json(Path(data_root) / 'retrieval/submission_progress.json', status)
        except OSError:
            pass
