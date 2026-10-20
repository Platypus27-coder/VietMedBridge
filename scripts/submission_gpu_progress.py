"""Bounded parallel map checks and visible GPU startup; producer files stay pinned."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import gzip
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import sys
import threading
import time

from vietmedbridge.artifacts import (atomic_json, digest_json, local_workspace, publish_file,
    read_json, sha256_file, utc_now, verify_file)
from vietmedbridge.content_embeddings import ContentBlocks as NativeContentBlocks, checked, seal
from vietmedbridge.portable_embeddings import resources_relative
from vietmedbridge.scale_vectors import bound

# Notebook cells embed the pure-stdlib mount helper before this adapter. Module
# imports in local tests load the same helper from scripts, outside producer src.
if 'safe_colab_remount' not in globals():
    from scripts.colab_drive_mount import safe_colab_remount
if 'authenticate_drive_receipts' not in globals():
    from scripts.drive_receipts_api import authenticate_drive_receipts


class ResumableContentBlocks(NativeContentBlocks):
    """Retain validated receipt lookups on local disk across Drive recovery.

    The native content key, encoder identity and vector payload validation stay
    unchanged. A compressed lookup is an optimization, never a coverage receipt.
    """
    def __init__(self, data_root, encoder, work_dir, *, progress=print, receipt_api=None):
        self.receipt_api = receipt_api
        if receipt_api is not None and Path(data_root).absolute() not in (receipt_api.requested_root, receipt_api.data_root):
            raise ValueError('Receipt API root differs from the embedding data root.')
        self.root = receipt_api.data_root if receipt_api is not None else Path(data_root).resolve()
        self.encoder = encoder
        self.identity = {'encoder': encoder.identity, 'dimension': encoder.dimension,
            'format': 'content-float32-l2-v1'}
        self.directory = self.root / 'model_cache' / digest_json(self.identity)[:24]
        if receipt_api is None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        self.binding = {'schema_version': 1, 'data_root': str(self.root), 'identity': self.identity}
        self.lookup = Path(work_dir) / 'receipt_lookups' / digest_json(self.binding)[:16]
        self.lookup.mkdir(parents=True, exist_ok=True)
        self.database = self.lookup / 'lookup.sqlite'
        self.remote = self.root / 'retrieval/content_lookups' / digest_json(self.binding)[:16]
        self.temporary = local_workspace(work_dir)
        self.local = Path(self.temporary.__enter__())
        self.con = None
        try:
            if not self.database.exists():
                self._restore()
            self.con = sqlite3.connect(self.database)
            self.con.execute('PRAGMA cache_size=-32768')
            self.con.execute('PRAGMA synchronous=FULL')
            self.con.execute('CREATE TABLE IF NOT EXISTS metadata(value TEXT NOT NULL)')
            row = self.con.execute('SELECT value FROM metadata').fetchone()
            signature = digest_json(self.binding)
            if row is not None and row[0] != signature:
                raise ValueError('Receipt lookup model/root binding changed.')
            if row is None:
                self.con.execute('INSERT INTO metadata VALUES (?)', [signature])
            self.con.execute('CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY,path TEXT,sha TEXT,row INTEGER)')
            self.con.execute('CREATE TABLE IF NOT EXISTS receipts(name TEXT PRIMARY KEY,digest TEXT NOT NULL)')
            self.con.commit()
            known = {row[0] for row in self.con.execute('SELECT name FROM receipts')}
            self.progress(f'BGE receipt inventory: retained {len(known):,} validated receipts; listing new names')
            markers = (receipt_api.receipts(self.identity) if receipt_api is not None
                else sorted(self.directory.glob('*.done.json')))
            pending = [path for path in markers if path.name not in known]
            self.progress(f'BGE receipt inventory: {len(markers)-len(pending):,}/{len(markers):,} reused'
                f' | {len(pending):,} unread | {"API" if receipt_api is not None else "mount"} transport | commit each receipt')
            self._read_pending(pending, len(markers)-len(pending), len(markers))
            self.progress(f'BGE receipt inventory ready: {len(markers):,} receipts | native vector SHA checked on use')
            if receipt_api is not None:
                self.progress('BGE API receipt lookup committed on local disk; no mount I/O for inventory')
            elif pending or not (self.remote / 'current.json').is_file():
                self._snapshot()
        except BaseException:
            self.close()
            raise

    def _read_pending(self, pending, reused, total):
        # Keep at most two reads in flight: pool.map queued the whole directory
        # and discarded its temporary SQLite lookup after any mount failure.
        readers = 4 if self.receipt_api is not None else 2
        pool = ThreadPoolExecutor(max_workers=readers)
        iterator, futures = iter(pending), []
        count, last_log = 0, time.monotonic()
        def refill():
            while len(futures) < readers:
                path = next(iterator, None)
                if path is None:
                    break
                futures.append((path, pool.submit(lambda p: checked(
                    p.read_json() if self.receipt_api is not None else read_json(p)), path)))
        try:
            refill()
            while futures:
                # Register in native filename order, preserving the first
                # existing vector reference when blocks contain overlapping text.
                path, future = futures.pop(0)
                block = future.result()
                if path.name != 'block-' + digest_json(block['keys']) + '.done.json':
                    raise ValueError('Content receipt name/key binding changed.')
                self.register(block)
                count += 1
                if count % 32 == 0 or time.monotonic()-last_log >= 15 or count == len(pending):
                    self.progress(f'BGE receipt inventory: {reused+count:,}/{total:,}'
                        f' | reused {reused:,}, newly indexed {count:,} | local checkpoint committed')
                    last_log = time.monotonic()
                refill()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    def register(self, block):
        checked(block)
        marker = 'block-' + digest_json(block['keys']) + '.done.json'
        old = self.con.execute('SELECT digest FROM receipts WHERE name=?', [marker]).fetchone()
        if old is not None and old[0] != block['manifest_sha256']:
            raise ValueError('Immutable content receipt changed in the retained lookup.')
        try:
            # Keys and their receipt commit together; interrupted reads never
            # advance the inventory or erase an earlier committed prefix.
            if self.receipt_api is None:
                super().register(block)
            else:
                if block['identity'] != self.identity or len(block['keys']) != block['rows']:
                    raise ValueError('Content vector producer/key mismatch.')
                relative = PurePosixPath(block['path'])
                if relative.is_absolute() or '..' in relative.parts or '\\' in block['path'] or not relative.parts:
                    raise ValueError('API content reference escapes its data root.')
                # No filesystem resolution through FUSE during API inventory.
                # Native vector readers still resolve the actual path and check
                # its SHA256/shape before consuming the vector payload.
                self.con.executemany('INSERT OR IGNORE INTO vectors VALUES (?,?,?,?)',
                    [(key,block['path'],block['sha256'],i) for i,key in enumerate(block['keys'])])
            self.con.execute('INSERT OR IGNORE INTO receipts VALUES (?,?)', [marker, block['manifest_sha256']])
            self.con.commit()
        except BaseException:
            self.con.rollback()
            raise

    def _restore(self):
        marker = self.remote / 'current.json'
        if self.receipt_api is None:
            if not marker.is_file():
                return
            saved = checked(read_json(marker))
        else:
            value = self.receipt_api.relative_json(marker.relative_to(self.root).as_posix())
            if value is None:
                return
            saved = checked(value)
        if saved.get('binding') != self.binding:
            raise ValueError('Published receipt lookup producer/root changed.')
        relative = PurePosixPath(saved['path'])
        if relative.is_absolute() or len(relative.parts) != 1 or '\\' in saved['path']:
            raise ValueError('Published receipt snapshot path escapes its directory.')
        self.progress('BGE receipt inventory: restore one compressed lookup; no old receipt JSON scan')
        local = self.local / 'lookup.sqlite.gz'
        if self.receipt_api is None:
            shutil.copyfile(bound(self.remote, saved['path']), local)
        else:
            source = self.receipt_api.relative((self.remote / saved['path']).relative_to(self.root).as_posix())
            if source is None:
                raise ValueError('Published API receipt lookup payload is missing.')
            self.receipt_api.download(source, local, expected_sha256=saved['sha256'])
        verify_file(local, saved['sha256'])
        stage = self.lookup / 'restore.sqlite'
        try:
            with gzip.open(local, 'rb') as src, stage.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
            verify_file(stage, saved['sqlite_sha256'])
            stage.replace(self.database)
        finally:
            stage.unlink(missing_ok=True)

    def _snapshot(self):
        self.progress('BGE receipt inventory: save compressed lookup for future runtimes')
        copy = self.local / 'lookup-backup.sqlite'
        payload = self.local / 'lookup.sqlite.gz'
        try:
            backup = sqlite3.connect(copy)
            try:
                self.con.backup(backup)
            finally:
                backup.close()
            checksum = sha256_file(copy)
            with copy.open('rb') as src, gzip.open(payload, 'wb', compresslevel=1) as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
            target = self.remote / ('lookup-' + sha256_file(payload) + '.sqlite.gz')
            published = publish_file(payload, target)
            atomic_json(self.remote / 'current.json', seal({'binding': self.binding, 'path': target.name,
                'sha256': published, 'sqlite_sha256': checksum,
                'receipts': self.con.execute('SELECT count(*) FROM receipts').fetchone()[0],
                'scope': 'Receipt lookup only; vector payloads remain subject to native SHA/shape checks'}))
            self.progress('BGE receipt inventory snapshot saved; next runtime reads the compressed lookup')
        except OSError as error:
            # Optional acceleration only: the local committed database survives,
            # and native reads/writes still surface actual mount/space failures.
            self.progress(f'BGE receipt lookup snapshot deferred: {type(error).__name__}; local checkpoint retained')
        finally:
            copy.unlink(missing_ok=True)
            payload.unlink(missing_ok=True)

    def close(self):
        if self.con is not None:
            self.con.close()
            self.con = None
        self.temporary.__exit__(None, None, None)


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


def ensure_submission_mount(data_root, expected_candidate_sha256):
    """Return the matching root through known Drive aliases; JSON only."""
    from vietmedbridge.colab_storage import _disconnected, _drive_mountpoint
    from vietmedbridge.scale_benchmark import load_handoff
    if not isinstance(expected_candidate_sha256, str) or not expected_candidate_sha256:
        raise ValueError('Expected frozen candidate is required before reconnecting.')
    print('SUBMISSION preflight: read frozen handoff JSON; preparation is not repeated', flush=True)
    root = Path(data_root)
    mount = _drive_mountpoint(root)
    # These are observed aliases of this shared project, not a Drive-wide search.
    aliases = [root]
    if mount is not None:
        aliases += [Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge/data'),
            Path('/content/drive/MyDrive/VietMedBridge/data')]
    aliases = list(dict.fromkeys(aliases))
    missing = None
    for attempt in range(2 if mount is not None else 1):
        for location in aliases:
            try:
                _, candidate, _ = load_handoff(location)
            except FileNotFoundError as error:
                missing = error
                continue
            except OSError as error:
                if mount is None or not _disconnected(error, mount):
                    raise
                if attempt:
                    raise
                break
            else:
                if candidate['candidate_manifest_sha256'] != expected_candidate_sha256:
                    raise ValueError('Active candidate changed; refusing to resume another submission corpus.')
                print('SUBMISSION preflight: same frozen candidate verified | DATA_ROOT:', location, flush=True)
                return location
        if mount is not None and not attempt:
            print('SUBMISSION preflight: handoff unavailable; reconnect Drive, retain runtime and local lookup', flush=True)
            safe_colab_remount(mount, readonly=False)
    if mount is None and missing is not None:
        raise missing
    raise FileNotFoundError('Frozen submission handoff not found after reconnect. Checked only: '
        + ', '.join(map(str, aliases)) + '. No data or checkpoint was created.')


def prepare_drive_receipt_lookup(receipt_api, checkout, work_dir, *, progress=print):
    """CPU metadata only, before mapping scans or any neural model allocation."""
    from types import SimpleNamespace
    config = read_json(Path(checkout) / 'configs/retrieval_full.json')
    relative = Path(resources_relative(receipt_api.candidate, config)) / 'bge/config.json'
    header = receipt_api.relative_json(relative.as_posix())
    if header is None:
        progress('BGE API lookup: no existing BGE view; no paid receipt inventory to prepare')
        return
    if (header.get('input_manifest_sha256') != receipt_api.active['index_inputs_manifest_sha256']
        or header.get('dimension') != 1024 or header.get('format') != 'content-reference-v1'
        or any(header.get('encoder', {}).get(k) != v for k,v in config['dense'].items())):
        raise ValueError('API BGE producer/header differs from the active frozen inputs.')
    progress('BGE API lookup preparation: CPU metadata only; no mapping payload scan or model weights')
    encoder = SimpleNamespace(identity=header['encoder'], dimension=header['dimension'])
    lookup = ResumableContentBlocks(receipt_api.requested_root, encoder, work_dir,
        progress=progress, receipt_api=receipt_api)
    lookup.close()
    progress('BGE API receipt lookup ready; continuing native submission checks')


class SubmissionProgress:
    def __init__(self, data_root, work_dir, label, *, interval=15, drive_reports=True):
        self.root, self.work, self.label = Path(data_root), Path(work_dir), label
        self.interval, self.main_id = interval, threading.get_ident()
        self.started, self.last_saved = time.monotonic(), 0
        self.phase = 'Entering submission workflow'
        self.binding = {}
        self.drive_reports = drive_reports
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
            if location is None and 'vietmedbridge' in frame.f_code.co_filename:
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
        if self.drive_reports and time.monotonic() - self.last_saved >= 15:
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
def submission_gpu_progress(data_root, work_dir, *, label='baseline', receipt_api=None):
    """Single-writer notebook adapter; restores every patched binding on exit."""
    from vietmedbridge import colab_storage, content_embeddings, scale_baseline, submission_workflow
    progress = SubmissionProgress(data_root, work_dir, label, drive_reports=receipt_api is None)
    original_maps = scale_baseline.imported_bge_manifest
    original_blocks = content_embeddings.ContentBlocks
    original_remount = colab_storage._remount
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
    content_embeddings.ContentBlocks = lambda *a, **k: ResumableContentBlocks(*a, **k,
        progress=progress.report, receipt_api=receipt_api)
    colab_storage._remount = safe_colab_remount
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
        content_embeddings.ContentBlocks = original_blocks
        colab_storage._remount = original_remount
        for name, function in originals.items():
            setattr(submission_workflow, name, function)
        status = progress.snapshot(state)
        try:
            atomic_json(Path(work_dir) / 'submission_progress.json', status)
        except OSError:
            pass
        if receipt_api is None or state == 'CALL_FINISHED':
            try:
                atomic_json(Path(data_root) / 'retrieval/submission_progress.json', status)
            except OSError:
                pass
