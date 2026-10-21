"""Released resumable receipt lookup; model producers are unchanged."""
from concurrent.futures import ThreadPoolExecutor
import gzip
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import time
from vietmedbridge.artifacts import (atomic_json, digest_json, local_workspace, publish_file,
    read_json, sha256_file, utc_now, verify_file)
from vietmedbridge.content_embeddings import ContentBlocks as NativeContentBlocks, checked, seal
from vietmedbridge.scale_vectors import bound
try:
    from .new_data_drive_api import LookupPublishUnavailable
except ImportError:
    from new_data_drive_api import LookupPublishUnavailable


class ResumableContentBlocks(NativeContentBlocks):
    """Retain validated receipt lookups on local disk across Drive recovery.

    The native content key, encoder identity and vector payload validation stay
    unchanged. A compressed lookup is an optimization, never a coverage receipt.
    """
    def __init__(self, data_root, encoder, work_dir, *, progress=print, receipt_api=None, inventory=True):
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
        self.snapshot_binding = (dict(schema_version=2, data_folder_id=receipt_api.folder_id, identity=self.identity)
            if receipt_api is not None and hasattr(receipt_api, 'folder_id') else self.binding)
        self.remote = self.root / 'retrieval/content_lookups' / digest_json(self.snapshot_binding)[:16]
        self.legacy_remote = self.root / 'retrieval/content_lookups' / digest_json(self.binding)[:16]
        self.temporary = local_workspace(work_dir)
        self.local = Path(self.temporary.__enter__())
        self.con = None
        self._snapshot_dirty = False
        self._snapshot_busy = False
        self._next_snapshot_count = 256
        self._receipt_count = 0
        try:
            if not self.database.exists():
                if not inventory:
                    raise FileNotFoundError('No local lookup to save; no receipt scan or model inference was started.')
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
            self._receipt_count = len(known)
            self._next_snapshot_count = max(256, len(known)*2)
            if not inventory:
                self._snapshot_dirty = True
                self._snapshot()
                return
            self.progress(f'BGE receipt inventory: retained {len(known):,} validated receipts; listing new names')
            markers = (receipt_api.receipts(self.identity) if receipt_api is not None
                else sorted(self.directory.glob('*.done.json')))
            pending = [path for path in markers if path.name not in known]
            self.progress(f'BGE receipt inventory: {len(markers)-len(pending):,}/{len(markers):,} reused'
                f' | {len(pending):,} unread | {"API" if receipt_api is not None else "mount"} transport | commit each receipt')
            self._read_pending(pending, len(markers)-len(pending), len(markers))
            self.progress(f'BGE receipt inventory ready: {len(markers):,} receipts | native vector SHA checked on use')
            if receipt_api is not None:
                if pending:
                    self._snapshot_dirty = True
                if hasattr(receipt_api, 'publish_lookup') and not self._snapshot_dirty:
                    saved = receipt_api.relative_json((self.remote / 'current.json').relative_to(self.root).as_posix())
                    if saved is None:
                        self._snapshot_dirty = True
                    else:
                        checked(saved)
                        if saved.get('binding') != self.snapshot_binding:
                            raise ValueError('Published lookup producer/folder changed.')
                        self._snapshot_dirty = saved['receipts'] < self._receipt_count
                if self._snapshot_dirty:
                    self._snapshot()
                self.progress('BGE API receipt lookup ready; no mount I/O for inventory')
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
            if old is None:
                self._receipt_count += 1
                self._snapshot_dirty = True
        except BaseException:
            self.con.rollback()
            raise
        if (self.receipt_api is not None and self._receipt_count >= self._next_snapshot_count
            and hasattr(self.receipt_api, 'publish_lookup')):
            self._next_snapshot_count = max(256, self._receipt_count*2)
            self._snapshot()

    def _restore(self):
        remote = self.remote
        marker = remote / 'current.json'
        if self.receipt_api is None:
            if not marker.is_file():
                return
            saved = checked(read_json(marker))
        else:
            value = self.receipt_api.relative_json(marker.relative_to(self.root).as_posix())
            if value is None and self.legacy_remote != self.remote:
                remote = self.legacy_remote
                value = self.receipt_api.relative_json((remote / 'current.json').relative_to(self.root).as_posix())
            if value is None:
                return
            saved = checked(value)
        if saved.get('binding') != (self.snapshot_binding if remote == self.remote else self.binding):
            raise ValueError('Published receipt lookup producer/root changed.')
        relative = PurePosixPath(saved['path'])
        if relative.is_absolute() or len(relative.parts) != 1 or '\\' in saved['path']:
            raise ValueError('Published receipt snapshot path escapes its directory.')
        self.progress('BGE receipt inventory: restore one compressed lookup; no old receipt JSON scan')
        local = self.local / 'lookup.sqlite.gz'
        if self.receipt_api is None:
            shutil.copyfile(bound(remote, saved['path']), local)
        else:
            source = self.receipt_api.relative((remote / saved['path']).relative_to(self.root).as_posix())
            if source is None:
                raise ValueError('Published API receipt lookup payload is missing.')
            self.receipt_api.download(source, local, expected_sha256=saved['sha256'])
        verify_file(local, saved['sha256'])
        stage = self.lookup / 'restore.sqlite'
        try:
            with gzip.open(local, 'rb') as src, stage.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
            verify_file(stage, saved['sqlite_sha256'])
            # A shared folder has the same identity on another account even when
            # Colab exposes it through a different MyDrive/shortcut path.
            old_binding = saved.get('lookup_binding', saved['binding'])
            restored = sqlite3.connect(stage)
            try:
                rows = restored.execute('SELECT value FROM metadata').fetchall()
                if rows != [(digest_json(old_binding),)] or old_binding.get('identity') != self.identity:
                    raise ValueError('Restored lookup SQLite binding changed.')
                if restored.execute('SELECT count(*) FROM receipts').fetchone()[0] != saved['receipts']:
                    raise ValueError('Restored lookup receipt count changed.')
                if old_binding != self.binding:
                    restored.execute('UPDATE metadata SET value=?', [digest_json(self.binding)])
                    restored.commit()
            finally:
                restored.close()
            stage.replace(self.database)
        finally:
            stage.unlink(missing_ok=True)

    def _snapshot(self):
        if self._snapshot_busy:
            return
        if self.receipt_api is not None and not hasattr(self.receipt_api, 'publish_lookup'):
            return  # Compatibility with older read-only transports; local prefix remains usable.
        self._snapshot_busy = True
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
            record = seal({'binding': self.snapshot_binding, 'lookup_binding': self.binding, 'path': target.name,
                'sha256': sha256_file(payload), 'sqlite_sha256': checksum,
                'receipts': self.con.execute('SELECT count(*) FROM receipts').fetchone()[0],
                'updated_at': utc_now(),
                'scope': 'Receipt lookup only; vector payloads remain subject to native SHA/shape checks'})
            if self.receipt_api is None:
                publish_file(payload, target)
                atomic_json(self.remote / 'current.json', record)
            else:
                self.receipt_api.publish_lookup((self.remote / 'current.json').relative_to(self.root).as_posix(),
                    payload, record)
            self._snapshot_dirty = False
            self.progress(f"BGE receipt lookup SAVED_ON_DRIVE: {record['receipts']:,} receipts; next runtime restores this lookup")
        except (OSError, LookupPublishUnavailable) as error:
            # Optional acceleration only: the local committed database survives,
            # and native reads/writes still surface actual mount/space failures.
            self.progress(f'BGE receipt lookup snapshot deferred: {type(error).__name__}; local checkpoint retained')
        finally:
            self._snapshot_busy = False
            copy.unlink(missing_ok=True)
            payload.unlink(missing_ok=True)

    def close(self):
        if self.con is not None:
            try:
                if self.receipt_api is not None and self._snapshot_dirty:
                    self._snapshot()
            finally:
                self.con.close()
                self.con = None
        self.temporary.__exit__(None, None, None)
