# Run only after the current processing cell has returned, before changing accounts.
import threading
import importlib.util
import subprocess
import sys
if any(t.name == "submission-progress" and t.is_alive() for t in threading.enumerate()):
    raise RuntimeError("Cell xử lý vẫn đang chạy. Chỉ lưu bảng tra sau khi cell đó kết thúc; không chạy đồng thời.")
if any(name not in globals() for name in ("DATA_ROOT", "WORK_DIR", "CHECKOUT", "PREPARATION")):
    raise RuntimeError("Cần runtime cũ còn bảng tra local và biến của notebook submit; không khởi động pipeline mới.")
_vmb_previous_receipts = globals().get("DRIVE_RECEIPTS")
if importlib.util.find_spec("googleapiclient") is None:
    subprocess.run([sys.executable,"-m","pip","install","google-api-python-client>=2,<3"],check=True)

"""Repair a nonempty, unmounted Colab Drive directory without deleting files.

Pure standard library so notebook setup can embed this before pip/package imports.
Kernel mount state must positively exclude this directory and any nested mounts
before an atomic rename preserves the complete local directory under /content.
"""
import os
from pathlib import Path
import re
from uuid import uuid4


def preserve_unmounted_drive_directory(mount, *, content_root='/content',
                                       mountinfo_file='/proc/self/mountinfo'):
    target, parent = Path(mount).absolute(), Path(content_root).resolve()
    if target.name != 'drive' or target.parent.resolve() != parent:
        raise ValueError('Drive recovery may preserve only the named content_root/drive directory.')
    if target.is_symlink():
        raise RuntimeError('Drive mountpoint is a symlink; refusing to move it.')
    try:
        entries = Path(mountinfo_file).read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise RuntimeError('Cannot verify kernel mount state; no Drive directory was moved.') from error
    if not entries:
        raise RuntimeError('Empty kernel mount information; no Drive directory was moved.')
    for line in entries:
        fields = line.split()
        if len(fields) < 10 or '-' not in fields[6:]:
            raise RuntimeError('Invalid kernel mount information; no Drive directory was moved.')
        name = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[4])
        point = Path(name).absolute()
        if point == target or point.is_relative_to(target):
            raise RuntimeError(f'Kernel mount still active at {point}; refusing to move Drive data.')
    if os.path.ismount(target):
        raise RuntimeError('Drive is still a mountpoint; refusing to move it.')
    if not target.exists():
        return None
    if not target.is_dir():
        raise RuntimeError('Drive mountpoint is a file; refusing to move it.')
    if next(target.iterdir(), None) is None:
        return None
    backup = parent / ('vmb_drive_unmounted_' + uuid4().hex)
    if backup.exists() or backup.is_symlink():
        raise RuntimeError('Preservation path already exists; no Drive directory was moved.')
    # Same-parent rename retains every nested file and never follows child links.
    # Only an unmounted local directory is moved; original remote data is untouched.
    target.rename(backup)
    target.mkdir()
    print('Local unmounted Drive directory preserved at:', backup, flush=True)
    return backup


def safe_colab_mount(mount='/content/drive', *, force_remount=False, readonly=False,
                     drive_api=None, content_root='/content', mountinfo_file='/proc/self/mountinfo'):
    if drive_api is None:
        from google.colab import drive as drive_api
    try:
        drive_api.mount(str(mount), force_remount=force_remount, readonly=readonly)
        return None
    except ValueError as error:
        if 'Mountpoint must not already contain files' not in str(error):
            raise
    backup = preserve_unmounted_drive_directory(mount, content_root=content_root,
        mountinfo_file=mountinfo_file)
    drive_api.mount(str(mount), force_remount=True, readonly=readonly)
    return backup


def safe_colab_remount(mount, *, readonly=False, drive_api=None,
                       content_root='/content', mountinfo_file='/proc/self/mountinfo'):
    if drive_api is None:
        from google.colab import drive as drive_api
    try:
        drive_api.flush_and_unmount()
    except Exception as error:
        print('Drive flush/unmount did not complete:', type(error).__name__, flush=True)
    return safe_colab_mount(mount, force_remount=True, readonly=readonly, drive_api=drive_api,
        content_root=content_root, mountinfo_file=mountinfo_file)

"""API receipt reads and narrowly scoped lookup checkpoints, without Drive FUSE."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time


class LookupPublishUnavailable(RuntimeError):
    """An optional lookup upload failed; paid vector receipts remain untouched."""


class RemoteReceipt:
    def __init__(self, source, metadata):
        self.source, self.metadata = source, metadata
        self.name = metadata['name']

    def read_json(self):
        return self.source.read_json(self.metadata)


class DriveReceiptSource:
    endpoint = 'https://www.googleapis.com/drive/v3/files'

    def __init__(self, session_factory, data_root, canonical_root, *,
                 data_folder_id='1Ez1S9jk6FO_puxC8r2vPSc0LJbuOkkdj', report=print):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', data_folder_id):
            raise ValueError('Invalid Drive data folder ID.')
        self.session_factory, self.sessions = session_factory, threading.local()
        self.requested_root, self.data_root = Path(data_root).absolute(), Path(canonical_root)
        self.folder_id, self.report, self.folders = data_folder_id, report, {}

    def _session(self):
        if not hasattr(self.sessions, 'session'):
            self.sessions.session = self.session_factory()
        return self.sessions.session

    def _get(self, url, params, *, stream=False):
        import requests
        session = self._session()
        for attempt in range(4):
            try:
                response = session.get(url, params=params, stream=stream, timeout=(10,60))
            except requests.RequestException:
                if attempt == 3:
                    raise
            else:
                if response.status_code not in (429,500,502,503,504) or attempt == 3:
                    response.raise_for_status()
                    return response
                response.close()
            self.report(f'Drive receipt API: transient request failure; retry {attempt+1}/3')
            time.sleep(2**attempt)

    def _list(self, query):
        token, seen = None, set()
        while True:
            params = {'q': query, 'pageSize': 1000, 'supportsAllDrives': 'true',
                'includeItemsFromAllDrives': 'true',
                'fields': 'nextPageToken,incompleteSearch,files(id,name,mimeType,size,md5Checksum)'}
            if token:
                params['pageToken'] = token
            with self._get(self.endpoint, params) as response:
                page = response.json()
            if page.get('incompleteSearch'):
                raise ValueError('Drive receipt listing is incomplete; refusing partial inventory.')
            yield from page.get('files', [])
            token = page.get('nextPageToken')
            if not token:
                return
            if token in seen:
                raise ValueError('Drive receipt pagination repeated; inventory is incomplete.')
            seen.add(token)

    def child(self, parent, name, *, folder=False):
        safe = name.replace('\\', '\\\\').replace("'", "\\'")
        query = f"'{parent}' in parents and trashed=false and name='{safe}'"
        if folder:
            query += " and mimeType='application/vnd.google-apps.folder'"
        values = list(self._list(query))
        if len(values) > 1:
            raise ValueError('Duplicate Drive path names; refusing ambiguous receipt source.')
        return values[0] if values else None

    def relative(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or not relative.parts:
            raise ValueError('Invalid relative Drive receipt path.')
        parent = self.folder_id
        for part in relative.parts[:-1]:
            key = parent, part
            if key not in self.folders:
                value = self.child(parent, part, folder=True)
                if value is None:
                    return None
                self.folders[key] = value['id']
            parent = self.folders[key]
        return self.child(parent, relative.name)

    def download(self, metadata, target, *, expected_sha256=None, max_bytes=None):
        size = int(metadata['size'])
        if max_bytes is not None and size > max_bytes:
            raise ValueError('Receipt exceeds the bounded JSON size.')
        sha, md5, count = hashlib.sha256(), hashlib.md5(), 0
        with self._get(self.endpoint+'/'+metadata['id'], {'alt':'media','supportsAllDrives':'true'}, stream=True) as response:
            with Path(target).open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    count += len(chunk)
                    if count > size or (max_bytes is not None and count > max_bytes):
                        raise ValueError('Drive receipt payload exceeds its declared size.')
                    sha.update(chunk)
                    md5.update(chunk)
                    output.write(chunk)
        if (count != size or (metadata.get('md5Checksum') and md5.hexdigest() != metadata['md5Checksum'])
            or (expected_sha256 and sha.hexdigest() != expected_sha256)):
            raise ValueError('Drive receipt size/checksum mismatch.')

    def read_json(self, metadata):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='vmb-api-json-') as directory:
            path = Path(directory) / 'receipt.json'
            self.download(metadata, path, max_bytes=16*1024*1024)
            return json.loads(path.read_text(encoding='utf-8'))

    def relative_json(self, path):
        metadata = self.relative(path)
        return None if metadata is None else self.read_json(metadata)

    def verify_candidate(self, expected):
        active = self.relative_json('active_data_candidate.json')
        if active is None or active.get('candidate_manifest_sha256') != expected:
            raise ValueError('Drive API data folder differs from the expected frozen candidate.')
        candidate = self.relative_json('processed/'+active['build_run']+'/'+active['candidate_name'])
        from vietmedbridge.artifacts import digest_json
        if (candidate is None or candidate.get('candidate_manifest_sha256') != expected
            or digest_json({k:v for k,v in candidate.items() if k != 'candidate_manifest_sha256'}) != expected):
            raise ValueError('Drive API candidate digest changed.')
        self.active, self.candidate = active, candidate

    def receipts(self, identity):
        from vietmedbridge.artifacts import digest_json
        # Resolve the folder itself without relying on any receipt being present.
        cache = self.child(self.folder_id, 'model_cache', folder=True)
        folder = self.child(cache['id'], digest_json(identity)[:24], folder=True) if cache else None
        if folder is None:
            return []
        result, names = [], set()
        # Drive name-contains is prefix matching, not a suffix predicate. List
        # the complete producer folder then filter receipt names locally.
        query = f"'{folder['id']}' in parents and trashed=false"
        for value in self._list(query):
            if not value['name'].endswith('.done.json'):
                continue
            if (not re.fullmatch(r'block-[0-9a-f]{64}\.done\.json', value['name'])
                or value['name'] in names or 'size' not in value):
                raise ValueError('Invalid or duplicate API receipt filename.')
            names.add(value['name'])
            result.append(RemoteReceipt(self, value))
        self.report(f'Drive receipt API: complete listing, {len(result):,} receipts; no mount reads')
        return sorted(result, key=lambda value: value.name)

    def publish_lookup(self, relative_marker, payload, marker):
        """Upload a checksum-addressed lookup, then its pointer. Never write vector/source files."""
        from vietmedbridge.artifacts import digest_json
        body = {k:v for k,v in marker.items() if k != 'manifest_sha256'}
        binding = marker.get('binding', {})
        expected_directory = 'retrieval/content_lookups/' + digest_json(binding)[:16]
        if (relative_marker != expected_directory + '/current.json'
            or binding.get('schema_version') != 2 or binding.get('data_folder_id') != self.folder_id
            or marker.get('manifest_sha256') != digest_json(body)
            or marker.get('path') != 'lookup-' + marker.get('sha256', '') + '.sqlite.gz'
            or not re.fullmatch(r'lookup-[0-9a-f]{64}\.sqlite\.gz', marker.get('path', ''))):
            raise ValueError('Lookup upload is outside its bound checkpoint directory.')
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with Path(payload).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                sha.update(chunk)
                md5.update(chunk)
        if sha.hexdigest() != marker['sha256']:
            raise ValueError('Lookup payload changed before upload.')
        try:
            return self._publish_lookup_files(expected_directory, Path(payload), marker, md5.hexdigest())
        except ValueError:
            raise
        except Exception as error:
            # Do not include provider messages/URLs: resumable upload URLs are private.
            raise LookupPublishUnavailable(type(error).__name__) from error

    def _publish_lookup_files(self, directory, payload, marker, md5):
        from vietmedbridge.artifacts import digest_json
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
        import io
        service = build('drive', 'v3', credentials=copy.copy(self._session().credentials), cache_discovery=False)
        files = service.files()
        fields = 'id,name,mimeType,size,md5Checksum'

        def new_id():
            return files.generateIds(count=1, space='drive', type='files').execute(num_retries=3)['ids'][0]

        def ensure_folder(parent, name):
            value = self.child(parent, name, folder=True)
            if value is None:
                identifier = new_id()
                try:
                    value = files.create(body={'id':identifier, 'name':name, 'parents':[parent],
                        'mimeType':'application/vnd.google-apps.folder'}, fields=fields,
                        supportsAllDrives=True).execute(num_retries=3)
                except Exception:
                    # A lost response may follow a successful create. Reconcile the
                    # preallocated ID; never retry a folder with a different ID.
                    value = files.get(fileId=identifier, fields=fields, supportsAllDrives=True).execute(num_retries=3)
                    if value.get('name') != name or value.get('mimeType') != 'application/vnd.google-apps.folder':
                        raise ValueError('Created lookup folder identity changed.')
            self.folders[parent, name] = value['id']
            return value['id']

        parent = self.folder_id
        for name in PurePosixPath(directory).parts:
            parent = ensure_folder(parent, name)
        current = self.child(parent, 'current.json')
        if current is not None:
            existing = self.read_json(current)
            if (existing.get('manifest_sha256') != digest_json(
                    {k:v for k,v in existing.items() if k != 'manifest_sha256'})
                or existing.get('binding') != marker['binding']):
                raise ValueError('Existing lookup checkpoint binding/digest changed.')
            if existing['receipts'] > marker['receipts']:
                return existing  # An older local prefix must not replace a newer saved lookup.

        def upload(media, name, existing=None):
            if existing is None:
                request = files.create(body={'id':new_id(), 'name':name, 'parents':[parent]},
                    media_body=media, fields=fields, supportsAllDrives=True)
            else:
                request = files.update(fileId=existing['id'], media_body=media,
                    fields=fields, supportsAllDrives=True)
            result = None
            while result is None:
                _, result = request.next_chunk(num_retries=3)
            # Query provider metadata independently before publishing a pointer.
            return files.get(fileId=result['id'], fields=fields, supportsAllDrives=True).execute(num_retries=3)

        saved = self.child(parent, marker['path'])
        if saved is None:
            saved = upload(MediaFileUpload(str(payload), mimetype='application/gzip',
                chunksize=8*1024*1024, resumable=True), marker['path'])
        if int(saved.get('size', -1)) != payload.stat().st_size or saved.get('md5Checksum') != md5:
            raise ValueError('Uploaded lookup payload size/checksum mismatch; pointer not advanced.')
        data = json.dumps(marker, ensure_ascii=False, sort_keys=True).encode('utf-8')
        saved_marker = upload(MediaIoBaseUpload(io.BytesIO(data), mimetype='application/json',
            chunksize=256*1024, resumable=True), 'current.json', current)
        if int(saved_marker.get('size', -1)) != len(data) or saved_marker.get('md5Checksum') != hashlib.md5(data).hexdigest():
            raise ValueError('Uploaded lookup pointer size/checksum mismatch.')
        return marker


def authenticate_drive_receipts(data_root, expected_candidate_sha256):
    """Authenticate on the main Colab thread; never print/save credential tokens."""
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    root = Path(data_root).resolve()
    print('Drive receipts: authenticate API; receipts read-only, lookup checkpoints saved in retrieval/content_lookups', flush=True)
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    client = DriveReceiptSource(lambda: AuthorizedSession(copy.copy(credentials)), data_root, root)
    client.verify_candidate(expected_candidate_sha256)
    print('Drive receipts: API source verified against the same frozen candidate', flush=True)
    return client

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
    from scripts.drive_receipts_api import authenticate_drive_receipts, LookupPublishUnavailable


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


def save_existing_drive_receipt_lookup(receipt_api, checkout, work_dir, *, progress=print):
    """Save an older cell's committed local lookup, without listing receipts or encoding."""
    from types import SimpleNamespace
    config = read_json(Path(checkout) / 'configs/retrieval_full.json')
    relative = Path(resources_relative(receipt_api.candidate, config)) / 'bge/config.json'
    header = receipt_api.relative_json(relative.as_posix())
    if (header is None or header.get('input_manifest_sha256') != receipt_api.active['index_inputs_manifest_sha256']
        or header.get('dimension') != 1024 or header.get('format') != 'content-reference-v1'
        or any(header.get('encoder', {}).get(k) != v for k,v in config['dense'].items())):
        raise ValueError('Saved lookup producer differs from the current frozen BGE view.')
    encoder = SimpleNamespace(identity=header['encoder'], dimension=header['dimension'])
    lookup = ResumableContentBlocks(receipt_api.requested_root, encoder, work_dir,
        progress=progress, receipt_api=receipt_api, inventory=False)
    lookup.close()
    if lookup._snapshot_dirty:
        raise LookupPublishUnavailable('Lookup remains local; Drive save was not confirmed. Keep this runtime.')


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


if _vmb_previous_receipts is None:
    DRIVE_RECEIPTS = authenticate_drive_receipts(DATA_ROOT, PREPARATION["candidate_manifest_sha256"])
else:
    DRIVE_RECEIPTS = DriveReceiptSource(_vmb_previous_receipts.session_factory,
        _vmb_previous_receipts.requested_root, _vmb_previous_receipts.data_root,
        data_folder_id=_vmb_previous_receipts.folder_id)
    DRIVE_RECEIPTS.verify_candidate(PREPARATION["candidate_manifest_sha256"])
save_existing_drive_receipt_lookup(DRIVE_RECEIPTS, CHECKOUT, WORK_DIR)
print("Đã xác nhận lưu bảng tra lên Drive. Không quét receipt, không tính embedding hoặc dựng catalog.",flush=True)
