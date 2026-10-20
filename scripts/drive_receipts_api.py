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
