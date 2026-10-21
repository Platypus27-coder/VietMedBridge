"""Collect all available BTC URL content from every AI Guru archive version.

Read metadata members through Drive Range, never unpack raw TARs. Original
snapshot/hash provenance is preserved; current BTC membership is a separate join.
This collects extraction candidates, not silently promoted chunks or embeddings.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import tarfile
import tempfile
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from urllib3.exceptions import ProtocolError, ReadTimeoutError
from vietmedbridge.artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, verify_file, utc_now
from vietmedbridge.content_embeddings import checked, seal
from vietmedbridge.external_import import _hold, MEMBERS
from vietmedbridge.isolated_intake import Window, DriveRange, DriveSource

FOLDER = '1P2ke-yIDwj443R_GMegaiH9RACf-Gzth'
OFFICIAL_SHA = '87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7'
CAMPAIGN = 'btc-all-ai-guru-71be0e12-v1'
RAW_PATH_POLICY = 'native-data-raw-and-portable-raw-v2'


def raw_record_problem(row, leaf_format, descriptor):
    """Validate provenance without confusing a path/schema problem with HTTP failure."""
    if row['status'] != 'SUCCESS_RAW':
        return row['status'] or 'MISSING_CRAWL_RESULT'
    if not isinstance(row['raw_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',row['raw_sha256']):
        return 'RAW_RECORD_INVALID_SHA256'
    path=row['raw_path']
    prefixes=('raw/','data/raw/') if leaf_format=='native' else ('raw/',)
    if (not isinstance(path,str) or not path.startswith(prefixes) or path.endswith('/') or '\\' in path or '\x00' in path
            or PurePosixPath(path).is_absolute() or '..' in path.split('/')):
        return 'RAW_RECORD_INVALID_LOCATOR'
    if not row['crawl_snapshot']:
        return 'RAW_RECORD_MISSING_SNAPSHOT'
    if leaf_format=='native' and row['crawl_snapshot'] != descriptor.get('snapshot_id'):
        return 'ARCHIVE_SNAPSHOT_BINDING_MISMATCH'
    return None


def repair_leaf_id(base_id):
    return digest_json({'original_leaf':base_id,'raw_path_policy':RAW_PATH_POLICY})


class CollectionDriveError(RuntimeError):
    def __init__(self, status, reasons, file_id, byte_range, *, cooldown=0, pause=False):
        self.details = {'http_status':status, 'reasons':list(reasons), 'file_id':file_id,
                        'byte_range':byte_range, 'cooldown_seconds':cooldown, 'pause_campaign':pause}
        super().__init__(f'Drive HTTP {status} | reason={",".join(reasons) or "UNKNOWN"} | '
            f'file={file_id} | Range={byte_range} | cooldown={cooldown:g}s | checkpoint retained')


def drive_reasons(response):
    # Error bodies only, bounded; never print tokens, headers or private bodies.
    try:
        body=json.loads(response.raw.read(65536))
        error=body.get('error',{}) if isinstance(body,dict) else {}
        if not isinstance(error,dict):
            return ()
        entries=[*(error.get('errors') or []),*(error.get('details') or [])]
        return tuple(sorted({r['reason'] for r in entries if isinstance(r,dict)
            and isinstance(r.get('reason'),str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,99}',r['reason'])}))
    except (ValueError,TypeError,AttributeError):
        return ()


def drive_delay(response, fallback):
    value=response.headers.get('Retry-After','').strip()
    if value.isdigit():
        return max(fallback,int(value))
    if value:
        try:
            until=parsedate_to_datetime(value)
            if until.tzinfo is None:
                until=until.replace(tzinfo=timezone.utc)
            return max(fallback,(until-datetime.now(timezone.utc)).total_seconds())
        except (ValueError,TypeError,OverflowError):
            pass
    return fallback


class CollectionDriveRange(DriveRange):
    """Retry the identical Range only for explicit transient provider failures."""
    def _get_block(self,start):
        end=min(self.size,start+self.block_size)-1
        requested=f'bytes={start}-{end}'
        # Google may send legacy error.errors and google.rpc.ErrorInfo together.
        # Both spellings must be recognized; permission/quota codes stay denied.
        rate_reasons={'rateLimitExceeded','userRateLimitExceeded',
                      'RATE_LIMIT_EXCEEDED','USER_RATE_LIMIT_EXCEEDED'}
        for attempt in range(4):
            delay=2**attempt
            try:
                with self.session.get(self.url,params={'alt':'media','supportsAllDrives':'true'},
                        headers={'Range':requested,'Accept-Encoding':'identity'},stream=True,timeout=(15,90)) as response:
                    status=response.status_code
                    if status>=400:
                        reasons=drive_reasons(response)
                        retryable=status in (429,500,502,503,504) or (status==403 and bool(reasons)
                            and set(reasons)<=rate_reasons)
                        delay=drive_delay(response,delay)
                        print('DRIVE_RANGE_ERROR:',status,'| reason='+(','.join(reasons) or 'UNKNOWN'),
                              '|',self.file_id,'|',requested,flush=True)
                        if not retryable or attempt==3 or delay>60:
                            pause=status==429 or (bool(reasons) and set(reasons)<=rate_reasons|{'dailyLimitExceeded'})
                            raise CollectionDriveError(status,reasons,self.file_id,requested,cooldown=delay,pause=pause)
                    else:
                        expected=f'bytes {start}-{end}/{self.size}'
                        if status!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError('Drive did not honor the bounded Range request; no full TAR consumed.')
                        payload=response.raw.read(end-start+2)
                        if len(payload)!=end-start+1:
                            raise EOFError('Incomplete Drive range; unpublished block retained for retry.')
                        self.block_start,self.block=start,payload
                        return
            except (requests.ConnectionError,requests.Timeout,requests.exceptions.ChunkedEncodingError,
                    ProtocolError,ReadTimeoutError,EOFError):
                if attempt==3:
                    raise
            print(f'DRIVE_RANGE_RETRY {attempt+1}/3 | same {requested} | wait {delay:g}s',flush=True)
            time.sleep(delay)


class CollectionDriveSource(DriveSource):
    def open(self,file_id,metadata):
        return CollectionDriveRange(self.session,file_id,metadata)


def fetch_url(value):
    return value.strip().split('#', 1)[0]


def inventory(session, folder=FOLDER):
    result, visited, queue = [], set(), [folder]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        token = None
        while True:
            params = {'q': f"'{current}' in parents and trashed=false", 'pageSize': 1000,
                      'fields': 'nextPageToken,files(id,name,mimeType,size,modifiedTime,md5Checksum,shortcutDetails)',
                      'supportsAllDrives': 'true', 'includeItemsFromAllDrives': 'true'}
            if token:
                params['pageToken'] = token
            response = session.get('https://www.googleapis.com/drive/v3/files', params=params, timeout=90)
            response.raise_for_status()
            body = response.json()
            for item in body.get('files', []):
                if item['mimeType'] == 'application/vnd.google-apps.shortcut':
                    target = item['shortcutDetails']
                    response = session.get('https://www.googleapis.com/drive/v3/files/'+target['targetId'], params={
                        'fields': 'id,name,mimeType,size,modifiedTime,md5Checksum', 'supportsAllDrives': 'true'}, timeout=90)
                    response.raise_for_status()
                    item = response.json()
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    queue.append(item['id'])
                elif item['name'].lower().endswith('.tar'):
                    if not item.get('size') or not item.get('md5Checksum'):
                        raise ValueError('Archive has no immutable byte identity.')
                    result.append(item)
            token = body.get('nextPageToken')
            if not token:
                break
    return sorted({r['id']: r for r in result}.values(), key=lambda r: (r.get('modifiedTime', ''), r['id']), reverse=True)


def identity(remote):
    return {k: remote[k] for k in ('id', 'name', 'size', 'modifiedTime', 'md5Checksum')}


@contextmanager
def metadata_leaves(reader, remote, retained, work):
    """Read native leaf metadata, or enumerate every child TAR of a container.

    A native source's four manifests are its authoritative URL inventory. Its
    raw members are payloads, not additional sources, and must not be enumerated
    after the manifests are complete. Containers have no native manifests; walk
    their complete member list so later resume TARs are still discovered.
    """
    with reader.open(remote['id'], identity(remote)) as stream:
        if hasattr(stream, 'block_size'):
            stream.block_size = 65536
        leaves = []
        def scan(start, size, name, depth=0):
            if depth > 3:
                raise ValueError('Nested archive depth exceeds the bounded scanner.')
            found, nested = {}, []
            headers, logged = 0, time.monotonic()
            with tarfile.open(fileobj=Window(stream, start, size), mode='r:') as archive:
                for member in archive:
                    headers += 1
                    if time.monotonic()-logged >= 15:
                        print('ARCHIVE_HEADER_PROGRESS:',name,'|',headers,'headers | metadata',len(found),'/4',flush=True)
                        logged = time.monotonic()
                    if not member.isfile():
                        continue
                    label = ('dataset' if member.name == MEMBERS['dataset'] else
                             'crawl' if member.name == MEMBERS['crawl'] else
                             'extracted' if member.name == MEMBERS['extracted'] else
                             'frontier' if re.fullmatch(r'data/crawl_shards/shard_[0-9]+[.]parquet', member.name) else None)
                    if label:
                        if nested:
                            raise ValueError('Mixed native/container layout; do not choose an arbitrary source.')
                        if label in found or not 0 < member.size <= 2*1024**3:
                            raise ValueError('Duplicate/oversize metadata member; no arbitrary version selected.')
                        found[label] = {'path': member.name, 'offset': start+member.offset_data,
                                        'bytes': member.size, 'sha256': member.pax_headers.get('vibiomir.sha256')}
                        if len(found) == 4:
                            print('NATIVE_METADATA_READY:',name,'|',headers,
                                  'headers | 4/4 manifests; raw member listing skipped',flush=True)
                            break
                    elif member.name.endswith('.tar') and not member.name.startswith('raw/'):
                        if found:
                            raise ValueError('Mixed native/container layout; source requires inspection.')
                        nested.append((start+member.offset_data, member.size, name+'!'+member.name))
            if found:
                if set(found) != {'dataset','crawl','extracted','frontier'}:
                    raise ValueError('Incomplete leaf metadata; preserve this archive for inspection.')
                leaves.append({'name': name, 'format': 'native', 'members': found})
            for entry in nested:
                scan(*entry, depth+1)
        if remote['id'] == retained['drive_file_id']:
            if int(remote['size']) != retained['archive_bytes']:
                raise ValueError('Pinned retained package changed; do not guess member offsets.')
            members = retained['members']
            for item in members.values():
                stream.seek(item['offset']-512)
                header = tarfile.TarInfo.frombuf(stream.read(512), 'utf-8', 'surrogateescape')
                if header.name != item['path'] or header.size != item['bytes'] or not header.isfile():
                    raise ValueError('Retained member header changed.')
            leaves.append({'name': remote['name'], 'format': 'retained', 'members': members})
        else:
            scan(0, int(remote['size']), remote['name'])
        if not leaves:
            raise ValueError('No supported metadata leaf discovered in this archive.')
        yield stream, leaves


def stage_member(stream, item, target):
    if hasattr(stream, 'block_size'):
        stream.block_size = 8*1024**2
        stream.block_start, stream.block = -1, b''
    stream.seek(item['offset'])
    digest, remaining = hashlib.sha256(), item['bytes']
    logged = time.monotonic()
    with Path(target).open('wb') as out:
        while remaining:
            payload = stream.read(min(8*1024**2, remaining))
            if not payload:
                raise EOFError('Truncated metadata member.')
            out.write(payload)
            digest.update(payload)
            remaining -= len(payload)
            if time.monotonic()-logged>=15 or remaining==0:
                print('METADATA_TRANSFER',item['path'],'|',round((item['bytes']-remaining)/1024**2,1),
                      '/',round(item['bytes']/1024**2,1),'MiB',flush=True)
                logged=time.monotonic()
    if item.get('sha256') and item['sha256'] != digest.hexdigest():
        raise ValueError('Source metadata checksum differs from the TAR receipt.')
    return digest.hexdigest()


def cached_member(stream,item,target,cache_dir):
    """Keep completed metadata members on Drive even if a later request fails.

    An in-flight Range retries in place. An incomplete member without a receipt
    may restart after a hard failure; complete members never reopen the TAR.
    """
    cache_dir=Path(cache_dir)
    receipt=cache_dir/'receipt.json'
    payload=cache_dir/'metadata.bin'
    binding=digest_json(item)
    if receipt.is_file():
        saved=checked(read_json(receipt))
        if saved['binding']!=binding or saved['bytes']!=item['bytes']:
            raise ValueError('Metadata member cache belongs to another source version.')
        shutil.copyfile(payload,target)
        verify_file(target,saved['sha256'])
        if Path(target).stat().st_size!=item['bytes']:
            raise ValueError('Cached metadata member length differs from its TAR header.')
        if item.get('sha256') and saved['sha256']!=item['sha256']:
            raise ValueError('Cached metadata differs from source TAR checksum.')
        print('REUSED_METADATA_FROM_DRIVE:',item['path'],'| no TAR payload read',flush=True)
        return saved['sha256']
    checksum=stage_member(stream,item,target)
    if publish_file(target,payload)!=checksum:
        raise ValueError('Metadata cache publication checksum mismatch.')
    atomic_json(receipt,seal({'binding':binding,'bytes':item['bytes'],'sha256':checksum}))
    print('METADATA_CHECKPOINT_SAVED_ON_DRIVE:',item['path'],flush=True)
    return checksum


class CollectAll:
    def __init__(self, project, work, *, official_sha=OFFICIAL_SHA):
        self.project, self.work = Path(project).resolve(), Path(work).resolve()
        self.root = self.project/'data_staging'/CAMPAIGN
        self.root.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.binding = {'schema': 'whole-folder-content-collection-v1', 'official_sha256': official_sha,
                        'folder_id': FOLDER, 'normalization': 'trim-and-remove-fragment-only'}
        self.database = self.work/'collection.sqlite'
        self.pointer = self.root/'checkpoint.json'
        if not self.database.exists() and self.pointer.exists():
            saved = checked(read_json(self.pointer))
            if saved['binding'] != self.binding:
                raise ValueError('Collection BTC snapshot changed.')
            payload = self.root/saved['path']
            verify_file(payload, saved['sha256'])
            print('RESTORE_COLLECTION: one SQL checkpoint; no old archives reopened', flush=True)
            with gzip.open(payload, 'rb') as source, self.database.open('wb') as out:
                shutil.copyfileobj(source, out, length=1024**2)
            verify_file(self.database, saved['sqlite_sha256'])
        self.con = sqlite3.connect(self.database)
        self.con.execute('PRAGMA cache_size=-32768')
        self.con.execute('PRAGMA synchronous=FULL')
        self.con.execute('CREATE TABLE IF NOT EXISTS contract(value TEXT)')
        row = self.con.execute('SELECT value FROM contract').fetchone()
        if row and row[0] != digest_json(self.binding):
            raise ValueError('Retained local collection belongs to another BTC snapshot.')
        if not row:
            self.con.execute('INSERT INTO contract VALUES (?)', [digest_json(self.binding)])
        self.con.executescript('''
            CREATE TABLE IF NOT EXISTS official(id INTEGER PRIMARY KEY,url TEXT NOT NULL,fetch TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS official_fetch ON official(fetch);
            CREATE TABLE IF NOT EXISTS content(fetch TEXT PRIMARY KEY,seen INTEGER DEFAULT 0,raw INTEGER DEFAULT 0,
                existing TEXT,candidate TEXT,reason TEXT);
            CREATE TABLE IF NOT EXISTS completed(id TEXT PRIMARY KEY,kind TEXT,receipt TEXT);
        ''')
        columns={r[1] for r in self.con.execute('PRAGMA table_info(content)')}
        if 'raw_status' not in columns:
            self.con.execute('ALTER TABLE content ADD COLUMN raw_status INTEGER DEFAULT 0')
            self.con.execute('UPDATE content SET raw_status=raw')
        self.con.commit()

    def saved(self, identifier):
        row = self.con.execute('SELECT receipt FROM completed WHERE id=?', [identifier]).fetchone()
        return json.loads(row[0]) if row else None

    def commit(self, identifier, kind, receipt):
        self.con.execute('INSERT INTO completed VALUES (?,?,?)', [identifier, kind, json.dumps(receipt)])
        self.con.commit()
        self.snapshot()

    def snapshot(self):
        copy = self.work/'backup.sqlite'
        destination = sqlite3.connect(copy)
        try:
            self.con.backup(destination)
        finally:
            destination.close()
        payload = self.work/'backup.sqlite.gz'
        with copy.open('rb') as source, gzip.open(payload, 'wb', compresslevel=1) as out:
            shutil.copyfileobj(source, out, length=1024**2)
        checksum = sha256_file(payload)
        target = self.root/('sql-'+checksum+'.gz')
        publish_file(payload, target)
        atomic_json(self.pointer, seal({'binding': self.binding, 'path': target.name, 'sha256': checksum,
            'sqlite_sha256': sha256_file(copy), 'updated_at': utc_now()}))
        copy.unlink()
        payload.unlink()
        print('COLLECTION_SQL_CHECKPOINT_SAVED_ON_DRIVE', flush=True)

    def official(self, path):
        if self.saved('official'):
            return
        verify_file(path, self.binding['official_sha256'])
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=['id','url']):
                rows = [(r['id'], r['url'], fetch_url(r['url'])) for r in batch.to_pylist()]
                self.con.executemany('INSERT INTO official VALUES (?,?,?)', rows)
                self.con.executemany('INSERT OR IGNORE INTO content(fetch) VALUES (?)', [(r[2],) for r in rows])
            self.commit('official', 'reference', {'sha256': self.binding['official_sha256']})
        except BaseException:
            self.con.rollback()
            raise

    def existing(self, path, checksum, label):
        identifier = 'existing:'+checksum
        if self.saved(identifier):
            print('SKIP_EXISTING_ALIAS_MAPPING:', label, flush=True)
            return
        verify_file(path, checksum)
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=['doc_id','url']):
                rows = [(json.dumps({'mapping': label, 'original_doc_id': r['doc_id'], 'original_url': r['url']}), fetch_url(r['url']))
                        for r in batch.to_pylist()]
                self.con.executemany('UPDATE content SET existing=coalesce(existing,?) WHERE fetch=?', rows)
            self.commit(identifier, 'existing', {'mapping': label, 'sha256': checksum})
        except BaseException:
            self.con.rollback()
            raise

    def leaf(self, stream, remote, leaf, *, version_check=None, repair=False, metadata_paths=None):
        base_id = digest_json({'remote': identity(remote), 'leaf': leaf})
        identifier = repair_leaf_id(base_id) if repair else base_id
        if self.saved(identifier):
            print('SKIP_COMPLETED_ARCHIVE_LEAF:', leaf['name'], '| receipt only', flush=True)
            return
        print('COLLECT_ARCHIVE_LEAF:', leaf['name'], '| metadata only, no raw TAR copy', flush=True)
        with tempfile.TemporaryDirectory(prefix='metadata-', dir=self.work) as temporary:
            local = Path(temporary)
            staged_checksums={}
            for kind, item in leaf['members'].items():
                if leaf['format'] == 'retained' and kind == 'mapping':
                    continue
                print('READ_METADATA', kind, '|', round(item['bytes']/1024**2, 1), 'MiB', flush=True)
                destination=local/(kind+'.'+('json' if kind == 'dataset' else 'parquet'))
                if metadata_paths is not None:
                    reference,checksum=metadata_paths[kind]
                    shutil.copyfile(reference,destination)
                    verify_file(destination,checksum)
                    if destination.stat().st_size!=item['bytes']:
                        raise ValueError('Saved metadata length differs from its source receipt.')
                    staged_checksums[kind]=checksum
                    print('REPAIR_METADATA_FROM_DRIVE:',kind,'| no TAR access',flush=True)
                else:
                    staged_checksums[kind]=cached_member(stream,item,destination,
                        self.root/'metadata_cache'/base_id/kind)
            descriptor = read_json(local/'dataset.json')
            con = duckdb.connect()
            con.execute("SET memory_limit='1GB'")
            con.execute('SET threads=1')
            con.execute('SET preserve_insertion_order=false')
            con.execute('SET temp_directory=?', [str(local/'spill')])
            try:
                if leaf['format'] == 'native':
                    for kind in ('frontier','crawl','extracted'):
                        con.read_parquet(str(local/(kind+'.parquet'))).create_view(kind)
                        duplicates=con.execute(f'SELECT count(*)-count(DISTINCT crawl_url_id) FROM {kind}').fetchone()[0]
                        if duplicates:
                            raise ValueError('Duplicate crawl_url_id in source metadata; no arbitrary row selected.')
                    query = '''SELECT f.fetch_url,c.status,c.raw_sha256,c.raw_path,c.raw_size_bytes,c.final_url,
                        c.content_type,c.snapshot_id AS crawl_snapshot,e.snapshot_id AS extract_snapshot,
                        e.raw_sha256 AS extract_raw_sha,e.extract_status,e.text,e.title,e.char_count,
                        to_json(c) AS crawl_json,to_json(e) AS extracted_json
                        FROM frontier f LEFT JOIN crawl c USING(crawl_url_id) LEFT JOIN extracted e USING(crawl_url_id)'''
                    if repair:
                        # Previously failed HTTP records are already in the SQL checkpoint.
                        # Only successful raw records can be affected by the locator bug.
                        query += " WHERE c.status='SUCCESS_RAW'"
                else:
                    con.read_parquet(str(local/'raw_catalog.parquet')).create_view('raw')
                    con.read_parquet(str(local/'documents.parquet')).create_view('docs')
                    for kind in ('raw','docs'):
                        if con.execute(f'SELECT count(*)-count(DISTINCT crawl_url_id) FROM {kind}').fetchone()[0]:
                            raise ValueError('Duplicate retained crawl_url_id; source requires inspection.')
                    query = '''SELECT r.fetch_url,'SUCCESS_RAW' AS status,r.raw_sha256,r.raw_path,r.raw_size_bytes,r.final_url,
                        r.content_type,r.original_snapshot_id AS crawl_snapshot,e.snapshot_id AS extract_snapshot,
                        e.raw_sha256 AS extract_raw_sha,e.extract_status,e.text,e.title,e.char_count,
                        r.original_crawl_record_json AS crawl_json,to_json(e) AS extracted_json,r.quality_tier AS raw_quality_tier
                        FROM raw r LEFT JOIN docs e USING(crawl_url_id)'''
                output = local/'recovered.parquet'
                schema = pa.schema([('fetch_url',pa.string()),('source_file_id',pa.string()),('leaf',pa.string()),
                    ('original_snapshot_id',pa.string()),('source_text_sha256',pa.string()),
                    ('crawl_json',pa.string()),('extracted_json',pa.large_string())])
                count, matched, raw_count, held = 0, 0, 0, {}
                logged=time.monotonic()
                with pq.ParquetWriter(output, schema, compression='zstd') as writer:
                    for batch in con.execute(query).fetch_record_batch(128):
                        candidates = []
                        for row in batch.to_pylist():
                            url = row['fetch_url']
                            state = self.con.execute('SELECT existing,candidate,raw FROM content WHERE fetch=?', [url]).fetchone()
                            if state is None:
                                continue
                            matched += 1
                            original = row['crawl_snapshot']
                            raw_problem=raw_record_problem(row,leaf['format'],descriptor)
                            raw_ok=raw_problem is None
                            raw_count += bool(raw_ok)
                            self.con.execute('UPDATE content SET seen=1,raw=max(raw,?),raw_status=max(raw_status,?) WHERE fetch=?',
                                [int(raw_ok),int(row['status']=='SUCCESS_RAW'),url])
                            if state[0] is not None or state[1] is not None:
                                continue
                            reason = None
                            if not raw_ok:
                                reason = raw_problem
                            elif row['extract_raw_sha'] != row['raw_sha256'] or row['extract_snapshot'] != original:
                                reason = 'MISSING_OR_UNBOUND_EXTRACTION'
                            elif row['extract_status'] == 'EXTRACT_SUCCESS' and len(row['text'] or '') != row['char_count']:
                                reason = 'INVALID_SOURCE_LENGTH'
                            else:
                                extra = json.loads(row['extracted_json']) if row['extracted_json'] else {}
                                gate_row = {**row, 'url':url, 'source_issue':extra.get('source_issue')}
                                hold = _hold(gate_row, 2_000_000)
                                reason = ':'.join(hold) if hold else None
                                if leaf['format'] == 'retained' and any(str(t).upper() in ('LOW','QUARANTINE')
                                    for t in (extra.get('quality_tier'),row.get('raw_quality_tier'))):
                                    reason = 'RETAINED_QUALITY_REVIEW'
                            if reason:
                                held[reason] = held.get(reason,0)+1
                                if raw_ok or not state[2]:
                                    self.con.execute('UPDATE content SET reason=? WHERE fetch=? AND candidate IS NULL', [reason,url])
                                continue
                            reference = {'path': 'parts/'+identifier+'.parquet', 'row':count,'receipt_id':identifier,
                                         'source_file_id':remote['id'], 'leaf':leaf['name'], 'original_snapshot_id':original}
                            candidates.append({'fetch_url':url,'source_file_id':remote['id'],'leaf':leaf['name'],
                                'original_snapshot_id':original,'source_text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
                                'crawl_json':row['crawl_json'],'extracted_json':row['extracted_json']})
                            self.con.execute('UPDATE content SET candidate=?,reason=NULL WHERE fetch=?', [json.dumps(reference),url])
                            count += 1
                        if candidates:
                            writer.write_table(pa.Table.from_pylist(candidates,schema=schema))
                        if time.monotonic()-logged>=15:
                            print('COLLECT_PROGRESS',leaf['name'],'| matched',matched,'| recovery candidates',count,flush=True)
                            logged=time.monotonic()
                target = self.root/'parts'/(identifier+'.parquet')
                if version_check is not None:
                    version_check()
                checksum = publish_file(output,target)
                self.commit(identifier,'raw_repair' if repair else 'leaf',{'archive':identity(remote),'leaf':leaf,'candidates':count,
                    'matched_rows':matched,'raw_rows':raw_count,'held':held,'path':target.relative_to(self.root).as_posix(),
                    'sha256':checksum,'metadata_sha256':staged_checksums,'original_snapshot_provenance_preserved':True,
                    'raw_path_policy':RAW_PATH_POLICY,'supersedes':base_id if repair else None,
                    'row_scope':'successful_crawl_records_only' if repair else 'full_source_frontier'})
                print('COLLECT_LEAF_SAVED:', count, 'new candidate URLs |', leaf['name'], flush=True)
            except BaseException:
                self.con.rollback()
                raise
            finally:
                con.close()

    def pending_raw_repairs(self):
        result=[]
        for identifier,raw in self.con.execute("SELECT id,receipt FROM completed WHERE kind='leaf' ORDER BY id"):
            receipt=json.loads(raw)
            if (receipt['leaf']['format']=='native' and receipt.get('raw_path_policy')!=RAW_PATH_POLICY
                    and not self.saved(repair_leaf_id(identifier))):
                result.append((identifier,receipt))
        return result

    def _repair_metadata_index(self, *, refresh=False):
        """Read only bounded metadata receipt patterns, never corpus/index payloads."""
        leaves=[(r[0],json.loads(r[1])) for r in self.con.execute(
            "SELECT id,receipt FROM completed WHERE kind='leaf' ORDER BY id")]
        binding=digest_json([(identifier,r['metadata_sha256']) for identifier,r in leaves])
        marker=self.root/'raw_path_repair'/'metadata_index.json'
        if marker.is_file() and not refresh:
            saved=checked(read_json(marker))
            if saved['binding']==binding:
                print('REUSED_REPAIR_METADATA_INDEX | receipt JSON only',flush=True)
                return saved['paths']
        paths={}
        print('REPAIR_METADATA_INDEX: saved receipt JSON only; no corpus/index/TAR scan',flush=True)
        def add(checksum,path):
            if not isinstance(checksum,str) or not re.fullmatch('[0-9a-f]{64}',checksum):
                return
            path=Path(path).resolve()
            if not path.is_relative_to(self.project):
                raise ValueError('Metadata reference escapes the project.')
            relative=path.relative_to(self.project).as_posix()
            values=paths.setdefault(checksum,[])
            if relative not in values:
                values.append(relative)
        for identifier,receipt in leaves:
            for kind,checksum in receipt['metadata_sha256'].items():
                add(checksum,self.root/'metadata_cache'/identifier/kind/'metadata.bin')
        patterns=(
            'data/processed/*/external_source/source.json',
            'data_staging/btc-new-71be0e12-*/sources/*/metadata/metadata.json',
            'data_staging/btc-new-71be0e12-*/*/sources/*/metadata/metadata.json',
            'data_staging/btc-new-71be0e12-*/parallel_cpu4/jobs/*/sources/*/metadata/metadata.json',
        )
        visited=set()
        logged=time.monotonic()
        for pattern in patterns:
            for manifest in self.project.glob(pattern):
                manifest=manifest.resolve()
                if not manifest.is_relative_to(self.project):
                    raise ValueError('Metadata receipt escapes the project.')
                if manifest in visited:
                    continue
                visited.add(manifest)
                saved=read_json(manifest)
                for item in saved.get('files',{}).values():
                    if isinstance(item,dict) and (item.get('path') or item.get('name')):
                        add(item.get('sha256'),manifest.parent/(item.get('path') or item['name']))
                if time.monotonic()-logged>=15:
                    print('REPAIR_METADATA_INDEX:',len(visited),'metadata receipts inspected',flush=True)
                    logged=time.monotonic()
        atomic_json(marker,seal({'binding':binding,'paths':paths}))
        print('REPAIR_METADATA_INDEX_READY:',len(visited),'saved metadata receipts; no corpus scan',flush=True)
        return paths

    def repair_raw_paths(self):
        """Replay successful records from saved metadata, without opening any TAR.

        Original receipts/candidate parts are retained. Each repaired leaf gets
        its own durable SQL receipt and immutable recovery part; a new VM skips
        committed repairs. Missing saved metadata is reported, never fetched.
        """
        previous=checked(read_json(self.root/'collection_report.json'))
        files=previous.get('archive_inventory') or [json.loads(r[0])['remote'] for r in self.con.execute(
            "SELECT receipt FROM completed WHERE kind='archive' ORDER BY id")]
        if (previous['archive_errors'] or len(files)!=previous['archive_files']
                or any(not self.saved('archive:'+digest_json(identity(r))) for r in files)):
            raise ValueError('Complete the source inventory before repairing raw classification.')
        pending=self.pending_raw_repairs()
        if not pending and previous.get('raw_path_classification',{}).get('state')=='CURRENT':
            print('RAW_PATH_REPAIR_ALREADY_COMPLETE | report only; no payload reads',flush=True)
            return previous
        index=self._repair_metadata_index() if pending else {}
        errors=[]
        for number,(identifier,receipt) in enumerate(pending,1):
            print(f'RAW_PATH_REPAIR {number}/{len(pending)} | {receipt["leaf"]["name"]}',flush=True)
            references={}
            missing=[]
            for kind,item in receipt['leaf']['members'].items():
                checksum=receipt['metadata_sha256'][kind]
                for relative in index.get(checksum,[]):
                    path=(self.project/relative).resolve()
                    if not path.is_relative_to(self.project):
                        raise ValueError('Saved metadata index escapes the project.')
                    if path.is_file():
                        references[kind]=(path,checksum)
                        break
                else:
                    missing.append({'kind':kind,'sha256':checksum,'bytes':item['bytes']})
            if missing:
                errors.append({'leaf':receipt['leaf']['name'],'leaf_id':identifier,'error':'SAVED_METADATA_MISSING',
                    'missing':missing,'tar_download_attempted':False})
                print('RAW_REPAIR_NEEDS_METADATA:',receipt['leaf']['name'],'| missing',
                      ','.join(r['kind'] for r in missing),'| no TAR reopened',flush=True)
                continue
            try:
                self.leaf(None,receipt['archive'],receipt['leaf'],repair=True,metadata_paths=references)
            except Exception as error:
                self.con.rollback()
                errors.append({'leaf':receipt['leaf']['name'],'leaf_id':identifier,'error':str(error)[:500],
                               'tar_download_attempted':False})
        # A receipt may have committed locally before a transient snapshot write
        # failed. Publish the final SQL snapshot before declaring completion.
        self.snapshot()
        remaining={identifier for identifier,_ in self.pending_raw_repairs()}
        repair={'state':'RAW_PATH_REPAIR_COMPLETE' if not remaining else 'RAW_PATH_REPAIR_INCOMPLETE',
            'policy':RAW_PATH_POLICY,'errors':[e for e in errors if e['leaf_id'] in remaining],
            'recovered_warnings':[e for e in errors if e['leaf_id'] not in remaining],
            'tar_accessed':False,'updated_at':utc_now()}
        atomic_json(self.root/'raw_path_repair'/'repair_report.json',seal(repair))
        return self.report(files,[],raw_repair=repair)

    def report(self, files, errors, *, raw_repair=None):
        # Count official IDs and distinct fetch URLs separately. Failed records never count as usable text.
        state = "CASE WHEN c.existing IS NOT NULL THEN 'REUSED_VALIDATED_CONTENT' WHEN c.candidate IS NOT NULL THEN 'RECOVERED_EXTRACTION_PENDING_CHUNK' WHEN c.seen=0 THEN 'NOT_FOUND_IN_ARCHIVES' WHEN c.raw=1 THEN 'RAW_AVAILABLE_NEEDS_EXTRACTION' WHEN c.raw_status=1 THEN 'RAW_METADATA_REVIEW' ELSE 'CRAWL_NOT_SUCCESSFUL' END"
        query = f'SELECT o.id,o.url,o.fetch,{state} AS state,c.existing,c.candidate,c.reason FROM official o JOIN content c ON o.fetch=c.fetch'
        counts = dict(self.con.execute(f'SELECT state,count(*) FROM ({query}) GROUP BY state'))
        total = self.con.execute('SELECT count(*) FROM official').fetchone()[0]
        unique = self.con.execute('SELECT count(*) FROM content').fetchone()[0]
        seen,raw=self.con.execute('SELECT sum(c.seen),sum(c.raw) FROM official o JOIN content c ON o.fetch=c.fetch').fetchone()
        files_out={}
        for name, sql in [('btc_membership.csv',query),('missing_or_unusable.csv',f"SELECT * FROM ({query}) WHERE state NOT IN ('REUSED_VALIDATED_CONTENT','RECOVERED_EXTRACTION_PENDING_CHUNK')")]:
            import csv
            path = self.work/name
            cursor = self.con.execute(sql)
            with path.open('w',encoding='utf-8',newline='') as out:
                writer = csv.writer(out)
                writer.writerow([c[0] for c in cursor.description])
                while rows := cursor.fetchmany(8192):
                    writer.writerows(rows)
            files_out[name]={'path':name,'sha256':publish_file(path,self.root/name)}
        pending_repairs=len(self.pending_raw_repairs())
        recovery_parts=[]
        for identifier,kind,raw_receipt in self.con.execute(
                "SELECT id,kind,receipt FROM completed WHERE kind IN ('leaf','raw_repair')"):
            receipt=json.loads(raw_receipt)
            superseded=(kind=='leaf' and receipt['leaf']['format']=='native'
                        and self.saved(repair_leaf_id(identifier)) is not None)
            recovery_parts.append({**receipt,'classification_effective':not superseded,
                'superseded_classification_by':repair_leaf_id(identifier) if superseded else None})
        result = seal({'state':'ALL_FOLDER_SOURCES_COLLECTED' if not errors else 'FOLDER_COLLECTION_INCOMPLETE',
            'official_ids':total,'official_unique_fetch_urls':unique,'counts_by_official_id':counts,
            'official_ids_seen_in_archive_metadata':seen,'official_ids_with_successful_raw_record':raw,
            'all_official_ids_represented_in_archives':not errors and seen==total,
            'all_official_raw_records_successful':not errors and not pending_repairs and raw==total,
            'archive_files':len(files),'archive_inventory':files,'archive_errors':errors,'files':files_out,
            'recovery_parts':recovery_parts,
            'all_official_content_available':not errors and not pending_repairs and sum(counts.get(k,0) for k in ('REUSED_VALIDATED_CONTENT','RECOVERED_EXTRACTION_PENDING_CHUNK'))==total,
            'raw_path_classification':{'policy':RAW_PATH_POLICY,'state':'REPAIR_REQUIRED' if pending_repairs else 'CURRENT',
                                       'pending_leaves':pending_repairs},
            'counts_provisional':bool(pending_repairs),'raw_path_repair':raw_repair,
            'recovered_text_requires_standard_chunk_pipeline':True,'raw_payloads_rehashed':False,
            'submission_candidate_changed':False,'updated_at':utc_now()})
        atomic_json(self.root/'collection_report.json',result)
        return result

    def run(self, reader, files, retained):
        errors = []
        for index, remote in enumerate(files,1):
            print(f'ALL_AI_GURU {index}/{len(files)} | {remote["name"]}',flush=True)
            archive_id = 'archive:'+digest_json(identity(remote))
            if self.saved(archive_id):
                print('SKIP_COMPLETED_ARCHIVE | durable SQL receipt; no TAR reopened',flush=True)
                continue
            try:
                def version_check():
                    if identity(reader.metadata(remote['id'])) != identity(remote):
                        raise ValueError('Archive changed during collection; this leaf was not committed.')
                with metadata_leaves(reader,remote,retained,self.work) as (stream,leaves):
                    for leaf in leaves:
                        self.leaf(stream,remote,leaf,version_check=version_check)
                if identity(reader.metadata(remote['id'])) != identity(remote):
                    raise ValueError('Archive changed during collection.')
                self.commit(archive_id,'archive',{'remote':identity(remote),'leaves':len(leaves)})
            except Exception as error:
                self.con.rollback()
                record={'file':remote['name'],'id':remote['id'],'error':str(error)[:500]}
                if isinstance(error,CollectionDriveError):
                    record['drive_error']=error.details
                errors.append(record)
                print('ARCHIVE_NEEDS_INSPECTION:',remote['name'],type(error).__name__,str(error)[:200],flush=True)
                if isinstance(error,CollectionDriveError) and error.details['pause_campaign']:
                    print('DRIVE_COOLDOWN: campaign paused; remaining archives not requested. Resume same checkpoint later.',flush=True)
                    errors.extend({'file':r['name'],'id':r['id'],
                        'error':'NOT_ATTEMPTED_DRIVE_COOLDOWN','blocked_by':remote['id']}
                        for r in files[index:] if not self.saved('archive:'+digest_json(identity(r))))
                    break
        return self.report(files,errors)

    def close(self):
        self.con.close()
