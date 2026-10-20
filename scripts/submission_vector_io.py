"""Cache-aware submission I/O; immutable producer/model source files stay pinned."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tarfile
import threading
import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file, verify_file
from vietmedbridge.content_embeddings import ContentVectorParts as OriginalVectorParts, checked, seal
from vietmedbridge.embeddings import validate_vectors
from vietmedbridge.portable_embeddings import resources_relative

if 'LookupPublishUnavailable' not in globals():
    from scripts.drive_receipts_api import LookupPublishUnavailable


def read_declared_vector_manifest(data_root, candidate, inputs, config, *, family='bge', api=None, progress=print):
    """Use the sealed manifest, validating actual payloads only when consumed.

    None means the caller must discover partial BGE receipts through the native
    fallback. A declared incomplete Qwen view can only defer that optional leg.
    """
    if family not in ('bge', 'qwen'):
        raise ValueError('Unknown embedding family.')
    checked(inputs)
    view = Path(data_root) / resources_relative(candidate, config) / family
    def load(name):
        if api is not None:
            return api.relative_json((view / name).relative_to(Path(data_root)).as_posix())
        path = view / name
        return read_json(path) if path.is_file() else None
    identity = load('config.json')
    if identity is None:
        return None
    spec = config['dense' if family == 'bge' else 'second_dense']
    if (identity.get('input_manifest_sha256') != inputs['manifest_sha256']
        or identity.get('input_count') != inputs['input_count']
        or identity.get('format') != 'content-reference-v1'
        or identity.get('dimension') != (1024 if family == 'bge' else 4096)
        or any(identity.get('encoder', {}).get(k) != v for k,v in spec.items())
        or (family == 'qwen' and identity.get('encoder', {}).get('input_role') != 'corpus')):
        raise ValueError('Declared embedding producer differs from the frozen candidate.')
    saved = load('embeddings.json')
    if saved is None:
        return None
    checked(saved)
    if (any(saved.get(k) != v for k,v in identity.items())
        or saved.get('signature') != digest_json(identity)):
        raise ValueError('Declared vector manifest identity changed.')
    if family == 'bge' and saved.get('state') != 'COMPLETE':
        return None
    if saved.get('state') not in ('COMPLETE', 'IN_PROGRESS'):
        raise ValueError('Invalid vector manifest state.')
    sources = {p['start']:p for p in inputs['parts']}
    seen, last = set(), -1
    for part in saved['parts']:
        checked(part)
        source = sources.get(part['start'])
        if source is None or part['start'] <= last or part['start'] in seen:
            raise ValueError('Vector manifest coverage/order changed.')
        expected = {'signature':saved['signature'], 'start':source['start'], 'rows':source['rows'],
            'input_sha256':source['sha256'], 'path':f"map-{source['start']:012d}.parquet"}
        if any(part.get(k) != v for k,v in expected.items()) or len(part.get('sha256', '')) != 64:
            raise ValueError('Vector manifest part differs from frozen input order.')
        seen.add(part['start']); last = part['start']
    if saved['state'] == 'COMPLETE' and seen != set(sources):
        raise ValueError('Complete vector manifest has a coverage gap.')
    progress(f"{family.upper()} sealed manifest: {len(saved['parts']):,}/{len(sources):,} parts"
        ' | no per-file Drive verification scan; payload SHA checked on use')
    return view, saved


class VerifiedSubmissionIO:
    """Hash legacy local copies once, then reuse unchanged local file identities."""
    def __init__(self, work_dir, *, api=None, progress=print):
        self.work = Path(work_dir).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.api, self.progress, self.lock = api, progress, threading.RLock()
        directory = self.work / 'io_checkpoints'
        directory.mkdir(exist_ok=True)
        self.con = sqlite3.connect(directory / 'local_proofs.sqlite', check_same_thread=False, timeout=30)
        self.con.execute('PRAGMA journal_mode=WAL')
        self.con.execute('CREATE TABLE IF NOT EXISTS proofs(path TEXT PRIMARY KEY,sha TEXT,stamp TEXT,shape TEXT)')
        self.con.commit()

    def target(self, path):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(self.work):
            raise ValueError('Submission cache target escaped its local workspace.')
        return path

    @staticmethod
    def stamp(path):
        s = path.stat()
        return digest_json([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns])

    def known(self, path, checksum):
        path = self.target(path)
        if not path.is_file():
            return False
        with self.lock:
            row = self.con.execute('SELECT sha,stamp FROM proofs WHERE path=?', [str(path)]).fetchone()
        return row == (checksum, self.stamp(path))

    def record(self, path, checksum, *, shape=None):
        with self.lock:
            self.con.execute('INSERT OR REPLACE INTO proofs VALUES (?,?,?,?)',
                [str(path), checksum, self.stamp(path), None if shape is None else digest_json(list(shape))])
            self.con.commit()

    def prove(self, path, checksum):
        path = self.target(path)
        if self.known(path, checksum):
            return
        before = self.stamp(path)
        verify_file(path, checksum)  # Local SSD only, never rehash the Drive source.
        if before != self.stamp(path):
            raise ValueError('Local cache changed while it was checked.')
        self.record(path, checksum)

    def validate(self, path, checksum, values, dimension):
        with self.lock:
            old = self.con.execute('SELECT sha,stamp,shape FROM proofs WHERE path=?', [str(path)]).fetchone()
        expected = (checksum,self.stamp(path),digest_json(list(values.shape)))
        if values.ndim != 2 or values.shape[1] != dimension:
            raise ValueError('Cached vector dimension changed.')
        if old != expected:
            validate_vectors(values,len(values),dimension)
            self.record(path,checksum,shape=values.shape)

    def relative(self, source):
        if self.api is None:
            return None
        for root in (self.api.data_root,self.api.requested_root):
            try:
                value = Path(source).absolute().relative_to(root).as_posix()
            except ValueError:
                continue
            if '..' not in PurePosixPath(value).parts:
                return value
        return None

    def fetch(self, source, target, checksum, metadata=None):
        target = self.target(target)
        if target.is_file():
            self.prove(target,checksum)
            return False,0
        target.parent.mkdir(parents=True,exist_ok=True)
        stage = target.with_name('.'+uuid.uuid4().hex+'.part')
        try:
            relative = self.relative(source)
            if relative is not None and hasattr(self.api,'payload_metadata'):
                metadata = metadata or self.api.payload_metadata(relative)
                self.api.download(metadata,stage,expected_sha256=checksum)
            else:
                sha = hashlib.sha256()
                with Path(source).open('rb') as src, stage.open('wb') as dst:
                    for chunk in iter(lambda:src.read(1024*1024),b''):
                        sha.update(chunk); dst.write(chunk)
                if sha.hexdigest() != checksum:
                    raise ValueError('Copied source payload checksum changed.')
            os.replace(stage,target)
            self.record(target,checksum)
            return True,target.stat().st_size
        finally:
            stage.unlink(missing_ok=True)

    def stage(self, items, label, *, consume=None, reserve=None, workers=4):
        """A bounded window: consume each window before it can be evicted."""
        items = list(items)
        if not items:
            return
        prepared, missing_bytes = [],0
        for source,target,checksum in items:
            metadata = None
            if not Path(target).is_file():
                relative = self.relative(source)
                if relative is not None and hasattr(self.api,'payload_metadata'):
                    metadata = self.api.payload_metadata(relative)
                    missing_bytes += int(metadata['size'])
            prepared.append((source,target,checksum,metadata))
        visible = len(items)>=32 or any(not Path(item[1]).is_file() for item in prepared)
        if visible:
            self.progress(f'{label}: 0/{len(items):,} files | existing cache retained | missing payload {missing_bytes/2**20:.1f} MiB')
        started, new_bytes, reused = time.monotonic(),0,0
        # Ranking repeatedly gathers a few already-cached blocks. Keep this hot
        # path synchronous; no new threads or API metadata calls are necessary.
        pool = None
        try:
            for offset in range(0,len(prepared),workers):
                window = prepared[offset:offset+workers]
                if reserve is not None:
                    reserve(window)
                if all(Path(item[1]).is_file() for item in window):
                    outcomes = (self.fetch(*item) for item in window)
                else:
                    if pool is None:
                        pool = ThreadPoolExecutor(max_workers=workers)
                    futures = [pool.submit(self.fetch,*item) for item in window]
                    outcomes = (future.result() for future in futures)
                for index,(item,outcome) in enumerate(zip(window,outcomes,strict=True),offset):
                    fresh, size = outcome
                    new_bytes += size; reused += not fresh
                    if consume is not None:
                        consume(index,Path(item[1]))
                done = offset+len(window)
                remaining = max(0,missing_bytes-new_bytes)
                elapsed = time.monotonic()-started
                eta = f' | transfer ETA {remaining/(new_bytes/elapsed):.0f}s' if new_bytes and remaining else ''
                if visible and (done % 32 == 0 or done == len(items)):
                    self.progress(f'{label}: {done:,}/{len(items):,} | reused {reused:,}'
                        f' | new {new_bytes/2**20:.1f} MiB | remaining {remaining/2**20:.1f} MiB{eta}')
        except BaseException:
            if pool is not None:
                pool.shutdown(wait=False,cancel_futures=True)
            raise
        else:
            if pool is not None:
                pool.shutdown(wait=True)


class SubmissionVectorParts(OriginalVectorParts):
    """Same vectors/order as native; parallel API transport and local proof reuse."""
    def __init__(self,*args,io,**kwargs):
        super().__init__(*args,**kwargs)
        self.io = io
        self._bundle_attempted = False

    def map_path(self,index):
        p = self.manifest['parts'][index]
        return self.cache / (digest_json([str(self.root),p['path'],p['sha256']])+'.parquet')

    def block_path(self,ref):
        return self.cache / (digest_json([ref['path'],ref['sha256']])+'.npy')

    def _touch(self,path):
        # Mapping files are small; don't evict them to make room for a vector.
        previous = self.disk.pop(path.name,None)
        if previous:
            self.disk_bytes -= previous[1]
        self.disk[path.name] = path,path.stat().st_size
        self.disk_bytes += self.disk[path.name][1]
        for name,(old,size) in list(self.disk.items()):
            if self.disk_bytes <= self.cache_budget:
                break
            if old.suffix == '.parquet' or old.stem in self.maps or name in getattr(self,'_protected',()):
                continue
            old.unlink(missing_ok=True)
            self.disk_bytes -= size; del self.disk[name]
            self.verified.discard(old.stem)

    def reserve(self,window):
        needed = sum(int(m['size']) if m else 16*2**20 for _,t,_,m in window if not Path(t).is_file())
        protected = {Path(t).name for _,t,_,_ in window}
        self._protected = protected
        for name,(path,size) in list(self.disk.items()):
            if shutil.disk_usage(self.cache).free >= needed+16*2**20:
                return
            if name in protected or path.suffix == '.parquet' or path.stem in self.maps:
                continue
            path.unlink(missing_ok=True); self.disk_bytes -= size; del self.disk[name]
        if shutil.disk_usage(self.cache).free < needed+16*2**20:
            raise RuntimeError('Insufficient Colab scratch space for the next vector window; existing Drive vectors retained.')

    def _block(self,ref):
        name = digest_json([ref['path'],ref['sha256']])
        if name not in self.maps:
            path = self.block_path(ref)
            self.io.fetch(self.data_root/ref['path'],path,ref['sha256'])
            values = np.load(path,allow_pickle=False,mmap_mode='r')
            self.io.validate(path,ref['sha256'],values,self.manifest['dimension'])
            self.maps[name] = values
            if len(self.maps)>4:
                self.maps.popitem(last=False)
        self.maps.move_to_end(name); self._touch(self.block_path(ref))
        return self.maps[name]

    def gather(self,positions):
        positions = np.asarray(positions,np.int64)
        if positions.ndim != 1 or np.any(positions<0) or np.any(positions>=self.manifest['input_count']):
            raise ValueError('Invalid content vector positions.')
        result = np.empty((len(positions),self.manifest['dimension']),np.float32)
        if not len(positions):
            return result
        selected = {}
        for out in np.argsort(positions,kind='stable'):
            index = int(np.searchsorted(self.starts,int(positions[out]),side='right'))-1
            selected.setdefault(index,[]).append((int(out),int(positions[out])-self.starts[index]))
        self.restore_mapping_bundle()
        indexes, groups = list(selected),{}
        def read_map(number,path):
            index = indexes[number]
            part = self.manifest['parts'][index]
            table = pq.read_table(path,columns=['path','sha256','row'])
            if table.num_rows != part['rows']:
                raise ValueError('Content map shape changed.')
            outputs,offsets = zip(*selected[index])
            for out,ref in zip(outputs,table.take(pa.array(offsets,type=pa.int64())).to_pylist(),strict=True):
                relative = PurePosixPath(ref['path'])
                if relative.is_absolute() or '..' in relative.parts or '\\' in ref['path'] or not relative.parts:
                    raise ValueError('Content reference escapes its declared corpus.')
                groups.setdefault((ref['path'],ref['sha256']),[]).append((out,ref['row']))
            self._touch(path)
        self.io.stage([(self.root/self.manifest['parts'][i]['path'],self.map_path(i),self.manifest['parts'][i]['sha256'])
            for i in indexes], 'Mapping cache',consume=read_map)
        self.save_mapping_bundle()
        references = list(groups)
        def read_block(number,path):
            relative, checksum = references[number]
            block = self._block({'path':relative,'sha256':checksum})
            outputs,offsets = zip(*groups[relative,checksum])
            if min(offsets)<0 or max(offsets)>=len(block):
                raise ValueError('Content map row escapes its vector block.')
            result[list(outputs)] = block[list(offsets)]
        self.io.stage([(self.data_root/name,self.block_path({'path':name,'sha256':sha}),sha)
            for name,sha in references], 'Vector cache',consume=read_block,reserve=self.reserve)
        return result

    def bundle_binding(self):
        return {'schema_version':1,'kind':'mapping-bundle-v1','data_folder_id':self.io.api.folder_id,
            'source_manifest_sha256':self.manifest['manifest_sha256']}

    def restore_mapping_bundle(self):
        if self._bundle_attempted or self.io.api is None or not hasattr(self.io.api,'publish_cache_checkpoint'):
            return
        self._bundle_attempted = True
        if all(self.map_path(i).is_file() for i in range(len(self.manifest['parts']))):
            return  # Preserve the old runtime's existing mapping cache.
        binding = self.bundle_binding()
        directory = 'retrieval/mapping_bundles/'+binding['source_manifest_sha256']
        saved = self.io.api.relative_json(directory+'/current.json')
        if saved is None:
            return
        checked(saved)
        if saved.get('binding') != binding or saved.get('parts') != len(self.manifest['parts']):
            raise ValueError('Mapping bundle differs from the current frozen view.')
        if saved.get('path') != 'maps-'+saved['sha256']+'.tar.gz':
            raise ValueError('Mapping bundle path changed.')
        payload = self.io.work/'mapping_bundles'/(saved['sha256']+'.tar.gz')
        self.io.fetch(self.data_root/directory/saved['path'],payload,saved['sha256'])
        expected = {p['path']:(i,p) for i,p in enumerate(self.manifest['parts'])}
        seen, total = set(),0
        with tarfile.open(payload,mode='r|gz') as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen or member.size>16*2**20:
                    raise ValueError('Unexpected file/link in mapping bundle.')
                total += member.size
                if total>512*2**20:
                    raise ValueError('Mapping bundle exceeds its bounded metadata budget.')
                i,p = expected[member.name]; target = self.map_path(i)
                if target.exists():
                    self.io.prove(target,p['sha256'])
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    stage = target.with_name('.'+uuid.uuid4().hex+'.part')
                    try:
                        src, sha = archive.extractfile(member),hashlib.sha256()
                        with src,stage.open('wb') as dst:
                            for chunk in iter(lambda:src.read(1024*1024),b''):
                                sha.update(chunk); dst.write(chunk)
                        if sha.hexdigest()!=p['sha256']:
                            raise ValueError('Mapping bundle member checksum changed.')
                        os.replace(stage,target); self.io.record(target,p['sha256'])
                    finally:
                        stage.unlink(missing_ok=True)
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError('Mapping bundle has missing parts.')
        self._bundle_saved = True
        self.io.progress(f'Mapping bundle restored: {len(seen):,} parts from one Drive payload')

    def save_mapping_bundle(self):
        if (getattr(self,'_bundle_saved',False) or getattr(self,'_bundle_publish_attempted',False) or self.io.api is None
            or not hasattr(self.io.api,'publish_cache_checkpoint')
            or not all(self.io.known(self.map_path(i),p['sha256']) for i,p in enumerate(self.manifest['parts']))):
            return
        binding = self.bundle_binding()
        directory = 'retrieval/mapping_bundles/'+binding['source_manifest_sha256']
        local = self.io.work/'mapping_bundles'/(binding['source_manifest_sha256']+'.tar.gz')
        local.parent.mkdir(parents=True,exist_ok=True)
        if sum(self.map_path(i).stat().st_size for i in range(len(self.manifest['parts'])))>512*2**20:
            return  # Optional bundle never grows without a metadata bound.
        self._bundle_publish_attempted = True
        with tarfile.open(local,mode='w:gz',compresslevel=1) as archive:
            for i,p in enumerate(self.manifest['parts']):
                path = self.map_path(i)
                info = tarfile.TarInfo(p['path']); info.size = path.stat().st_size
                with path.open('rb') as stream:
                    archive.addfile(info,stream)
        checksum = sha256_file(local)
        saved = seal({'binding':binding,'path':'maps-'+checksum+'.tar.gz','sha256':checksum,
            'parts':len(self.manifest['parts'])})
        try:
            self.io.api.publish_cache_checkpoint(directory+'/current.json',local,saved)
            self._bundle_saved = True
            self.io.progress(f'Mapping bundle SAVED_ON_DRIVE: {saved["parts"]:,} parts; future runtimes restore one payload')
        except LookupPublishUnavailable:
            self.io.progress('Mapping bundle upload deferred; local mapping cache retained')


def index_sample(store,positions,io):
    """Checkpoint the exact deterministic ANN sample, not a second corpus copy."""
    positions = np.asarray(positions,np.int64)
    source = store.manifest['manifest_sha256']
    positions_sha = hashlib.sha256(positions.tobytes()).hexdigest()
    binding = {'schema_version':1,'kind':'index-sample-v1','source_manifest_sha256':source,
        'positions_sha256':positions_sha,'data_folder_id':getattr(io.api,'folder_id',None)}
    directory = 'retrieval/index_samples/'+source+'/'+positions_sha
    local = io.work/'index_samples'/digest_json([source,positions_sha])[:32]
    local.mkdir(parents=True,exist_ok=True)
    marker, payload = local/'sample.json',local/'sample.npy'
    saved = checked(read_json(marker)) if marker.is_file() else None
    if saved is None and io.api is not None and hasattr(io.api,'publish_cache_checkpoint'):
        saved = io.api.relative_json(directory+'/current.json')
        if saved is not None:
            checked(saved)
    if saved is not None:
        if saved.get('binding') != binding or saved.get('rows') != len(positions):
            raise ValueError('ANN sample checkpoint source/positions changed.')
        if saved.get('path') != 'sample-'+saved['sha256']+'.npy':
            raise ValueError('ANN sample checkpoint path changed.')
        io.fetch(store.data_root/directory/saved['path'],payload,saved['sha256'])
        values = np.load(payload,allow_pickle=False)
        io.validate(payload,saved['sha256'],values,store.manifest['dimension'])
        if len(values)!=len(positions):
            raise ValueError('ANN sample row count changed.')
        atomic_json(marker,saved)
        io.progress(f'ANN training sample reused: {len(values):,} vectors; no corpus read')
        return values
    io.progress(f'ANN sample preparation: {len(positions):,} existing vectors; bounded parallel payload reads')
    values = np.ascontiguousarray(store.gather(positions),dtype=np.float32)
    validate_vectors(values,len(positions),store.manifest['dimension'])
    stage = payload.with_suffix('.tmp.npy')
    np.save(stage,values,allow_pickle=False); os.replace(stage,payload)
    checksum = sha256_file(payload); io.record(payload,checksum,shape=values.shape)
    saved = seal({'binding':binding,'path':'sample-'+checksum+'.npy','sha256':checksum,
        'parts':1,'rows':len(positions),'dimension':store.manifest['dimension']})
    atomic_json(marker,saved)
    if io.api is not None and hasattr(io.api,'publish_cache_checkpoint'):
        try:
            io.api.publish_cache_checkpoint(directory+'/current.json',payload,saved)
            io.progress(f'ANN sample SAVED_ON_DRIVE: {len(values):,} vectors; source vectors retained')
        except LookupPublishUnavailable:
            io.progress('ANN sample upload deferred; local sample retained')
    return values
