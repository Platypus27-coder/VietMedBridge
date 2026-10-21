"""CPU-only delta recovery. Paid corpus and active submission remain immutable."""
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from collect_all_btc_sources import CollectAll, identity, raw_record_problem
except ModuleNotFoundError:
    from scripts.collect_all_btc_sources import CollectAll, identity, raw_record_problem
from vietmedbridge.artifacts import atomic_json,code_fingerprint,digest_json,publish_file,read_json,runtime_versions,utc_now,verify_file
from vietmedbridge.content_embeddings import checked,seal
from vietmedbridge.build import verify_spans
from vietmedbridge.chunks import chunk_source
from vietmedbridge.external_import import SCHEMAS,_document,_hold
from vietmedbridge.isolated_intake import Window
from vietmedbridge.quality import document_quality
from vietmedbridge.structure import parse_structure
from vietmedbridge.text import EXPECTED_PARSE_ERRORS,extract_source,normalize_for_retrieval
from vietmedbridge.validation import SCHEMA_VERSION,validate_snapshot
from vietmedbridge.health import health_report,freeze_candidate
from vietmedbridge.index_inputs import prepare_index_inputs

POLICY='collected-delta-raw-recovery-v1'
CAMPAIGN='btc-maximize-71be0e12-v1'
MAX_BODY=64*1024**2
RECOVERY_SCHEMA=pa.schema([('fetch_url',pa.string()),('source_file_id',pa.string()),('leaf',pa.string()),
    ('original_snapshot_id',pa.string()),('source_text_sha256',pa.string()),('crawl_json',pa.string()),
    ('extracted_json',pa.large_string())])


def inside(root,relative):
    path=(Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Recovery artifact escapes its root.')
    return path


def unpack_raw(payload,path,record):
    """Check compressed OR decoded SHA/length explicitly; never guess the convention."""
    size=record.get('raw_size_bytes')
    if type(size) is not int or not 0<size<=MAX_BODY:
        raise ValueError('RAW_SIZE_REQUIRES_REVIEW')
    if path.endswith('.zst'):
        # zstd frame content size may be absent: use a bounded streaming decoder.
        with pa.CompressedInputStream(pa.BufferReader(payload),'zstd') as stream:
            body=stream.read(MAX_BODY+1)
    elif path.endswith('.gz'):
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            body=stream.read(MAX_BODY+1)
    else:
        body=payload
    if len(body)>MAX_BODY:
        raise ValueError('RAW_SIZE_REQUIRES_REVIEW')
    expected=record['raw_sha256']
    if len(body)==size and hashlib.sha256(body).hexdigest()==expected:
        convention='decoded-body'
    elif len(payload)==size and hashlib.sha256(payload).hexdigest()==expected:
        convention='stored-payload'
    else:
        raise ValueError('RAW_PAYLOAD_HASH_OR_LENGTH_MISMATCH')
    return body,convention


class MaximizeCollection:
    snapshot=CollectAll.snapshot
    saved=CollectAll.saved
    commit=CollectAll.commit

    def __init__(self,collection,work,*,tokenizer_spec,chunk_config):
        self.collection=collection
        self.project=collection.project
        self.root=self.project/'data_staging'/CAMPAIGN
        self.work=Path(work)
        self.root.mkdir(parents=True,exist_ok=True)
        self.work.mkdir(parents=True,exist_ok=True)
        self.report=checked(read_json(collection.root/'collection_report.json'))
        if self.report['counts_provisional'] or self.report['archive_errors']:
            raise ValueError('Complete collection/classification before delta recovery.')
        self.binding={'policy':POLICY,'collection':digest_json(self.report),
            'tokenizer':tokenizer_spec,'chunks':asdict(chunk_config),
            'producer_code':code_fingerprint(),'runtime':runtime_versions()}
        self.tokenizer_spec,self.chunk_config=tokenizer_spec,chunk_config
        self.pointer=self.root/'checkpoint.json'
        self.database=self.work/'collection.sqlite'
        if not self.database.is_file() and self.pointer.is_file():
            saved=checked(read_json(self.pointer))
            if saved['binding']!=self.binding:
                raise ValueError('Recovery inputs/policy changed; preserve this campaign.')
            source=inside(self.root,saved['path'])
            verify_file(source,saved['sha256'])
            print('RESTORE_MAXIMIZE: one SQL snapshot; completed sources not replayed',flush=True)
            temporary=self.database.with_suffix('.restore.partial')
            try:
                with gzip.open(source,'rb') as incoming,temporary.open('wb') as out:
                    shutil.copyfileobj(incoming,out,1024**2)
                verify_file(temporary,saved['sqlite_sha256'])
                temporary.replace(self.database)
            finally:
                temporary.unlink(missing_ok=True)
        self.con=sqlite3.connect(self.database)
        self.con.execute('PRAGMA cache_size=-32768')
        self.con.execute('PRAGMA synchronous=FULL')
        self.con.executescript('''CREATE TABLE IF NOT EXISTS contract(value TEXT);
            CREATE TABLE IF NOT EXISTS completed(id TEXT PRIMARY KEY,kind TEXT,receipt TEXT);
            CREATE TABLE IF NOT EXISTS aliases(id INTEGER PRIMARY KEY,url TEXT,fetch TEXT);
            CREATE INDEX IF NOT EXISTS aliases_fetch ON aliases(fetch);
            CREATE TABLE IF NOT EXISTS todo(fetch TEXT PRIMARY KEY,state TEXT,candidate TEXT,reason TEXT);
            CREATE TABLE IF NOT EXISTS tasks(leaf_id TEXT,fetch TEXT,raw_path TEXT,sha TEXT,crawl TEXT,
                quality_review INTEGER,PRIMARY KEY(leaf_id,fetch));
            CREATE TABLE IF NOT EXISTS raw_results(sha TEXT,source_url TEXT,quality_review INTEGER,result TEXT,
                PRIMARY KEY(sha,source_url,quality_review));
        ''')
        row=self.con.execute('SELECT value FROM contract').fetchone()
        if row and row[0]!=digest_json(self.binding):
            raise ValueError('Local recovery SQL belongs to another frozen input.')
        if not row:
            self.con.execute('INSERT INTO contract VALUES (?)',[digest_json(self.binding)])
        self.con.commit()
        if not self.saved('seed'):
            # URL membership and pointers only. Existing document/vector payloads are never opened.
            cursor=collection.con.execute('''SELECT o.id,o.url,o.fetch,c.candidate,c.reason,c.raw
                FROM official o JOIN content c ON o.fetch=c.fetch
                WHERE c.existing IS NULL''')
            seeded=0; logged=time.monotonic()
            for rows in iter(lambda:cursor.fetchmany(2048),[]):
                for doc_id,url,fetch,candidate,reason,raw in rows:
                    self.con.execute('INSERT OR IGNORE INTO aliases VALUES (?,?,?)',[doc_id,url,fetch])
                    state='TEXT_READY' if candidate else 'NEEDS_RAW' if raw else 'NEEDS_CRAWL'
                    self.con.execute('INSERT OR IGNORE INTO todo VALUES (?,?,?,?)',[fetch,state,candidate,reason])
                seeded+=len(rows)
                if time.monotonic()-logged>=15:
                    print('DELTA_MEMBERSHIP:',f'{seeded:,}','official IDs | pointers only; no corpus/vector reads',flush=True)
                    logged=time.monotonic()
            self.commit('seed','seed',{'counts':self.report['counts_by_official_id']})

    def close(self):
        if getattr(self,'_candidate_db',None) is not None:
            self._candidate_db.close()
            self._candidate_db=None
        self.con.close()

    def status(self):
        counts=dict(self.con.execute('''SELECT t.state,count(*) FROM aliases a JOIN todo t USING(fetch)
            GROUP BY t.state'''))
        result=seal({'policy':POLICY,'state':'DELTA_RECOVERY_STATUS','counts_by_official_id':counts,
            'reused_validated_ids':self.report['counts_by_official_id'].get('REUSED_VALIDATED_CONTENT',0),
            'raw_pass_complete':self.saved('raw-pass') is not None,
            'raw_source_errors':self.source_errors,'submission_candidate_changed':False,
            'scope':'CPU extraction/chunks only; vectors and submission are separate','updated_at':utc_now()})
        atomic_json(self.root/'status.json',result)
        print('DELTA_STATUS:',json.dumps(counts,ensure_ascii=False),flush=True)
        return result

    @property
    def source_errors(self):
        return getattr(self,'_source_errors',[])

    def _candidate(self,reference):
        root=self.root if reference.get('recovery_root')=='maximize' else self.collection.root
        item=reference
        if 'sha256' not in item:
            receipt=self.collection.saved(item['receipt_id'])
            item={**item,'sha256':receipt['sha256']}
        if getattr(self,'_candidate_db',None) is None:
            self._candidate_db=sqlite3.connect(self.work/'candidate_rows.sqlite')
            self._candidate_db.execute('PRAGMA cache_size=-32768')
            self._candidate_db.executescript('''CREATE TABLE IF NOT EXISTS parts(sha TEXT PRIMARY KEY,rows INTEGER);
                CREATE TABLE IF NOT EXISTS records(sha TEXT,position INTEGER,record TEXT,PRIMARY KEY(sha,position));''')
        cache=self._candidate_db
        sha=item['sha256']
        if cache.execute('SELECT 1 FROM parts WHERE sha=?',[sha]).fetchone() is None:
            local=self.work/'candidate_cache'/sha
            if local.is_file():
                verify_file(local,sha)
            else:
                if publish_file(inside(root,item['path']),local)!=sha:
                    raise ValueError('Recovery candidate checksum changed.')
            position=0
            try:
                # Decode each small delta part once, with bounded batches and a disk row index.
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    rows=[(sha,position+i,json.dumps(row,ensure_ascii=False)) for i,row in enumerate(batch.to_pylist())]
                    cache.executemany('INSERT INTO records VALUES (?,?,?)',rows)
                    position+=len(rows)
                cache.execute('INSERT INTO parts VALUES (?,?)',[sha,position]); cache.commit()
            except BaseException:
                cache.rollback(); raise
        row=cache.execute('SELECT record FROM records WHERE sha=? AND position=?',[sha,item['row']]).fetchone()
        if row is None: raise ValueError('Candidate row exceeds its receipt.')
        return json.loads(row[0])

    def chunk_ready(self,tokenizer,*,phase):
        """Normal source spans/BGE token policy; one durable delta shard per 128 URLs."""
        if phase not in ('metadata-text-v1','raw-text-v1'):
            raise ValueError('Use separate immutable phases for collected text and raw recovery.')
        origin="json_extract(candidate,'$.recovery_root')='maximize'" if phase=='raw-text-v1' else "coalesce(json_extract(candidate,'$.recovery_root'),'')!='maximize'"
        target=self.root/'processed'/phase
        target.mkdir(parents=True,exist_ok=True)
        config={'schema_version':SCHEMA_VERSION,'code_sha256':code_fingerprint(),
            'official_links_sha256':self.collection.binding['official_sha256'],
            'tokenizer':self.tokenizer_spec,'chunks':asdict(self.chunk_config),
            'recovery_binding':self.binding,'phase':phase}
        signature=digest_json(config)
        atomic_json(target/'config.json',{**config,'signature':signature})
        while rows:=self.con.execute("SELECT fetch,candidate FROM todo WHERE state='TEXT_READY' AND "+origin+" ORDER BY fetch LIMIT 128").fetchall():
            key='chunk:'+digest_json({'phase':phase,'rows':rows,'signature':signature})
            saved=self.saved(key)
            if saved:
                raise ValueError('Chunk receipt exists but its SQL outcomes were not committed together.')
            count=Counter()
            with tempfile.TemporaryDirectory(prefix='delta-chunk-',dir=self.work) as temporary:
                local=Path(temporary)
                paths={k:local/(k+'.parquet') for k in SCHEMAS}
                writers={k:pq.ParquetWriter(paths[k],s,compression='zstd') for k,s in SCHEMAS.items()}
                outcomes=[]
                try:
                    for fetch,reference in rows:
                        source=self._candidate(json.loads(reference))
                        if source['fetch_url']!=fetch:
                            raise ValueError('Candidate URL differs from official membership.')
                        crawl=json.loads(source['crawl_json']); extra=json.loads(source['extracted_json'])
                        text=extra.get('text') or ''
                        if (hashlib.sha256(text.encode()).hexdigest()!=source['source_text_sha256']
                                or crawl.get('fetch_url')!=fetch
                                or crawl['snapshot_id']!=source['original_snapshot_id']
                                or extra['snapshot_id']!=crawl['snapshot_id']
                                or extra['raw_sha256']!=crawl['raw_sha256'] or len(text)!=extra['char_count']):
                            raise ValueError('Recovered extraction provenance/length changed.')
                        success=True
                        for doc_id,url in self.con.execute('SELECT id,url FROM aliases WHERE fetch=? ORDER BY id',[fetch]):
                            row={**crawl,**extra,'status':crawl['status'],'url':url}
                            row.update(language=extra.get('language'),language_method=extra.get('language_method'),
                                language_confidence=extra.get('language_confidence'),extractor=extra.get('extractor'),
                                extractor_version=extra.get('extractor_version'))
                            hold=_hold(row,2_000_000)
                            if hold:
                                raise ValueError('A previously accepted candidate now fails its extraction gate: '+str(hold))
                            row['doc_id']=doc_id
                            doc=_document(row,source['source_file_id']+'!'+source['leaf'])
                            structure=parse_structure(doc,doc['heading_hints'])
                            doc.update(document_quality(text,title=doc['title'],section_count=structure['known_heading_count'],
                                raw_bytes=crawl.get('raw_size_bytes')),section_count=structure['known_heading_count'])
                            content_hash=hashlib.sha256(normalize_for_retrieval(text).encode()).hexdigest()
                            doc.update(content_hash=content_hash,canonical_content_id=content_hash)
                            ledger={'doc_id':doc_id,'url':url,'domain':doc['domain'],'status':'ok',
                                'content_type':doc['content_type'],'raw_bytes':crawl.get('raw_size_bytes') or 0,
                                'elapsed_ms':None,'http_status':crawl.get('http_status')}
                            tables={'ledger':[ledger]}
                            count['input_records']+=1
                            try:
                                children,parents=chunk_source(doc,tokenizer,self.tokenizer_spec,self.chunk_config,structure=structure)
                                verify_spans(doc,children,parents)
                                if not children: raise ValueError('NO_INDEXABLE_CHILDREN')
                                tables.update(documents=[doc],children=children,parents=parents,sections=structure['sections'])
                                for kind in ('documents','children','parents','sections'): count[kind]+=len(tables[kind])
                            except ValueError as error:
                                success=False
                                tables['failures']=[{'doc_id':doc_id,'url':url,'phase':'chunk','reason':str(error),
                                    'reason_code':'CHUNK_REVIEW','body_sha256':doc['body_sha256'],'raw_file':doc['raw_file'],
                                    'domain':doc['domain'],'content_type':doc['content_type']}]
                                count['failures']+=1
                            for kind,values in tables.items():
                                writers[kind].write_table(pa.Table.from_pylist(values,schema=SCHEMAS[kind]))
                        outcomes.append(('CHUNKED' if success else 'CHUNK_REVIEW',fetch))
                finally:
                    for writer in writers.values(): writer.close()
                files={}
                for kind,path in paths.items():
                    relative='parts/'+key[6:]+'/'+kind+'.parquet'
                    files[kind]={'path':relative,'sha256':publish_file(path,target/relative)}
            self.con.executemany('UPDATE todo SET state=? WHERE fetch=?',outcomes)
            self.commit(key,'chunk',{'phase':phase,'signature':signature,'files':files,'counts':dict(count),
                'records':count['input_records'],'completed_at':utc_now()})
            print('DELTA_CHUNK_SAVED:',phase,'|',count['documents'],'documents | existing corpus untouched',flush=True)
        parts=[json.loads(r[0]) for r in self.con.execute("SELECT receipt FROM completed WHERE kind='chunk' ORDER BY id")
               if json.loads(r[0])['phase']==phase]
        if not parts: return None
        counts=Counter()
        for p in parts: counts.update(p['counts'])
        build={'schema_version':SCHEMA_VERSION,'signature':signature,'parts':parts,'counts':dict(counts),
            'run_name':phase,'selected_range_complete':True,'selected_shards':len(parts),
            'chunking':asdict(self.chunk_config),
            'requested_input_records':counts['input_records'],'input_kind':'COLLECTED_RECOVERY_DELTA',
            'state':'DATA_VALIDATED','snapshot_sha256':digest_json({'signature':signature,'parts':parts})}
        atomic_json(target/'build.json',build)
        return target,build

    def prepare_raw_tasks(self,reader):
        # Each native source version is inspected once, using already cached metadata.
        originals=[(r[0],json.loads(r[1])) for r in self.collection.con.execute(
            "SELECT id,receipt FROM completed WHERE kind='leaf' ORDER BY id")]
        originals=[(identifier,receipt) for identifier,receipt in originals if not self.saved('tasks:'+identifier)]
        if not originals:
            print('RAW_TASKS_REUSED: receipt inventory only; no metadata payload replay',flush=True)
            return
        index=self.collection._repair_metadata_index()
        for identifier,receipt in originals:
            key='tasks:'+identifier
            if self.saved(key): continue
            leaf=receipt['leaf']; members=leaf['members']
            print('RAW_TASK_SOURCE_START:',leaf['name'],'| saved metadata only',flush=True)
            native=leaf['format']=='native'
            kinds=('dataset','frontier','crawl') if native else ('dataset','raw_catalog')
            with tempfile.TemporaryDirectory(prefix='raw-task-',dir=self.work) as temporary:
                local=Path(temporary)
                for kind in kinds:
                    sha=receipt['metadata_sha256'][kind]
                    source=next((inside(self.project,p) for p in index.get(sha,[]) if inside(self.project,p).is_file()),None)
                    if source is None:
                        restored=self.collection._restore_repair_members(reader,identifier,receipt,[{'kind':kind}],[])
                        source=restored[kind][0]
                    destination=local/kind
                    if publish_file(source,destination)!=sha: raise ValueError('Task metadata checksum mismatch.')
                descriptor=read_json(local/'dataset')
                con=duckdb.connect()
                try:
                    con.execute("SET memory_limit='1GB'"); con.execute('SET threads=1')
                    con.execute('SET preserve_insertion_order=false')
                    con.execute('SET temp_directory=?',[str(local/'spill')])
                    if native:
                        con.read_parquet(str(local/'frontier')).create_view('f')
                        con.read_parquet(str(local/'crawl')).create_view('c')
                        query="SELECT f.fetch_url,to_json(c) AS crawl,NULL::VARCHAR AS quality FROM f JOIN c USING(crawl_url_id) WHERE c.status='SUCCESS_RAW'"
                    else:
                        con.read_parquet(str(local/'raw_catalog')).create_view('r')
                        query='''SELECT fetch_url,original_crawl_record_json AS crawl,quality_tier AS quality,
                            raw_path,raw_sha256,original_snapshot_id FROM r'''
                    tasks=0
                    for batch in con.execute(query).fetch_record_batch(256):
                        for item in batch.to_pylist():
                            fetch=item['fetch_url']
                            state=self.con.execute('SELECT state FROM todo WHERE fetch=?',[fetch]).fetchone()
                            if not state or state[0]!='NEEDS_RAW': continue
                            crawl=json.loads(item['crawl'])
                            if crawl.get('fetch_url')!=fetch:
                                raise ValueError('Raw crawl URL differs from native/BTC membership.')
                            locator=crawl if native else {**crawl,'raw_path':item['raw_path']}
                            if (not native and (item['raw_sha256']!=crawl.get('raw_sha256')
                                    or item['original_snapshot_id']!=crawl.get('snapshot_id'))):
                                raise ValueError('Retained locator and original crawl binding differ.')
                            if raw_record_problem({**locator,'crawl_snapshot':crawl.get('snapshot_id')},leaf['format'],descriptor): continue
                            self.con.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?)',
                                [identifier,fetch,locator['raw_path'],crawl['raw_sha256'],item['crawl'],
                                 int(str(item['quality']).upper() in ('LOW','QUARANTINE'))])
                            tasks+=1
                finally: con.close()
            self.commit(key,'tasks',{'leaf_id':identifier,'tasks':tasks})
            print('RAW_TASK_SOURCE_SAVED:',leaf['name'],'|',tasks,'raw versions; completed source skipped on resume',flush=True)

    def _save_extracted(self,source,rows):
        if not rows: return
        key=digest_json(rows)
        with tempfile.TemporaryDirectory(dir=self.work) as temporary:
            path=Path(temporary)/'recovered.parquet'
            pq.write_table(pa.Table.from_pylist(rows,schema=RECOVERY_SCHEMA),path,compression='zstd')
            relative='recovered/'+key+'.parquet'
            sha=publish_file(path,self.root/relative)
        for number,row in enumerate(rows):
            ref={'path':relative,'row':number,'sha256':sha,'recovery_root':'maximize',
                 'source_file_id':source['archive']['id'],'original_snapshot_id':row['original_snapshot_id']}
            self.con.execute("UPDATE todo SET state='TEXT_READY',candidate=?,reason=NULL WHERE fetch=? AND state='NEEDS_RAW'",
                [json.dumps(ref),row['fetch_url']])

    def recover_raw(self,reader,*,checkpoint_headers=1024):
        """One bounded TAR header pass per needed source; resume at saved next header."""
        if self.saved('raw-pass'):
            print('RAW_PASS_REUSED: completed recovery checkpoint; no source/payload reads',flush=True)
            return self.status()
        self.prepare_raw_tasks(reader)
        self._source_errors=[]
        sources=[(r[0],json.loads(r[1])) for r in self.collection.con.execute(
            "SELECT id,receipt FROM completed WHERE kind='leaf' ORDER BY id")]
        for leaf_id,receipt in sources:
            if self.saved('raw:'+leaf_id): continue
            tasks=self.con.execute("""SELECT t.fetch,t.raw_path,t.sha,t.crawl,t.quality_review FROM tasks t
                JOIN todo o USING(fetch) WHERE t.leaf_id=? AND o.state='NEEDS_RAW' AND NOT EXISTS (
                SELECT 1 FROM raw_results r WHERE r.sha=t.sha AND r.source_url=t.fetch
                AND r.quality_review=t.quality_review)""",[leaf_id]).fetchall()
            if not tasks: continue
            remote=receipt['archive']
            if identity(reader.metadata(remote['id']))!=identity(remote):
                raise ValueError('Raw archive changed since collection: '+remote['name'])
            by_path={}
            for task in tasks: by_path.setdefault(task[1],[]).append(task)
            remaining_paths=set(by_path)
            cursor_key='cursor:'+leaf_id
            state=self.saved(cursor_key) or {'offset':0,'leaf_start':None,'leaf_size':None,'pax':{}}
            if state.get('eof'):
                self.commit('raw:'+leaf_id,'raw_source',{'leaf_id':leaf_id,'complete':True})
                continue
            # Find a nested leaf via TAR headers only, keeping payload reads bounded.
            with reader.open(remote['id'],remote) as stream:
                if hasattr(stream,'block_size'): stream.block_size=8*1024**2
                start,size=0,int(remote['size'])
                if state['leaf_start'] is not None:
                    start,size=state['leaf_start'],state['leaf_size']
                else:
                    for child in receipt['leaf']['name'].split('!')[1:]:
                        with tarfile.open(fileobj=Window(stream,start,size),mode='r:') as archive:
                            member=next((m for m in archive if m.name==child and m.isfile()),None)
                            if member is None: raise ValueError('Nested raw leaf missing: '+child)
                            start,size=start+member.offset_data,member.size
                offset=state['offset']; pending=[]; headers=0; logged=time.monotonic()
                if offset==size:
                    self.commit('raw:'+leaf_id,'raw_source',{'leaf_id':leaf_id,'complete':True})
                    continue
                stream.seek(start+offset)
                boundary=stream.read(min(512,size-offset))
                if boundary and not any(boundary):
                    self.commit('raw:'+leaf_id,'raw_source',{'leaf_id':leaf_id,'complete':True})
                    continue
                print('RAW_SOURCE_RESUME:',receipt['leaf']['name'],'| header byte',offset,'| pending URLs',len(tasks),flush=True)
                def save(next_offset,*,eof=False):
                    self._save_extracted(receipt,pending); pending.clear()
                    value={'offset':next_offset,'leaf_start':start,'leaf_size':size,'pax':dict(archive.pax_headers),'eof':eof}
                    self.con.execute('INSERT OR REPLACE INTO completed VALUES (?,?,?)',
                        [cursor_key,'cursor',json.dumps(value)])
                    self.con.commit(); self.snapshot()
                    print('RAW_CURSOR_SAVED_ON_DRIVE:',receipt['leaf']['name'],'| next header',next_offset,flush=True)
                with tarfile.open(fileobj=Window(stream,start+offset,size-offset),mode='r:') as archive:
                    archive.pax_headers.update(state.get('pax',{}))
                    for member in archive:
                        headers+=1
                        if member.isfile() and member.name in by_path:
                            needed=[t for t in by_path[member.name] if self.con.execute('SELECT state FROM todo WHERE fetch=?',[t[0]]).fetchone()[0]=='NEEDS_RAW']
                            if needed:
                                if not 0<member.size<=MAX_BODY:
                                    for task in needed: self._raw_result(task,'RAW_SIZE_REQUIRES_REVIEW')
                                else:
                                    payload=archive.extractfile(member).read(MAX_BODY+1)
                                    for task in needed:
                                        fetch,path,sha,raw,quality_review=task
                                        if self.con.execute('SELECT 1 FROM raw_results WHERE sha=? AND source_url=? AND quality_review=?',
                                                [sha,fetch,quality_review]).fetchone(): continue
                                        crawl=json.loads(raw)
                                        try:
                                            body,convention=unpack_raw(payload,path,crawl)
                                            if quality_review: raise ValueError('RETAINED_QUALITY_REVIEW')
                                            extracted=extract_source(body,crawl.get('content_type') or '',
                                                source_url=crawl.get('final_url') or fetch,requested_url=fetch)
                                            text=extracted['source_text']
                                            extra={'snapshot_id':crawl['snapshot_id'],'raw_sha256':sha,
                                                'extract_status':'EXTRACT_SUCCESS','text':text,'char_count':len(text),
                                                'title':extracted['title'],'language':extracted['language'],
                                                'language_method':extracted['language_method'],'language_confidence':extracted['language_confidence'],
                                                'extractor':extracted['parser'],'extractor_version':POLICY,
                                                'source_issue':None,'text_markdown':'','raw_hash_convention':convention}
                                            hold=_hold({**crawl,**extra,'url':fetch},2_000_000)
                                            if hold: raise ValueError(':'.join(hold))
                                            pending.append({'fetch_url':fetch,'source_file_id':remote['id'],
                                                'leaf':receipt['leaf']['name'],'original_snapshot_id':crawl['snapshot_id'],
                                                'source_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                                                'crawl_json':raw,'extracted_json':json.dumps(extra,ensure_ascii=False)})
                                            self._raw_result(task,'EXTRACTED')
                                        except EXPECTED_PARSE_ERRORS as error:
                                            self._raw_result(task,str(error)[:300])
                            remaining_paths.discard(member.name)
                        if headers%checkpoint_headers==0 or len(pending)>=32:
                            save(offset+archive.offset)
                        if time.monotonic()-logged>=15:
                            print('RAW_RECOVERY_PROGRESS:',receipt['leaf']['name'],'|',headers,'headers |',len(pending),'new text pending save',flush=True)
                            logged=time.monotonic()
                        archive.members.clear()
                        if not remaining_paths:
                            break
                    for missing in remaining_paths:
                        for task in by_path[missing]: self._raw_result(task,'RAW_MEMBER_NOT_FOUND_IN_SOURCE')
                    save(offset+archive.offset,eof=True)
            if identity(reader.metadata(remote['id']))!=identity(remote):
                raise ValueError('Raw source version changed during recovery; do not freeze.')
            self.commit('raw:'+leaf_id,'raw_source',{'leaf_id':leaf_id,'complete':True})
        self.commit('raw-pass','raw_campaign',{'complete':True,'policy':POLICY,'updated_at':utc_now()})
        return self.status()

    def _raw_result(self,task,result):
        fetch,path,sha,raw,review=task
        self.con.execute('INSERT OR REPLACE INTO raw_results VALUES (?,?,?,?)',[sha,fetch,review,result])
        if result!='EXTRACTED':
            self.con.execute("UPDATE todo SET reason=? WHERE fetch=? AND state='NEEDS_RAW'",[result,fetch])

    def export_remaining(self):
        import csv
        for name,query in (
            ('remaining_for_crawl.csv',"SELECT a.id,a.url,t.state,t.reason FROM aliases a JOIN todo t USING(fetch) WHERE t.state IN ('NEEDS_CRAWL','NEEDS_RAW','CHUNK_REVIEW') ORDER BY a.id"),
            ('raw_recovery_outcomes.csv','SELECT sha,source_url,quality_review,result FROM raw_results ORDER BY source_url,sha')):
            local=self.work/name
            cursor=self.con.execute(query)
            with local.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream); writer.writerow([c[0] for c in cursor.description])
                for rows in iter(lambda:cursor.fetchmany(2048),[]): writer.writerows(rows)
            publish_file(local,self.root/name)
        return self.status()

    def freeze_delta(self,phase,links,tokenizer,golden,*,audit_size=20):
        key='freeze:'+phase
        if saved:=self.saved(key):
            print('DELTA_FREEZE_REUSED:',phase,'| manifest only; no old corpus/inputs replay',flush=True)
            self._publish_delta_manifest()
            return saved
        if phase=='raw-text-v1' and not self.saved('raw-pass'):
            raise ValueError('Finish/resume raw recovery before freezing its delta.')
        origin="json_extract(candidate,'$.recovery_root')='maximize'" if phase=='raw-text-v1' else "coalesce(json_extract(candidate,'$.recovery_root'),'')!='maximize'"
        if self.con.execute("SELECT count(*) FROM todo WHERE state='TEXT_READY' AND "+origin).fetchone()[0]:
            raise ValueError('Complete pending delta chunks before freezing.')
        target=self.root/'processed'/phase
        if not (target/'build.json').is_file(): return None
        build=read_json(target/'build.json')
        if not build['counts'].get('documents'): return None
        health=health_report(target,official_links=links,work_dir=self.work,audit_size=audit_size)
        candidate=freeze_candidate(target,health,golden_report=golden)
        name='candidate-'+candidate['candidate_manifest_sha256'][:16]+'.json'
        inputs=prepare_index_inputs(target,name,tokenizer,work_dir=self.work)
        saved={'data_root':str(self.root.relative_to(self.project)), 'build_run':phase,'candidate_name':name,
            'candidate_manifest_sha256':candidate['candidate_manifest_sha256'],
            'snapshot_sha256':candidate['snapshot_sha256'],'index_inputs_manifest_sha256':inputs['manifest_sha256'],
            'documents':build['counts']['documents'],'unique_dense_inputs':inputs['input_count'],
            'state':'DELTA_READY_FOR_EMBEDDING','submission_candidate_changed':False,'updated_at':utc_now()}
        self.commit(key,'freeze',saved)
        self._publish_delta_manifest()
        return saved

    def _publish_delta_manifest(self):
        candidates=[json.loads(r[0]) for r in self.con.execute("SELECT receipt FROM completed WHERE kind='freeze' ORDER BY id")]
        atomic_json(self.root/'delta_candidates.json',seal({'builds':candidates,'scope':'New recovery delta only',
            'submission_candidate_changed':False}))
