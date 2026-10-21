"""Full-folder recovery across nested archives, old snapshots and failed versions."""
import ast
import io
import json
from pathlib import Path
import shutil
import tarfile

import nbformat
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.collect_all_btc_sources import CollectAll,metadata_leaves,identity
from vietmedbridge.artifacts import atomic_json,read_json,sha256_file
from vietmedbridge.external_import import MEMBERS
from vietmedbridge.isolated_intake import DriveRange
from test_external_import import external,write_parquet
from test_isolated_intake import LocalReader,make_tar


def archive(tmp_path,external,name,*,successful=True,nested=False,original='old-snapshot'):
    directory=tmp_path/(name+'-source')
    shutil.copytree(external[0],directory)
    for kind in ('crawl','extracted'):
        path=directory/MEMBERS[kind]
        rows=pq.read_table(path).to_pylist()
        for r in rows:
            r['snapshot_id']=original
            if not successful and kind=='extracted' and r['extract_status']=='EXTRACT_SUCCESS':
                r['extract_status']='JAVASCRIPT_CHALLENGE'
        write_parquet(path,rows)
    descriptor=read_json(directory/MEMBERS['dataset'])
    descriptor['snapshot_id']=original
    atomic_json(directory/MEMBERS['dataset'],descriptor)
    path=make_tar(directory,tmp_path/(name+'.tar'))
    if nested:
        target=tmp_path/(name+'-outer.tar')
        with tarfile.open(target,'w') as outer:
            outer.add(path,arcname=path.name)
        path=target
    return path


def fixture(tmp_path,external):
    links=tmp_path/'btc.parquet'
    # Old source IDs differ; exact current BTC URL membership must be preserved.
    write_parquet(links,[{'id':100,'url':'https://example.org/article'},
        {'id':101,'url':'https://example.org/article#evidence'},
        {'id':102,'url':'https://example.org/failed'},
        {'id':103,'url':'https://example.org/parse'},
        {'id':104,'url':'https://example.org/absent'}])
    project=tmp_path/'project'
    atomic_json(project/'data/active_data_candidate.json',{'keep':'submission'})
    flow=CollectAll(project,tmp_path/'work',official_sha=sha256_file(links))
    flow.official(links)
    return flow,links


def test_failed_new_version_recovered_from_old_nested_archive_and_current_ids_preserved(tmp_path,external):
    flow,links=fixture(tmp_path,external)
    files={'new':archive(tmp_path,external,'new',successful=False,original='new-snapshot'),
           'old':archive(tmp_path,external,'old',nested=True)}
    reader=LocalReader(files)
    remotes=[reader.metadata(k) for k in files]
    report=flow.run(reader,remotes,{'drive_file_id':'unused'})
    assert report['archive_errors']==[]
    assert report['counts_by_official_id']=={'CRAWL_NOT_SUCCESSFUL':1,'NOT_FOUND_IN_ARCHIVES':1,
        'RAW_AVAILABLE_NEEDS_EXTRACTION':1,'RECOVERED_EXTRACTION_PENDING_CHUNK':2}
    assert report['all_official_content_available'] is False
    assert report['official_ids_seen_in_archive_metadata']==4
    assert report['official_ids_with_successful_raw_record']==3
    parts=list((flow.root/'parts').glob('*.parquet'))
    recovered=[r for p in parts for r in pq.read_table(p).to_pylist()]
    assert len(recovered)==1 and recovered[0]['original_snapshot_id']=='old-snapshot'
    assert json.loads(recovered[0]['extracted_json'])['text']==external[2]
    assert read_json(flow.project/'data/active_data_candidate.json')=={'keep':'submission'}
    assert set(r[0] for r in flow.con.execute('SELECT id FROM official'))=={100,101,102,103,104}
    root=flow.root
    project=flow.project
    flow.close()
    shutil.rmtree(tmp_path/'work')
    # Another account/runtime restores one SQL payload, then skips old archives.
    resumed=CollectAll(project,tmp_path/'work2',official_sha=sha256_file(links))
    def fail(*args,**kwargs):
        pytest.fail('Completed archive or official file reopened during cold resume')
    reader.open=fail
    resumed.official(tmp_path/'missing-links.parquet')
    again=resumed.run(reader,remotes,{'drive_file_id':'unused'})
    assert again['counts_by_official_id']==report['counts_by_official_id']
    resumed.close()


def test_existing_content_mapping_reused_by_url_without_rechunk_or_id_relabel(tmp_path,external):
    flow,links=fixture(tmp_path,external)
    aliases=tmp_path/'aliases.parquet'
    write_parquet(aliases,[{'doc_id':583,'url':'https://example.org/article'}])
    checksum=sha256_file(aliases)
    flow.existing(aliases,checksum,'old/validated/aliases.parquet')
    aliases.unlink()
    flow.existing(aliases,checksum,'old/validated/aliases.parquet')  # receipt-only skip
    path=archive(tmp_path,external,'old')
    reader=LocalReader({'old':path})
    report=flow.run(reader,[reader.metadata('old')],{'drive_file_id':'unused'})
    assert report['counts_by_official_id']['REUSED_VALIDATED_CONTENT']==2
    assert not any(pq.read_table(p).num_rows for p in (flow.root/'parts').glob('*.parquet'))
    ref=json.loads(flow.con.execute('SELECT existing FROM content WHERE fetch=?',['https://example.org/article']).fetchone()[0])
    assert ref['original_doc_id']==583
    flow.close()


def test_bad_metadata_does_not_count_archive_as_complete_and_other_archive_still_collected(tmp_path,external):
    flow,links=fixture(tmp_path,external)
    bad=tmp_path/'bad.tar'
    with tarfile.open(bad,'w') as out:
        value=b'not a supported metadata leaf'
        member=tarfile.TarInfo('readme.txt');member.size=len(value)
        out.addfile(member,io.BytesIO(value))
    reader=LocalReader({'bad':bad,'good':archive(tmp_path,external,'good')})
    report=flow.run(reader,[reader.metadata(k) for k in ('bad','good')],{'drive_file_id':'unused'})
    assert len(report['archive_errors'])==1 and report['state']=='FOLDER_COLLECTION_INCOMPLETE'
    assert report['counts_by_official_id']['RECOVERED_EXTRACTION_PENDING_CHUNK']==2
    flow.close()


def test_official_checksum_change_never_reuses_another_dataset(tmp_path,external):
    flow,links=fixture(tmp_path,external)
    flow.close()
    with pytest.raises(ValueError,match='snapshot'):
        CollectAll(flow.project,tmp_path/'fresh-work',official_sha='0'*64)


def test_notebook_compiles_and_embedded_source_matches():
    from scripts.write_collect_all_btc_notebook import notebook
    book=notebook()
    for c in book.cells:
        if c.cell_type=='code':
            compile(c.source,'<collect-notebook>','exec')
    tree=ast.parse(book.cells[2].source)
    source=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCE' for t in n.targets))
    assert source==Path('scripts/collect_all_btc_sources.py').read_text(encoding='utf-8')
    assert 'KAGGLE_JSON' not in '\n'.join(c.source for c in book.cells)


def test_retained_package_preserves_original_snapshot_and_low_quality_review(tmp_path,external):
    from test_full_intake import portable_fixture
    spec,path=portable_fixture(tmp_path,external)
    flow=CollectAll(tmp_path/'project',tmp_path/'work',official_sha=sha256_file(external[1]))
    flow.official(external[1])
    reader=LocalReader({'retained':path})
    report=flow.run(reader,[reader.metadata('retained')],spec)
    assert report['archive_errors']==[]
    assert report['counts_by_official_id']['RECOVERED_EXTRACTION_PENDING_CHUNK']==3
    assert report['counts_by_official_id']['RAW_AVAILABLE_NEEDS_EXTRACTION']==3
    assert report['all_official_content_available'] is False
    parts=[r for p in (flow.root/'parts').glob('*.parquet') for r in pq.read_table(p).to_pylist()]
    assert all(r['original_snapshot_id']=='old-capture' for r in parts)
    assert not any(r['fetch_url'].endswith('low-retained') for r in parts)
    flow.close()


def test_source_change_mid_read_cannot_commit_candidates(tmp_path,external):
    flow,links=fixture(tmp_path,external)
    path=archive(tmp_path,external,'changed')
    reader=LocalReader({'changed':path})
    remote=reader.metadata('changed')
    with metadata_leaves(reader,remote,{'drive_file_id':'unused'},flow.work) as (stream,leaves):
        def changed():
            raise ValueError('Archive changed during collection')
        with pytest.raises(ValueError,match='changed'):
            flow.leaf(stream,remote,leaves[0],version_check=changed)
    assert flow.con.execute('SELECT count(*) FROM content WHERE candidate IS NOT NULL').fetchone()[0]==0
    assert flow.con.execute("SELECT count(*) FROM completed WHERE kind='leaf'").fetchone()[0]==0
    flow.close()


def test_native_metadata_prefix_does_not_list_hundreds_of_raw_members(tmp_path,external):
    source=archive(tmp_path,external,'native-with-raw')
    target=tmp_path/'native-many-raw.tar'
    with tarfile.open(source,'r:') as original,tarfile.open(target,'w') as out:
        for member in original:
            out.addfile(member,original.extractfile(member))
        first_raw=out.offset
        for index in range(300):
            member=tarfile.TarInfo(f'raw/{index:06}.zst');member.size=4096
            out.addfile(member,io.BytesIO(b'x'*member.size))
    class Guarded(io.BytesIO):
        def read(self,size=-1):
            assert self.tell()<first_raw,'Native manifest discovery must not enumerate raw headers'
            return super().read(size)
    reader=LocalReader({'raw':target})
    reader.open=lambda *a:Guarded(target.read_bytes())
    with metadata_leaves(reader,reader.metadata('raw'),{'drive_file_id':'unused'},tmp_path) as (_,leaves):
        assert len(leaves)==1 and set(leaves[0]['members'])==set(MEMBERS)


def test_metadata_after_128_headers_and_later_resume_children_are_not_omitted(tmp_path,external):
    first=archive(tmp_path,external,'first')
    second=archive(tmp_path,external,'second')
    leaf=tmp_path/'late-metadata.tar'
    with tarfile.open(first,'r:') as source,tarfile.open(leaf,'w') as out:
        for index in range(140):
            out.addfile(tarfile.TarInfo(f'notes/{index:05}.txt'),io.BytesIO())
        for member in source:
            out.addfile(member,source.extractfile(member))
    outer=tmp_path/'many-header-container.tar'
    with tarfile.open(outer,'w') as out:
        out.add(leaf,arcname='resume-first.tar')
        for index in range(140):
            out.addfile(tarfile.TarInfo(f'notes/{index:05}.txt'),io.BytesIO())
        out.add(second,arcname='resume-later.tar')
    reader=LocalReader({'container':outer})
    with metadata_leaves(reader,reader.metadata('container'),{'drive_file_id':'unused'},tmp_path) as (_,leaves):
        assert [r['name'].split('!')[-1] for r in leaves]==['resume-first.tar','resume-later.tar']


def test_range_transport_skips_native_raw_tail_and_handles_all_container_children(tmp_path,external):
    import hashlib
    leaf=archive(tmp_path,external,'range-leaf')
    outer=tmp_path/'range-container.tar'
    with tarfile.open(outer,'w') as out:
        out.add(leaf,arcname='first.tar')
        out.add(leaf,arcname='second.tar')
    payload=outer.read_bytes()
    ranges=[]
    class Response:
        status_code=206
        def __init__(self,start,end):
            self.raw=io.BytesIO(payload[start:end+1])
            self.headers={'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def raise_for_status(self):pass
    class Session:
        def get(self,url,**kw):
            start,end=map(int,kw['headers']['Range'].split('=')[1].split('-'))
            ranges.append((start,end))
            return Response(start,end)
    remote={'id':'range','name':outer.name,'size':str(len(payload)),
        'modifiedTime':'fixed','md5Checksum':hashlib.md5(payload).hexdigest()}
    class Reader:
        def open(self,*a):return DriveRange(Session(),'range',remote)
    with metadata_leaves(Reader(),remote,{'drive_file_id':'unused'},tmp_path) as (stream,leaves):
        assert len(leaves)==2
        for leaf in leaves:
            for item in leaf['members'].values():
                stream.seek(item['offset'])
                assert len(stream.read(item['bytes']))==item['bytes']
    assert ranges and max(end-start+1 for start,end in ranges)<=65536


def test_failed_archive_retry_keeps_committed_sql_and_skips_finished_sources(tmp_path,external):
    flow,links=fixture(tmp_path,external)
    good=archive(tmp_path,external,'committed')
    retry=archive(tmp_path,external,'retry',nested=True)
    reader=LocalReader({'good':good,'retry':retry})
    remotes=[reader.metadata(k) for k in ('good','retry')]
    real_open=reader.open
    def fail_retry(file_id,*a):
        if file_id=='retry':raise ValueError('Previous reader stopped at 128 headers')
        return real_open(file_id,*a)
    reader.open=fail_retry
    report=flow.run(reader,remotes,{'drive_file_id':'unused'})
    assert len(report['archive_errors'])==1
    project=flow.project
    flow.close()
    resumed=CollectAll(project,tmp_path/'new-runtime',official_sha=sha256_file(links))
    def resume_open(file_id,*a):
        assert file_id!='good','Completed source must remain receipt-only'
        return real_open(file_id,*a)
    reader.open=resume_open
    report=resumed.run(reader,remotes,{'drive_file_id':'unused'})
    assert report['archive_errors']==[] and report['state']=='ALL_FOLDER_SOURCES_COLLECTED'
    assert resumed.con.execute("SELECT count(*) FROM completed WHERE kind='archive'").fetchone()[0]==2
    resumed.close()
