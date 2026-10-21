"""Real native data/raw locators, legacy SQL repair and metadata-only cold resume."""
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq
import pytest

import scripts.collect_all_btc_sources as collect
from scripts.collect_all_btc_sources import CollectAll,raw_record_problem,RAW_PATH_POLICY
from test_collect_all_btc_sources import fixture as flow_fixture
from test_external_import import external,write_parquet
from test_isolated_intake import LocalReader,make_tar
from vietmedbridge.artifacts import atomic_json,read_json,sha256_file
from vietmedbridge.external_import import MEMBERS


def data_raw_archive(tmp_path,external,name,*,source_issue=None):
    directory=tmp_path/(name+'-metadata')
    shutil.copytree(external[0],directory)
    rows=pq.read_table(directory/MEMBERS['crawl']).to_pylist()
    for row in rows:
        sha=row['raw_sha256']
        row['raw_path']=f'data/raw/html/{sha[:2]}/{sha[2:4]}/{sha}.html.zst'
    write_parquet(directory/MEMBERS['crawl'],rows)
    if source_issue:
        rows=pq.read_table(directory/MEMBERS['extracted']).to_pylist()
        rows[0]['source_issue']=source_issue
        write_parquet(directory/MEMBERS['extracted'],rows)
    return make_tar(directory,tmp_path/(name+'.tar'))


def legacy_checkpoint(tmp_path,external,monkeypatch,*,two=False,existing=False):
    flow,links=flow_fixture(tmp_path,external)
    if existing:
        aliases=tmp_path/'aliases.parquet'
        write_parquet(aliases,[{'doc_id':583,'url':'https://example.org/article'}])
        flow.existing(aliases,sha256_file(aliases),'paid/validated/aliases.parquet')
    files={name:data_raw_archive(tmp_path,external,name) for name in (('one','two') if two else ('one',))}
    reader=LocalReader(files)
    remotes=[reader.metadata(k) for k in files]
    original=collect.raw_record_problem
    def old_gate(row,kind,descriptor):
        if kind=='native' and row['status']=='SUCCESS_RAW' and row['raw_path'].startswith('data/raw/'):
            return 'SUCCESS_RAW'  # Exact legacy predicate failure shown in the live report.
        return original(row,kind,descriptor)
    with monkeypatch.context() as scope:
        scope.setattr(collect,'raw_record_problem',old_gate)
        flow.run(reader,remotes,{'drive_file_id':'unused'})
    for identifier,raw in list(flow.con.execute("SELECT id,receipt FROM completed WHERE kind='leaf'")):
        receipt=json.loads(raw)
        receipt.pop('raw_path_policy')
        flow.con.execute('UPDATE completed SET receipt=? WHERE id=?',[json.dumps(receipt),identifier])
    flow.con.execute('UPDATE content SET raw_status=raw')
    flow.con.commit()
    report=flow.report(remotes,[])
    flow.snapshot()
    return flow,links,reader,remotes,report


def never_tar(monkeypatch):
    def fail(*a,**kw):
        pytest.fail('Classification repair must not open TARs, call Drive API or download source members')
    for name in ('metadata_leaves','stage_member','inventory'):
        monkeypatch.setattr(collect,name,fail)


def test_exact_native_path_is_valid_and_retained_contract_remains_strict():
    sha='4a334587388a8da0a9fd974830e339632da4a6a9f2a83fec7612b296c2a080e4'
    snapshot='5586f423916005c11ffa8afb96895017d672af51d68aee8a88d28dd2e0dc39c6'
    row={'status':'SUCCESS_RAW','raw_path':f'data/raw/html/4a/33/{sha}.html.zst',
         'raw_sha256':sha,'crawl_snapshot':snapshot}
    assert raw_record_problem(row,'native',{'snapshot_id':snapshot}) is None
    assert raw_record_problem(row,'retained',{})=='RAW_RECORD_INVALID_LOCATOR'


@pytest.mark.parametrize('path',['data/raw/../../secret','raw/../secret','/data/raw/body.zst',
    r'data/raw/..\secret',r'data/raw/body\secret','data/raw/body\x00.zst','raw/','data/raw/'])
def test_unsafe_locators_do_not_become_recovery_candidates(path):
    row={'status':'SUCCESS_RAW','raw_path':path,'raw_sha256':'a'*64,'crawl_snapshot':'same'}
    assert raw_record_problem(row,'native',{'snapshot_id':'same'})=='RAW_RECORD_INVALID_LOCATOR'


def test_success_with_invalid_metadata_is_not_reported_as_failed_http_crawl(tmp_path,external):
    flow,links=flow_fixture(tmp_path,external)
    source=data_raw_archive(tmp_path,external,'invalid')
    # Integrity check differs from HTTP status: use a snapshot mismatch in a native source.
    directory=tmp_path/'invalid-metadata'
    rows=pq.read_table(directory/MEMBERS['crawl']).to_pylist()
    for row in rows:
        row['snapshot_id']='unbound'
    write_parquet(directory/MEMBERS['crawl'],rows)
    source=make_tar(directory,tmp_path/'invalid-final.tar')
    reader=LocalReader({'source':source})
    report=flow.run(reader,[reader.metadata('source')],{'drive_file_id':'unused'})
    assert report['counts_by_official_id']=={'CRAWL_NOT_SUCCESSFUL':1,'NOT_FOUND_IN_ARCHIVES':1,'RAW_METADATA_REVIEW':3}
    assert not any(r['candidates'] for r in report['recovery_parts'])
    flow.close()


def test_valid_native_data_raw_recovers_text_without_weakening_quality_gates(tmp_path,external):
    flow,links=flow_fixture(tmp_path,external)
    source=data_raw_archive(tmp_path,external,'challenge',source_issue='JAVASCRIPT_CHALLENGE')
    reader=LocalReader({'source':source})
    report=flow.run(reader,[reader.metadata('source')],{'drive_file_id':'unused'})
    assert report['counts_by_official_id']=={'CRAWL_NOT_SUCCESSFUL':1,'NOT_FOUND_IN_ARCHIVES':1,
        'RAW_AVAILABLE_NEEDS_EXTRACTION':3}
    assert report['recovery_parts'][0]['held']['quality:JAVASCRIPT_CHALLENGE']==1
    flow.close()


def test_legacy_sql_repaired_from_drive_metadata_no_tar_then_cold_resume_skips_all(tmp_path,external,monkeypatch):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch)
    original_receipts=list(flow.con.execute("SELECT id,receipt FROM completed WHERE kind='leaf'"))
    assert before['counts_by_official_id']=={'CRAWL_NOT_SUCCESSFUL':4,'NOT_FOUND_IN_ARCHIVES':1}
    assert before['counts_provisional'] and before['raw_path_classification']['pending_leaves']==1
    project=flow.project
    flow.close()
    never_tar(monkeypatch)
    resumed=CollectAll(project,tmp_path/'new-runtime',official_sha=sha256_file(links))
    report=resumed.repair_raw_paths()
    assert report['raw_path_repair']['state']=='RAW_PATH_REPAIR_COMPLETE'
    assert report['raw_path_repair']['tar_accessed'] is False
    assert report['counts_by_official_id']=={'CRAWL_NOT_SUCCESSFUL':1,'NOT_FOUND_IN_ARCHIVES':1,
        'RAW_AVAILABLE_NEEDS_EXTRACTION':1,'RECOVERED_EXTRACTION_PENDING_CHUNK':2}
    assert not report['counts_provisional']
    assert [r['classification_effective'] for r in report['recovery_parts']]==[False,True]
    assert report['official_ids_with_successful_raw_record']==3
    assert list(resumed.con.execute("SELECT id,receipt FROM completed WHERE kind='leaf'"))==original_receipts
    candidate=json.loads(resumed.con.execute('SELECT candidate FROM content WHERE fetch=?',
        ['https://example.org/article']).fetchone()[0])
    payload=pq.read_table(resumed.root/candidate['path']).to_pylist()[candidate['row']]
    assert json.loads(payload['extracted_json'])['text']==external[2]
    assert json.loads(payload['crawl_json'])['raw_path'].startswith('data/raw/html/')
    assert read_json(project/'data/active_data_candidate.json')=={'keep':'submission'}
    resumed.close()
    again=CollectAll(project,tmp_path/'runtime-3',official_sha=sha256_file(links))
    monkeypatch.setattr(again,'_repair_metadata_index',lambda **kw:pytest.fail('Index reread after completion'))
    monkeypatch.setattr(again,'leaf',lambda *a,**kw:pytest.fail('Completed source replayed'))
    assert again.repair_raw_paths()==report
    again.close()


@pytest.mark.parametrize('origin',['external','parallel'])
def test_missing_collection_cache_uses_hash_matched_old_metadata_receipts(tmp_path,external,monkeypatch,origin):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch)
    identifier,receipt=next((r[0],json.loads(r[1])) for r in flow.con.execute(
        "SELECT id,receipt FROM completed WHERE kind='leaf'"))
    if origin=='external':
        root=flow.project/'data/processed/paid-build/external_source'
    else:
        root=flow.project/'data_staging/btc-new-71be0e12-20261019-v1/parallel_cpu4/jobs/new-00006/sources/new-00006/metadata'
    root.mkdir(parents=True)
    files={}
    for kind,checksum in receipt['metadata_sha256'].items():
        payload=flow.root/'metadata_cache'/identifier/kind/'metadata.bin'
        name=kind+'.bin'
        shutil.copyfile(payload,root/name)
        files[kind]={'sha256':checksum,'bytes':payload.stat().st_size,
                     'name' if origin=='external' else 'path':name}
        payload.unlink()
    atomic_json(root/('source.json' if origin=='external' else 'metadata.json'),{'files':files})
    never_tar(monkeypatch)
    report=flow.repair_raw_paths()
    assert report['raw_path_repair']['state']=='RAW_PATH_REPAIR_COMPLETE'
    assert report['counts_by_official_id']['RECOVERED_EXTRACTION_PENDING_CHUNK']==2
    flow.close()


def test_missing_saved_metadata_is_reported_without_downloading_tar(tmp_path,external,monkeypatch):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch)
    identifier=flow.con.execute("SELECT id FROM completed WHERE kind='leaf'").fetchone()[0]
    (flow.root/'metadata_cache'/identifier/'extracted/metadata.bin').unlink()
    never_tar(monkeypatch)
    report=flow.repair_raw_paths()
    assert report['raw_path_repair']['state']=='RAW_PATH_REPAIR_INCOMPLETE'
    assert report['raw_path_repair']['errors'][0]['missing'][0]['kind']=='extracted'
    assert report['raw_path_classification']['pending_leaves']==1 and report['counts_provisional']
    assert report['archive_errors']==[] and report['state']=='ALL_FOLDER_SOURCES_COLLECTED'
    flow.close()


def test_previously_paid_valid_documents_not_recovered_again(tmp_path,external,monkeypatch):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch,existing=True)
    never_tar(monkeypatch)
    report=flow.repair_raw_paths()
    assert report['counts_by_official_id']['REUSED_VALIDATED_CONTENT']==2
    assert not any(r['candidates'] for r in report['recovery_parts'])
    flow.close()


def test_interrupted_repair_keeps_each_leaf_checkpoint_and_recovers_no_duplicate_text(tmp_path,external,monkeypatch):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch,two=True)
    never_tar(monkeypatch)
    original=flow.leaf
    called=[]
    def interrupted(*args,**kwargs):
        called.append(args[2]['name'])
        if len(called)==2:
            raise KeyboardInterrupt('New Colab runtime')
        return original(*args,**kwargs)
    monkeypatch.setattr(flow,'leaf',interrupted)
    with pytest.raises(KeyboardInterrupt):
        flow.repair_raw_paths()
    assert len(flow.pending_raw_repairs())==1
    project=flow.project
    flow.close()
    resumed=CollectAll(project,tmp_path/'after-interrupt',official_sha=sha256_file(links))
    assert len(resumed.pending_raw_repairs())==1
    report=resumed.repair_raw_paths()
    assert report['raw_path_repair']['state']=='RAW_PATH_REPAIR_COMPLETE'
    assert sum(r['candidates'] for r in report['recovery_parts'])==1
    resumed.close()


def test_old_sql_schema_gains_success_status_without_resetting_paid_mapping(tmp_path,external,monkeypatch):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch,existing=True)
    flow.con.execute('ALTER TABLE content DROP COLUMN raw_status')
    flow.con.commit()
    flow.snapshot()
    project=flow.project
    flow.close()
    never_tar(monkeypatch)
    resumed=CollectAll(project,tmp_path/'old-schema-resume',official_sha=sha256_file(links))
    report=resumed.repair_raw_paths()
    assert report['raw_path_classification']['state']=='CURRENT'
    assert report['counts_by_official_id']['REUSED_VALIDATED_CONTENT']==2
    assert report['counts_by_official_id']['RAW_AVAILABLE_NEEDS_EXTRACTION']==1
    resumed.close()


def test_corrupt_metadata_cannot_repair_source_or_publish_recovery(tmp_path,external,monkeypatch):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch)
    identifier=flow.con.execute("SELECT id FROM completed WHERE kind='leaf'").fetchone()[0]
    (flow.root/'metadata_cache'/identifier/'extracted/metadata.bin').write_bytes(b'corrupt')
    never_tar(monkeypatch)
    report=flow.repair_raw_paths()
    assert report['counts_provisional']
    assert 'changed artifact' in report['raw_path_repair']['errors'][0]['error']
    assert len(flow.pending_raw_repairs())==1
    assert flow.con.execute('SELECT count(*) FROM content WHERE candidate IS NOT NULL').fetchone()[0]==0
    flow.close()


def test_transient_snapshot_failure_is_republished_before_completion(tmp_path,external,monkeypatch):
    flow,links,reader,remotes,before=legacy_checkpoint(tmp_path,external,monkeypatch)
    never_tar(monkeypatch)
    original=flow.snapshot
    calls=[]
    def first_snapshot_fails():
        calls.append(1)
        if len(calls)==1:
            raise OSError('Drive snapshot temporarily unavailable')
        return original()
    monkeypatch.setattr(flow,'snapshot',first_snapshot_fails)
    report=flow.repair_raw_paths()
    assert report['raw_path_repair']['state']=='RAW_PATH_REPAIR_COMPLETE'
    assert not report['raw_path_repair']['errors'] and report['raw_path_repair']['recovered_warnings']
    project=flow.project
    flow.close()
    cold=CollectAll(project,tmp_path/'snapshot-recovery',official_sha=sha256_file(links))
    assert not cold.pending_raw_repairs()
    assert cold.repair_raw_paths()==report
    cold.close()
