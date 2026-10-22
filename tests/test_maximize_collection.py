"""Actual delta chunks, raw integrity, immutable paid data and cold resume."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.maximize_collection as maximum
from scripts.maximize_collection import MaximizeCollection,unpack_raw
from test_external_import import external,CharacterTokenizer,CONFIG,SPEC,write_parquet
from test_collect_all_btc_sources import fixture as collection_fixture
from test_isolated_intake import LocalReader,make_tar
from vietmedbridge.artifacts import code_fingerprint,read_json,sha256_file
from vietmedbridge.external_import import MEMBERS
from vietmedbridge.health import freeze_candidate,health_report
from vietmedbridge.validation import validate_snapshot


def collected(tmp_path,external,*,corrupt=False,challenge=False,nested=False):
    source,_,_=external
    body=(b'<html><head><title>Just a moment</title></head><body>Robot check</body></html>' if challenge else
        b'<html><head><title>Medical treatment</title></head><body><article><h1>Medical treatment</h1>'
        +b'<p>Patients require regular medical follow up and treatment to control disease symptoms.</p>'*12
        +b'</article></body></html>')
    crawl=pq.read_table(source/MEMBERS['crawl']).to_pylist()
    record=crawl[2]
    record['raw_sha256']=hashlib.sha256(body).hexdigest()
    record['raw_path']='data/raw/html/'+record['raw_sha256']+'.html.zst'
    record['raw_size_bytes']=len(body)
    write_parquet(source/MEMBERS['crawl'],crawl)
    extracted=pq.read_table(source/MEMBERS['extracted']).to_pylist()
    extracted[1]['raw_sha256']=record['raw_sha256']
    write_parquet(source/MEMBERS['extracted'],extracted)
    archive=make_tar(source,tmp_path/'data.tar')
    with tarfile.open(archive,'a') as tar:
        payload=pa.compress(b'corrupt body' if corrupt else body,codec='zstd',asbytes=True)
        item=tarfile.TarInfo(record['raw_path']); item.size=len(payload)
        tar.addfile(item,io.BytesIO(payload))
    if nested:
        outer=tmp_path/'outer.tar'
        with tarfile.open(outer,'w') as tar: tar.add(archive,arcname=archive.name)
        archive=outer
    collection,links=collection_fixture(tmp_path,external)
    reader=LocalReader({'source':archive})
    report=collection.run(reader,[reader.metadata('source')],{'drive_file_id':'unused'})
    assert not report['counts_provisional']
    controller=MaximizeCollection(collection,tmp_path/'maximum',tokenizer_spec=SPEC,chunk_config=CONFIG)
    return collection,controller,reader,links,body


def test_delta_chunk_preserves_exact_text_aliases_and_can_validate_and_freeze(tmp_path,external):
    collection,flow,reader,links,body=collected(tmp_path,external)
    old_candidate=(collection.project/'data/active_data_candidate.json').read_bytes()
    target,build=flow.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')
    assert build['counts']['documents']==2
    docs=pq.read_table(target/build['parts'][0]['files']['documents']['path']).to_pylist()
    assert {d['doc_id'] for d in docs}=={100,101}
    assert all(d['source_text']==external[2] for d in docs)
    assert validate_snapshot(target,build,official_links=links,work_dir=flow.work)['passed']
    health=health_report(target,official_links=links,work_dir=flow.work,audit_size=2)
    golden={'passed':True,'code_sha256':code_fingerprint(),'tokenizer':SPEC,'chunking':asdict(CONFIG)}
    candidate=freeze_candidate(target,health,golden_report=golden)
    assert candidate['state']=='FROZEN_CANDIDATE'
    record=flow.freeze_delta('metadata-text-v1',links,CharacterTokenizer(),golden,audit_size=2)
    assert record['state']=='DELTA_READY_FOR_EMBEDDING' and record['unique_dense_inputs']>0
    (flow.root/'delta_candidates.json').unlink()
    assert flow.freeze_delta('metadata-text-v1',links,CharacterTokenizer(),golden,audit_size=2)==record
    assert read_json(flow.root/'delta_candidates.json')['builds']==[record]
    assert (collection.project/'data/active_data_candidate.json').read_bytes()==old_candidate
    flow.close(); collection.close()


@pytest.mark.parametrize('nested',[False,True])
def test_saved_raw_becomes_only_new_delta_with_original_provenance(tmp_path,external,nested):
    collection,flow,reader,links,body=collected(tmp_path,external,nested=nested)
    first=flow.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')
    status=flow.recover_raw(reader,checkpoint_headers=2)
    assert status['counts_by_official_id']['TEXT_READY']==1
    assert status['counts_by_official_id']['CHUNKED']==2
    # A rerun of the first stage cannot append raw text into an already frozen phase.
    assert flow.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')[1]==first[1]
    target,build=flow.chunk_ready(CharacterTokenizer(),phase='raw-text-v1')
    assert build['counts']['documents']==1
    assert validate_snapshot(target,build,official_links=links,work_dir=flow.work)['passed']
    doc=pq.read_table(target/build['parts'][0]['files']['documents']['path']).to_pylist()[0]
    assert doc['doc_id']==103 and doc['body_sha256']==hashlib.sha256(body).hexdigest()
    assert 'regular medical follow up' in doc['source_text']
    candidate=json.loads(flow.con.execute("SELECT candidate FROM todo WHERE state='CHUNKED' AND fetch LIKE '%parse'").fetchone()[0])
    recovered=flow._candidate(candidate)
    assert recovered['original_snapshot_id']=='team-test'
    assert json.loads(recovered['extracted_json'])['raw_hash_convention']=='decoded-body'
    flow.export_remaining()
    assert (flow.root/'remaining_for_crawl.csv').is_file()
    flow.close(); collection.close()


@pytest.mark.parametrize('kind',['corrupt','challenge'])
def test_corrupt_or_challenge_raw_is_never_counted_as_new_document(tmp_path,external,kind):
    collection,flow,reader,links,body=collected(tmp_path,external,**{kind:True})
    flow.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')
    status=flow.recover_raw(reader)
    assert status['counts_by_official_id']['NEEDS_RAW']==1
    assert flow.chunk_ready(CharacterTokenizer(),phase='raw-text-v1') is None
    result=flow.con.execute('SELECT result FROM raw_results').fetchone()[0]
    assert 'MISMATCH' in result if kind=='corrupt' else 'challenge' in result
    flow.close(); collection.close()


def test_cold_resume_skips_old_metadata_raw_and_chunk_payloads(tmp_path,external,monkeypatch):
    collection,flow,reader,links,body=collected(tmp_path,external)
    first=flow.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')
    flow.recover_raw(reader,checkpoint_headers=1)
    flow.chunk_ready(CharacterTokenizer(),phase='raw-text-v1')
    flow.close()
    resumed=MaximizeCollection(collection,tmp_path/'new-runtime',tokenizer_spec=SPEC,chunk_config=CONFIG)
    def fail(*a,**kw): pytest.fail('Completed source/payload was replayed')
    monkeypatch.setattr(reader,'open',fail)
    monkeypatch.setattr(resumed,'_candidate',fail)
    monkeypatch.setattr(collection,'_repair_metadata_index',fail)
    # Task receipts must be checked before constructing the metadata index.
    resumed.recover_raw(reader)
    assert resumed.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')[1]==first[1]
    resumed.close(); collection.close()


def test_raw_cursor_survives_interruption_after_last_payload_without_reextract(tmp_path,external,monkeypatch):
    collection,flow,reader,links,body=collected(tmp_path,external)
    flow.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')
    original=flow.snapshot
    def interrupted():
        original()
        if flow.con.execute("SELECT count(*) FROM raw_results WHERE result='EXTRACTED'").fetchone()[0]:
            raise KeyboardInterrupt('Simulated runtime end after durable cursor')
    with monkeypatch.context() as patch:
        patch.setattr(flow,'snapshot',interrupted)
        with pytest.raises(KeyboardInterrupt): flow.recover_raw(reader,checkpoint_headers=1)
    flow.close()
    resumed=MaximizeCollection(collection,tmp_path/'after-interrupt',tokenizer_spec=SPEC,chunk_config=CONFIG)
    monkeypatch.setattr(maximum,'extract_source',lambda *a,**kw:pytest.fail('Paid extraction replayed'))
    status=resumed.recover_raw(reader)
    assert status['counts_by_official_id']['TEXT_READY']==1
    assert resumed.chunk_ready(CharacterTokenizer(),phase='raw-text-v1')[1]['counts']['documents']==1
    resumed.close(); collection.close()


def test_unpack_requires_exact_hash_and_size_and_records_storage_convention():
    body=b'body '*100; payload=pa.compress(body,codec='zstd',asbytes=True)
    for value,convention in ((body,'decoded-body'),(payload,'stored-payload')):
        assert unpack_raw(payload,'raw/file.zst',{'raw_size_bytes':len(value),
            'raw_sha256':hashlib.sha256(value).hexdigest()})==(body,convention)
    with pytest.raises(ValueError,match='MISMATCH'):
        unpack_raw(payload,'raw/file.zst',{'raw_size_bytes':len(body)+1,'raw_sha256':hashlib.sha256(body).hexdigest()})


def test_retained_tasks_keep_portable_locator_and_do_not_promote_quarantine(tmp_path,external):
    from test_full_intake import portable_fixture
    from scripts.collect_all_btc_sources import CollectAll
    spec,archive=portable_fixture(tmp_path,external)
    collection=CollectAll(tmp_path/'project',tmp_path/'collector',official_sha=sha256_file(external[1]))
    collection.official(external[1])
    reader=LocalReader({'retained':archive})
    collection.run(reader,[reader.metadata('retained')],spec)
    flow=MaximizeCollection(collection,tmp_path/'maximum',tokenizer_spec=SPEC,chunk_config=CONFIG)
    flow.prepare_raw_tasks(reader)
    tasks=flow.con.execute('SELECT raw_path,quality_review FROM tasks').fetchall()
    assert tasks and all(p.startswith('raw/') for p,q in tasks)
    assert any(q for p,q in tasks)
    flow.close(); collection.close()


def test_notebook_embeds_current_helpers_and_never_promotes_active_candidate():
    import ast
    from scripts.write_maximize_collection_notebook import notebook
    book=notebook()
    for cell in book.cells:
        if cell.cell_type=='code': compile(cell.source,'<maximise-notebook>','exec')
    tree=ast.parse(book.cells[2].source)
    assigned={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('COLLECT_SOURCE','MAXIMIZE_SOURCE')}
    assert assigned['MAXIMIZE_SOURCE']==Path('scripts/maximize_collection.py').read_text(encoding='utf-8')
    assert assigned['COLLECT_SOURCE']==Path('scripts/collect_all_btc_sources.py').read_text(encoding='utf-8')
    assert 'publish_data_handoff' not in '\n'.join(c.source for c in book.cells)


def test_candidate_part_is_decoded_once_for_multiple_row_lookups(tmp_path,external,monkeypatch):
    collection,flow,reader,links,body=collected(tmp_path,external)
    reference=json.loads(flow.con.execute("SELECT candidate FROM todo WHERE state='TEXT_READY'").fetchone()[0])
    original=maximum.pq.ParquetFile
    calls=[]
    def counted(*a,**kw):
        calls.append(a[0]); return original(*a,**kw)
    monkeypatch.setattr(maximum.pq,'ParquetFile',counted)
    first=flow._candidate(reference)
    assert flow._candidate(reference)==first
    assert flow._candidate(reference)==first
    assert len(calls)==1
    flow.close(); collection.close()


def test_archive_version_change_does_not_start_raw_recovery(tmp_path,external,monkeypatch):
    collection,flow,reader,links,body=collected(tmp_path,external)
    flow.prepare_raw_tasks(reader)
    original=reader.metadata
    monkeypatch.setattr(reader,'metadata',lambda key:{**original(key),'md5Checksum':'changed'})
    opens=reader.opens
    with pytest.raises(ValueError,match='changed since collection'): flow.recover_raw(reader)
    assert reader.opens==opens and not flow.saved('raw-pass')
    flow.close(); collection.close()


def test_unindexed_local_candidate_copy_must_still_match_receipt(tmp_path,external):
    collection,flow,reader,links,body=collected(tmp_path,external)
    reference=json.loads(flow.con.execute("SELECT candidate FROM todo WHERE state='TEXT_READY'").fetchone()[0])
    sha=collection.saved(reference['receipt_id'])['sha256']
    cached=flow.work/'candidate_cache'/sha
    cached.parent.mkdir(); cached.write_bytes(b'corrupt unpublished cache')
    with pytest.raises(ValueError,match='changed artifact'): flow._candidate(reference)
    flow.close(); collection.close()


def test_ready_cutoff_freezes_committed_text_while_raw_remains_incomplete(tmp_path,external,monkeypatch):
    collection,flow,reader,links,body=collected(tmp_path,external)
    flow.chunk_ready(CharacterTokenizer(),phase='metadata-text-v1')
    original=flow.snapshot
    def stop_after_saved_text():
        original()
        if flow.con.execute("SELECT count(*) FROM raw_results WHERE result='EXTRACTED'").fetchone()[0]:
            raise KeyboardInterrupt('Rate limit after durable payload/cursor')
    with monkeypatch.context() as patch:
        patch.setattr(flow,'snapshot',stop_after_saved_text)
        with pytest.raises(KeyboardInterrupt): flow.recover_raw(reader,checkpoint_headers=1)
    assert not flow.saved('raw-pass')
    cursor=flow.con.execute("SELECT id,receipt FROM completed WHERE kind='cursor'").fetchall()
    active=(collection.project/'data/active_data_candidate.json').read_bytes()
    opens=reader.opens
    with monkeypatch.context() as patch:
        patch.setattr(reader,'open',lambda *a,**k:pytest.fail('Cutoff reopened raw TAR'))
        phase=flow.stage_ready_batch()
        assert flow.stage_ready_batch()==phase
        target,build=flow.chunk_ready(CharacterTokenizer(),phase=phase)
        golden={'passed':True,'code_sha256':code_fingerprint(),'tokenizer':SPEC,'chunking':asdict(CONFIG)}
        record=flow.freeze_delta(phase,links,CharacterTokenizer(),golden,audit_size=2)
        assert record['documents']==1 and not record['source_campaign_raw_pass_complete']
        assert record['selection']['official_ids']==1
        assert validate_snapshot(target,build,official_links=links,work_dir=flow.work)['passed']
        assert flow.stage_ready_batch() is None
    assert reader.opens==opens
    assert not flow.saved('raw-pass')
    assert flow.con.execute("SELECT id,receipt FROM completed WHERE kind='cursor'").fetchall()==cursor
    assert (collection.project/'data/active_data_candidate.json').read_bytes()==active
    # The original raw pass can finish later, without duplicating the exported text.
    monkeypatch.setattr(maximum,'extract_source',lambda *a,**k:pytest.fail('Already exported text re-extracted'))
    flow.recover_raw(reader)
    assert flow.saved('raw-pass') and flow.chunk_ready(CharacterTokenizer(),phase='raw-text-v1') is None
    flow.close(); collection.close()


def test_ready_cutoff_detects_membership_tampering(tmp_path,external):
    collection,flow,reader,links,body=collected(tmp_path,external)
    flow.recover_raw(reader)
    phase=flow.stage_ready_batch()
    flow.con.execute('DELETE FROM ready_members WHERE phase=?',[phase])
    with pytest.raises(ValueError,match='membership differs'):
        flow.chunk_ready(CharacterTokenizer(),phase=phase)
    flow.close(); collection.close()


def test_full_raw_phase_still_requires_completed_raw_pass(tmp_path,external):
    collection,flow,reader,links,body=collected(tmp_path,external)
    with pytest.raises(ValueError,match='Finish/resume raw'):
        flow.freeze_delta('raw-text-v1',links,CharacterTokenizer(),{})
    flow.close(); collection.close()


def test_finalize_notebook_never_starts_raw_recovery():
    from scripts.write_maximize_collection_notebook import finalize_notebook
    book=finalize_notebook()
    assert 'FLOW.recover_raw(' not in book.cells[-1].source
    assert 'stage_ready_batch()' in book.cells[-1].source
    for cell in book.cells:
        if cell.cell_type=='code': compile(cell.source,'<finalize-notebook>','exec')
