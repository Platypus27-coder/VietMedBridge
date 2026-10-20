"""The completed parent is immutable; only the later source goes through intake."""
import ast
import copy
import json
from pathlib import Path
import shutil
import tarfile

import nbformat
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.intake_00008_extension import (audit_00008,completed_parent_mappings,
    extension_00008_plan,load_00008_metadata,SOURCE_00008)
from scripts.intake_resume import IntakeResume
from test_isolated_intake import LocalReader,make_source
from test_external_import import external,CharacterTokenizer,SPEC,CONFIG
from vietmedbridge.artifacts import atomic_json,digest_json,read_json,sha256_file
from vietmedbridge.external_import import MEMBERS
from dataclasses import asdict


@pytest.fixture
def source(tmp_path,external):
    archive,spec=make_source(tmp_path,external,'new-00008')
    renamed=tmp_path/'renamed.tar'
    with tarfile.open(archive,'r:') as incoming,tarfile.open(renamed,'w',format=tarfile.PAX_FORMAT) as outgoing:
        for member in incoming:
            if member.name==MEMBERS['frontier']:
                member.name='data/crawl_shards/shard_00008.parquet'
            with incoming.extractfile(member) as stream:
                outgoing.addfile(member,stream)
    shutil.move(renamed,archive)
    spec.update(frontier='data/crawl_shards/shard_00008.parquet',archive_bytes=archive.stat().st_size,
        mode='NEW_SNAPSHOT_SOURCE')
    spec.pop('frontier_sha256')
    return archive,spec


def parent_at(root,external):
    sources=[{'job_id':f'new-{i:05d}','snapshot_id':'team-test','drive_file_id':f'id-{i}',
        'frontier_sha256':digest_json(i)} for i in range(8)]
    retained={'job_id':'retained-test','snapshot_id':'team-test','drive_file_id':'retained'}
    baseline=root/'references/baseline.parquet'
    baseline.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{'doc_id':900000,'url':'https://old.test/a'}]),baseline)
    scope={'base_plan':{'sources':sources[:6],'baseline':{'sha256':sha256_file(baseline)},
        'snapshots':{'team-test':{'sha256':sha256_file(external[1])}},'required_snapshot_id':'team-test',
        'chunking':asdict(CONFIG),'tokenizer_revision':'test'},
        'additional_sources':sources[6:],'retained':retained,'audited_coverage':{'union_native_ids':9}}
    for phase,group in (('.',sources[:6]),('additional-frontiers',sources[6:]),('retained-recovery',[retained])):
        plan={'sources':group}
        atomic_json(root/phase/'intake_plan.json',plan)
        for i,s in enumerate(group):
            folder=root/phase/'sources'/s['job_id'];folder.mkdir(parents=True)
            path=folder/'added_ids.parquet'
            pq.write_table(pa.Table.from_pylist([{'doc_id':100000+len(s['job_id'])+i,'url':'https://old.test/'+s['job_id']}]),path)
            receipt={'state':'COMPLETE','source':s,'plan_sha256':digest_json(plan),'build_run':s['job_id'],
                'counts':{'ADDED_VALID_DOCUMENT':1},'added_ids_sha256':sha256_file(path),
                'files':{'added_ids':{'path':path.name,'sha256':sha256_file(path),'bytes':path.stat().st_size}}}
            atomic_json(folder/'complete.json',receipt)
    report={'state':'FULL_INTAKE_COMPLETE','completed_sources':9,'total_sources':9,'pending_sources':[],
        'scope_sha256':digest_json(scope),'builds':[{'build_run':s['job_id']} for s in sources+[retained]],
        'counts':{'ADDED_VALID_DOCUMENT':9}}
    atomic_json(root/'full_intake_status.json',report)
    return scope,baseline


def hashes(root):
    return {p.relative_to(root).as_posix():sha256_file(p) for p in root.rglob('*') if p.is_file()}


def test_source_audit_is_bounded_and_resume_never_reopens_tar(tmp_path,source):
    archive,spec=source
    reader=LocalReader({spec['drive_file_id']:archive})
    root=tmp_path/'audit'
    actual=audit_00008(root,lambda:reader,source=spec)
    assert len(actual['frontier_sha256'])==64 and reader.opens==1
    saved=read_json(root/'source_audit.json')
    assert set(saved['metadata'])=={'dataset','frontier'}
    assert not list(root.rglob('*.tar')) and not list(root.rglob('*.parquet'))
    def unavailable():
        pytest.fail('Completed audit touched the remote archive')
    assert audit_00008(root,unavailable,source=spec)==actual


def test_wrong_snapshot_or_archive_version_is_refused(tmp_path,source):
    archive,spec=source
    reader=LocalReader({spec['drive_file_id']:archive})
    with pytest.raises(ValueError,match='different BTC snapshot'):
        audit_00008(tmp_path/'bad',lambda:reader,source=dict(spec,snapshot_id='wrong'))
    bound=audit_00008(tmp_path/'good',lambda:reader,source=spec)
    original=reader.metadata
    reader.metadata=lambda key:dict(original(key),modifiedTime='new-version')
    with pytest.raises(ValueError,match='version changed'):
        load_00008_metadata(reader,bound,tmp_path/'metadata',tmp_path/'good')


@pytest.mark.parametrize('fault',['counts','missing','size'])
def test_parent_receipts_fail_closed_without_replaying_sources(tmp_path,external,fault):
    root=tmp_path/'parent'
    scope,_=parent_at(root,external)
    refs,status=completed_parent_mappings(root,scope)
    assert len(refs)==9 and status['completed_sources']==9
    if fault=='counts':
        value=read_json(root/'full_intake_status.json');value['counts']['ADDED_VALID_DOCUMENT']=8
        atomic_json(root/'full_intake_status.json',value)
    elif fault=='missing':
        (root/'sources/new-00000/complete.json').unlink()
    else:
        refs[0][0].write_bytes(b'bad mapping')
    with pytest.raises((ValueError,FileNotFoundError)):
        completed_parent_mappings(root,scope)


def test_real_extra_source_import_and_complete_resume_preserve_parent_and_submission(tmp_path,external,source):
    archive,spec=source
    parent=tmp_path/'project/data_staging/old'
    scope,baseline=parent_at(parent,external)
    protected=tmp_path/'project/data'
    atomic_json(protected/'active_data_candidate.json',{'keep':'current submit'})
    before_parent,before_active=hashes(parent),hashes(protected)
    root=parent.parent/'extra';work=tmp_path/'work'
    reader=LocalReader({spec['drive_file_id']:archive})
    plan,refs,_=extension_00008_plan(parent,root,work,scope,lambda:reader,source=spec)
    controller=IntakeResume(plan,root=root,work_dir=work,tokenizer_spec=SPEC,chunk_config=CONFIG,prior_mappings=refs)
    kwargs=dict(load_reader=lambda:reader,load_links=lambda _:external[1],load_baseline=lambda:baseline,
        load_tokenizer=lambda:(CharacterTokenizer(),SPEC),
        load_metadata=lambda src,target:load_00008_metadata(reader,src,target,root))
    result=controller.run(**kwargs)
    assert result['state']=='INTAKE_COMPLETE' and result['completed_sources']==1
    assert (root/'sources/new-00008/complete.json').is_file()
    assert hashes(parent)==before_parent and hashes(protected)==before_active
    def paid_replay(*a,**k):
        pytest.fail('Completed source loaded data or tokenizer again')
    reopened=IntakeResume(plan,root=root,work_dir=work,tokenizer_spec=SPEC,chunk_config=CONFIG,prior_mappings=refs)
    repeat=reopened.run(load_reader=paid_replay,load_links=paid_replay,load_baseline=paid_replay,load_tokenizer=paid_replay)
    assert repeat['counts']==result['counts'] and hashes(parent)==before_parent


def test_new_notebook_only_runs_extra_source_with_parent_locks():
    repo=Path(__file__).resolve().parents[1]
    book=nbformat.read(repo/'notebooks/02_colab_import_new_00008.ipynb',as_version=4)
    cells=[c.source for c in book.cells if c.cell_type=='code']
    for cell in cells:
        ast.parse(cell)
    setup,prepare,run=cells
    assert 'else parent_lock' in setup and 'else PARENT_ROOT / "intake_runtime_lock.json"' in setup
    assert 'if not lock_path.is_file():' in setup
    assert 'btc-new-71be0e12-00008-20261020-v1' in setup
    assert 'shared = PARENT_ROOT / "references' in prepare
    assert 'CONTROLLER = IntakeResume(' in prepare and 'CONTROLLER = FullIntake(' not in prepare
    assert 'source_metadata' in run and 'freeze_candidate(' not in '\n'.join(cells)
    assert SOURCE_00008['archive']=='vibiomir_retain_shard_00008.tar'
