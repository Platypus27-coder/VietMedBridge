"""Collection resumes exact ranges and persists complete metadata before SQL work."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from urllib3.exceptions import ProtocolError,ReadTimeoutError

import scripts.collect_all_btc_sources as collect
from scripts.collect_all_btc_sources import CollectionDriveRange,CollectionDriveError,CollectAll,cached_member
from test_collect_all_btc_sources import archive,fixture
from test_external_import import external
from test_isolated_intake import LocalReader
from vietmedbridge.artifacts import sha256_file
from vietmedbridge.external_import import MEMBERS


class Reply:
    def __init__(self,status,reasons=(),*,headers=None,payload=None):
        self.status_code=status
        self.headers=headers or {}
        self.raw=io.BytesIO(payload if payload is not None else json.dumps(
            {'error':{'errors':[{'reason':r} for r in reasons],
                      'message':'PRIVATE_BODY_MUST_NOT_BE_LOGGED'}}).encode())
        self.closed=False
    def __enter__(self):return self
    def __exit__(self,*a):self.closed=True


def range_client(monkeypatch,replies):
    calls,sleeps=[],[]
    def get(url,**kw):
        calls.append((url,kw))
        reply=replies.pop(0)
        if isinstance(reply,Exception):raise reply
        return reply
    monkeypatch.setattr(collect.time,'sleep',sleeps.append)
    reader=CollectionDriveRange(SimpleNamespace(get=get),'file_1',{'size':'10'},block_size=4)
    reader.seek(3)
    return reader,calls,sleeps


def success():
    return Reply(206,headers={'Content-Range':'bytes 3-6/10'},payload=b'defg')


@pytest.mark.parametrize('reason',['rateLimitExceeded','userRateLimitExceeded',
    'RATE_LIMIT_EXCEEDED','USER_RATE_LIMIT_EXCEEDED'])
def test_rate_403_retries_identical_range_without_restarting_download(monkeypatch,capsys,reason):
    denied=Reply(403,[reason])
    reader,calls,sleeps=range_client(monkeypatch,[denied,success()])
    assert reader.read(4)==b'defg'
    assert len(calls)==2 and calls[0]==calls[1]
    assert calls[0][1]['headers']['Range']=='bytes=3-6' and sleeps==[1]
    assert denied.closed and reader.position==7
    logged=capsys.readouterr().out
    assert reason in logged and 'PRIVATE_BODY' not in logged


def test_mixed_legacy_and_google_rpc_reason_retries_as_rate_limit(monkeypatch):
    denied=Reply(403,payload=json.dumps({'error':{
        'errors':[{'reason':'rateLimitExceeded'}],
        'details':[{'@type':'type.googleapis.com/google.rpc.ErrorInfo','reason':'RATE_LIMIT_EXCEEDED'}]
    }}).encode())
    reader,calls,sleeps=range_client(monkeypatch,[denied,success()])
    assert reader.read(4)==b'defg' and calls[0]==calls[1] and sleeps==[1]


def test_small_alias_cell_is_repeatable_and_never_normalizes_permission_codes():
    import ast
    from types import ModuleType
    source=Path('scripts/resume_collection_rate_alias_cell.py').read_text(encoding='utf-8')
    compile(source,'resume_collection_rate_alias_cell.py','exec')
    tree=ast.parse(source)
    definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    legacy=ModuleType('old_collection')
    legacy.drive_reasons=collect.drive_reasons
    namespace={'collector':legacy}
    for _ in range(2):
        exec(compile(ast.Module(body=[definition],type_ignores=[]),'small_alias_cell.py','exec'),namespace)
        legacy.drive_reasons=namespace['_collection_reason_aliases']
    assert legacy.drive_reasons(Reply(403,['RATE_LIMIT_EXCEEDED','rateLimitExceeded']))==('rateLimitExceeded',)
    assert legacy.drive_reasons(Reply(403,['USER_RATE_LIMIT_EXCEEDED']))==('userRateLimitExceeded',)
    assert legacy.drive_reasons(Reply(403,['RATE_LIMIT_EXCEEDED','insufficientFilePermissions']))==(
        'insufficientFilePermissions','rateLimitExceeded')


@pytest.mark.parametrize('reasons',[[],['insufficientFilePermissions'],['downloadQuotaExceeded'],
    ['dailyLimitExceeded'],['appNotAuthorizedToFile'],['rateLimitExceeded','insufficientFilePermissions'],
    ['RATE_LIMIT_EXCEEDED','insufficientFilePermissions']])
def test_unknown_permission_and_download_quota_do_not_retry(monkeypatch,reasons):
    denied=Reply(403,reasons)
    reader,calls,sleeps=range_client(monkeypatch,[denied])
    with pytest.raises(CollectionDriveError) as caught:
        reader.read(4)
    assert len(calls)==1 and not sleeps and denied.closed
    assert reader.position==3 and reader.block_start==-1
    assert caught.value.details['reasons']==sorted(reasons)
    assert caught.value.details['byte_range']=='bytes=3-6'
    assert 'reason=' in str(caught.value)


@pytest.mark.parametrize('reasons',[['rateLimitExceeded'],['RATE_LIMIT_EXCEEDED'],
    ['RATE_LIMIT_EXCEEDED','rateLimitExceeded']])
def test_retry_after_and_exhaustion_are_bounded(monkeypatch,reasons):
    reader,calls,sleeps=range_client(monkeypatch,[
        Reply(403,['userRateLimitExceeded'],headers={'Retry-After':'9'}),success()])
    assert reader.read(4)==b'defg' and sleeps==[9]
    reader,calls,sleeps=range_client(monkeypatch,[Reply(403,reasons) for _ in range(4)])
    with pytest.raises(CollectionDriveError) as caught:
        reader.read(4)
    assert sleeps==[1,2,4] and len(calls)==4 and caught.value.details['pause_campaign']


def test_long_cooldown_stops_without_retrying_early(monkeypatch):
    reader,calls,sleeps=range_client(monkeypatch,[Reply(429,headers={'Retry-After':'3600'})])
    with pytest.raises(CollectionDriveError) as caught:
        reader.read(4)
    assert len(calls)==1 and sleeps==[]
    assert caught.value.details['cooldown_seconds']==3600


@pytest.mark.parametrize('failure',[requests.ConnectionError('reset'),
    requests.Timeout('timeout'),ProtocolError('raw connection reset'),
    ReadTimeoutError(None,None,'raw read timed out'),
    Reply(206,headers={'Content-Range':'bytes 3-6/10'},payload=b'd')])
def test_transport_or_truncated_block_retries_same_range(monkeypatch,failure):
    reader,calls,sleeps=range_client(monkeypatch,[failure,success()])
    assert reader.read(4)==b'defg' and calls[0]==calls[1] and sleeps==[1]


@pytest.mark.parametrize('status,header',[(200,'bytes 3-6/10'),(206,'bytes 0-3/10')])
def test_ignored_or_wrong_range_never_consumes_full_archive(monkeypatch,status,header):
    reply=Reply(status,headers={'Content-Range':header},payload=b'defg')
    reader,calls,sleeps=range_client(monkeypatch,[reply])
    with pytest.raises(ValueError,match='bounded Range'):
        reader.read(4)
    assert reply.raw.tell()==0 and len(calls)==1 and not sleeps


def test_complete_metadata_survives_failed_leaf_and_new_runtime(tmp_path,external,monkeypatch):
    flow,links=fixture(tmp_path,external)
    source=archive(tmp_path,external,'metadata-retry')
    reader=LocalReader({'source':source})
    remotes=[reader.metadata('source')]
    original=collect.stage_member
    def first(stream,item,target):
        if item['path']==MEMBERS['extracted']:
            raise CollectionDriveError(403,['downloadQuotaExceeded'],'source','bytes=100-199')
        return original(stream,item,target)
    monkeypatch.setattr(collect,'stage_member',first)
    report=flow.run(reader,remotes,{'drive_file_id':'unused'})
    assert report['archive_errors'][0]['drive_error']['reasons']==['downloadQuotaExceeded']
    assert len(list((flow.root/'metadata_cache').glob('*/**/receipt.json')))==3
    project=flow.project
    flow.close()
    resumed=CollectAll(project,tmp_path/'cold-runtime',official_sha=sha256_file(links))
    calls=[]
    def second(stream,item,target):
        calls.append(item['path'])
        assert item['path']==MEMBERS['extracted'],'Committed metadata must not reopen TAR payload'
        return original(stream,item,target)
    monkeypatch.setattr(collect,'stage_member',second)
    report=resumed.run(reader,remotes,{'drive_file_id':'unused'})
    assert not report['archive_errors'] and calls==[MEMBERS['extracted']]
    assert report['counts_by_official_id']['RECOVERED_EXTRACTION_PENDING_CHUNK']==2
    resumed.close()


def test_corrupt_cached_metadata_is_not_silently_reused(tmp_path):
    item={'path':'data/test.parquet','offset':0,'bytes':4,'sha256':hashlib.sha256(b'data').hexdigest()}
    cached_member(io.BytesIO(b'data'),item,tmp_path/'first',tmp_path/'cache')
    (tmp_path/'cache/metadata.bin').write_bytes(b'bad!')
    with pytest.raises(ValueError,match='changed artifact'):
        cached_member(io.BytesIO(b'data'),item,tmp_path/'second',tmp_path/'cache')


def test_global_rate_cooldown_does_not_request_remaining_archives(tmp_path,external):
    flow,links=fixture(tmp_path,external)
    reader=LocalReader({'blocked':archive(tmp_path,external,'blocked'),
                        'later':archive(tmp_path,external,'later')})
    remotes=[reader.metadata(k) for k in ('blocked','later')]
    calls=[]
    def denied(file_id,*a):
        calls.append(file_id)
        raise CollectionDriveError(403,['rateLimitExceeded'],file_id,'bytes=0-511',pause=True)
    reader.open=denied
    report=flow.run(reader,remotes,{'drive_file_id':'unused'})
    assert calls==['blocked'] and len(report['archive_errors'])==2
    assert report['archive_errors'][1]['error']=='NOT_ATTEMPTED_DRIVE_COOLDOWN'
    assert report['state']=='FOLDER_COLLECTION_INCOMPLETE'
    flow.close()
