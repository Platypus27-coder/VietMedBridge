"""Independent measurements and API-only report writes without a real GPU."""
import hashlib
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import nbformat as nb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import colab_qwen_benchmark as bench
from scripts.qwen_token_batching import TokenBatchEncoder
from scripts.write_colab_qwen_benchmark_notebook import notebook,ROOT
from test_qwen_token_batching import Clock,Native,attention_native
from vietmedbridge.artifacts import atomic_json,digest_json,read_json,sha256_file,verify_file
from vietmedbridge.content_embeddings import key,seal


def fixture_sources(tmp_path):
    root=tmp_path/'remote'
    root.mkdir()
    tasks,assets=[],{}
    for i in range(9):
        rows=[{'id':key(f'{i*200+j}:32'),'text':f'{i*200+j}:32'} for j in range(128)]
        name=f'missing-qwen-{i:06d}.parquet'
        path=root/name
        pq.write_table(pa.Table.from_pylist(rows),path)
        checksum=sha256_file(path)
        relative=f'selection/qwen/build-{i}/missing.parquet'
        assets[name]={'rows':len(rows),'source_path':relative,'sha256':checksum}
        tasks.append({'id':digest_json(i),'family':'qwen','asset':name,'rows':128,'offset':0,
                      'sha256':checksum,'keys_digest':digest_json(sorted(r['id'] for r in rows))})
    calls=[]
    class Source:
        def relative(self,path):
            name=next(n for n,v in assets.items() if v['source_path']==path)
            calls.append(path)
            return {'name':name}
        def download(self,metadata,target,*,expected_sha256,max_bytes):
            assert max_bytes==64*1024*1024
            verify_file(root/metadata['name'],expected_sha256)
            shutil.copyfile(root/metadata['name'],target)
    return seal({'tasks':tasks,'assets':assets}),Source(),calls


def test_sample_reads_only_three_saved_assets_and_cold_retry_reads_one_small_json(tmp_path,monkeypatch):
    job,source,calls=fixture_sources(tmp_path)
    work=tmp_path/'work'
    path,sample=bench.prepare_sample(source,job,work,expected_library=job['manifest_sha256'])
    assert len(calls)==sample['assets_read']==3
    assert calls==[f'selection/qwen/build-{i}/missing.parquet' for i in (0,4,8)]
    calibration={r['id'] for r in sample['calibration']}
    evaluation={r['id'] for r in sample['evaluation']}
    assert len(calibration)==len(evaluation)==144 and not calibration&evaluation
    assert sample['tasks_committed']==0 and read_json(path)==sample
    monkeypatch.setattr(pq,'read_table',lambda *a,**k:pytest.fail('Saved sample replayed source text.'))
    assert bench.prepare_sample(source,job,work,expected_library=job['manifest_sha256'])==(path,sample)
    assert len(calls)==3


@pytest.mark.parametrize('change',['wrong-key-digest','unsafe-source-path','corrupt-source'])
def test_sample_binding_fails_before_model_work(tmp_path,change):
    job,source,calls=fixture_sources(tmp_path)
    if change=='wrong-key-digest':job['tasks'][0]['keys_digest']='f'*64
    elif change=='unsafe-source-path':job['assets'][job['tasks'][0]['asset']]['source_path']='../outside.parquet'
    else:(tmp_path/'remote'/job['tasks'][0]['asset']).write_bytes(b'corrupt')
    job=seal({k:v for k,v in job.items() if k!='manifest_sha256'})
    with pytest.raises((ValueError,AssertionError)):
        bench.prepare_sample(source,job,tmp_path/'work',expected_library=job['manifest_sha256'])
    assert not (tmp_path/'work/sample.json').exists()


def sample_pair():
    def rows(start):
        return [{'id':key(t),'text':t} for t in [f'{i}:{32 if i%2 else 768}' for i in range(start,start+64)]]
    return seal({'calibration':rows(0),'evaluation':rows(100)})


class Cuda:
    def __init__(self):self.syncs,self.resets=0,0
    def synchronize(self):self.syncs+=1
    def reset_peak_memory_stats(self):self.resets+=1
    def max_memory_allocated(self):return 2*1024**3
    def max_memory_reserved(self):return 3*1024**3


def test_measurement_uses_held_out_texts_alternating_order_and_synchronized_cuda(tmp_path):
    clock=Clock()
    native,state=attention_native(clock)
    tuned=TokenBatchEncoder(native,tmp_path/'tuning.json',attention_tuning=True,clock=clock,
                           report=lambda *a,**k:None)
    cuda=Cuda()
    result=bench.compare_execution(native,tuned,sample_pair(),cuda,clock=clock,report=lambda *a,**k:None)
    assert [r['mode'] for r in result['runs']]==['native','tuned','tuned','native']
    assert cuda.resets==4 and cuda.syncs>=9
    assert all(r['peak_allocated_bytes']==2*1024**3 for r in result['runs'])
    assert result['candidate_accepted'] and result['held_out_speedup']>=1.05
    assert all(r['compatible'] for r in result['agreements'])
    assert result['evaluation_texts']==64 and not state['enabled']


def test_invalid_or_slower_candidates_are_not_recommended(tmp_path):
    clock=Clock();native=Native(clock,drift=True)
    tuned=TokenBatchEncoder(native,tmp_path/'tuning.json',clock=clock,report=lambda *a,**k:None)
    result=bench.compare_execution(native,tuned,sample_pair(),Cuda(),clock=clock,report=lambda *a,**k:None)
    assert result['recommendation']=='KEEP_NATIVE_FOR_THIS_SAMPLE'
    assert result['candidate_accepted'] is False


def test_overlapping_evaluation_is_rejected_before_native_encode(tmp_path):
    sample=sample_pair();sample['evaluation'][0]=sample['calibration'][0]
    sample=seal({k:v for k,v in sample.items() if k!='manifest_sha256'})
    clock=Clock();native=Native(clock)
    tuned=TokenBatchEncoder(native,tmp_path/'tuning.json',clock=clock,report=lambda *a,**k:None)
    with pytest.raises(ValueError,match='disjoint'):
        bench.compare_execution(native,tuned,sample,Cuda(),clock=clock)
    assert not native.calls


class Request:
    def __init__(self,fn):self.fn=fn
    def execute(self,**kwargs):return self.fn()


class DriveFiles:
    def __init__(self,*,lost=None):self.store,self.creates,self.ids,self.lost={},[],0,lost
    def generateIds(self,**kwargs):
        self.ids+=1
        return Request(lambda:{'ids':[f'generated-{self.ids}']})
    def create(self,*,body,media_body=None,**kwargs):
        def execute():
            value=dict(body)
            if media_body is not None:
                payload=media_body.getbytes(0,media_body.size())
                value.update(size=str(len(payload)),md5Checksum=hashlib.md5(payload).hexdigest(),
                             webViewLink='https://drive.google.com/file/d/'+value['id']+'/view')
            self.store[value['id']]=value
            self.creates.append(value['id'])
            kind='file' if media_body is not None else 'folder'
            if self.lost==kind:
                self.lost=None
                raise ConnectionError('synthetic lost create response')
            return value
        return Request(execute)
    def get(self,*,fileId,**kwargs):return Request(lambda:self.store[fileId])


@pytest.mark.parametrize('lost',[None,'folder','file'])
def test_report_upload_only_creates_benchmark_artifacts_and_reconciles_lost_responses(tmp_path,lost):
    files=DriveFiles(lost=lost)
    def child(parent,name,**kwargs):
        return next((v for v in files.store.values() if v['parents']==[parent] and v['name']==name),None)
    source=SimpleNamespace(child=child)
    path=tmp_path/'benchmark.json'
    atomic_json(path,seal({'state':'BENCHMARK_COMPLETE_NO_TASKS_COMMITTED',
        'binding':{'job':bench.LIBRARY},'tasks_committed':0,'vectors_published':0}))
    saved=bench.publish_report(source,path,files=files)
    assert len(files.creates)==2
    folder=next(v for v in files.store.values() if v['name']=='qwen_performance_benchmarks')
    assert folder['parents']==[bench.STAGING] and saved['parents']==[folder['id']]
    assert bench.publish_report(source,path,files=files)==saved
    assert len(files.creates)==2


def test_upload_rejects_a_report_that_claims_corpus_progress(tmp_path):
    path=tmp_path/'benchmark.json'
    atomic_json(path,seal({'state':'BENCHMARK_COMPLETE_NO_TASKS_COMMITTED',
        'binding':{'job':bench.LIBRARY},'tasks_committed':1,'vectors_published':0}))
    files=DriveFiles()
    with pytest.raises(ValueError,match='Only a bounded benchmark'):
        bench.publish_report(SimpleNamespace(),path,files=files)
    assert not files.creates


@pytest.mark.parametrize('gpu,match',[('Tesla T4',True),('NVIDIA L4',False)])
def test_gpu_report_records_hardware_and_reuses_measurement_without_loading_model_again(tmp_path,monkeypatch,gpu,match):
    from vietmedbridge import qwen_models
    clock=Clock();native=Native(clock)
    checkout=tmp_path/'checkout'
    model_code=checkout/'src/vietmedbridge/qwen_models.py'
    model_code.parent.mkdir(parents=True)
    model_code.write_text('frozen model code')
    job=seal({'schema':'full-data-text-library-v1','repo_url':'https://github.com/Platypus27-coder/VietMedBridge.git',
        'code_commit':'a'*40,'input_export_complete':True,'coverage_scope':{'official_ids':802028},
        'runtime':{},'config':{'second_dense':{}},'model_code':{'qwen_models.py':sha256_file(model_code)},
        'models':{'qwen':{'encoder':native.identity,'dimension':native.dimension}}})
    monkeypatch.setattr(bench,'LIBRARY',job['manifest_sha256'])
    monkeypatch.setattr(bench,'benchmark_runtime',lambda:{})
    cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda a:gpu,
        get_device_capability=lambda a:(7,5),get_device_properties=lambda a:SimpleNamespace(total_memory=16*1024**3))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    events=[]
    class Owner:
        def __init__(self,*args):events.append('load')
        def for_role(self,role):return native
        def close(self):events.append('close')
    monkeypatch.setattr(qwen_models,'TorchQwenEncoder',Owner)
    def compare(n,t,s,c):
        atomic_json(t.path,{'mode':'native','batch_cap':16})
        return {'native_texts_per_second':6,'candidate_texts_per_second':6,'candidate_accepted':False,
                'held_out_speedup':1,'recommendation':'KEEP_NATIVE_FOR_THIS_SAMPLE'}
    monkeypatch.setattr(bench,'compare_execution',compare)
    sample=sample_pair()
    sample=seal({**{k:v for k,v in sample.items() if k!='manifest_sha256'},'job':job['manifest_sha256'],'source_tasks':[]})
    job_path,sample_path,output=tmp_path/'job.json',tmp_path/'sample.json',tmp_path/'benchmark.json'
    atomic_json(job_path,job);atomic_json(sample_path,sample)
    result=bench.run_benchmark(job_path,sample_path,checkout,output)
    assert result['kaggle_t4_hardware_match'] is match
    assert result['tasks_committed']==result['vectors_published']==0
    assert bench.run_benchmark(job_path,sample_path,checkout,output)==result
    assert events==['load','close']


def test_benchmark_helpers_do_not_import_full_pipeline_or_crawl_dependencies():
    source='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in {'duckdb','datasets','trafilatura','langdetect','kagglehub','pypdf'}:
            raise AssertionError('Full pipeline dependency imported: '+fullname)
sys.meta_path.insert(0,Block())
from scripts.colab_qwen_benchmark import benchmark_runtime,compare_execution
from vietmedbridge.qwen_models import TorchQwenEncoder
'''
    subprocess.run([sys.executable,'-c',source],cwd=ROOT,check=True,capture_output=True,text=True)


def test_delivered_colab_notebook_pins_runtime_and_never_mounts_or_dispatches_corpus_jobs():
    expected=notebook()
    saved=nb.read(ROOT/'notebooks/04_colab_benchmark_qwen_t4.ipynb',as_version=4)
    assert [c.source for c in saved.cells]==[c.source for c in expected.cells]
    for c in saved.cells:
        if c.cell_type=='code':compile(c.source,'<colab-benchmark>','exec')
    code='\n'.join(c.source for c in saved.cells if c.cell_type=='code')
    assert 'drive.mount' not in code and 'KAGGLE_JSON' not in code
    calls={n.func.attr for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&{'create_plan','install','receipts','refs','save_block','publish_lookup','publish_cache_checkpoint'}
    assert "'CUDA_VISIBLE_DEVICES']='0'" in code
    assert "JOB['runtime']" in code and "JOB['code_commit']" in code
    assert 'subprocess.Popen(command,env=env)' in code
    assert 'publish_report(api,REPORT_PATH)' in saved.cells[4].source
    assert 'Popen' not in saved.cells[4].source
