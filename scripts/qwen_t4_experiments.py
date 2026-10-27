"""Benchmark-only execution trials; the live Kaggle v2 worker stays unchanged."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import re
import statistics
import time

import numpy as np

from vietmedbridge.artifacts import atomic_json,read_json,sha256_file,utc_now,verify_file
from vietmedbridge.content_embeddings import checked,seal
from vietmedbridge.embeddings import validate_vectors
try:
    from .qwen_token_batching import TokenBatchEncoder,vector_agreement
    from .colab_qwen_benchmark import validate_job,benchmark_runtime
except ImportError:
    from qwen_token_batching import TokenBatchEncoder,vector_agreement
    from colab_qwen_benchmark import validate_job,benchmark_runtime

POLICY='qwen-t4-extra-trials-v1'
CONFIGS=(
    {'name':'auto-b8','cap':8,'backend':'auto','threads':None},
    {'name':'auto-b12','cap':12,'backend':'auto','threads':None},
    {'name':'auto-b24','cap':24,'backend':'auto','threads':None},
    {'name':'efficient-b16','cap':16,'backend':'efficient','threads':None},
    {'name':'efficient-b24','cap':24,'backend':'efficient','threads':None},
    {'name':'threads1-b16','cap':16,'backend':'auto','threads':1},
    {'name':'threads1-efficient-b16','cap':16,'backend':'efficient','threads':1},
)


@contextmanager
def efficient_attention():
    from transformers.integrations import sdpa_attention as integration
    from torch.nn.attention import SDPBackend,sdpa_kernel
    original=integration.use_gqa_in_sdpa
    try:
        # Equivalent K/V head expansion lets the efficient backend be tried
        # on torch 2.8, where fused GQA is otherwise restricted to flash/math.
        # Masks, causality and all model/tokenizer parameters are untouched.
        integration.use_gqa_in_sdpa=lambda *a,**k:False
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            yield
    finally:
        integration.use_gqa_in_sdpa=original


@contextmanager
def execution_context(torch,config):
    if config not in CONFIGS:
        raise ValueError('Unknown benchmark execution configuration.')
    threads=config['threads']
    previous=torch.get_num_threads() if threads is not None else None
    try:
        if threads is not None:torch.set_num_threads(threads)
        if config['backend']=='efficient':
            with efficient_attention():yield
        else:yield
    finally:
        if previous is not None:torch.set_num_threads(previous)


def _known_unavailable(error,torch):
    if isinstance(error,torch.OutOfMemoryError):return True
    return isinstance(error,RuntimeError) and any(s in str(error).lower() for s in
        ('no available kernel','no kernel found','no viable backend','not compiled with','requires sm','does not support this'))


def run_suite(native,reference,sample,cuda,*,clock=time.perf_counter,report=print):
    checked(sample)
    calibration,evaluation=sample['calibration'],sample['evaluation']
    from vietmedbridge.content_embeddings import key
    if ({r['id'] for r in calibration}&{r['id'] for r in evaluation}
            or min(len(calibration),len(evaluation))<32
            or any(r['id']!=key(r['text']) for r in [*calibration,*evaluation])):
        raise ValueError('Independent validated calibration/evaluation texts are required.')
    calibration=[r['text'] for r in calibration]
    evaluation=[r['text'] for r in evaluation]
    reference._calibrate(calibration,reference.lengths(calibration))
    count=min(64,len(calibration))
    trial_texts=[calibration[i*(len(calibration)-1)//max(1,count-1)] for i in range(count)]
    runners={}
    owner=getattr(native,'encoder',native)
    def variant(config,texts):
        name=config['name']
        if name not in runners:
            runners[name]=TokenBatchEncoder(native,reference.path.with_name(name+'.json'),report=report)
        runner=runners[name]
        lengths=runner.lengths(texts)
        with execution_context(owner.torch,config):
            return runner._run(texts,lengths,config['cap'],'native',24_576)
    def measure(operation,rows):
        cuda.synchronize();cuda.reset_peak_memory_stats()
        started=clock()
        value=np.asarray(operation(),dtype=np.float32)
        cuda.synchronize()
        elapsed=max(clock()-started,1e-9)
        validate_vectors(value,rows,native.dimension)
        return value,elapsed,{'peak_allocated_bytes':cuda.max_memory_allocated(),
                              'peak_reserved_bytes':cuda.max_memory_reserved()}
    frozen,_,_=measure(lambda:native.encode(trial_texts,batch_size=16),len(trial_texts))
    _,reference_seconds,_=measure(lambda:reference.encode(trial_texts,batch_size=64),len(trial_texts))
    best,best_seconds,trials=None,reference_seconds,[]
    started=clock()
    for config in CONFIGS:
        if clock()-started>240:
            trials.append({'name':config['name'],'state':'SKIPPED_TIME_BUDGET'})
            continue
        if config['threads']==1 and owner.torch.get_num_threads()==1:
            trials.append({'name':config['name'],'state':'SKIPPED_ALREADY_ONE_THREAD'})
            continue
        try:
            variant(config,trial_texts[:4])  # backend warm-up is outside timing
            value,seconds,memory=measure(lambda:variant(config,trial_texts),len(trial_texts))
            compatible,agreement=vector_agreement(frozen,value)
            item={**config,'state':'MEASURED','seconds':seconds,'compatible':compatible,
                  'speedup_over_v2':reference_seconds/seconds,**agreement,**memory,
                  'oom_backoffs':runners[config['name']].ooms}
            if compatible and seconds<best_seconds and seconds<=reference_seconds/1.05:
                best,best_seconds=config,seconds
        except (RuntimeError,ValueError) as error:
            if isinstance(error,RuntimeError) and not _known_unavailable(error,owner.torch):raise
            item={**config,'state':'UNAVAILABLE' if isinstance(error,RuntimeError) else 'INVALID_VECTORS',
                  'error_type':type(error).__name__}
        trials.append(item)
        report('T4_EXTRA_TRIAL: '+json.dumps(item),flush=True)
    if best is None:
        return {'measurement_scope':'calibration_only_no_faster_candidate','candidate_accepted':False,
                'reference_v2_texts_per_second':len(trial_texts)/reference_seconds,
                'candidate_texts_per_second':len(trial_texts)/reference_seconds,
                'additional_speedup_over_v2':1.0,'selected_config':None,'trials':trials,
                'evaluation_texts':0,'recommendation':'KEEP_EXISTING_V2'}
    runs,agreements=[],[]
    for repeat,order in enumerate((('original_native','reference_v2','candidate'),
                                   ('candidate','reference_v2','original_native'))):
        vectors={}
        for mode in order:
            operation=(lambda:native.encode(evaluation,batch_size=16)) if mode=='original_native' else (
                (lambda:reference.encode(evaluation,batch_size=64)) if mode=='reference_v2' else
                (lambda:variant(best,evaluation)))
            try:
                value,seconds,memory=measure(operation,len(evaluation))
            except (RuntimeError,ValueError) as error:
                if mode!='candidate' or (isinstance(error,RuntimeError) and not _known_unavailable(error,owner.torch)):
                    raise
                # A backend that fails on held-out shapes cannot be recommended.
                # Do not turn a partial timing into a claimed speedup.
                return {'measurement_scope':'held_out_candidate_failed','candidate_accepted':False,
                    'reference_v2_texts_per_second':len(trial_texts)/reference_seconds,
                    'candidate_texts_per_second':None,'additional_speedup_over_v2':None,
                    'selected_config':None,'best_calibration_candidate':best,
                    'evaluation_texts':len(evaluation),'trials':trials,'runs':runs,
                    'error_type':type(error).__name__,'recommendation':'KEEP_EXISTING_V2'}
            vectors[mode]=value
            item={'repeat':repeat,'mode':mode,'seconds':seconds,'texts_per_second':len(evaluation)/seconds,**memory}
            runs.append(item)
            report('T4_EXTRA_PASS: '+json.dumps(item),flush=True)
        compatible,agreement=vector_agreement(vectors['original_native'],vectors['candidate'])
        agreements.append({'repeat':repeat,'compatible':compatible,**agreement})
    def seconds(mode):return statistics.median(r['seconds'] for r in runs if r['mode']==mode)
    consistent=all(next(r['seconds'] for r in runs if r['repeat']==i and r['mode']=='candidate')
        <=next(r['seconds'] for r in runs if r['repeat']==i and r['mode']=='reference_v2')/1.05 for i in range(2))
    accepted=consistent and all(a['compatible'] for a in agreements)
    return {'measurement_scope':'held_out_two_passes_against_existing_v2','candidate_accepted':accepted,
            'reference_v2_texts_per_second':len(evaluation)/seconds('reference_v2'),
            'original_native_texts_per_second':len(evaluation)/seconds('original_native'),
            'candidate_texts_per_second':len(evaluation)/seconds('candidate'),
            'additional_speedup_over_v2':seconds('reference_v2')/seconds('candidate'),
            'selected_config':best if accepted else None,'best_calibration_candidate':best,
            'evaluation_texts':len(evaluation),'trials':trials,'runs':runs,'agreements':agreements,
            'recommendation':'NEW_CANDIDATE_VALIDATED_ON_SAMPLE' if accepted else 'KEEP_EXISTING_V2'}


def run(job_path,sample_path,checkout,output,*,force=False):
    import torch
    from vietmedbridge.qwen_models import TorchQwenEncoder
    from vietmedbridge.qwen_execution import configure_embedding_execution
    job,sample=validate_job(read_json(job_path)),checked(read_json(sample_path))
    if sample['job']!=job['manifest_sha256'] or not torch.cuda.is_available() or benchmark_runtime()!=job['runtime']:
        raise ValueError('Original sample and CUDA runtime are required.')
    for name,checksum in job['model_code'].items():verify_file(Path(checkout)/'src/vietmedbridge'/name,checksum)
    gpu=torch.cuda.get_device_name(0)
    binding={'job':job['manifest_sha256'],'sample':sample['manifest_sha256'],'runtime':job['runtime'],'gpu':gpu,
             'policy':POLICY,'experiment_code_sha256':sha256_file(__file__),
             'tuning_code_sha256':sha256_file(Path(__file__).with_name('qwen_token_batching.py'))}
    output=Path(output)
    if output.is_file() and not force:
        previous=checked(read_json(output))
        if previous.get('binding')==binding and previous.get('state')=='BENCHMARK_COMPLETE_NO_TASKS_COMMITTED':
            print('T4_EXTRA_REUSED_SAVED_MEASUREMENT:',str(output),flush=True)
            return previous
    owner=None
    try:
        owner=TorchQwenEncoder(job['config']['second_dense'])
        native=owner.for_role('corpus')
        if native.identity!=job['models']['qwen']['encoder'] or native.dimension!=job['models']['qwen']['dimension']:
            raise ValueError('Native model differs from the existing vector producer.')
        configure_embedding_execution(native)
        reference=TokenBatchEncoder(native,output.with_name('reference-v2-tuning.json'),attention_tuning=True,
                                   deadline=time.monotonic()+1200)
        evaluation=run_suite(native,reference,sample,torch.cuda)
        result=seal({'state':'BENCHMARK_COMPLETE_NO_TASKS_COMMITTED','binding':binding,'gpu':gpu,
            'kaggle_t4_hardware_match':bool(re.search(r'\bT4\b',gpu,re.I)),'full_corpus_performance_certified':False,
            'evaluation':evaluation,'calibration':read_json(reference.path),'source_tasks':sample['source_tasks'],
            'tasks_committed':0,'vectors_published':0,'production_worker_changed':False,'updated_at':utc_now()})
        atomic_json(output,result)
        print('T4_EXTRA_RESULT:',json.dumps(evaluation),flush=True)
        return result
    except Exception as error:
        atomic_json(output,seal({'state':'BENCHMARK_FAILED','binding':binding,'gpu':gpu,
            'error_type':type(error).__name__,'tasks_committed':0,'vectors_published':0,'updated_at':utc_now()}))
        raise
    finally:
        if owner is not None:owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('job','sample','checkout','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    run(args.job,args.sample,args.checkout,args.output,force=args.force)
