"""Bounded Colab Qwen benchmark; never installs vectors or changes task journals."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import re
import statistics
import time

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file, utc_now, verify_file
from vietmedbridge.content_embeddings import checked, key, seal

LIBRARY = 'd6ab9a432ac64e5aadb39a826b7dc2176094cc66d4f4e3d9e17c121e071c0711'
STAGING = '1UMiIBJBkeFF4cTVif0QPRmjxTUDI_sYg'
SAMPLE_POLICY = 'first-middle-last-96-alternate-v1'


def benchmark_runtime():
    import importlib.metadata
    return {name:importlib.metadata.version(name) for name in ('torch','transformers','bitsandbytes')}


def validate_job(job):
    checked(job)
    if (job.get('manifest_sha256') != LIBRARY or job.get('schema') != 'full-data-text-library-v1'
            or job.get('repo_url') != 'https://github.com/Platypus27-coder/VietMedBridge.git'
            or not re.fullmatch('[0-9a-f]{40}', job.get('code_commit', ''))
            or not job.get('input_export_complete') or job.get('coverage_scope', {}).get('official_ids') != 802028):
        raise ValueError('Benchmark requires the original frozen 802,028-ID input library.')
    return job


def prepare_sample(source, job, work, *, expected_library=LIBRARY):
    """Read at most three saved missing-text assets, never the original corpus."""
    import pyarrow.parquet as pq
    checked(job)
    if job.get('manifest_sha256') != expected_library:
        raise ValueError('Benchmark sample belongs to another library.')
    tasks = [t for t in job['tasks'] if t['family'] == 'qwen' and t['rows'] >= 64]
    if not tasks:
        raise ValueError('No sufficiently sized Qwen task available for a held-out benchmark.')
    selected = [tasks[i] for i in sorted({0, len(tasks)//2, len(tasks)-1})]
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    provenance = [{'task':t['id'],'asset':t['asset'],'source_sha256':t['sha256']} for t in selected]
    sample_path = work/'sample.json'
    if sample_path.is_file():
        saved = checked(read_json(sample_path))
        if (saved.get('job') != job['manifest_sha256'] or saved.get('policy') != SAMPLE_POLICY
                or saved.get('source_tasks') != provenance or saved.get('tasks_committed') != 0):
            raise ValueError('Saved benchmark sample belongs to another source or policy.')
        print('COLAB_BENCHMARK_SAMPLE_REUSED: one small sample JSON; no source asset replay',flush=True)
        return sample_path, saved
    tables, values, provenance = {}, {}, []
    for task in selected:
        name, asset = task['asset'], job['assets'][task['asset']]
        relative = PurePosixPath(asset['source_path'])
        if (not re.fullmatch('missing-qwen-[0-9]{6}[.]parquet', name) or relative.is_absolute()
                or '..' in relative.parts or '\\' in asset['source_path']
                or task['sha256'] != asset['sha256']):
            raise ValueError('Unsafe or changed benchmark input asset.')
        if name not in tables:
            path = work/name
            if path.is_file():
                verify_file(path, asset['sha256'])
            else:
                metadata = source.relative(asset['source_path'])
                if metadata is None:
                    raise FileNotFoundError('Saved benchmark text is missing from the input-library directory.')
                temporary = path.with_suffix('.partial')
                source.download(metadata, temporary, expected_sha256=asset['sha256'], max_bytes=64*1024*1024)
                temporary.replace(path)
            table = pq.read_table(path, columns=['id', 'text'])
            if len(table) != asset['rows']:
                raise ValueError('Benchmark asset row coverage changed.')
            tables[name] = table
        rows = tables[name].slice(task['offset'], task['rows']).to_pylist()
        if (len(rows) != task['rows'] or digest_json(sorted(r['id'] for r in rows)) != task['keys_digest']
                or any(r['id'] != key(r['text']) for r in rows)):
            raise ValueError('Benchmark input text/key binding changed.')
        count = min(96, len(rows))
        for i in range(count):
            row = rows[i*(len(rows)-1)//max(1,count-1)]
            values.setdefault(row['id'], row['text'])
        provenance.append({'task': task['id'], 'asset': name, 'source_sha256': task['sha256']})
        print('COLAB_BENCHMARK_SAMPLE:',len(tables),'/3 assets |',len(values),'unique texts',flush=True)
    items = list(values.items())
    calibration, evaluation = items[::2], items[1::2]
    if min(len(calibration), len(evaluation)) < 32:
        raise ValueError('Not enough independent calibration/evaluation texts; no model was loaded.')
    result = seal({'schema':'qwen-colab-sample-v1','policy':SAMPLE_POLICY,'job':job['manifest_sha256'],
        'calibration':[{'id':k,'text':v} for k,v in calibration],
        'evaluation':[{'id':k,'text':v} for k,v in evaluation],
        'source_tasks':provenance,'assets_read':len(tables),'tasks_committed':0})
    atomic_json(sample_path, result)
    return sample_path, result


def compare_execution(native, tuned, sample, cuda, *, clock=time.perf_counter, report=print):
    """Alternate two held-out native/tuned passes with CUDA synchronization."""
    import numpy as np
    try:
        from .qwen_token_batching import vector_agreement
    except ImportError:
        from qwen_token_batching import vector_agreement
    from vietmedbridge.embeddings import validate_vectors
    checked(sample)
    calibration, evaluation = sample['calibration'], sample['evaluation']
    if (min(len(calibration),len(evaluation)) < 32
            or {r['id'] for r in calibration} & {r['id'] for r in evaluation}
            or any(r['id'] != key(r['text']) for r in [*calibration,*evaluation])):
        raise ValueError('Calibration and evaluation must have disjoint validated text IDs.')
    calibration = [r['text'] for r in calibration]
    evaluation = [r['text'] for r in evaluation]
    lengths = tuned.lengths(calibration)
    eval_lengths = tuned.lengths(evaluation)
    tuned._calibrate(calibration, lengths)
    native.encode(evaluation[:4], batch_size=16)
    cuda.synchronize()
    runs, agreements = [], []
    for repeat, order in enumerate((('native','tuned'),('tuned','native'))):
        vectors = {}
        for mode in order:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
            start = clock()
            value = native.encode(evaluation,batch_size=16) if mode == 'native' else tuned.encode(evaluation,batch_size=64)
            cuda.synchronize()
            elapsed = max(clock()-start,1e-9)
            validate_vectors(np.asarray(value),len(evaluation),native.dimension)
            vectors[mode] = value
            item = {'repeat':repeat,'mode':mode,'seconds':elapsed,'texts_per_second':len(evaluation)/elapsed,
                    'tokens_per_second':sum(eval_lengths)/elapsed,
                    'peak_allocated_bytes':cuda.max_memory_allocated(),'peak_reserved_bytes':cuda.max_memory_reserved()}
            runs.append(item)
            report('COLAB_BENCHMARK_PASS: '+json.dumps(item),flush=True)
        compatible, error = vector_agreement(vectors['native'],vectors['tuned'])
        agreements.append({'repeat':repeat,'compatible':compatible,**error})
    baseline = statistics.median(r['seconds'] for r in runs if r['mode'] == 'native')
    candidate = statistics.median(r['seconds'] for r in runs if r['mode'] == 'tuned')
    consistent = all(next(r['seconds'] for r in runs if r['repeat']==i and r['mode']=='tuned')
                     <= next(r['seconds'] for r in runs if r['repeat']==i and r['mode']=='native')/1.05 for i in range(2))
    accepted = bool(tuned.selected is not None and consistent and all(a['compatible'] for a in agreements))
    return {'calibration_texts':len(calibration),'evaluation_texts':len(evaluation),
            'evaluation_tokens':sum(eval_lengths),'evaluation_token_range':[min(eval_lengths),max(eval_lengths)],
            'native_texts_per_second':len(evaluation)/baseline,'candidate_texts_per_second':len(evaluation)/candidate,
            'held_out_speedup':baseline/candidate,'candidate_accepted':accepted,'runs':runs,'agreements':agreements,
            'recommendation':'TOKEN_BATCHES_VALIDATED_ON_SAMPLE' if accepted else 'KEEP_NATIVE_FOR_THIS_SAMPLE'}


def run_benchmark(job_path, sample_path, checkout, output, *, force=False):
    import torch
    try:
        from .qwen_token_batching import TokenBatchEncoder
    except ImportError:
        from qwen_token_batching import TokenBatchEncoder
    from vietmedbridge.qwen_models import TorchQwenEncoder
    from vietmedbridge.qwen_execution import configure_embedding_execution
    job, sample = validate_job(read_json(job_path)), checked(read_json(sample_path))
    if sample['job'] != job['manifest_sha256']:
        raise ValueError('Sample and job differ.')
    if not torch.cuda.is_available() or benchmark_runtime() != job['runtime']:
        raise RuntimeError('Select Colab GPU with the exact original CUDA/runtime before benchmarking.')
    for name, checksum in job['model_code'].items():
        verify_file(Path(checkout)/'src/vietmedbridge'/name,checksum)
    gpu = torch.cuda.get_device_name(0)
    binding = {'job':job['manifest_sha256'],'sample':sample['manifest_sha256'],'runtime':job['runtime'],
               'gpu':gpu,'benchmark_code_sha256':sha256_file(__file__),
               'tuning_code_sha256':sha256_file(Path(__file__).with_name('qwen_token_batching.py'))}
    output = Path(output)
    if output.is_file() and not force:
        previous = checked(read_json(output))
        if previous.get('binding') == binding and previous.get('state') == 'BENCHMARK_COMPLETE_NO_TASKS_COMMITTED':
            print('BENCHMARK_REUSED_SAVED_MEASUREMENT:',str(output),'| no GPU work repeated',flush=True)
            return previous
    owner, stage = None, 'model_load'
    try:
        loading_started = time.perf_counter()
        owner = TorchQwenEncoder(job['config']['second_dense'])
        model_load_seconds = time.perf_counter()-loading_started
        native = owner.for_role('corpus')
        expected = job['models']['qwen']
        if native.identity != expected['encoder'] or native.dimension != expected['dimension']:
            raise ValueError('Benchmark encoder differs from existing corpus vectors.')
        configure_embedding_execution(native)
        stage = 'calibration_and_evaluation'
        tuned = TokenBatchEncoder(native,output.with_name('tuning.json'),attention_tuning=True,
                                 deadline=time.monotonic()+1200)
        evaluation = compare_execution(native,tuned,sample,torch.cuda)
        is_t4 = bool(re.search(r'\bT4\b',gpu,re.I))
        result = seal({'state':'BENCHMARK_COMPLETE_NO_TASKS_COMMITTED','binding':binding,
            'gpu':gpu,'compute_capability':list(torch.cuda.get_device_capability(0)),
            'total_vram_bytes':torch.cuda.get_device_properties(0).total_memory,'model_load_seconds':model_load_seconds,
            'vram_metric':'Peak allocated includes the resident model; reserved also includes allocator cache.',
            'kaggle_t4_hardware_match':is_t4,'full_corpus_performance_certified':False,
            'evaluation':evaluation,'calibration':read_json(tuned.path),
            'source_tasks':sample['source_tasks'],'tasks_committed':0,'vectors_published':0,
            'updated_at':utc_now()})
        atomic_json(output,result)
        print('COLAB_QWEN_BENCHMARK_RESULT:',json.dumps({
            'gpu':gpu,'kaggle_t4_hardware_match':is_t4,**{k:v for k,v in evaluation.items() if k not in ('runs','agreements')},
            'batch_cap':tuned.safe_batch_size,'kernel':tuned.selected_kernel,'tasks_committed':0}),flush=True)
        return result
    except Exception as error:
        atomic_json(output,seal({'state':'BENCHMARK_FAILED','binding':binding,'gpu':gpu,
            'stage':stage,'error_type':type(error).__name__,'tasks_committed':0,'vectors_published':0,'updated_at':utc_now()}))
        raise
    finally:
        if owner is not None:
            owner.close()


def publish_report(source, path, *, files=None):
    """Create one report under the project staging folder; never update other data."""
    path = Path(path)
    value = checked(read_json(path))
    if (value.get('state') not in ('BENCHMARK_COMPLETE_NO_TASKS_COMMITTED','BENCHMARK_FAILED')
            or value.get('tasks_committed') != 0 or value.get('vectors_published') != 0
            or value.get('binding',{}).get('job') != LIBRARY or path.stat().st_size > 1024*1024):
        raise ValueError('Only a bounded benchmark report can be published.')
    json.dumps(value,allow_nan=False)
    if files is None:
        import copy
        from googleapiclient.discovery import build
        files = build('drive','v3',credentials=copy.copy(source._session().credentials),cache_discovery=False).files()
    from googleapiclient.http import MediaIoBaseUpload
    fields = 'id,name,mimeType,size,md5Checksum,parents,webViewLink'
    folder = source.child(STAGING,'qwen_performance_benchmarks',folder=True)
    if folder is None:
        identifier = files.generateIds(count=1,space='drive',type='files').execute(num_retries=3)['ids'][0]
        try:
            folder = files.create(body={'id':identifier,'name':'qwen_performance_benchmarks','parents':[STAGING],
                'mimeType':'application/vnd.google-apps.folder'},fields=fields,supportsAllDrives=True).execute(num_retries=3)
        except Exception:
            folder = files.get(fileId=identifier,fields=fields,supportsAllDrives=True).execute(num_retries=3)
        if folder.get('name') != 'qwen_performance_benchmarks' or folder.get('parents') != [STAGING]:
            raise ValueError('Benchmark destination differs from the project staging folder.')
    name = 'qwen-benchmark-'+value['manifest_sha256']+'.json'
    existing = source.child(folder['id'],name)
    payload = path.read_bytes()
    md5 = hashlib.md5(payload).hexdigest()
    if existing is None:
        identifier = files.generateIds(count=1,space='drive',type='files').execute(num_retries=3)['ids'][0]
        media = MediaIoBaseUpload(io.BytesIO(payload),mimetype='application/json',resumable=False)
        try:
            existing = files.create(body={'id':identifier,'name':name,'parents':[folder['id']]},media_body=media,
                fields=fields,supportsAllDrives=True).execute(num_retries=3)
        except Exception:
            existing = files.get(fileId=identifier,fields=fields,supportsAllDrives=True).execute(num_retries=3)
    saved = files.get(fileId=existing['id'],fields=fields,supportsAllDrives=True).execute(num_retries=3)
    if (saved.get('name') != name or saved.get('parents') != [folder['id']]
            or saved.get('md5Checksum') != md5 or int(saved.get('size',-1)) != len(payload)):
        raise ValueError('Drive benchmark report readback differs from the measured result.')
    print('BENCHMARK_SAVED_ON_DRIVE:',saved.get('webViewLink',saved['id']),flush=True)
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('job','sample','checkout','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--force',action='store_true')
    args = parser.parse_args()
    run_benchmark(args.job,args.sample,args.checkout,args.output,force=args.force)
