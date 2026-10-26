"""Measured execution-only batching around the immutable corpus encoder.

No weights, tokenizer options, text, pooling, vector dimensions or cache identity
are changed. The native forward is used, with a bounded on-device canary before
selecting any faster batching order. There is no local model inference here.
"""
from contextlib import contextmanager
import gc
import time
from pathlib import Path

import numpy as np

from vietmedbridge.artifacts import atomic_json
from vietmedbridge.embeddings import validate_vectors


POLICY = 'qwen-token-batches-v2'


def token_groups(lengths, cap, budget):
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    group = []
    for i in order:
        if group and (len(group) >= cap or (len(group)+1)*lengths[i] > budget):
            yield group
            group = []
        group.append(i)
    if group:
        yield group


def vector_agreement(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        return False, {'reason':'shape_or_finite'}
    cosine = np.sum(reference*candidate, axis=1) / np.maximum(
        np.linalg.norm(reference,axis=1)*np.linalg.norm(candidate,axis=1), 1e-12)
    minimum = float(cosine.min())
    maximum = float(np.max(np.abs(reference-candidate)))
    return minimum >= 0.99999 and maximum <= 0.001, {'min_cosine':minimum,'max_abs_error':maximum}


class TokenBatchEncoder:
    def __init__(self, native, report_path, *, deadline=None, clock=time.perf_counter, report=print,
                 attention_tuning=False):
        self.native = native
        self.encoder = getattr(native,'encoder',native)
        self.identity, self.dimension = native.identity, native.dimension
        self.path = Path(report_path)
        self.deadline, self.clock, self.report = deadline, clock, report
        self.selected, self.safe_batch_size = None, 16
        self.selected_kernel,self.selected_budget = 'native',24_576
        self.attention_tuning = attention_tuning
        self.caps, self.ooms = {}, 0
        self.attempted = False

    def _attention_api(self):
        backend=getattr(getattr(self.encoder.torch,'backends',None),'cuda',None)
        setter=getattr(backend,'allow_fp16_bf16_reduction_math_sdp',None)
        getter=getattr(backend,'fp16_bf16_reduction_math_sdp_allowed',None)
        return (setter,getter) if callable(setter) and callable(getter) else (None,None)

    @contextmanager
    def _attention(self,kernel):
        if kernel=='native':
            yield
            return
        setter,getter=self._attention_api()
        if kernel!='fp16_math' or setter is None:
            raise ValueError('Unsupported attention execution policy.')
        previous=getter()
        try:
            setter(True)
            yield
        finally:
            setter(previous)

    def lengths(self, texts):
        # Exactly the native tokenizer's options, without batch padding.
        ids = self.encoder.tokenizer(list(texts),padding=False,truncation=False)['input_ids']
        lengths = [len(row) for row in ids]
        if any(n > self.encoder.spec['max_length'] for n in lengths):
            raise ValueError('Qwen input exceeds the frozen max_length; no text was truncated.')
        return lengths

    def _forward(self, texts, indices, lengths, kernel):
        bucket = max(lengths[i] for i in indices).bit_length()
        cap = self.caps.get((kernel,bucket),len(indices))
        if len(indices) > cap:
            return np.concatenate([self._forward(texts,indices[i:i+cap],lengths,kernel)
                                   for i in range(0,len(indices),cap)],axis=0)
        try:
            self.encoder.input_role = 'corpus'
            return np.asarray(self.encoder._batch([texts[i] for i in indices]),dtype=np.float32)
        except self.encoder.torch.OutOfMemoryError:
            if len(indices) == 1:
                raise
            self.ooms += 1
        # Drop the exception/traceback before freeing CUDA allocations.
        cap = max(1,len(indices)//2)
        self.caps[kernel,bucket] = cap
        gc.collect()
        self.encoder.torch.cuda.empty_cache()
        self.report(f'QWEN_BATCH_OOM: token bucket {bucket}; retry same texts with batch {cap}',flush=True)
        return np.concatenate([self._forward(texts,indices[i:i+cap],lengths,kernel)
                               for i in range(0,len(indices),cap)],axis=0)

    def _run(self, texts, lengths, cap, kernel='native', budget=24_576):
        result = np.empty((len(texts),self.dimension),np.float32)
        with self._attention(kernel):
            for indices in token_groups(lengths,cap,budget):
                values = self._forward(texts,indices,lengths,kernel)
                validate_vectors(values,len(indices),self.dimension)
                result[indices] = values
        return result

    def _calibrate(self, texts, lengths):
        self.attempted = True
        started = self.clock()
        order = sorted(range(len(texts)),key=lambda i:lengths[i])
        indices = sorted({order[int(i)] for i in np.linspace(0,len(order)-1,min(64,len(order)))})
        sample = [texts[i] for i in indices]
        sample_lengths = [lengths[i] for i in indices]
        self.report(f'QWEN_TUNING: {len(sample)} existing task texts; native batch 16 versus token batches 16/32/64',flush=True)
        # Warm up only existing task texts. These vectors are never published.
        self.native.encode(sample[:4],batch_size=16)
        tick = self.clock()
        reference = np.asarray(self.native.encode(sample,batch_size=16),dtype=np.float32)
        baseline = max(self.clock()-tick,1e-9)
        validate_vectors(reference,len(sample),self.dimension)
        best, trials = baseline, []
        candidates=[(16,'native',24_576),(32,'native',24_576),(64,'native',24_576)]
        if self.attention_tuning:
            candidates.append((64,'native',49_152))
            setter,getter=self._attention_api()
            if setter is not None and not getter():
                candidates.insert(1,(32,'fp16_math',24_576))
                candidates.insert(2,(64,'fp16_math',49_152))
        for cap,kernel,budget in candidates:
            if self.clock()-started > (120 if self.attention_tuning else 90) or (self.deadline is not None and time.monotonic()+180 >= self.deadline):
                break
            tick = self.clock()
            # Include token-length preparation in the candidate measurement.
            self.lengths(sample)
            try:
                values = self._run(sample,sample_lengths,cap,kernel,budget)
            except ValueError:
                trials.append({'batch_cap':cap,'kernel':kernel,'compatible':False,'reason':'vector_validation_failed'})
                continue
            seconds = max(self.clock()-tick,1e-9)
            compatible, agreement = vector_agreement(reference,values)
            trial = {'batch_cap':cap,'kernel':kernel,'max_padded_tokens':budget,
                     'seconds':seconds,'compatible':compatible,**agreement}
            trials.append(trial)
            self.report(f'QWEN_TUNING_TRIAL: {kernel} cap {cap} | {seconds:.2f}s | compatible {compatible} | native {baseline:.2f}s',flush=True)
            if compatible and seconds < best and seconds <= baseline/1.05:
                best, self.selected = seconds, cap
                self.selected_kernel,self.selected_budget=kernel,budget
        confirmation=None
        if self.selected is not None and self.attention_tuning:
            # A second comparison prevents choosing a single noisy timing win.
            tick=self.clock()
            ref2=np.asarray(self.native.encode(sample,batch_size=16),dtype=np.float32)
            baseline2=max(self.clock()-tick,1e-9)
            tick=self.clock()
            self.lengths(sample)
            try:
                values=self._run(sample,sample_lengths,self.selected,self.selected_kernel,self.selected_budget)
                seconds2=max(self.clock()-tick,1e-9)
                compatible,agreement=vector_agreement(ref2,values)
            except ValueError:
                compatible,agreement,seconds2=False,{'reason':'vector_validation_failed'},baseline2
            confirmation={'compatible':compatible,'native_seconds':baseline2,'selected_seconds':seconds2,**agreement}
            if not compatible or seconds2>baseline2/1.05:
                self.selected,self.selected_kernel,self.selected_budget=None,'native',24_576
                best=baseline
            else:
                baseline,best=baseline2,seconds2
        self.safe_batch_size = self.selected or 16
        record = {'policy':POLICY,'encoder':self.identity,'mode':'token_batches' if self.selected else 'native',
            'sample_texts':len(sample),'baseline_seconds':baseline,'selected_seconds':best,
            'batch_cap':self.safe_batch_size,'kernel':self.selected_kernel,
            'max_padded_tokens':self.selected_budget,'trials':trials,'confirmation':confirmation,
            'sample_tokens':sum(sample_lengths),'sample_token_range':[min(sample_lengths),max(sample_lengths)],
            'measured_sample_speedup':baseline/best,'oom_backoffs':self.ooms,
            'baseline_texts_per_second':len(sample)/baseline,'selected_texts_per_second':len(sample)/best,
            'scope':'Startup canary, not a full-corpus quality or throughput certification.'}
        atomic_json(self.path,record)
        self.report(f'QWEN_TUNING_SELECTED: {record["mode"]} {self.selected_kernel} | cap {self.safe_batch_size} | measured sample speedup {baseline/best:.2f}x | {len(sample)/best:.2f} texts/s | report {self.path}',flush=True)

    def encode(self, texts, *, batch_size=16):
        if not texts:
            return np.empty((0,self.dimension),np.float32)
        if not self.attempted and len(texts) >= 32:
            self._calibrate(texts,self.lengths(texts))
        if self.selected is None:
            return self.native.encode(texts,batch_size=16)
        lengths=self.lengths(texts)
        try:
            return self._run(texts,lengths,self.selected,self.selected_kernel,self.selected_budget)
        except ValueError:
            # Only the current uncommitted task is retried if later inputs expose
            # a numerical issue; previous vector blocks remain unchanged.
            self.selected,self.selected_kernel,self.safe_batch_size=None,'native',16
            from vietmedbridge.artifacts import read_json
            if self.path.is_file():
                saved=read_json(self.path)
                saved.update(active_mode='native',runtime_fallback='vector_validation_failed')
                atomic_json(self.path,saved)
            self.report('QWEN_TUNING_FALLBACK: native batch 16 for current uncommitted texts; previous checkpoints retained.',flush=True)
            return self.native.encode(texts,batch_size=16)


class TokenBatchOwner:
    def __init__(self, owner, report_path, *, deadline=None, attention_tuning=False):
        self.owner, self.path, self.deadline = owner, report_path, deadline
        self.attention_tuning=attention_tuning

    def for_role(self, role):
        native = self.owner.for_role(role)
        if role != 'corpus' or native.identity.get('model_id') != 'Qwen/Qwen3-Embedding-8B':
            return native
        return TokenBatchEncoder(native,self.path,deadline=self.deadline,attention_tuning=self.attention_tuning)

    def close(self):
        self.owner.close()


def benchmark_texts(directory,library):
    """Sample three bounded Qwen task slices; never walk corpus/vector files."""
    import pyarrow.parquet as pq
    from vietmedbridge.artifacts import digest_json,verify_file
    from vietmedbridge.content_embeddings import key
    tasks=[t for t in library['tasks'] if t['family']=='qwen']
    if not tasks:
        raise ValueError('Library has no Qwen task to benchmark.')
    selected=[tasks[i] for i in sorted({0,len(tasks)//2,len(tasks)-1})]
    values=[]
    for task in selected:
        if Path(task['asset']).name!=task['asset']:
            raise ValueError('Unsafe benchmark text asset path.')
        path=Path(directory)/task['asset']
        verify_file(path,task['sha256'])
        table=pq.read_table(path,columns=['id','text'])
        if len(table)!=library['assets'][task['asset']]['rows']:
            raise ValueError('Benchmark asset row count changed.')
        rows=table.slice(task['offset'],task['rows']).to_pylist()
        if (len(rows)!=task['rows'] or digest_json(sorted(r['id'] for r in rows))!=task['keys_digest']
                or any(r['id']!=key(r['text']) for r in rows)):
            raise ValueError('Benchmark task text/key binding changed.')
        indices=sorted({int(i) for i in np.linspace(0,len(rows)-1,min(24,len(rows)))})
        values.extend(rows[i]['text'] for i in indices)
    return values


def benchmark(directory,checkout,output,worker_id):
    import torch
    from full_queue_embeddings import validate_library
    from vietmedbridge.artifacts import read_json
    from vietmedbridge.portable_embeddings import embedding_runtime
    from vietmedbridge.qwen_models import TorchQwenEncoder
    from vietmedbridge.qwen_execution import configure_embedding_execution
    job=validate_library(directory,checkout=checkout)
    if not torch.cuda.is_available() or embedding_runtime()!=job['runtime']:
        raise RuntimeError('Benchmark must use the original producer CUDA runtime.')
    texts=benchmark_texts(directory,job)
    owner=TorchQwenEncoder(job['config']['second_dense'])
    try:
        native=owner.for_role('corpus')
        expected=job['models']['qwen']
        if native.identity!=expected['encoder'] or native.dimension!=expected['dimension']:
            raise ValueError('Benchmark encoder differs from the existing vector producer.')
        configure_embedding_execution(native)
        output=Path(output)
        path=output/f'worker-{worker_id}.json'
        tuned=TokenBatchEncoder(native,path,attention_tuning=True,deadline=time.monotonic()+900)
        result=tuned.encode(texts,batch_size=64)
        validate_vectors(result,len(texts),native.dimension)
        report=read_json(path) if path.is_file() else {'mode':'native','reason':'not_enough_sample_texts'}
        operators=[]
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                native.encode([texts[0],texts[-1]],batch_size=2)
            operators=sorted({item.key for item in profile.key_averages() if 'scaled_dot_product' in item.key})
        except RuntimeError:
            operators=['profiler_unavailable']
        report.update(state='BENCHMARK_COMPLETE_NO_TASKS_COMMITTED',job=job['manifest_sha256'],
            gpu=torch.cuda.get_device_name(0),worker_id=worker_id,validated_texts=len(texts),
            tasks_committed=0,native_sdpa_operators=operators)
        atomic_json(path,report)
        import json
        print('QWEN_BENCHMARK_RESULT:',json.dumps({k:v for k,v in report.items() if k!='encoder'},ensure_ascii=False),flush=True)
        return report
    finally:
        owner.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--checkout',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker-id',type=int,required=True)
    args=parser.parse_args()
    benchmark(args.job,args.checkout,args.output,args.worker_id)
