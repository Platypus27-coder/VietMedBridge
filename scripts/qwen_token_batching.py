"""Measured execution-only batching around the immutable corpus encoder.

No weights, tokenizer options, text, pooling, vector dimensions or cache identity
are changed. The native forward is used, with a bounded on-device canary before
selecting any faster batching order. There is no local model inference here.
"""
import gc
import time
from pathlib import Path

import numpy as np

from vietmedbridge.artifacts import atomic_json
from vietmedbridge.embeddings import validate_vectors


POLICY = 'qwen-token-batches-v1'


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
    def __init__(self, native, report_path, *, deadline=None, clock=time.perf_counter, report=print):
        self.native = native
        self.encoder = getattr(native,'encoder',native)
        self.identity, self.dimension = native.identity, native.dimension
        self.path = Path(report_path)
        self.deadline, self.clock, self.report = deadline, clock, report
        self.selected, self.safe_batch_size = None, 16
        self.caps, self.ooms = {}, 0
        self.attempted = False

    def lengths(self, texts):
        # Exactly the native tokenizer's options, without batch padding.
        ids = self.encoder.tokenizer(list(texts),padding=False,truncation=False)['input_ids']
        lengths = [len(row) for row in ids]
        if any(n > self.encoder.spec['max_length'] for n in lengths):
            raise ValueError('Qwen input exceeds the frozen max_length; no text was truncated.')
        return lengths

    def _forward(self, texts, indices, lengths):
        bucket = max(lengths[i] for i in indices).bit_length()
        cap = self.caps.get(bucket,len(indices))
        if len(indices) > cap:
            return np.concatenate([self._forward(texts,indices[i:i+cap],lengths)
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
        self.caps[bucket] = cap
        gc.collect()
        self.encoder.torch.cuda.empty_cache()
        self.report(f'QWEN_BATCH_OOM: token bucket {bucket}; retry same texts with batch {cap}',flush=True)
        return np.concatenate([self._forward(texts,indices[i:i+cap],lengths)
                               for i in range(0,len(indices),cap)],axis=0)

    def _run(self, texts, lengths, cap):
        result = np.empty((len(texts),self.dimension),np.float32)
        for indices in token_groups(lengths,cap,24_576):
            values = self._forward(texts,indices,lengths)
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
        for cap in (16,32,64):
            if self.clock()-started > 90 or (self.deadline is not None and time.monotonic()+120 >= self.deadline):
                break
            tick = self.clock()
            # Include token-length preparation in the candidate measurement.
            self.lengths(sample)
            try:
                values = self._run(sample,sample_lengths,cap)
            except ValueError:
                trials.append({'batch_cap':cap,'compatible':False,'reason':'vector_validation_failed'})
                continue
            seconds = max(self.clock()-tick,1e-9)
            compatible, agreement = vector_agreement(reference,values)
            trial = {'batch_cap':cap,'seconds':seconds,'compatible':compatible,**agreement}
            trials.append(trial)
            self.report(f'QWEN_TUNING_TRIAL: cap {cap} | {seconds:.2f}s | compatible {compatible} | native {baseline:.2f}s',flush=True)
            if compatible and seconds < best and seconds <= baseline/1.05:
                best, self.selected = seconds, cap
        self.safe_batch_size = self.selected or 16
        record = {'policy':POLICY,'encoder':self.identity,'mode':'token_batches' if self.selected else 'native',
            'sample_texts':len(sample),'baseline_seconds':baseline,'selected_seconds':best,
            'batch_cap':self.safe_batch_size,'max_padded_tokens':24_576,'trials':trials,
            'sample_tokens':sum(sample_lengths),'sample_token_range':[min(sample_lengths),max(sample_lengths)],
            'measured_sample_speedup':baseline/best,'oom_backoffs':self.ooms,
            'scope':'Startup canary, not a full-corpus quality or throughput certification.'}
        atomic_json(self.path,record)
        self.report(f'QWEN_TUNING_SELECTED: {record["mode"]} | cap {self.safe_batch_size} | measured sample speedup {baseline/best:.2f}x | report {self.path}',flush=True)

    def encode(self, texts, *, batch_size=16):
        if not texts:
            return np.empty((0,self.dimension),np.float32)
        if not self.attempted and len(texts) >= 32:
            self._calibrate(texts,self.lengths(texts))
        if self.selected is None:
            return self.native.encode(texts,batch_size=16)
        return self._run(texts,self.lengths(texts),self.selected)


class TokenBatchOwner:
    def __init__(self, owner, report_path, *, deadline=None):
        self.owner, self.path, self.deadline = owner, report_path, deadline

    def for_role(self, role):
        native = self.owner.for_role(role)
        if role != 'corpus' or native.identity.get('model_id') != 'Qwen/Qwen3-Embedding-8B':
            return native
        return TokenBatchEncoder(native,self.path,deadline=self.deadline)

    def close(self):
        self.owner.close()
