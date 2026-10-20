"""No verification-only Drive scan; same vectors, bounded I/O and durable resume."""
import ast
import copy
import hashlib
import io as byteio
from pathlib import Path
from types import SimpleNamespace
import tarfile
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.submission_vector_io as optimized
from scripts.drive_receipts_api import DriveReceiptSource
from scripts.submission_vector_io import (VerifiedSubmissionIO, SubmissionVectorParts,
    read_declared_vector_manifest, index_sample)
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file, verify_file
from vietmedbridge.content_embeddings import seal, ContentVectorParts
from vietmedbridge.portable_embeddings import resources_relative


@pytest.fixture
def vectors(tmp_path):
    root, work = tmp_path/'d',tmp_path/'w'
    config = {'dense':{'model_id':'test','revision':'fixed'},'second_dense':{}}
    candidate = {'candidate_manifest_sha256':'a'*64}
    view = root/resources_relative(candidate,config)/'bge'
    view.mkdir(parents=True)
    inputs = seal({'state':'COMPLETE','input_count':12,'parts':[
        {'start':i,'rows':4,'sha256':digest_json(i)} for i in range(0,12,4)]})
    identity = {'input_manifest_sha256':inputs['manifest_sha256'],'input_count':12,
        'encoder':config['dense'],'dimension':1024,'format':'content-reference-v1'}
    signature = digest_json(identity)
    values = np.random.default_rng(79).normal(size=(12,1024)).astype(np.float32)
    values /= np.linalg.norm(values,axis=1,keepdims=True)
    refs,blocks = [],[]
    for start in range(0,12,3):
        path = root/'model_cache'/'producer'/f'block-{start}.npy'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,values[start:start+3],allow_pickle=False)
        checksum = sha256_file(path)
        blocks.append(path)
        refs.extend({'path':path.relative_to(root).as_posix(),'sha256':checksum,'row':i} for i in range(3))
    parts = []
    for source in inputs['parts']:
        start = source['start']
        path = view/f'map-{start:012d}.parquet'
        pq.write_table(pa.Table.from_pylist(refs[start:start+4]),path)
        parts.append(seal({'start':start,'rows':4,'input_sha256':source['sha256'],
            'signature':signature,'path':path.name,'sha256':sha256_file(path)}))
    manifest = seal(identity | {'signature':signature,'state':'COMPLETE','parts':parts})
    atomic_json(view/'config.json',identity);atomic_json(view/'embeddings.json',manifest)
    return SimpleNamespace(root=root,work=work,view=view,manifest=manifest,inputs=inputs,
        candidate=candidate,config=config,values=values,blocks=blocks)


class MemoryAPI:
    def __init__(self,p):
        self.data_root = self.requested_root = p.root
        self.folder_id = 'same-shared-folder'
        self.active = {'build_run':'build','index_inputs_manifest_sha256':p.inputs['manifest_sha256']}
        self.candidate = p.candidate
        self.json = {'processed/build/index_inputs/units.json':p.inputs}
        self.payloads = {}
        for path in p.root.rglob('*'):
            if path.is_file():
                name = path.relative_to(p.root).as_posix()
                if path.suffix=='.json':
                    self.json[name] = read_json(path)
                else:
                    self.payloads[name] = path.read_bytes()
        self.downloads,self.metadata,self.publications = [],[],[]
        self.active_requests,self.peak_requests = 0,0
        self.lock = threading.Lock()
    def relative_json(self,name):
        assert not name.endswith('.done.json'), 'Per-part receipt requested'
        return copy.deepcopy(self.json.get(name))
    def payload_metadata(self,name):
        self.metadata.append(name)
        return {'id':name,'size':str(len(self.payloads[name]))}
    def download(self,meta,target,*,expected_sha256):
        with self.lock:
            self.active_requests += 1
            self.peak_requests = max(self.peak_requests,self.active_requests)
        try:
            time.sleep(.005)
            self.downloads.append(meta['id'])
            Path(target).write_bytes(self.payloads[meta['id']])
            verify_file(target,expected_sha256)
        finally:
            with self.lock:
                self.active_requests -= 1
    def publish_cache_checkpoint(self,name,payload,marker):
        # Exercise the real narrow path/seal validator, with an in-memory upload.
        return DriveReceiptSource.publish_cache_checkpoint(self,name,payload,marker)
    def _publish_lookup_files(self,directory,payload,marker,md5):
        assert hashlib.md5(payload.read_bytes()).hexdigest()==md5
        self.json[directory+'/current.json'] = copy.deepcopy(marker)
        self.payloads[directory+'/'+marker['path']] = payload.read_bytes()
        self.publications.append(directory)
        return marker


def store(p,*,work=None,api=None):
    work = work or p.work
    transport = VerifiedSubmissionIO(work,api=api,progress=lambda _:None)
    return SubmissionVectorParts(p.root,p.view,p.manifest,work/'vector_cache',io=transport),transport


def test_complete_manifest_avoids_every_per_part_read_and_rejects_coverage_hole(vectors,monkeypatch):
    p = vectors
    api = MemoryAPI(p)
    monkeypatch.setattr(optimized,'verify_file',lambda *a:pytest.fail('Preflight opened a payload'))
    view,result = read_declared_vector_manifest(p.root,p.candidate,p.inputs,p.config,api=api)
    assert view==p.view and result==p.manifest and not api.downloads
    name = (p.view/'embeddings.json').relative_to(p.root).as_posix()
    partial = {k:v for k,v in p.manifest.items() if k!='manifest_sha256'}
    partial['parts'] = partial['parts'][:-1]
    api.json[name] = seal(partial)
    with pytest.raises(ValueError,match='coverage gap'):
        read_declared_vector_manifest(p.root,p.candidate,p.inputs,p.config,api=api)


def test_complete_preparation_skips_receipt_inventory_and_weights(vectors,monkeypatch):
    import scripts.submission_gpu_progress as adapter
    p = vectors
    api = MemoryAPI(p)
    checkout = p.work/'checkout'
    atomic_json(checkout/'configs/retrieval_full.json',p.config)
    monkeypatch.setattr(adapter,'ResumableContentBlocks',lambda *a,**k:pytest.fail('Paid receipt inventory repeated'))
    adapter.prepare_drive_receipt_lookup(api,checkout,p.work)
    assert not api.downloads


def test_exact_native_order_duplicates_and_legacy_vm_cache_reused(vectors,monkeypatch):
    p = vectors
    positions = np.array([11,0,7,7,3,5])
    legacy = ContentVectorParts(p.root,p.view,p.manifest,p.work/'vector_cache')
    expected = legacy.gather(positions)
    changed,transport = store(p)
    assert changed.cache==legacy.cache
    np.testing.assert_array_equal(changed.gather(positions),expected)
    # New object/same VM and persistent proof DB: no remote read or rehash.
    resumed,_ = store(p)
    monkeypatch.setattr(optimized,'verify_file',lambda *a:pytest.fail('Unchanged local cache rehashed'))
    monkeypatch.setattr(transport,'relative',lambda *a:pytest.fail('Remote lookup for local cache'))
    np.testing.assert_array_equal(resumed.gather(positions),p.values[positions])


def test_altered_local_cache_is_not_trusted(vectors):
    p = vectors
    current,_ = store(p)
    current.gather([0])
    block = current.block_path({'path':p.blocks[0].relative_to(p.root).as_posix(),'sha256':sha256_file(p.blocks[0])})
    with block.open('ab') as stream:
        stream.write(b'corrupt')
    resumed,_ = store(p)
    with pytest.raises(ValueError,match='changed artifact'):
        resumed.gather([0])


def test_payload_checks_on_use_and_bounded_parallel_api_download(vectors):
    p = vectors
    api = MemoryAPI(p)
    current,_ = store(p,api=api)
    np.testing.assert_array_equal(current.gather(np.arange(12)),p.values)
    assert 1<api.peak_requests<=4
    assert len(api.downloads)==7  # Three maps and four blocks, once each.
    api.downloads.clear();api.metadata.clear()
    np.testing.assert_array_equal(current.gather([11,0,7]),p.values[[11,0,7]])
    assert not api.downloads and not api.metadata
    bad = MemoryAPI(p)
    name = p.blocks[0].relative_to(p.root).as_posix()
    bad.payloads[name] = b'corrupt vector'
    broken,_ = store(p,work=p.work/'broken',api=bad)
    with pytest.raises(ValueError,match='changed artifact'):
        broken.gather([0])
    assert not broken.block_path({'path':name,'sha256':sha256_file(p.blocks[0])}).exists()
    assert not list(broken.cache.glob('*.part'))


def test_new_vm_restores_one_mapping_bundle_and_exact_training_sample(vectors,monkeypatch):
    p = vectors
    api = MemoryAPI(p)
    first,transport = store(p,api=api)
    sample = np.arange(12,dtype=np.int64)
    expected = index_sample(first,sample,transport)
    assert len(api.publications)==2  # One mapping archive, one exact ANN sample.
    api.downloads.clear()
    fresh,transport2 = store(p,work=p.work/'fresh',api=api)
    monkeypatch.setattr(fresh,'gather',lambda _:pytest.fail('Training sample rescanned corpus'))
    np.testing.assert_array_equal(index_sample(fresh,sample,transport2),expected)
    assert len(api.downloads)==1 and api.downloads[0].endswith('.npy')
    third,_ = store(p,work=p.work/'third',api=api)
    api.downloads.clear()
    np.testing.assert_array_equal(third.gather([0,7]),p.values[[0,7]])
    assert sum(name.endswith('.tar.gz') for name in api.downloads)==1
    assert not any(name.endswith('.parquet') for name in api.downloads)


@pytest.mark.parametrize('fault',['member','missing','link'])
def test_invalid_mapping_bundle_cannot_supply_vectors(vectors,fault):
    p = vectors
    api = MemoryAPI(p)
    first,_ = store(p,api=api)
    first.gather(np.arange(12))
    directory = 'retrieval/mapping_bundles/'+p.manifest['manifest_sha256']
    body = byteio.BytesIO()
    with tarfile.open(fileobj=body,mode='w:gz') as archive:
        for index,part in enumerate(p.manifest['parts']):
            if fault=='missing' and index==0:
                continue
            payload = b'bad map' if fault=='member' and index==0 else (p.view/part['path']).read_bytes()
            info = tarfile.TarInfo(part['path']);info.size=len(payload)
            if fault=='link' and index==0:
                info.type = tarfile.SYMTYPE;info.linkname='../outside'
            archive.addfile(info,byteio.BytesIO(payload))
    payload = body.getvalue();checksum=hashlib.sha256(payload).hexdigest()
    marker = api.json[directory+'/current.json']
    marker = seal({k:v for k,v in marker.items() if k!='manifest_sha256'} | {
        'sha256':checksum,'path':'maps-'+checksum+'.tar.gz'})
    api.json[directory+'/current.json']=marker
    api.payloads[directory+'/'+marker['path']]=payload
    resumed,_ = store(p,work=p.work/'corrupt-bundle',api=api)
    with pytest.raises(ValueError,match='Unexpected|checksum|missing parts'):
        resumed.gather([0])


def test_corrupt_ann_sample_is_rejected_without_corpus_rescan(vectors,monkeypatch):
    p=vectors
    api=MemoryAPI(p)
    first,transport=store(p,api=api)
    positions=np.arange(12)
    index_sample(first,positions,transport)
    name=next(name for name in api.payloads if name.startswith('retrieval/index_samples/'))
    api.payloads[name]=b'changed sample'
    resumed,other=store(p,work=p.work/'new-vm',api=api)
    monkeypatch.setattr(resumed,'gather',lambda _:pytest.fail('Fallback corpus rescan'))
    with pytest.raises(ValueError,match='changed artifact'):
        index_sample(resumed,positions,other)


def test_scoped_faiss_builder_uses_checkpointed_sample_and_restores_aliases(vectors,monkeypatch):
    from scripts.submission_gpu_progress import submission_gpu_progress
    from vietmedbridge import (vector_search,content_embeddings,scale_baseline,
        full_scale_runtime,cached_strong_runtime)
    p=vectors
    api=MemoryAPI(p)
    positions=np.array([0,4,9])
    seen=[]
    def build_sample(current,*args,**kwargs):
        seen.append(current.gather(positions))
        np.testing.assert_array_equal(current.part(1),p.values[4:8])
        return {'state':'test double'}
    monkeypatch.setattr(vector_search,'build_faiss_index',build_sample)
    modules=(content_embeddings,scale_baseline,full_scale_runtime,cached_strong_runtime)
    old=[m.ContentVectorParts for m in modules]
    with submission_gpu_progress(p.root,p.work,receipt_api=api):
        current=scale_baseline.ContentVectorParts(p.root,p.view,p.manifest,p.work/'vector_cache')
        vector_search.build_faiss_index(current,p.work/'index')
    assert [m.ContentVectorParts for m in modules]==old
    assert vector_search.build_faiss_index is build_sample
    np.testing.assert_array_equal(seen[0],p.values[positions])
    with submission_gpu_progress(p.root,p.work,receipt_api=api):
        current=scale_baseline.ContentVectorParts(p.root,p.view,p.manifest,p.work/'vector_cache')
        monkeypatch.setattr(current,'gather',lambda _:pytest.fail('Training corpus sampled again'))
        # Building the index also consumes part(): delegate it explicitly to test
        # the sampler proxy independently from native per-shard source reads.
        monkeypatch.setattr(current,'part',lambda _:p.values[4:8])
        vector_search.build_faiss_index(current,p.work/'index')
    np.testing.assert_array_equal(seen[1],seen[0])


def test_resume_patch_and_notebook_embed_current_io_and_no_pc_download():
    import nbformat
    root=Path(__file__).resolve().parents[1]
    helper=(root/'scripts/submission_vector_io.py').read_text(encoding='utf-8').strip()
    for path in ('scripts/resume_submission_optimized_cell.py','scripts/save_submission_lookup_cell.py'):
        source=(root/path).read_text(encoding='utf-8')
        ast.parse(source)
        assert helper in source and 'files.download' not in source
    book=nbformat.read(root/'notebooks/04_colab_submit_available_data.ipynb',as_version=4)
    code=[cell.source for cell in book.cells if cell.cell_type=='code']
    for source in code:
        ast.parse(source)
        assert 'files.download' not in source
    assert all(helper in source for source in code[2:4])
