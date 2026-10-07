"""Real disk/source/export tests. GPU models are explicit CPU test doubles."""
from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_external_import import external, run_import, SPEC, CONFIG, CharacterTokenizer
from vietmedbridge.artifacts import atomic_json, code_fingerprint, digest_json, read_json, sha256_file
from vietmedbridge.competition_pilot import MASTER_PLAN
from vietmedbridge.disk_catalog import prepare_disk_catalog
from vietmedbridge.health import freeze_candidate, health_report
from vietmedbridge.index_inputs import prepare_index_inputs, publish_data_handoff
from vietmedbridge.medical_lexical import MedicalAnalyzer
from vietmedbridge.scale_vectors import VectorParts, search_parts, stream_embeddings


class Tokenizer(CharacterTokenizer):
    def __call__(self,text,**kwargs):
        return {**super().__call__(text,**kwargs),"input_ids":[ord(c) for c in text]}


class Encoder:
    dimension = 4
    def __init__(self,spec=None):
        self.identity = {"test_double":"CPU_NO_RELEVANCE_OR_GPU_EVIDENCE","model_id":"test"}
        self.tokenizer = Tokenizer()
        self.calls = 0
    def encode(self,texts,**kwargs):
        self.calls += 1
        values = np.asarray([[1+b for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts],np.float32)
        return values/np.linalg.norm(values,axis=1,keepdims=True)
    def close(self):
        pass


@pytest.fixture
def frozen(tmp_path,external):
    run_import(tmp_path,external)
    build = tmp_path / "processed/team-100k-data-v1"
    health = health_report(build,official_links=external[1],work_dir=tmp_path / "work",audit_size=2)
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    candidate = freeze_candidate(build,health,golden_report=golden)
    inputs = prepare_index_inputs(build,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",
        Tokenizer(),work_dir=tmp_path / "work",part_size=1)
    publish_data_handoff(tmp_path,"team-100k-data-v1",candidate,inputs)
    return build,candidate,inputs


def test_streamed_order_resume_and_corruption(tmp_path,frozen):
    build,_,_ = frozen
    source = tmp_path / "unordered-inputs"
    source.mkdir()
    parts = []
    for start,texts in ((0,["longer original text","短","abc"]),(3,["z","an even longer text than before","中文"] )):
        path = source / f"inputs-{start}.parquet"
        pq.write_table(pa.Table.from_pylist([{"id":hashlib.sha256(t.encode()).hexdigest(),"text":t} for t in texts]),path)
        parts.append({"path":path.name,"start":start,"rows":len(texts),"sha256":sha256_file(path)})
    inputs = {"state":"COMPLETE","input_count":6,"parts":parts}
    inputs["manifest_sha256"] = digest_json(inputs)
    encoder = Encoder()
    first = stream_embeddings(source,inputs,encoder,tmp_path / "vectors",
        max_new_parts=1,work_dir=tmp_path / "work")
    assert first["state"] == "IN_PROGRESS"
    manifest = stream_embeddings(source,inputs,encoder,tmp_path / "vectors",work_dir=tmp_path / "work")
    assert manifest["state"] == "COMPLETE" and encoder.calls == len(inputs["parts"])
    for source,part in zip(inputs["parts"],manifest["parts"],strict=True):
        texts = pq.read_table(tmp_path / "unordered-inputs" / source["path"]).column("text").to_pylist()
        np.testing.assert_allclose(np.load(tmp_path / "vectors" / part["path"]),Encoder().encode(texts))
    before = encoder.calls
    assert stream_embeddings(tmp_path / "unordered-inputs",inputs,encoder,tmp_path / "vectors",batch_size=8,work_dir=tmp_path / "work") == manifest
    assert encoder.calls == before  # batch change never invalidates semantic checkpoints
    (tmp_path / "vectors" / manifest["parts"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="changed artifact"):
        stream_embeddings(tmp_path / "unordered-inputs",inputs,encoder,tmp_path / "vectors",work_dir=tmp_path / "work")


def test_part_search_matches_bruteforce_and_checks_cache(tmp_path,frozen):
    build,_,inputs = frozen
    encoder = Encoder()
    manifest = stream_embeddings(build / "index_inputs",inputs,encoder,tmp_path / "vectors",work_dir=tmp_path / "work")
    store = VectorParts(tmp_path / "vectors",manifest,tmp_path / "cache")
    all_vectors = store.gather(range(inputs["input_count"]))
    queries = encoder.encode(["医学","triệu chứng"])
    ss,ii,saved = search_parts(store,queries,tmp_path / "search",k=5,query_batch_size=1,device="cpu",work_dir=tmp_path / "work")
    brute = queries @ all_vectors.T
    expected = np.argsort(-brute,axis=1,kind="stable")[:,:5]
    np.testing.assert_array_equal(ii,expected)
    np.testing.assert_allclose(ss,np.take_along_axis(brute,expected,axis=1),atol=1e-6)
    cached = search_parts(store,queries,tmp_path / "search",k=5,query_batch_size=2,device="cpu")
    np.testing.assert_array_equal(cached[1],ii)
    assert cached[2] == saved
    with pytest.raises(ValueError,match="positions"):
        store.gather([-1])
    (tmp_path / "search/positions.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="changed artifact"):
        search_parts(store,queries,tmp_path / "search",k=5,device="cpu")


def test_disk_catalog_preserves_aliases_unicode_and_contentless_postings(tmp_path,frozen,external):
    build,candidate,inputs = frozen
    analyzer = MedicalAnalyzer(segmentation=False)
    catalog = prepare_disk_catalog(build,candidate,inputs,tmp_path / "catalogs",analyzer,work_dir=tmp_path / "work")
    try:
        assert catalog.documents[583]["source_text"] == external[2]
        assert 921 in catalog.documents and 4000 not in catalog.documents
        dense = catalog.dense_documents(range(inputs["input_count"]),np.ones(inputs["input_count"]),200)
        assert [d for d,_ in dense] == [583,921]
        assert {d for d,_ in catalog.sparse("中文医学 triệu chứng","vi",200)} == {583,921}
        assert catalog.con.execute("SELECT body FROM fts_vi LIMIT 1").fetchone()[0] is None
        children = catalog.document_children(921)
        assert children and all(r["text"] == external[2][r["start_char"]:r["end_char"]] for r,_ in children)
        for row,_ in children:
            parent = catalog.parents[row["parent_id"]]
            assert parent["start_char"] <= row["start_char"] < row["end_char"] <= parent["end_char"]
        identity = catalog.identity
    finally:
        catalog.close()
    repeated = prepare_disk_catalog(build,candidate,inputs,tmp_path / "catalogs",analyzer,work_dir=tmp_path / "work")
    assert repeated.identity == identity
    repeated.close()
    path = next((tmp_path / "work").glob("catalog-*/catalog.sqlite"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="changed artifact"):
        prepare_disk_catalog(build,candidate,inputs,tmp_path / "catalogs",analyzer,work_dir=tmp_path / "work")


def test_cuda_block_wiring_with_numpy_tensor_double(tmp_path,frozen,monkeypatch):
    """Exercise CUDA control flow without claiming a GPU measurement."""
    build,_,inputs = frozen
    encoder = Encoder()
    manifest = stream_embeddings(build / "index_inputs",inputs,encoder,tmp_path / "vectors",work_dir=tmp_path / "work")
    store = VectorParts(tmp_path / "vectors",manifest,tmp_path / "cache")
    matmuls = []

    class Tensor:
        def __init__(self,values):
            self.values = np.asarray(values)
        def to(self,_):
            return self
        @property
        def T(self):
            return Tensor(self.values.T)
        def __matmul__(self,other):
            matmuls.append(self.values.shape)
            return Tensor(self.values @ other.values)
        def __getitem__(self,index):
            return Tensor(self.values[index])
        def cpu(self):
            return self
        def numpy(self):
            return self.values

    matmul_policy = SimpleNamespace(allow_tf32=True)
    fake = SimpleNamespace(__version__="CPU_TENSOR_DOUBLE_NOT_GPU",from_numpy=Tensor,
        cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda _:"CPU-double"),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul_policy)),inference_mode=nullcontext,
        argsort=lambda t,**kw:Tensor(np.argsort(-t.values,axis=1,kind="stable")),
        gather=lambda t,dim,ids:Tensor(np.take_along_axis(t.values,ids.values,axis=dim)))
    monkeypatch.setitem(sys.modules,"torch",fake)
    queries = encoder.encode(["医学","triệu chứng"])
    ss,ii,_ = search_parts(store,queries,tmp_path / "cuda",k=2,query_batch_size=1,device="cuda",work_dir=tmp_path / "work")
    cpu = search_parts(store,queries,tmp_path / "cpu",k=2,device="cpu",work_dir=tmp_path / "work")
    np.testing.assert_allclose(ss,cpu[0],atol=1e-6)
    np.testing.assert_array_equal(ii,cpu[1])
    assert len(matmuls) == len(manifest["parts"])*len(queries)
    assert matmul_policy.allow_tf32 is True


def test_entrypoint_all_1200_queries_resume_source_zip_and_feedback(tmp_path,frozen,monkeypatch):
    from vietmedbridge import scale_baseline as runtime
    build,candidate,inputs = frozen
    checkout = Path(__file__).resolve().parents[1]
    config = read_json(checkout / "configs/retrieval_large_baseline.json")
    config.update(dense=SPEC,lexical_segmentation=False)
    full = {"dense":SPEC}
    test_checkout = tmp_path / "checkout"
    atomic_json(test_checkout / "configs/retrieval_large_baseline.json",config)
    atomic_json(test_checkout / "configs/retrieval_full.json",full)
    atomic_json(test_checkout / "configs/strong_model_manifest.json",{"models":[SPEC | {"parameters":4}]})
    (test_checkout / MASTER_PLAN).write_text("CPU wiring fixture, never upload these predictions",encoding="utf-8")
    queries = [{"id":10001+i*7,"query":"Người bệnh triệu chứng 中文医学"} for i in range(1200)]
    query_path = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pylist(queries),query_path)
    atomic_json(tmp_path / "raw/snapshot.json",{"files":{
        "query.parquet":{"path":"query.parquet","sha256":sha256_file(query_path)},
        "links_corpus.parquet":{"sha256":candidate["origin_corpus_sha256"]}}})
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
    monkeypatch.setattr(runtime,"review_model_registry",lambda *_:None)
    encoder = Encoder()
    monkeypatch.setattr(runtime,"TorchDenseEncoder",lambda _:encoder)
    real_search = runtime.search_parts
    monkeypatch.setattr(runtime,"search_parts",lambda *a,**kw:real_search(*a,**kw,device="cpu"))
    embedding_partial = runtime.run_large_baseline(tmp_path,test_checkout,code_commit="CPU-test",work_dir=tmp_path / "work",max_new_embedding_parts=1)
    assert embedding_partial["ready"]["state"] == "EMBEDDING_IN_PROGRESS"
    partial = runtime.run_large_baseline(tmp_path,test_checkout,code_commit="CPU-test",work_dir=tmp_path / "work",max_new_queries=3)
    assert partial["ready"]["state"] == "QUERIES_IN_PROGRESS" and partial["ready"]["query_count"] == 3
    calls = encoder.calls
    result = runtime.run_large_baseline(tmp_path,test_checkout,code_commit="CPU-test",work_dir=tmp_path / "work")
    assert result["ready"]["state"] == "READY_FOR_MANUAL_UPLOAD"
    assert encoder.calls == calls
    assert result["status"]["query_count"] == 1200 and result["status"]["complete_master_plan"] is False
    zip_path = Path(result["ready"]["zip_path"])
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["results.json"]
        predictions = json.loads(archive.read("results.json"))
    assert [p["id"] for p in predictions] == [q["id"] for q in queries]
    assert all(set(p["relevant_docs"]) == {583,921} for p in predictions)
    sources = {d:build_source(tmp_path,d) for d in (583,921)}
    assert all(c["chunk_text"] in sources[c["doc_id"]] for p in predictions for c in p["relevant_chunks"])
    feedback_path = zip_path.parent / "score_feedback.json"
    feedback = read_json(feedback_path)
    feedback["overall_score"] = .12345
    atomic_json(feedback_path,feedback)
    repeated = runtime.run_large_baseline(tmp_path,test_checkout,code_commit="CPU-test",work_dir=tmp_path / "work")
    assert repeated["ready"]["zip_sha256"] == result["ready"]["zip_sha256"]
    assert read_json(feedback_path)["overall_score"] == .12345 and encoder.calls == calls


def build_source(root,doc_id):
    # Test helper only; production validation reads through the bounded disk catalog.
    build = root / "processed/team-100k-data-v1"
    manifest = read_json(build / "build.json")
    for part in manifest["parts"]:
        for row in pq.read_table(build / part["files"]["documents"]["path"]).to_pylist():
            if row["doc_id"] == doc_id:
                return row["source_text"]
    raise AssertionError("missing source")
