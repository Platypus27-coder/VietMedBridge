"""Real frozen catalog/export tests with CPU model doubles, not relevance tests."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_scale_baseline import Encoder, frozen, external
from test_full_scale import FullTokenizer
from vietmedbridge.artifacts import atomic_json, read_json, sha256_file
from vietmedbridge.content_embeddings import content_embeddings
from vietmedbridge.portable_embeddings import resources_relative
from vietmedbridge.qwen_models import RoleEncoder


@pytest.fixture
def prepared(tmp_path, frozen, monkeypatch):
    from vietmedbridge import full_scale_runtime as runtime
    repo = Path(__file__).resolve().parents[1]
    config = read_json(repo / "configs/retrieval_full.json")
    config["dense"] = {"model_id":"test", "revision":"test", "max_length":512}
    config["lexical_segmentation"] = False
    config["retrieval"].update(doc_candidate_k=2, detail_doc_k=2, doc_min_k=1, doc_top_k=2,
        chunk_min_k=1, chunk_top_k=2)
    target = tmp_path / "checkout"
    for name, value in (("retrieval_full",config), ("retrieval_scale",read_json(repo / "configs/retrieval_scale.json")),
                        ("strong_model_manifest",{"test_only":True})):
        atomic_json(target / f"configs/{name}.json", value)
    (target / runtime.MASTER_PLAN).write_text("CPU doubles. No quality or GPU claims.", encoding="utf-8")
    queries = [{"id":10001+i*7, "query":"triệu chứng 中文医学"} for i in range(1200)]
    query_path = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pylist(queries), query_path)
    atomic_json(tmp_path / "raw/snapshot.json", {"files":{
        "query.parquet":{"path":"query.parquet", "sha256":sha256_file(query_path)},
        "links_corpus.parquet":{"sha256":frozen[1]["origin_corpus_sha256"]}}})
    counters = {name:0 for name in ("bge", "qwen", "translation", "reranker")}
    encoded = []
    class Dense(Encoder):
        dimension = 1024
        def __init__(self, spec):
            super().__init__()
            self.identity.update(spec)
            self.tokenizer = FullTokenizer()
        def encode(self, texts, **kwargs):
            counters["bge"] += len(texts)
            encoded.extend(texts)
            values = np.tile(Encoder.encode(self,texts), (1,self.dimension//4))
            return values/np.linalg.norm(values,axis=1,keepdims=True)
    class Secondary(Dense):
        dimension = 4096
        def __init__(self, spec):
            super().__init__(spec)
            self.identity.update(role="second_dense")
        def for_role(self, role):
            return RoleEncoder(self,role)
        def encode(self, texts, **kwargs):
            counters["qwen"] += len(texts)
            values = np.tile(Encoder.encode(self,texts),(1,self.dimension//4))
            return values/np.linalg.norm(values,axis=1,keepdims=True)
    class Translation:
        def __init__(self,spec):
            self.identity = spec | {"inference_code_sha256":sha256_file(repo / "src/vietmedbridge/translation_model.py")}
        def translate(self, query, prompt):
            counters["translation"] += 1
            return json.dumps({"original_vi":query,"entities":[],"constraints":[],"query_en":query,"query_zh":query})
        def close(self):
            pass
    class Reranker:
        def __init__(self,spec,adapter_path=None):
            self.identity = spec | {"fine_tuned":False}
        def score(self,pairs,**kwargs):
            counters["reranker"] += len(pairs)
            return np.ones(len(pairs),np.float32)
        def windows(self,query,text,overlap):
            return [{"text":text,"start_char":0,"end_char":len(text)}]
        def close(self):
            pass
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:FullTokenizer())))
    monkeypatch.setattr(runtime,"TorchDenseEncoder",Dense)
    monkeypatch.setattr(runtime,"TorchQwenEncoder",Secondary)
    monkeypatch.setattr(runtime,"TorchQueryTranslator",Translation)
    monkeypatch.setattr(runtime,"TorchQwenReranker",Reranker)
    monkeypatch.setattr(runtime,"review_model_registry",lambda *a:None)
    monkeypatch.setattr(runtime,"model_budget_report",lambda *a,**k:{"test_double":True})
    real_search = runtime.search_parts
    monkeypatch.setattr(runtime,"search_parts",lambda *a,**kw:real_search(*a,**(kw | {"device":"cpu"})))
    build,candidate,inputs = frozen
    resources = tmp_path / resources_relative(candidate,config)
    content_embeddings(tmp_path,build / "index_inputs",inputs,Dense(config["dense"]),resources / "bge",work_dir=tmp_path / "work")
    # Create a complete remote catalog once. The inference path must restore it
    # without modifying any source file or rebuilding corpus/canonical data.
    catalog,_ = runtime.load_scale_catalog(tmp_path,target,tmp_path / "work",FullTokenizer())
    catalog.close()
    counters.update({name:0 for name in counters})
    encoded.clear()
    work = tmp_path.parent / (tmp_path.name+"-readonly-work")
    outputs = tmp_path.parent / (tmp_path.name+"-results")
    return SimpleNamespace(root=tmp_path,checkout=target,work=work,outputs=outputs,config=config,resources=resources,
        frozen=frozen,queries=queries,counters=counters,encoded=encoded,secondary=Secondary)


def invoke(p, **kwargs):
    from vietmedbridge.full_scale_runtime import run_scale_full_pipeline
    return run_scale_full_pipeline(p.root,p.checkout,code_commit="CPU-test",work_dir=p.work,
        output_root=p.outputs,execution_mode="cached_strong",**kwargs)


def source_hashes(root):
    return {p.relative_to(root).as_posix():sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_cached_strong_no_qwen_full_submission_and_resume_without_source_writes(prepared):
    p = prepared
    before_source = source_hashes(p.root)
    partial = invoke(p,max_new_queries=3)
    assert partial["ready"] == {"state":"QUERIES_IN_PROGRESS","query_count":3}
    assert p.counters["qwen"] == 0 and p.counters["translation"] and p.counters["reranker"]
    assert partial["status"]["branches"]["document_dense"] == "COMPLETE"
    before = dict(p.counters)
    result = invoke(p)
    assert result["ready"]["state"] == "READY_FOR_MANUAL_UPLOAD"
    assert result["status"]["query_count"] == 1200 and result["status"]["strong_inference"]
    assert result["status"]["complete_master_plan"] is False
    assert result["status"]["full_plan_inference"] is False
    assert result["status"]["coverage"]["child_corpus_reencoded"] is False
    assert result["status"]["branches"]["auxiliary_dense_family"] == "bge"
    assert result["status"]["branches"]["source_schema_submission"] == "COMPLETE"
    assert all(p.counters[k] == before[k] for k in ("bge","translation","qwen"))
    corpus_texts = {t for part in p.frozen[2]["parts"] for t in pq.read_table(
        p.frozen[0] / "index_inputs" / part["path"],columns=["text"]).column("text").to_pylist()}
    assert not corpus_texts.intersection(p.encoded)
    with ZipFile(result["ready"]["zip_path"]) as z:
        assert z.namelist() == ["results.json"]
        predictions = json.loads(z.read("results.json"))
    assert [r["id"] for r in predictions] == [q["id"] for q in p.queries]
    assert all(set(r["relevant_docs"]) == {583,921} for r in predictions)
    assert source_hashes(p.root) == before_source
    before = dict(p.counters)
    repeated = invoke(p)
    assert repeated["ready"]["zip_sha256"] == result["ready"]["zip_sha256"]
    assert p.counters == before and source_hashes(p.root) == before_source


@pytest.mark.parametrize("complete",[False,True])
def test_qwen_complete_cache_optional_and_partial_cache_never_searched(prepared,complete):
    p = prepared
    content_embeddings(p.root,p.frozen[0] / "index_inputs",p.frozen[2],p.secondary(p.config["second_dense"]).for_role("corpus"),
        p.resources / "qwen",work_dir=p.root / "work",max_new_parts=None if complete else 1)
    p.counters["qwen"] = 0
    before_source = source_hashes(p.root)
    result = invoke(p,document_dense_enabled=False,max_new_queries=2)
    assert result["ready"]["query_count"] == 2
    assert result["status"]["coverage"]["qwen_dense_used"] == complete
    assert bool(p.counters["qwen"]) == complete
    assert result["status"]["branches"]["document_dense"] == "DISABLED"
    assert result["status"]["branches"]["auxiliary_dense_family"] == ("qwen" if complete else "bge")
    assert source_hashes(p.root) == before_source
    if not complete:
        before = dict(p.counters)
        waiting = invoke(p,second_dense_mode="require")
        assert waiting["ready"]["state"] == "WAITING_FOR_COMPLETE_QWEN_CACHE" and p.counters == before


def test_cached_strong_bge_coverage_gate_precedes_models_and_catalog(prepared,monkeypatch):
    p = prepared
    from vietmedbridge import full_scale_runtime as full
    (p.resources / "bge" / f"map-{p.frozen[2]['parts'][0]['start']:012d}.done.json").unlink()
    monkeypatch.setattr(full,"load_scale_catalog",lambda *a,**k:pytest.fail("Incomplete BGE must stop before catalog/models"))
    before = dict(p.counters)
    result = invoke(p)
    assert result["ready"]["state"] == "WAITING_FOR_COMPLETE_BGE_CACHE" and p.counters == before


def test_bge_auxiliary_branches_preserve_original_search_without_qwen(prepared,monkeypatch):
    p = prepared
    from vietmedbridge import full_scale_runtime as runtime
    from vietmedbridge.cached_strong_runtime import prepare_cached_strong_resources
    from vietmedbridge.query_expansion import PICO_FIELDS
    from vietmedbridge.strong_retrieval import StrongConfig
    original_translator = runtime.TorchQueryTranslator
    class ComplexTranslator(original_translator):
        def translate(self,query,prompt):
            if "pico" not in prompt:
                return super().translate(query,prompt)
            p.counters["translation"] += 1
            return json.dumps({"original_vi":query,"pico":{k:[] for k in PICO_FIELDS},
                "subqueries":[query+" tìm kiếm"],"hyde_en":query+" bằng chứng"})
    monkeypatch.setattr(runtime,"TorchQueryTranslator",ComplexTranslator)
    query = {"id":123,"query":"Tôi có các triệu chứng kéo dài và cần tìm thông tin về tình trạng sức khỏe, "
        "các yếu tố ảnh hưởng, phương pháp kiểm tra và cách theo dõi tại nhà."}
    before_source = source_hashes(p.root)
    prepared = prepare_cached_strong_resources(p.root,p.checkout,p.work,[query],output_root=p.outputs,
        document_dense_enabled=False,search_device="cpu")
    index = prepared["index"]
    try:
        policy = StrongConfig(**prepared["config"]["retrieval"])
        documents,_ = index.document_candidates(query["query"],prepared["vectors"][0],prepared["translations"][0]["variants"],policy)
        assert documents and index.secondary is None and policy.second_dense_weight == 0
        assert all({"dense","subquery_0","hyde"} <= set(d["branches"]) for d in documents)
        assert all("dense_qwen" not in d["branches"] for d in documents)
        assert index.primary.manifest["input_count"] == p.frozen[2]["input_count"]
    finally:
        index.catalog.close()
    assert source_hashes(p.root) == before_source


def test_document_input_resume_skips_already_tokenized_documents(prepared,monkeypatch):
    p = prepared
    from vietmedbridge import full_scale_runtime as runtime
    class CountingTokenizer(FullTokenizer):
        calls = 0
        def __call__(self,*a,**kw):
            self.calls += 1
            return super().__call__(*a,**kw)
    tokenizer = CountingTokenizer()
    catalog,_ = runtime.load_scale_catalog(p.root,p.checkout,p.work,tokenizer,read_only=True)
    real_write = runtime.atomic_json
    def interrupt_after_part(path,value):
        real_write(path,value)
        if path.name.startswith("units-") and path.name.endswith(".done.json"):
            raise RuntimeError("interrupted after completed document input part")
    try:
        monkeypatch.setattr(runtime,"atomic_json",interrupt_after_part)
        with pytest.raises(RuntimeError,match="interrupted"):
            runtime._document_inputs(catalog,tokenizer,p.outputs / "interrupted_inputs",part_size=1)
        monkeypatch.setattr(runtime,"atomic_json",real_write)
        resumed = runtime._document_inputs(catalog,tokenizer,p.outputs / "interrupted_inputs",part_size=1)
        calls_with_resume = tokenizer.calls
        fresh_tokenizer = CountingTokenizer()
        fresh = runtime._document_inputs(catalog,fresh_tokenizer,p.outputs / "fresh_inputs",part_size=1)
        assert resumed == fresh and calls_with_resume == fresh_tokenizer.calls
        assert resumed["input_count"] == len(catalog.documents)
    finally:
        catalog.close()


def test_strong_reuses_baseline_local_bge_cache_without_copying_vectors_again(prepared,monkeypatch):
    p = prepared
    from vietmedbridge import content_embeddings as content
    from vietmedbridge.scale_baseline import imported_bge_manifest
    view,manifest = imported_bge_manifest(p.root,p.frozen[1],p.frozen[2],p.config)
    store = content.ContentVectorParts(p.root,view,manifest,p.work / "vector_cache")
    for i in range(len(manifest["parts"])):
        store.part(i)
    def forbidden(*a,**kw):
        pytest.fail("Baseline has already copied the immutable child blocks/maps to local disk")
    monkeypatch.setattr(content,"publish_file",forbidden)
    result = invoke(p,document_dense_enabled=False,max_new_queries=1)
    assert result["ready"]["query_count"] == 1


@pytest.mark.parametrize("storage",["drive","hf"])
def test_notebook_default_cached_strong_routes_without_union_or_cpu_rebuild(prepared,monkeypatch,storage):
    p = prepared
    from vietmedbridge import full_scale_runtime as runtime, index_inputs, scale_baseline
    def forbidden(*a,**kw):
        pytest.fail("Strong cache mode must bypass CPU preparation, lineage and baseline")
    monkeypatch.setattr(runtime,"prepare_scale_cpu_resources",forbidden)
    monkeypatch.setattr(index_inputs,"read_candidate_lineage",forbidden)
    monkeypatch.setattr(scale_baseline,"run_large_baseline",forbidden)
    calls = []
    monkeypatch.setattr(runtime,"run_scale_full_pipeline",lambda *a,**kw:(calls.append(kw) or {
        "ready":{"state":"QUERIES_IN_PROGRESS","query_count":3},"status":{},"samples":[]}))
    repo = Path(__file__).resolve().parents[1]
    cell = next(c["source"] for c in read_json(repo / "notebooks/04_colab_retrieval_baseline.ipynb")["cells"]
        if c["cell_type"] == "code" and "PREPARE_ONLY =" in c["source"])
    context = {"DATA_ROOT":p.root,"CHECKOUT":p.checkout,"WORK_DIR":p.work,"RESULTS_ROOT":p.outputs,
        "CODE_COMMIT":"CPU-test","RUN_CACHED_BASELINE_FROM_DRIVE":storage == "drive",
        "USE_HF_STORAGE":storage == "hf","HF_STORAGE_PROFILE":"bge_submit"}
    exec(compile(cell,"04-cached-strong-routing","exec"),context)
    assert len(calls) == 1 and calls[0]["execution_mode"] == "cached_strong"
    assert calls[0]["output_root"] == p.outputs and calls[0]["second_dense_mode"] == "if_complete"
    assert calls[0]["document_dense_enabled"] is True
    assert context["TEAM_SIZE"] == 1 and context["PREPARE_ONLY"] is False
