"""Disk integration and restart tests; model doubles do not measure GPU quality."""
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

from test_scale_baseline import Encoder,Tokenizer,frozen,external
from vietmedbridge.artifacts import atomic_json,digest_json,read_json,sha256_file
from vietmedbridge.content_embeddings import ContentVectorParts,content_embeddings,seal
from vietmedbridge.disk_catalog import prepare_disk_catalog
from vietmedbridge.disk_full import AdaptiveDiskCatalog
from vietmedbridge.medical_lexical import MedicalAnalyzer
from vietmedbridge.scale_vectors import VectorParts,search_parts,stream_embeddings


def inputs_at(root,texts,part_size=2):
    root.mkdir(parents=True,exist_ok=True)
    parts = []
    for start in range(0,len(texts),part_size):
        selected = texts[start:start+part_size]
        path = root / f"units-{start}.parquet"
        pq.write_table(pa.Table.from_pylist([{"id":hashlib.sha256(t.encode()).hexdigest(),"text":t} for t in selected]),path)
        parts.append({"path":path.name,"sha256":sha256_file(path),"rows":len(selected),"start":start})
    return seal({"state":"COMPLETE","input_count":len(texts),"parts":parts})


def test_content_cache_seeds_legacy_then_encodes_only_new_texts(tmp_path):
    old = inputs_at(tmp_path / "inputs-old",["abc","中文医学","long original source"])
    encoder = Encoder()
    stream_embeddings(tmp_path / "inputs-old",old,encoder,tmp_path / "legacy",work_dir=tmp_path / "work")
    calls = encoder.calls
    manifest = content_embeddings(tmp_path,tmp_path / "inputs-old",old,encoder,tmp_path / "view-old",
        work_dir=tmp_path / "work",seed_root=tmp_path / "legacy")
    assert encoder.calls == calls
    assert not list((tmp_path / "view-old").rglob("*.npy"))
    newer = inputs_at(tmp_path / "inputs-new",["brand new","long original source","abc","中文医学","brand new"])
    manifest = content_embeddings(tmp_path,tmp_path / "inputs-new",newer,encoder,tmp_path / "view-new",work_dir=tmp_path / "work")
    assert manifest["state"] == "COMPLETE" and encoder.calls == calls+1
    store = ContentVectorParts(tmp_path,tmp_path / "view-new",manifest,tmp_path / "local")
    expected = Encoder().encode(["brand new","long original source","abc","中文医学","brand new"])
    np.testing.assert_allclose(store.gather([4,0,3,1,2]),expected[[4,0,3,1,2]])
    checksum = manifest["manifest_sha256"]
    assert content_embeddings(tmp_path,tmp_path / "inputs-new",newer,encoder,tmp_path / "view-new",work_dir=tmp_path / "work")["manifest_sha256"] == checksum
    assert encoder.calls == calls+1
    map_path = tmp_path / "view-new" / manifest["parts"][0]["path"]
    map_path.write_bytes(b"broken")
    with pytest.raises(ValueError,match="changed artifact"):
        content_embeddings(tmp_path,tmp_path / "inputs-new",newer,encoder,tmp_path / "view-new",work_dir=tmp_path / "work")


def test_three_workers_resume_disjoint_parts_and_coordinator_merges(tmp_path):
    inputs = inputs_at(tmp_path / "inputs",[f"source {i}" for i in range(12)])
    workers = [Encoder() for _ in range(3)]
    for i,encoder in enumerate(workers):
        result = content_embeddings(tmp_path,tmp_path / "inputs",inputs,encoder,tmp_path / "view",
            work_dir=tmp_path / "work",worker_id=i,workers=3,max_new_parts=1)
        assert result["state"] == "IN_PROGRESS"
    assert not (tmp_path / "view/embeddings.json").exists()
    coordinator = Encoder()
    result = content_embeddings(tmp_path,tmp_path / "inputs",inputs,coordinator,tmp_path / "view",
        work_dir=tmp_path / "work",workers=3,encode_missing=False)
    assert result["state"] == "IN_PROGRESS" and coordinator.calls == 0
    for i,encoder in enumerate(workers):
        content_embeddings(tmp_path,tmp_path / "inputs",inputs,encoder,tmp_path / "view",work_dir=tmp_path / "work",worker_id=i,workers=3)
        assert encoder.calls == 2
    result = content_embeddings(tmp_path,tmp_path / "inputs",inputs,coordinator,tmp_path / "view",work_dir=tmp_path / "work",encode_missing=False)
    assert result["state"] == "COMPLETE" and coordinator.calls == 0
    store = ContentVectorParts(tmp_path,tmp_path / "view",result,tmp_path / "local")
    np.testing.assert_allclose(store.gather(range(12)),Encoder().encode([f"source {i}" for i in range(12)]))
    wrong = Encoder()
    wrong.identity = wrong.identity | {"wrong_producer":True}
    with pytest.raises(ValueError,match="changed"):
        content_embeddings(tmp_path,tmp_path / "inputs",inputs,wrong,tmp_path / "view",work_dir=tmp_path / "work",worker_id=0,workers=3)


def test_dense_search_mid_scan_checkpoint_skips_completed_parts(tmp_path):
    inputs = inputs_at(tmp_path / "inputs",[f"source {i}" for i in range(10)],part_size=1)
    encoder = Encoder()
    manifest = stream_embeddings(tmp_path / "inputs",inputs,encoder,tmp_path / "vectors",work_dir=tmp_path / "work")
    store = VectorParts(tmp_path / "vectors",manifest,tmp_path / "local")
    query = encoder.encode(["query"])
    partial = search_parts(store,query,tmp_path / "search",k=5,device="cpu",max_new_parts=2,checkpoint_every=1)
    assert partial[2]["state"] == "IN_PROGRESS"
    visits = []
    original = store.part
    store.part = lambda i:(visits.append(i),original(i))[1]
    ss,ii,_ = search_parts(store,query,tmp_path / "search",k=5,device="cpu",checkpoint_every=1)
    assert visits == list(range(2,10))
    fresh = search_parts(store,query,tmp_path / "fresh",k=5,device="cpu")
    np.testing.assert_array_equal(ii,fresh[1])
    np.testing.assert_allclose(ss,fresh[0])
    assert len(list((tmp_path / "search").glob("progress-*.npy"))) <= 4


def test_lazy_source_parents_can_be_recovered_after_restart(tmp_path,frozen):
    build,candidate,inputs = frozen
    def open_catalog():
        base = prepare_disk_catalog(build,candidate,inputs,tmp_path / "catalogs",MedicalAnalyzer(segmentation=False),work_dir=tmp_path / "work")
        return AdaptiveDiskCatalog(base,Tokenizer())
    catalog = open_catalog()
    first = next(iter(catalog.children))
    child = catalog.children[first]
    parent_id = child["source_parent_alternatives"]["640"]
    parent = catalog.parents[parent_id]
    identity = catalog.identity
    assert parent["text"] == catalog.documents[parent["doc_id"]]["source_text"][parent["start_char"]:parent["end_char"]]
    catalog.close()
    catalog = open_catalog()
    try:
        assert catalog.identity == identity
        assert catalog.parents[parent_id] == parent  # parent requested before its anchor
        assert catalog.children[first]["source_parent_alternatives"]["640"] == parent_id
        from vietmedbridge.full_scale_runtime import training_source_pool
        pool = training_source_pool(catalog,1)
        assert len(pool.documents) == 1 and pool.children
    finally:
        catalog.close()


class FullTokenizer(Tokenizer):
    def num_special_tokens_to_add(self,pair=False):
        return 4 if pair else 2


def test_notebook_cpu_preparation_builds_reusable_catalog_without_gpu_weights(tmp_path,frozen,monkeypatch):
    from vietmedbridge import full_scale_runtime as runtime, full_plan_runtime
    checkout = Path(__file__).resolve().parents[1]
    config = read_json(checkout / "configs/retrieval_full.json")
    config["dense"] = {"model_id":"test","revision":"test","max_length":512}
    config["lexical_segmentation"] = False
    config["pilot_limits"] = {"max_documents":0,"max_children":0}
    target = tmp_path / "checkout"
    atomic_json(target / "configs/retrieval_full.json",config)
    atomic_json(tmp_path / "raw/snapshot.json",{"files":{
        "links_corpus.parquet":{"sha256":frozen[1]["origin_corpus_sha256"]}}})
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:FullTokenizer())))
    def forbidden(*args,**kwargs):
        pytest.fail("CPU preparation must never enter inference or load GPU weights")
    for name in ("TorchDenseEncoder","TorchQwenEncoder","TorchQueryTranslator","TorchQwenReranker","run_scale_full_pipeline"):
        monkeypatch.setattr(runtime,name,forbidden)
    monkeypatch.setattr(full_plan_runtime,"run_full_pipeline",forbidden)
    notebook = read_json(checkout / "notebooks/04_colab_retrieval_baseline.ipynb")
    cell = next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and "PREPARE_ONLY =" in c["source"])
    context = {"DATA_ROOT":tmp_path,"CHECKOUT":target,"WORK_DIR":tmp_path / "work","CODE_COMMIT":"CPU-test"}
    exec(compile(cell,"04-cpu-preparation","exec"),context)
    assert context["READY"]["state"] == "CPU_PREPARATION_COMPLETE"
    report = read_json(tmp_path / "retrieval/cpu_preparation" / (frozen[1]["candidate_manifest_sha256"][:16]+".json"))
    assert report["documents"] == 2 and report["unique_dense_inputs"] == frozen[2]["input_count"]
    marker = next((tmp_path / "retrieval/catalogs").glob("*/catalog.json"))
    before = (sha256_file(marker),marker.stat().st_mtime_ns)
    exec(compile(cell,"04-cpu-preparation-resume","exec"),context)
    assert (sha256_file(marker),marker.stat().st_mtime_ns) == before


def test_union_preserves_official_aliases_outcomes_and_resume(tmp_path,frozen,external,monkeypatch):
    from dataclasses import asdict
    from test_external_import import CONFIG,SPEC
    from vietmedbridge.artifacts import code_fingerprint
    from vietmedbridge.corpus_union import compose_candidates
    from vietmedbridge.scale_benchmark import load_handoff
    build,candidate,old_inputs = frozen
    source = (build.name,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json")
    encoder = Encoder()
    legacy = tmp_path / "retrieval" / (build.name+"-bge-bm25-v1-"+candidate["candidate_manifest_sha256"][:8]) / "corpus_embeddings"
    stream_embeddings(build / "index_inputs",old_inputs,encoder,legacy,work_dir=tmp_path / "work")
    previous_calls = encoder.calls
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    result = compose_candidates(tmp_path,[source,source],Tokenizer(),run_name="combined",golden_report=golden,
        official_links=external[1],work_dir=tmp_path / "work",part_size=2)
    merged,manifest,inputs = load_handoff(tmp_path)
    assert manifest["counts"]["documents"] == 2 and manifest["counts"]["failures"] == 2
    assert manifest["counts"]["input_records"] == 4 and manifest["integrity"]["passed"]
    assert all(v == 0 for v in manifest["integrity"]["checks"].values())
    again = compose_candidates(tmp_path,[source,source],Tokenizer(),run_name="combined",golden_report=golden,
        official_links=external[1],work_dir=tmp_path / "work",part_size=2)
    assert again["candidate_manifest_sha256"] == result["candidate_manifest_sha256"]
    assert again["index_inputs_manifest_sha256"] == inputs["manifest_sha256"]
    from vietmedbridge.full_scale_runtime import baseline_seed_sources
    cached = content_embeddings(tmp_path,merged / "index_inputs",inputs,encoder,tmp_path / "combined-vectors",
        work_dir=tmp_path / "work",seed_sources=baseline_seed_sources(tmp_path,manifest))
    assert cached["state"] == "COMPLETE" and encoder.calls == previous_calls
    with pytest.raises(ValueError,match="source/policy"):
        compose_candidates(tmp_path,[source],Tokenizer(),run_name="combined",golden_report=golden,
            official_links=external[1],work_dir=tmp_path / "work",part_size=2)
    from vietmedbridge import corpus_union as union
    original_prepare = union.prepare_index_inputs
    def interrupt_after_index(*args,**kwargs):
        original_prepare(*args,**kwargs)
        raise RuntimeError("simulated interruption before handoff")
    monkeypatch.setattr(union,"prepare_index_inputs",interrupt_after_index)
    with pytest.raises(RuntimeError,match="simulated interruption"):
        compose_candidates(tmp_path,[source],Tokenizer(),run_name="interrupted",golden_report=golden,
            official_links=external[1],work_dir=tmp_path / "work",part_size=2)
    frozen_checkpoint = read_json(tmp_path / "processed/interrupted/union_frozen_candidate.json")
    assert read_json(tmp_path / "active_data_candidate.json")["build_run"] == "combined"
    monkeypatch.setattr(union,"prepare_index_inputs",original_prepare)
    resumed = compose_candidates(tmp_path,[source],Tokenizer(),run_name="interrupted",golden_report=golden,
        official_links=external[1],work_dir=tmp_path / "work",part_size=2)
    assert resumed["candidate_manifest_sha256"] == frozen_checkpoint["candidate_manifest_sha256"]


def test_union_accepts_recovered_ids_and_rejects_conflicting_successes(tmp_path,frozen,external):
    from dataclasses import asdict
    from test_external_import import CONFIG,SPEC,run_import,MEMBERS,write_parquet
    from vietmedbridge.artifacts import code_fingerprint
    from vietmedbridge.corpus_union import compose_candidates
    from vietmedbridge.health import health_report,freeze_candidate
    from vietmedbridge.scale_benchmark import load_handoff
    old_build,old,_ = frozen
    source,links,text = external
    crawl = pq.read_table(source / MEMBERS["crawl"]).to_pylist()
    crawl[1]["status"],crawl[1]["http_status"] = "SUCCESS_RAW",200
    write_parquet(source / MEMBERS["crawl"],crawl)
    extracted = pq.read_table(source / MEMBERS["extracted"]).to_pylist()
    extracted.append(extracted[0] | {"crawl_url_id":"1","title":"Recovered article"})
    write_parquet(source / MEMBERS["extracted"],extracted)
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    def imported(name):
        run_import(tmp_path,external,run_name=name)
        build = tmp_path / "processed" / name
        candidate = freeze_candidate(build,health_report(build,official_links=links,work_dir=tmp_path / "work"),golden_report=golden)
        return (name,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json")
    initial = (old_build.name,f"candidate-{old['candidate_manifest_sha256'][:16]}.json")
    recovered = imported("recovered")
    compose_candidates(tmp_path,[initial,recovered],Tokenizer(),run_name="recovery-union",golden_report=golden,
        official_links=links,work_dir=tmp_path / "work")
    _,candidate,_ = load_handoff(tmp_path)
    assert candidate["counts"]["documents"] == 3 and candidate["counts"]["failures"] == 1
    assert candidate["counts"]["input_records"] == 4 and candidate["integrity"]["passed"]
    extracted[0]["text"] += " Changed source."
    extracted[0]["char_count"] = len(extracted[0]["text"])
    write_parquet(source / MEMBERS["extracted"],extracted)
    conflicting = imported("conflicting")
    active_before = read_json(tmp_path / "active_data_candidate.json")
    with pytest.raises(ValueError,match="Conflicting successful sources"):
        compose_candidates(tmp_path,[initial,conflicting],Tokenizer(),run_name="conflict-union",golden_report=golden,
            official_links=links,work_dir=tmp_path / "work")
    assert read_json(tmp_path / "active_data_candidate.json") == active_before


def test_adapter_reuses_weights_but_invalidates_cutoffs_for_new_corpus(tmp_path):
    from vietmedbridge.full_scale_runtime import scale_adapter
    from vietmedbridge.full_plan_runtime import calibration_context
    from vietmedbridge.content_embeddings import seal
    config = read_json(Path(__file__).resolve().parents[1] / "configs/retrieval_full.json")
    analyzer = MedicalAnalyzer(segmentation=False)
    catalog = SimpleNamespace(identity={"corpus":"old"},base=SimpleNamespace(analyzer=analyzer))
    queries = [{"id":123,"query":"contest only"}]
    run = tmp_path / "training/stage-a-qlora-v3-per-model-15b"
    file = run / "selected_adapter/adapter_config.json"
    atomic_json(file,{"test_only":True})
    adapter = seal({"base_model":config["reranker"]["model_id"],"base_revision":config["reranker"]["revision"],
        "files":{"adapter_config.json":sha256_file(file)}})
    atomic_json(run / "selected_adapter/adapter_manifest.json",adapter)
    calibration = seal({"reviewed":True,"split":"dev","catalog":catalog.identity,
        "reranker_spec":config["reranker"],"adapter_manifest_sha256":adapter["manifest_sha256"],
        "dev_query_ids":[456],"dev_query_texts_sha256":[digest_json("independent dev")],
        "contest_queries_sha256":digest_json(queries),"calibration_context":calibration_context(config,analyzer),
        "policy":{"doc_top_k":10,"chunk_top_k":10,"doc_score_margin":1.,"chunk_score_margin":1.}})
    atomic_json(run / "calibrated_policy.json",calibration)
    assert scale_adapter(tmp_path,catalog,queries,config)[0]["applies_to_current_corpus"]
    catalog.identity = {"corpus":"new"}
    result,path = scale_adapter(tmp_path,catalog,queries,config)
    assert not result["applies_to_current_corpus"] and path == run / "selected_adapter"
    with pytest.raises(ValueError,match="leakage"):
        scale_adapter(tmp_path,catalog,[{"id":456,"query":"x"}],config)


def test_local_vector_cache_eviction_preserves_durable_blocks(tmp_path):
    inputs = inputs_at(tmp_path / "inputs",[f"source {i}" for i in range(20)])
    encoder = Encoder()
    manifest = content_embeddings(tmp_path,tmp_path / "inputs",inputs,encoder,tmp_path / "view",work_dir=tmp_path / "work")
    blocks_before = {p:sha256_file(p) for p in (tmp_path / "model_cache").rglob("*.npy")}
    store = ContentVectorParts(tmp_path,tmp_path / "view",manifest,tmp_path / "local",local_cache_bytes=800)
    for i in range(20):
        np.testing.assert_allclose(store.gather([i]),Encoder().encode([f"source {i}"]))
    assert store.disk_bytes <= 800 and len(list(store.cache.glob("*.npy"))) <= 5
    np.testing.assert_allclose(store.gather([0,19]),Encoder().encode(["source 0","source 19"]))
    assert all(p.exists() and sha256_file(p) == sha for p,sha in blocks_before.items())


def test_training_uses_active_large_catalog_and_stops_before_gpu_for_missing_gold(tmp_path,frozen,monkeypatch):
    from vietmedbridge import training_workflow as workflow
    checkout = Path(__file__).resolve().parents[1]
    config = read_json(checkout / "configs/retrieval_full.json")
    config["dense"] = {"model_id":"test","revision":"test","max_length":512}
    config["lexical_segmentation"] = False
    config["pilot_limits"] = {"max_documents":0,"max_children":0}
    target = tmp_path / "checkout"
    for filename,value in (("retrieval_full",config),("retrieval_scale",read_json(checkout / "configs/retrieval_scale.json")),("strong_model_manifest",{})):
        atomic_json(target / f"configs/{filename}.json",value)
    queries = [{"id":10000+i,"query":f"contest {i}"} for i in range(1200)]
    query_path = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pylist(queries),query_path)
    atomic_json(tmp_path / "raw/snapshot.json",{"files":{
        "query.parquet":{"path":"query.parquet","sha256":sha256_file(query_path)},
        "links_corpus.parquet":{"sha256":frozen[1]["origin_corpus_sha256"]}}})
    for i,split in enumerate(("train","dev")):
        atomic_json(tmp_path / f"labels/retrieval_{split}.json",{"reviewed":True,"split":split,"queries":[{
            "id":i+1,"query":f"independent source question {split}","relevant_docs":[999999],
            "relevant_chunks":[{"doc_id":999999,"chunk_text":"missing source"}]}]})
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:FullTokenizer())))
    monkeypatch.setattr(workflow,"review_model_registry",lambda *a:None)
    monkeypatch.setattr(workflow,"model_budget_report",lambda *a,**k:{})
    monkeypatch.setattr(workflow,"load_catalog",lambda *a,**k:pytest.fail("Must use active disk corpus, not old pilot"))
    monkeypatch.setattr(workflow,"TorchQwenReranker",lambda *a,**k:pytest.fail("Missing labels must not load GPU weights"))
    result = workflow.run_training_workflow(tmp_path,target,work_dir=tmp_path / "work")
    assert result["state"] == "WAITING_FOR_LABEL_SOURCE_RECONCILIATION"
    assert result["unavailable_reference_documents"] == [999999]
    assert result["gpu_models_loaded_this_call"] == 0


def test_full_disk_entrypoint_all_model_roles_resume_and_source_zip(tmp_path,frozen,monkeypatch):
    from vietmedbridge import full_scale_runtime as runtime
    from vietmedbridge.qwen_models import RoleEncoder
    from vietmedbridge.competition_pilot import MASTER_PLAN
    checkout = Path(__file__).resolve().parents[1]
    config = read_json(checkout / "configs/retrieval_full.json")
    config["dense"] = {"model_id":"test","revision":"test","max_length":512}
    config["lexical_segmentation"] = False
    config["retrieval"].update(doc_candidate_k=2,detail_doc_k=2,doc_min_k=1,doc_top_k=2,chunk_min_k=1,chunk_top_k=2)
    config["pilot_limits"] = {"max_documents":0,"max_children":0}
    target = tmp_path / "checkout"
    atomic_json(target / "configs/retrieval_full.json",config)
    atomic_json(target / "configs/retrieval_scale.json",read_json(checkout / "configs/retrieval_scale.json"))
    atomic_json(target / "configs/strong_model_manifest.json",{"test_only":True})
    (target / MASTER_PLAN).write_text("CPU model doubles. No medical/relevance/GPU claims.",encoding="utf-8")
    queries = [{"id":10001+i*7,"query":"triệu chứng 中文医学"} for i in range(1200)]
    query_path = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pylist(queries),query_path)
    atomic_json(tmp_path / "raw/snapshot.json",{"files":{
        "query.parquet":{"path":"query.parquet","sha256":sha256_file(query_path)},
        "links_corpus.parquet":{"sha256":frozen[1]["origin_corpus_sha256"]}}})
    counters = {k:0 for k in ("bge","qwen","translation","reranker")}
    class Dense(Encoder):
        def __init__(self,spec):
            super().__init__()
            self.tokenizer = FullTokenizer()
            self.identity = self.identity | spec
        def encode(self,texts,**kwargs):
            counters["bge"] += len(texts)
            return super().encode(texts,**kwargs)
    class Secondary(Dense):
        def for_role(self,role):
            return RoleEncoder(self,role)
        def encode(self,texts,**kwargs):
            counters["qwen"] += len(texts)
            return Encoder.encode(self,texts,**kwargs)
    class Translation:
        def __init__(self,spec):
            self.identity = spec | {"inference_code_sha256":sha256_file(Path(runtime.__file__).with_name("translation_model.py"))}
        def translate(self,query,prompt):
            counters["translation"] += 1
            return json.dumps({"original_vi":query,"entities":[],"constraints":[],"query_en":query,"query_zh":query})
        def close(self):
            pass
    class Reranker:
        def __init__(self,spec,adapter_path=None):
            self.identity = spec | {"fine_tuned":adapter_path is not None}
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
    monkeypatch.setattr(runtime,"model_budget_report",lambda *a,**k:{"max_model_parameters":8_188_548_096,"limit_parameters":15_000_000_000})
    real_search = runtime.search_parts
    monkeypatch.setattr(runtime,"search_parts",lambda *a,**kw:real_search(*a,**(kw | {"device":"cpu"})))
    partial = runtime.run_scale_full_pipeline(tmp_path,target,code_commit="CPU-test",work_dir=tmp_path / "work",max_new_queries=3)
    assert partial["ready"]["state"] == "QUERIES_IN_PROGRESS"
    before = dict(counters)
    result = runtime.run_scale_full_pipeline(tmp_path,target,code_commit="CPU-test",work_dir=tmp_path / "work")
    assert result["ready"]["state"] == "READY_FOR_MANUAL_UPLOAD" and result["ready"]["query_count"] == 1200
    assert counters["bge"] == before["bge"] and counters["qwen"] == before["qwen"] and counters["translation"] == before["translation"]
    assert result["status"]["full_plan_inference"] is True
    with ZipFile(result["ready"]["zip_path"]) as z:
        predictions = json.loads(z.read("results.json"))
    assert [p["id"] for p in predictions] == [q["id"] for q in queries]
    before = dict(counters)
    repeated = runtime.run_scale_full_pipeline(tmp_path,target,code_commit="CPU-test",work_dir=tmp_path / "work")
    assert repeated["ready"]["zip_sha256"] == result["ready"]["zip_sha256"] and counters == before
    # 05 mines the same disk corpus, reuses corpus vectors, and can replay a
    # pending review without loading the models a second time.
    from vietmedbridge import training_workflow as workflow
    monkeypatch.setattr(workflow,"review_model_registry",lambda *a:None)
    monkeypatch.setattr(workflow,"model_budget_report",lambda *a,**k:{})
    monkeypatch.setattr(workflow,"load_catalog",lambda *a,**k:pytest.fail("Large training must not load the old pilot"))
    for i,split in enumerate(("train","dev")):
        source = pq.read_table(frozen[0] / frozen[1]["parts"][0]["files"]["documents"]["path"]).to_pylist()[0]
        atomic_json(tmp_path / f"labels/retrieval_{split}.json",{"reviewed":True,"split":split,"queries":[{
            "id":-(i+1),"query":f"Independent reviewed {split} question", "relevant_docs":[source["doc_id"]],
            "relevant_chunks":[{"doc_id":source["doc_id"],"chunk_text":source["source_text"][:100]}]}]})
    training = workflow.run_training_workflow(tmp_path,target,work_dir=tmp_path / "work")
    assert training["state"] == "MINING_REQUIRES_MORE_GOLD_COVERAGE_OR_REVIEWED_NEGATIVES"
    assert "corpora" in training["review_path"] and not training["fine_tuned"]
    before = dict(counters)
    pending = workflow.run_training_workflow(tmp_path,target,work_dir=tmp_path / "work")
    assert pending["gpu_models_loaded_this_call"] == 0 and pending["mining_reused"]
    assert counters == before
