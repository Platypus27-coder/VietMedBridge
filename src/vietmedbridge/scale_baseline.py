"""Practical large-corpus baseline: streamed BGE, disk BM25, source-validated ZIP."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .competition_pilot import MASTER_PLAN, bind_pilot_run, finish_pilot
from .dataset import parquet_path
from .disk_catalog import lexical_field, prepare_disk_catalog
from .embeddings import embed_units, embedding_matrix
from .medical_lexical import MedicalAnalyzer
from .qwen_models import review_model_registry
from .retrieval_cascade import PostingBM25, weighted_rrf
from .retrieval_data import load_queries
from .retrieval_models import TorchDenseEncoder
from .retrieval_policy import select_parents
from .runtime_profile import RuntimeProfile
from .scale_benchmark import load_handoff
from .scale_vectors import VectorParts, search_parts, stream_embeddings


def baseline_ranking(catalog, store, query, vector, positions, dense_values, policy):
    dense = catalog.dense_documents(positions,dense_values,policy["document_candidates"])
    orders = {"dense":[d for d,_ in dense]}
    weights = {"dense":policy["dense_weight"]}
    for language in ("vi","en","zh"):
        # Original text is always retained. This baseline has no query LLM.
        hits = catalog.sparse(query,language,policy["sparse_top_k"])
        if hits:
            orders[language] = [d for d,_ in hits]
            weights[language] = policy[language+"_weight"]
    fused = weighted_rrf(orders,weights,policy["rrf_k"])
    selected = sorted(fused,key=lambda d:(-fused[d],d))[:policy["doc_top_k"]]
    documents = [{"doc_id":d,"reranker_score":fused[d]} for d in selected]
    children = []
    for doc_id in selected:
        pairs = catalog.document_children(doc_id)
        rows,offsets = zip(*pairs)
        values = np.einsum("ij,j->i",store.gather(offsets),vector,optimize=False)
        dense_order = sorted(range(len(rows)),key=lambda i:(-float(values[i]),rows[i]["chunk_id"]))
        language = catalog.documents[doc_id]["language"]
        texts = [lexical_field(catalog.analyzer,row["text"],language) for row in rows]
        scores = PostingBM25(texts).scores(lexical_field(catalog.analyzer,query,language))
        sparse_order = sorted((i for i,v in enumerate(scores) if v > 0),key=lambda i:(-float(scores[i]),rows[i]["chunk_id"]))
        local = weighted_rrf({"dense":dense_order,"sparse":sparse_order},{"dense":1.,"sparse":.8},policy["rrf_k"])
        for i in sorted(local,key=lambda i:(-local[i],rows[i]["chunk_id"]))[:5]:
            # select_parents uses this legacy field name for any numeric ranking.
            # Evidence explicitly identifies these values as RRF, not neural logits.
            children.append({"child_id":rows[i]["chunk_id"],"doc_id":doc_id,
                "reranker_score":fused[doc_id]+local[i]})
    children.sort(key=lambda r:(-r["reranker_score"],r["doc_id"],r["child_id"]))
    return {"documents":documents,"children":children}, {name:len(ids) for name,ids in orders.items()}


def predict_baseline(catalog,store,queries,query_vectors,positions,scores,search_manifest,output_dir,
                     *, tokenizer,policy,max_new_queries=None):
    if max_new_queries is not None and (type(max_new_queries) is not int or max_new_queries < 0):
        raise ValueError("Invalid query checkpoint limit.")
    root = Path(output_dir)
    identity = {"catalog":catalog.identity,"search":search_manifest["manifest_sha256"],
        "queries_sha256":digest_json(queries),"policy":policy,"score_semantics":"weighted_rrf_no_neural_reranker",
        "code_sha256":sha256_file(Path(__file__)),
        "selection_code_sha256":sha256_file(Path(__file__).with_name("retrieval_policy.py"))}
    signature = digest_json(identity)
    if (root / "config.json").exists() and read_json(root / "config.json") != identity:
        raise ValueError("Prediction policy changed; use a new baseline run.")
    atomic_json(root / "config.json",identity)
    selection = SimpleNamespace(doc_min_k=0,doc_top_k=policy["doc_top_k"],doc_score_margin=None,
        doc_score_floor=None,chunk_min_k=0,chunk_top_k=policy["chunk_top_k"],chunk_score_margin=None,
        chunk_score_floor=None,max_chunks_per_doc=policy["max_chunks_per_doc"],dedup_threshold=policy["dedup_threshold"])
    records,written = [],0
    for i,q in enumerate(tqdm(queries,desc="100k baseline query checkpoints")):
        path = root / f"query-{q['id']}.done.json"
        if path.exists():
            saved = read_json(path)
            if (saved["signature"] != signature or saved["query_sha256"] != digest_json(q)
                or digest_json({k:v for k,v in saved.items() if k != "record_sha256"}) != saved["record_sha256"]):
                raise ValueError("Baseline query checkpoint changed.")
        else:
            if max_new_queries is not None and written >= max_new_queries:
                break
            ranking,branches = baseline_ranking(catalog,store,q["query"],query_vectors[i],positions[i],scores[i],policy)
            prediction,provenance = select_parents(catalog,ranking,selection,tokenizer,q["id"])
            saved = {"signature":signature,"query_sha256":digest_json(q),"prediction":prediction,
                "provenance":provenance,"branch_document_counts":branches,
                "scoring_method":"weighted_rrf_no_neural_reranker"}
            saved["record_sha256"] = digest_json(saved)
            atomic_json(path,saved)
            written += 1
        records.append(saved)
    report = {"signature":signature,"completed_queries":len(records),"requested_queries":len(queries),
        "state":"COMPLETE" if len(records) == len(queries) else "IN_PROGRESS",
        "evaluation":"NOT_EVALUATED_NO_REFERENCE_LABELS"}
    atomic_json(root / "predictions.json",report)
    return records,report


def run_large_baseline(data_root,checkout,*,code_commit,work_dir,batch_size=32,
                       max_new_embedding_parts=None,max_new_queries=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU for large-corpus BGE inference.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Invalid GPU embedding batch size.")
    root,checkout,work = Path(data_root),Path(checkout),Path(work_dir)
    build,candidate,inputs = load_handoff(root)
    config = read_json(checkout / "configs/retrieval_large_baseline.json")
    full = read_json(checkout / "configs/retrieval_full.json")
    registry = read_json(checkout / "configs/strong_model_manifest.json")
    review_model_registry(full,registry)
    if config["dense"] != full["dense"] or any(candidate["golden"]["tokenizer"][k] != config["dense"][k] for k in ("model_id","revision")):
        raise ValueError("Large baseline/tokenizer/model policy mismatch.")
    snapshot = read_json(root / "raw/snapshot.json")
    if snapshot["files"]["links_corpus.parquet"]["sha256"] != candidate["origin_corpus_sha256"]:
        raise ValueError("Large baseline/official corpus snapshot mismatch.")
    queries = load_queries(parquet_path(root,"query.parquet"),expected_count=1200)
    glossary = root / "labels/medical_aliases.json"
    analyzer = MedicalAnalyzer(segmentation=config["lexical_segmentation"],glossary_path=glossary if glossary.is_file() else None)
    config["analyzer"] = analyzer.identity
    run = root / "retrieval" / (build.name+"-bge-bm25-v1-"+candidate["candidate_manifest_sha256"][:8])
    profile = RuntimeProfile(run / "runtime_profile.json")
    print(f"100k baseline: {candidate['counts']['documents']:,} documents, {inputs['input_count']:,} unique inputs",flush=True)
    print("Executed model: BAAI/bge-m3. Qwen models are deferred for this baseline.",flush=True)
    with profile.stage("disk_catalog_multilingual_bm25"):
        catalog = prepare_disk_catalog(build,candidate,inputs,root / "retrieval/catalogs",analyzer,work_dir=work)
    try:
        contract = bind_pilot_run(run,plan_path=checkout / MASTER_PLAN,catalog=catalog,queries=queries,
            config=config,code_commit=code_commit)
        with profile.stage("bge_loading"):
            encoder = TorchDenseEncoder(config["dense"])
        try:
            with profile.stage("streamed_child_embeddings"):
                cm = stream_embeddings(build / "index_inputs",inputs,encoder,run / "corpus_embeddings",
                    batch_size=batch_size,work_dir=work,max_new_parts=max_new_embedding_parts)
            if cm["state"] != "COMPLETE":
                profile.finish("EMBEDDING_IN_PROGRESS")
                return {"ready":{"state":"EMBEDDING_IN_PROGRESS","query_count":0},"status":{
                    "state":"EMBEDDING_IN_PROGRESS","completed_inputs":sum(p["rows"] for p in cm["parts"]),
                    "requested_inputs":inputs["input_count"]},"samples":[],"diagnostics_path":str(run / "runtime_profile.json")}
            with profile.stage("query_embeddings"):
                qm = embed_units([{"id":q["id"],"text":q["query"]} for q in queries],encoder,
                    run / "query_embeddings",batch_size=batch_size,work_dir=work)
                qv = embedding_matrix(run / "query_embeddings",qm)
            tokenizer = encoder.tokenizer
        finally:
            encoder.close()
        store = VectorParts(run / "corpus_embeddings",cm,work / "vector_cache")
        with profile.stage("all_query_exact_dense_search"):
            scores,positions,search = search_parts(store,qv,run / "dense_search",k=config["retrieval"]["dense_top_k"],work_dir=work)
        with profile.stage("bm25_rrf_source_parent_predictions"):
            records,report = predict_baseline(catalog,store,queries,qv,positions,scores,search,run / "queries",
                tokenizer=tokenizer,policy=config["retrieval"],max_new_queries=max_new_queries)
        if report["state"] != "COMPLETE":
            profile.finish("QUERIES_IN_PROGRESS")
            return {"ready":{"state":"QUERIES_IN_PROGRESS","query_count":len(records)},"status":report,
                "samples":[],"diagnostics_path":str(run / "runtime_profile.json")}
        budget = next(m for m in registry["models"] if m["model_id"] == config["dense"]["model_id"] and m["revision"] == config["dense"]["revision"])
        evidence = {"git_commit":code_commit,"master_plan":MASTER_PLAN,"dense":cm["encoder"],
            "executed_models":[config["dense"]],"parameters_before_quantization":budget["parameters"],
            "model_limit_parameters":15_000_000_000,"fine_tuned":False,"neural_reranker_used":False,
            "inference_scope":"LARGE_PARTIAL_CORPUS_BGE_BM25_BASELINE",
            "corpus_embeddings":cm["manifest_sha256"],"query_embeddings":qm["manifest_sha256"],
            "search":search["manifest_sha256"],"config":config,"score_semantics":"weighted_rrf_no_neural_reranker"}
        with profile.stage("source_schema_zip_validation"):
            _,ready = finish_pilot(records,queries,catalog,report,run,contract=contract,tokenizer=tokenizer,
                evidence=evidence,work_dir=work)
        status = {"state":ready["state"],"query_count":len(records),"documents":candidate["counts"]["documents"],
            "children":candidate["counts"]["children"],"unique_child_inputs":inputs["input_count"],
            "complete_master_plan":False,"fine_tuned":False,"official_score":None,
            "architecture":config["architecture"],"executed_models":[config["dense"]],
            "deferred":["Qwen embedding","query translation/expansion","neural reranker","document dense","adaptive source parents","supervised training"],
            "limitations":["Only the imported partial corpus is searchable","Dense document candidates derive from top child hits; not a document-dense branch",
                "Uncalibrated top-k; no relevance labels or score guarantee"],
            "baseline_scope":config["scope"]}
        atomic_json(run / "baseline_status.json",status)
        profile.finish(ready["state"])
        return {"ready":ready,"status":status,"diagnostics_path":str(run / "baseline_status.json"),
            "samples":[{"query":q,"prediction":r["prediction"]} for q,r in zip(queries[:3],records[:3],strict=True)]}
    finally:
        catalog.close()
