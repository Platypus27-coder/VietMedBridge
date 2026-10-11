"""One search contract for portable ANN, Qdrant, and explicit exact reference.

ANN results are reranked using original vectors. A reproducible actual-query
sample must meet recall against full-corpus exact top-k before inference starts.
This is an ANN approximation check, not a relevance evaluation or F2 gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, sha256_file, verify_file
from .content_embeddings import seal
from .embeddings import validate_vectors
from .scale_vectors import bound, check_manifest, search_parts
from .vector_index import FaissVectorIndex, build_faiss_index, policy, refine_stream


def preflight(options):
    """Reject missing index libraries/server access before spending GPU work."""
    config = policy(options)
    if config["backend"] == "faiss_ivf_sq8":
        from .vector_index import _faiss
        _faiss(config)
    elif config["backend"] == "qdrant":
        from .qdrant_vectors import QdrantVectorIndex
        probe = QdrantVectorIndex(None,None,config)
        try:
            probe.request("GET","/collections")
        finally:
            probe.close()


def search_vectors(store,query_vectors,output_dir,*,k=2048,work_dir=None,device="cuda",
                   options=None,index_root=None,reference_search=None,max_new_batches=None):
    config = policy(options)
    if config["backend"] == "exact":
        return (reference_search or search_parts)(store,query_vectors,output_dir,k=k,work_dir=work_dir,device=device)
    if type(k) is not int or k < 1 or (max_new_batches is not None and (type(max_new_batches) is not int or max_new_batches < 0)):
        raise ValueError("Invalid vector search limits.")
    validate_vectors(query_vectors,len(query_vectors),store.manifest["dimension"])
    if not len(query_vectors):
        raise ValueError("Vector search requires queries.")
    k = min(k,store.manifest["input_count"])
    root = Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    durable = Path(index_root) if index_root is not None else root.parent / "vector_indexes"
    if config["backend"] == "faiss_ivf_sq8":
        manifest = build_faiss_index(store,durable,options=config,work_dir=work_dir)
        index = FaissVectorIndex(durable,manifest,config)
    else:
        from .qdrant_vectors import QdrantVectorIndex
        index = QdrantVectorIndex(store,durable,config)
        try:
            manifest = index.prepare()
        except BaseException:
            index.close()
            raise
    try:
        identity = {"index":manifest["manifest_sha256"],"queries":hashlib.sha256(query_vectors.tobytes()).hexdigest(),
            "policy":config,"k":k,"format":"ann-refined-positions-v1",
            "code":{n:sha256_file(Path(__file__).with_name(n)) for n in ("vector_index.py","vector_search.py","qdrant_vectors.py")}}
        signature = digest_json(identity)
        run = root / signature[:20]
        marker = run / "search.json"
        if marker.exists():
            saved = read_json(marker)
            check_manifest(saved)
            if saved["signature"] != signature:
                raise ValueError("Vector search identity changed.")
            for key in ("scores","positions"):
                verify_file(bound(run,saved[key]["path"]),saved[key]["sha256"])
            return np.load(bound(run,saved["scores"]["path"]),allow_pickle=False),np.load(bound(run,saved["positions"]["path"]),allow_pickle=False),saved
        sample_ids = np.unique(np.linspace(0,len(query_vectors)-1,min(config["recall_queries"],len(query_vectors)),dtype=np.int64))
        sample = query_vectors[sample_ids]
        gate_k = min(config["recall_k"],k)
        gate_path = run / "ann_quality.json"
        if gate_path.exists():
            gate = read_json(gate_path)
            check_manifest(gate)
            if gate["signature"] != signature or gate["recall"] < config["minimum_recall"]:
                raise ValueError("ANN quality checkpoint changed or did not pass.")
        else:
            print(f"ANN recall check: {len(sample)} actual queries, exact top-{gate_k} over all vectors",flush=True)
            _,truth,_ = (reference_search or search_parts)(store,sample,run / "exact_reference",k=gate_k,work_dir=work_dir,device=device)
            effort = config["nprobe"] if config["backend"] == "faiss_ivf_sq8" else config["qdrant_hnsw_ef"]
            max_effort = config["max_nprobe"] if config["backend"] == "faiss_ivf_sq8" else max(effort,2048)
            while True:
                _,ids = index.search(sample,k,effort)
                gate_candidates = hashlib.sha256(ids.tobytes()).hexdigest()[:12]
                _,ids = refine_stream(store,sample,ids,run / ("gate-"+gate_candidates),work_dir=work_dir)
                recalls = [len(set(a)&set(b[:gate_k]))/gate_k for a,b in zip(truth,ids,strict=True)]
                recall = float(np.mean(recalls))
                print(f"ANN recall@{gate_k}: {recall:.4f} | effort {effort}",flush=True)
                if recall >= config["minimum_recall"]:
                    break
                if effort >= max_effort:
                    atomic_json(run / "ann_quality_failed.json",seal({"state":"ANN_RECALL_FAILED", "recall":recall,
                        "minimum_recall":config["minimum_recall"], "effort":effort, "signature":signature}))
                    raise RuntimeError("ANN recall gate failed; increase search effort or explicitly select backend='exact'. No submission was produced with this index.")
                effort = min(max_effort,effort*2)
            gate = seal({"state":"COMPLETE", "signature":signature,"recall":recall,
                "recall_per_query":recalls,"sample_query_positions":sample_ids.tolist(),"k":gate_k,
                "effort":effort,"minimum_recall":config["minimum_recall"],
                "scope":"ANN approximation recall on actual query sample; not relevance/F2 evaluation"})
            atomic_json(gate_path,gate)
        scores = np.empty((len(query_vectors),k),np.float32)
        positions = np.empty((len(query_vectors),k),np.int64)
        written = 0
        for start in range(0,len(query_vectors),config["query_batch_size"]):
            end = min(len(query_vectors),start+config["query_batch_size"])
            batch_path = run / f"batch-{start:08d}.json"
            if batch_path.exists():
                batch = read_json(batch_path)
                check_manifest(batch)
                if batch["signature"] != signature or batch["start"] != start or batch["rows"] != end-start:
                    raise ValueError("ANN query batch checkpoint mismatch.")
                verify_file(bound(run,batch["path"]),batch["sha256"])
                with np.load(bound(run,batch["path"]),allow_pickle=False) as arrays:
                    ss,ii = arrays["scores"],arrays["positions"]
            else:
                if max_new_batches is not None and written >= max_new_batches:
                    return scores[:start],positions[:start],seal({"state":"IN_PROGRESS","completed_queries":start,"signature":signature})
                q = query_vectors[start:end]
                ss,ii = index.search(q,k,gate["effort"])
                with local_workspace(work_dir) as temporary:
                    local = Path(temporary) / f"batch-{start:08d}.npz"
                    np.savez(local,scores=ss,positions=ii)
                    checksum = publish_file(local,run / local.name)
                atomic_json(batch_path,seal({"state":"COMPLETE","signature":signature,"start":start,
                    "rows":end-start,"path":local.name,"sha256":checksum}))
                written += 1
            if ss.shape != (end-start,k) or ii.shape != ss.shape or not np.isfinite(ss).all() or np.any(ii < 0) or np.any(ii >= store.manifest["input_count"]):
                raise ValueError("Invalid ANN search checkpoint arrays.")
            scores[start:end],positions[start:end] = ss,ii
            print(f"Vector search ({config['backend']}): {end:,}/{len(query_vectors):,} queries",flush=True)
        scores,positions = refine_stream(store,query_vectors,positions,run / "refine",work_dir=work_dir)
        saved = identity | {"signature":signature,"state":"COMPLETE","backend":manifest["backend"],
            "index_manifest":manifest,"quality":gate,"exact_candidate_rescoring":True}
        with local_workspace(work_dir) as temporary:
            for key,array in (("scores",scores),("positions",positions)):
                local = Path(temporary) / (key+".npy")
                np.save(local,array,allow_pickle=False)
                saved[key] = {"path":local.name,"sha256":publish_file(local,run / local.name)}
        saved = seal(saved)
        atomic_json(marker,saved)
        return scores,positions,saved
    finally:
        if hasattr(index,"close"):
            index.close()
