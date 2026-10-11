"""Strong inference over immutable imported vectors, with optional second dense.

All new artifacts live under output_root. In particular, an incomplete Qwen
view never triggers corpus encoding or becomes a smaller searchable corpus.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from . import full_scale_runtime as full
from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .content_embeddings import ContentVectorParts, seal
from .disk_full import DiskStrongIndex
from .embeddings import embed_units, embedding_matrix
from .query_expansion import cached_expansions, expand_queries
from .query_translation import cached_translations, translate_queries
from .runtime_profile import RuntimeProfile, inference_batches
from .scale_baseline import imported_content_manifest
from .scale_benchmark import load_handoff
from .strong_retrieval import auxiliary_units
from .vector_search import search_vectors, preflight
from .vector_index import policy as vector_policy, VectorIndexPending


def prepare_cached_strong_resources(root, checkout, work, queries, *, output_root,
                                    second_dense_mode="if_complete", document_dense_enabled=True,
                                    max_new_embedding_parts=None, max_new_translations=None,
                                    search_device="cuda", vector_search_options=None):
    from transformers import AutoTokenizer

    root, checkout, work, outputs = map(Path, (root, checkout, work, output_root))
    if second_dense_mode not in ("if_complete", "require", "off"):
        raise ValueError("second_dense_mode must be if_complete, require or off.")
    if type(document_dense_enabled) is not bool:
        raise ValueError("document_dense_enabled must be boolean.")
    if any(p.resolve().is_relative_to(root.resolve()) for p in (outputs, work)):
        raise ValueError("Cached strong outputs/work must stay outside read-only DATA_ROOT.")
    work.mkdir(parents=True, exist_ok=True)
    build, candidate, inputs = load_handoff(root)
    config = read_json(checkout / "configs/retrieval_full.json")
    scale = read_json(checkout / "configs/retrieval_scale.json")
    vector_options = vector_policy(scale.get("vector_search", {}) | (vector_search_options or {}))
    registry = read_json(checkout / "configs/strong_model_manifest.json")
    full.review_model_registry(config, registry)
    bge_view, cm = imported_content_manifest(root, candidate, inputs, config, family="bge")
    qwen_view, sm = (None, None) if second_dense_mode == "off" else imported_content_manifest(
        root, candidate, inputs, config, family="qwen")
    second_complete = sm is not None and sm["state"] == "COMPLETE"
    coverage = {"official_input_ids":candidate["counts"]["input_records"],
        "searchable_documents":candidate["counts"]["documents"],
        "unique_child_inputs":inputs["input_count"], "total_parts":len(inputs["parts"]),
        "bge_completed_parts":len(cm["parts"]) if cm else 0,
        "qwen_completed_parts":len(sm["parts"]) if sm else 0,
        "qwen_mode":second_dense_mode, "qwen_dense_used":second_complete,
        "child_corpus_reencoded":False, "source_read_only":True}
    resources = outputs / "cached_strong_resources" / candidate["candidate_manifest_sha256"][:16] / digest_json({
        "dense":config["dense"], "second_dense":config["second_dense"],
        "document_builder":"document-title-bounded-opening-v1"})[:12]
    resources.mkdir(parents=True, exist_ok=True)
    atomic_json(resources / "cache_coverage.json", coverage)
    profile = RuntimeProfile(resources / "coordinator-profile.json")
    def waiting(state, **details):
        status = {"state":state, "resources":str(resources), "coverage":coverage, **details}
        atomic_json(resources / "preparation_status.json", status)
        profile.finish(state)
        return status
    if cm is None or cm["state"] != "COMPLETE":
        return waiting("WAITING_FOR_COMPLETE_BGE_CACHE")
    if second_dense_mode == "require" and not second_complete:
        return waiting("WAITING_FOR_COMPLETE_QWEN_CACHE")
    preflight(vector_options)
    print(f"Cached strong: {coverage['searchable_documents']:,} documents; "
          f"BGE {coverage['bge_completed_parts']}/{coverage['total_parts']}; "
          f"Qwen {coverage['qwen_completed_parts']}/{coverage['total_parts']} "
          f"({'enabled' if second_complete else 'deferred'}). No child corpus encoding.", flush=True)
    config["retrieval"] = dict(config["retrieval"])
    if not second_complete:
        config["retrieval"]["second_dense_weight"] = 0.
    if not document_dense_enabled:
        config["retrieval"]["document_dense_weight"] = 0.
    config["scale_execution"] = scale
    config["vector_search"] = vector_options
    config["architecture"] = "master-plan-cached-strong-" + ("dual-dense" if second_complete else "bge-dense")
    config["execution_policy"] = {"mode":"cached_strong", "second_dense_mode":second_dense_mode,
        "second_dense_active":second_complete, "document_dense_enabled":document_dense_enabled,
        "auxiliary_dense_family":"qwen" if second_complete else "bge",
        "child_corpus_reencoded":False, "source_read_only":True}
    query_root = outputs / "full_query_cache" / digest_json({"queries":queries,
        "models":{k:config[k] for k in ("dense", "second_dense", "translation")},
        "expansion":config["expansion_enabled"],
        "query_code":[sha256_file(Path(__file__).with_name(n)) for n in (
            "query_translation.py", "query_expansion.py", "translation_model.py")]})[:20]
    tokenizer = AutoTokenizer.from_pretrained(config["dense"]["model_id"],
        revision=config["dense"]["revision"], trust_remote_code=False)
    catalog, analyzer = full.load_scale_catalog(root, checkout, work, tokenizer, read_only=True)
    config["analyzer"] = analyzer.identity
    handed_off = False
    try:
        with profile.stage("query_translation_expansion"):
            cached = cached_translations(queries, config["translation"], query_root / "translations")
            translations = cached[0] if cached else None
            expansions = cached_expansions(queries, translations, config["translation"], query_root / "expansions",
                enabled=config["expansion_enabled"]) if translations is not None else None
            if expansions is None:
                translator = full.TorchQueryTranslator(config["translation"])
                try:
                    if translations is None:
                        translations, report = translate_queries(queries, translator, query_root / "translations",
                            max_new_queries=max_new_translations)
                        if report["state"] != "COMPLETE":
                            return waiting("TRANSLATIONS_IN_PROGRESS", report=report)
                    expansions = expand_queries(queries, translator, query_root / "expansions",
                        translations=translations, enabled=config["expansion_enabled"])
                finally:
                    translator.close()
        aux = auxiliary_units(queries, expansions)
        dense = full.TorchDenseEncoder(config["dense"])
        try:
            if dense.identity != cm["encoder"] or dense.dimension != cm["dimension"]:
                raise ValueError("Query BGE differs from imported producer; use the embedding runtime lock.")
            with profile.stage("bge_query_embeddings"):
                qm = embed_units([{"id":q["id"], "text":q["query"]} for q in queries], dense,
                    query_root / "bge", work_dir=work, **config["embedding"])
                qv = embedding_matrix(query_root / "bge", qm)
            di, dm = None, None
            if document_dense_enabled:
                with profile.stage("document_inputs"):
                    di = full._document_inputs(catalog, tokenizer, resources / "document_inputs", config["dense"]["max_length"])
                with profile.stage("bge_document_embeddings"):
                    dm = full.content_embeddings(outputs, resources / "document_inputs", di, dense,
                        resources / "document_dense", work_dir=work, batch_size=scale["bge_batch_size"],
                        max_new_parts=max_new_embedding_parts)
                if dm["state"] != "COMPLETE":
                    return waiting("DOCUMENT_EMBEDDINGS_IN_PROGRESS", completed_parts=len(dm["parts"]),
                        requested_parts=len(di["parts"]), document_inputs=di["input_count"])
            if not second_complete:
                with profile.stage("bge_auxiliary_query_embeddings"):
                    am = embed_units(aux, dense, query_root / "bge_auxiliary", work_dir=work, **config["embedding"])
                    av = embedding_matrix(query_root / "bge_auxiliary", am)
        finally:
            dense.close()
        if second_complete:
            encoder = full.TorchQwenEncoder(config["second_dense"])
            try:
                with profile.stage("qwen_query_embeddings"):
                    am = embed_units(aux, encoder.for_role("query"), query_root / "qwen", work_dir=work,
                        **(config["qwen_embedding"] | {"batch_size":inference_batches()["embedding"]}))
                    av = embedding_matrix(query_root / "qwen", am)
                    if {k:v for k,v in am["encoder"].items() if k != "input_role"} != {
                        k:v for k,v in sm["encoder"].items() if k != "input_role"}:
                        raise ValueError("Qwen query/corpus encoder identity mismatch.")
            finally:
                encoder.close()
        # Retain the paid-for BGE child vectors locally when space permits.
        cache_dir = work / "vector_cache"  # shared with the cached BGE/BM25 entrypoint
        retained_path = cache_dir / digest_json([str(bge_view), cm["manifest_sha256"]])[:16]
        retained = sum(p.stat().st_size for p in retained_path.glob("*") if p.is_file())
        desired = int(inputs["input_count"] * cm["dimension"] * 4 * 1.1) + 256 * 2**20
        primary_budget = max(1, min(desired, shutil.disk_usage(work).free + retained - 8 * 2**30))
        primary = ContentVectorParts(root, bge_view, cm, cache_dir, local_cache_bytes=primary_budget)
        secondary = ContentVectorParts(root, qwen_view, sm, cache_dir,
            local_cache_bytes=scale["local_vector_cache_bytes_per_branch"]) if second_complete else None
        documents = ContentVectorParts(outputs, resources / "document_dense", dm, cache_dir,
            local_cache_bytes=min(scale["local_vector_cache_bytes_per_branch"], 2**30)) if dm else None
        search_root = outputs / "vector_search" / digest_json({"candidate":candidate["candidate_manifest_sha256"],"queries":queries, "aux":aux,
            "auxiliary_family":config["execution_policy"]["auxiliary_dense_family"]})[:16]
        searches = []
        def search(name, store, vectors, k):
            with profile.stage("dense_search_"+name):
                result = search_vectors(store, vectors, search_root / name, k=k, work_dir=work, device=search_device,
                    options=vector_options, index_root=outputs / "vector_indexes" / vector_options["backend"],
                    reference_search=full.search_parts)
            searches.append(result)
            return result
        if secondary is None:
            # Query, subquery and HyDE scans share one complete BGE corpus pass.
            combined = search("bge_and_auxiliary", primary, np.concatenate((qv, av)), scale["dense_top_k"])
            ps = (combined[0][:len(queries)], combined[1][:len(queries)])
            ss = (combined[0][len(queries):], combined[1][len(queries):])
        else:
            ps = search("bge", primary, qv, scale["dense_top_k"])
            ss = search("qwen", secondary, av, scale["dense_top_k"])
        primary_search = {q["query"]:(ps[0][i], ps[1][i]) for i,q in enumerate(queries)}
        secondary_search = {u["id"]:(ss[0][i], ss[1][i]) for i,u in enumerate(aux)}
        document_search, document_ids = {}, []
        if documents is not None:
            ds = search("documents", documents, qv, config["retrieval"]["doc_candidate_k"])
            document_search = {q["query"]:(ds[0][i], ds[1][i]) for i,q in enumerate(queries)}
            for part in di["parts"]:
                path = resources / "document_inputs" / part["path"]
                full.verify_file(path,part["sha256"])
                document_ids.extend(int(u) for u in pq.read_table(path, columns=["id"]).column("id").to_pylist())
        branches = {"bge_child_dense":"COMPLETE", "bm25_vi_en_zh":"COMPLETE",
            "query_translation":"COMPLETE", "pico_subqueries_hyde":"COMPLETE" if config["expansion_enabled"] else "DISABLED",
            "document_dense":"COMPLETE" if dm else "DISABLED",
            "qwen_child_dense":"COMPLETE" if secondary is not None else "DEFERRED_INCOMPLETE_CACHE" if sm else "DISABLED_OR_UNAVAILABLE",
            "auxiliary_dense_family":config["execution_policy"]["auxiliary_dense_family"]}
        branches["vector_index"] = vector_options["backend"]
        vector_reports = [{"backend":s[2]["backend"],"index_manifest_sha256":s[2].get("index"),
            "quality":s[2].get("quality"),"exact_candidate_rescoring":s[2].get("exact_candidate_rescoring",False)} for s in searches]
        identity = {"catalog":catalog.identity, "primary":cm["manifest_sha256"],
            "secondary":sm["manifest_sha256"] if secondary is not None else None,
            "document_dense":dm["manifest_sha256"] if dm else None, "auxiliary":am["manifest_sha256"],
            "expansions":digest_json(expansions), "searches":[s[2]["manifest_sha256"] for s in searches],
            "analyzer":analyzer.identity, "execution_policy":config["execution_policy"],
            "code_sha256":sha256_file(Path(__file__).with_name("disk_full.py"))}
        manifest = seal(identity | {"signature":digest_json(identity)})
        index = DiskStrongIndex(catalog, primary, secondary, qv, {u["id"]:v for u,v in zip(aux,av,strict=True)},
            queries, expansions, primary_search, secondary_search, document_search, document_ids, tokenizer, manifest,
            auxiliary_family=config["execution_policy"]["auxiliary_dense_family"])
        atomic_json(resources / "preparation_status.json", {"state":"FULL_RESOURCES_READY", "branches":branches,
            "coverage":coverage, "resources":str(resources)})
        profile.finish("FULL_RESOURCES_READY")
        handed_off = True
        return {"state":"FULL_RESOURCES_READY", "index":index, "vectors":qv, "query_manifest":qm,
            "translations":translations, "config":config, "registry":registry, "resources":str(resources),
            "branches":branches, "coverage":coverage,"vector_search_reports":vector_reports}
    except VectorIndexPending as pending:
        return waiting(pending.report["state"],vector_index=pending.report)
    finally:
        if not handed_off:
            catalog.close()
