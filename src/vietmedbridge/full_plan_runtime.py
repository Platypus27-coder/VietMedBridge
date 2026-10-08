"""The single Colab entrypoint for the master-plan architecture and submission."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import re
import time

from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .competition_pilot import MASTER_PLAN, bind_pilot_run, finish_pilot
from .dataset import parquet_path
from .document_dense import prepare_document_vectors, document_units
from .embeddings import embed_units, embedding_matrix
from .medical_lexical import MedicalAnalyzer
from .model_budget import model_budget_report
from .query_expansion import cached_expansions, expand_queries
from .query_translation import cached_translations, translate_queries
from .qwen_models import TorchQwenEncoder, TorchQwenReranker, cached_qwen_embeddings, review_model_registry
from .retrieval_data import load_catalog, load_queries
from .retrieval_diagnostics import diagnose
from .retrieval_models import TorchDenseEncoder
from .source_parents import derive_parents
from .strong_retrieval import StrongConfig, StrongIndex, auxiliary_units, predict_strong
from .translation_model import TorchQueryTranslator
from .shared_embeddings import find_embeddings
from .runtime_profile import RuntimeProfile, inference_batches


def calibration_context(config, analyzer):
    return {"model_and_retrieval_policy": {k: config[k] for k in
        ("dense", "second_dense", "translation", "retrieval", "expansion_enabled")},
        "analyzer": analyzer.identity,
        "scoring_code": {name: sha256_file(Path(__file__).with_name(name)) for name in
            ("retrieval_eval.py", "retrieval_policy.py", "strong_retrieval.py")},
        **({"scale_execution":config["scale_execution"],"disk_scoring_code":sha256_file(Path(__file__).with_name("disk_full.py"))}
            if "scale_execution" in config else {})}


def reviewed_calibration(root, catalog, queries, spec, *, context):
    path = Path(root) / "calibrated_policy.json"
    if not path.exists():
        return None, None
    document = read_json(path)
    if (document.get("split") != "dev" or document.get("reviewed") is not True
        or document.get("catalog") != catalog.identity or document.get("reranker_spec") != spec
        or document.get("calibration_context") != context
        or document.get("contest_queries_sha256") != digest_json(queries)
        or digest_json({k: v for k, v in document.items() if k != "manifest_sha256"}) != document["manifest_sha256"]):
        raise ValueError("Selected adapter/dev calibration does not match this corpus/model/query set.")
    if set(document["dev_query_ids"]) & {q["id"] for q in queries} or set(document["dev_query_texts_sha256"]) & {digest_json(q["query"]) for q in queries}:
        raise ValueError("Calibration leaked contest queries.")
    adapter = read_json(Path(root) / "selected_adapter/adapter_manifest.json")
    if digest_json({k: v for k, v in adapter.items() if k != "manifest_sha256"}) != adapter["manifest_sha256"] or adapter["manifest_sha256"] != document["adapter_manifest_sha256"]:
        raise ValueError("Selected adapter/calibration hash mismatch.")
    allowed = {"doc_top_k", "chunk_top_k", "doc_score_margin", "chunk_score_margin"}
    if set(document["policy"]) != allowed:
        raise ValueError("Unexpected calibrated parameters.")
    return document, Path(root) / "selected_adapter"


def run_full_pipeline(data_root, checkout, *, code_commit, work_dir=None,
                      build_run="stage-a-data-v3-laodong", candidate_name="candidate-1cd220a4be956d5a.json",
                      run_name="stage-a-full-plan-v3-per-model-15b", embedding_cache_run="stage-a-retrieval-v1",
                      max_new_embedding_parts=None, max_new_translations=None, max_new_queries=None):
    import torch
    from transformers import AutoTokenizer

    for value in (build_run, candidate_name, run_name, embedding_cache_run):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("Unsafe run/candidate name.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU before Run all.")
    root, checkout = Path(data_root), Path(checkout)
    config = read_json(checkout / "configs/retrieval_full.json")
    registry = read_json(checkout / "configs/strong_model_manifest.json")
    review_model_registry(config, registry)
    config["strong_model_review_sha256"] = sha256_file(checkout / "configs/strong_model_manifest.json")
    policy = StrongConfig(**config["retrieval"])
    policy.validate()
    tokenizer = AutoTokenizer.from_pretrained(config["dense"]["model_id"], revision=config["dense"]["revision"],
        use_fast=True, trust_remote_code=False)
    catalog = load_catalog(root / "processed" / build_run, candidate_name, tokenizer, **config["pilot_limits"])
    if any(catalog.candidate["golden"]["tokenizer"][k] != config["dense"][k] for k in ("model_id", "revision")):
        raise ValueError("Frozen tokenizer differs from BGE encoding policy.")
    snapshot = read_json(root / "raw/snapshot.json")
    if snapshot["files"]["links_corpus.parquet"]["sha256"] != catalog.build_config["official_links_sha256"]:
        raise ValueError("Frozen candidate/official snapshot mismatch.")
    queries = load_queries(parquet_path(root, "query.parquet"), expected_count=1200)
    catalog = derive_parents(catalog, tokenizer, (policy.parent_short_tokens, policy.parent_long_tokens))
    glossary = root / "labels/medical_aliases.json"
    analyzer = MedicalAnalyzer(segmentation=config["lexical_segmentation"], glossary_path=glossary if glossary.is_file() else None)
    config["analyzer"] = analyzer.identity
    calibration, adapter_path = reviewed_calibration(root / "training/stage-a-qlora-v3-per-model-15b", catalog, queries, config["reranker"],
        context=calibration_context(config, analyzer))
    budget = model_budget_report(config, registry, adapter_paths=[adapter_path] if adapter_path else [])
    config["model_parameter_budget"] = budget
    base_run = root / "retrieval" / run_name
    if calibration is not None:
        config["adapter_manifest_sha256"] = calibration["adapter_manifest_sha256"]
        policy = replace(policy, **calibration["policy"])
        config["retrieval"] = asdict(policy)
        run_name += "-ft-" + calibration["manifest_sha256"][:8]
    run = root / "retrieval" / run_name
    contract = bind_pilot_run(run, plan_path=checkout / MASTER_PLAN, catalog=catalog,
        queries=queries, config=config, code_commit=code_commit)
    atomic_json(run / "model_parameter_budget.json", budget)
    profile = RuntimeProfile(run / "runtime_profile.json")
    batches = inference_batches()
    atomic_json(run / "execution_policy.json", batches)
    print(f"Largest model: {budget['max_model_parameters']:,} / {budget['limit_parameters']:,}; total inventory: {budget['total_parameters']:,}")
    print("GPU:", torch.cuda.get_device_name(0), "| Run:", run)
    query_units = [{"id": q["id"], "text": q["query"]} for q in queries]
    cache = root / "retrieval" / embedding_cache_run
    stage_started = time.perf_counter()
    corpus_cache = find_embeddings(root, catalog.units, config["dense"], family="bge",
        preferred=[cache / "corpus_embeddings", run / "corpus_embeddings"])
    query_cache = find_embeddings(root, query_units, config["dense"], family="bge", role="query",
        preferred=[cache / "query_embeddings", run / "query_embeddings"])
    if corpus_cache is not None and query_cache is not None:
        corpus_vectors, cm = corpus_cache[:2]
        query_vectors, qm = query_cache[:2]
        if cm["encoder"] != qm["encoder"]:
            raise ValueError("BGE corpus/query producer identities differ.")
        atomic_json(run / "embedding_reuse.json", {"corpus_producer": str(corpus_cache[2]), "corpus": cm["manifest_sha256"], "queries": qm["manifest_sha256"]})
    else:
        dense = TorchDenseEncoder(config["dense"])
        try:
            if corpus_cache is None:
                cm = embed_units(catalog.units, dense, run / "corpus_embeddings", work_dir=work_dir,
                    max_new_parts=max_new_embedding_parts, **config["embedding"])
                corpus_vectors = embedding_matrix(run / "corpus_embeddings", cm)
            else:
                corpus_vectors, cm = corpus_cache[:2]
            if query_cache is None:
                qm = embed_units(query_units, dense, run / "query_embeddings", work_dir=work_dir,
                    max_new_parts=max_new_embedding_parts, **config["embedding"])
                query_vectors = embedding_matrix(run / "query_embeddings", qm)
            else:
                query_vectors, qm = query_cache[:2]
        finally:
            dense.close()
        if cm["encoder"] != qm["encoder"]:
            raise ValueError("BGE corpus/query producer runtimes differ; use compatible runtimes.")
    doc_units = document_units(catalog, tokenizer, config["dense"]["max_length"])
    doc_cache = find_embeddings(root, doc_units, config["dense"], family="bge", preferred=[base_run / "document_embeddings"])
    if doc_cache is not None:
        document_vectors, dm = doc_cache[:2]
    else:
        document_vectors, dm, doc_units = prepare_document_vectors(catalog, tokenizer, config["dense"],
            base_run / "document_embeddings", TorchDenseEncoder, embedding=config["embedding"], work_dir=work_dir)
    profile.record("bge_and_document_embeddings", stage_started,
        corpus_reused_from=str(corpus_cache[2]) if corpus_cache else None,
        document_reused_from=str(doc_cache[2]) if doc_cache else None)
    stage_started = time.perf_counter()
    translation_root = run / "translations"
    cached = cached_translations(queries, config["translation"], translation_root)
    if cached is None:
        for previous in (base_run, root / "retrieval/stage-a-retrieval-v2.1"):
            available = cached_translations(queries, config["translation"], previous / "translations")
            if available is not None:
                translation_root, cached = previous / "translations", available
                break
    translations = cached[0] if cached is not None else None
    expansion_root = base_run / "expansions"
    expansions = cached_expansions(queries, translations, config["translation"], expansion_root, enabled=config["expansion_enabled"]) if translations is not None else None
    if expansions is None:
        translator = TorchQueryTranslator(config["translation"])
        started = time.monotonic()
        try:
            if translations is None:
                translations, report = translate_queries(queries, translator, translation_root, max_new_queries=max_new_translations)
                if report["state"] != "COMPLETE":
                    raise RuntimeError("Translations checkpointed; Run all again to finish before expansion.")
            expansions = expand_queries(queries, translator, expansion_root, translations=translations, enabled=config["expansion_enabled"])
            atomic_json(run / "query_llm_runtime.json", {"model": translator.identity, "seconds_this_call": time.monotonic() - started})
        finally:
            translator.close()
    profile.record("query_translation_expansion", stage_started)
    stage_started = time.perf_counter()
    aux = auxiliary_units(queries, expansions)
    corpus_root, auxiliary_root = base_run / "qwen_corpus", base_run / "qwen_queries"
    secondary_cache = find_embeddings(root, catalog.units, config["second_dense"], family="qwen", preferred=[corpus_root])
    auxiliary_cache = cached_qwen_embeddings(auxiliary_root, aux, config["second_dense"], "query")
    if secondary_cache is None or auxiliary_cache is None:
        encoder = TorchQwenEncoder(config["second_dense"])
        started = time.monotonic()
        try:
            options = {**config["qwen_embedding"], "batch_size": batches["embedding"]}
            if secondary_cache is None:
                sm = embed_units(catalog.units, encoder.for_role("corpus"), corpus_root, work_dir=work_dir,
                    max_new_parts=max_new_embedding_parts, **options)
                secondary_vectors = embedding_matrix(corpus_root, sm)
            else:
                secondary_vectors, sm = secondary_cache[:2]
            if auxiliary_cache is None:
                am = embed_units(aux, encoder.for_role("query"), auxiliary_root, work_dir=work_dir,
                    max_new_parts=max_new_embedding_parts, **options)
                auxiliary_vectors = embedding_matrix(auxiliary_root, am)
            else:
                auxiliary_vectors, am = auxiliary_cache
            atomic_json(run / "qwen_dense_runtime.json", {"model": encoder.identity,
                "seconds_this_call": time.monotonic() - started, "oom_backoffs": encoder.oom_backoffs})
        finally:
            encoder.close()
    else:
        secondary_vectors, sm = secondary_cache[:2]
        auxiliary_vectors, am = auxiliary_cache
    profile.record("qwen_embeddings", stage_started,
        corpus_reused_from=str(secondary_cache[2]) if secondary_cache else None)
    stage_started = time.perf_counter()
    index = StrongIndex(catalog, corpus_vectors, cm, run / "index", tokenizer,
        secondary_vectors=secondary_vectors, secondary_manifest=sm, auxiliary_vectors=auxiliary_vectors,
        auxiliary_manifest=am, auxiliary_inputs=aux, queries=queries, expansions=expansions, analyzer=analyzer, work_dir=work_dir,
        document_vectors=document_vectors, document_manifest=dm, document_inputs=doc_units)
    profile.record("index", stage_started)
    with profile.stage("reranker_loading"):
        reranker = TorchQwenReranker(config["reranker"], adapter_path=adapter_path)
    started = time.monotonic()
    try:
        with profile.stage("competition_reranking"):
            records, report = predict_strong(index, queries, query_vectors, reranker, run / "queries",
                query_embedding_manifest=qm, translations=translations, config=policy,
                batch_size=batches["reranker"], max_new_queries=max_new_queries)
        reranker_identity = dict(reranker.identity)
        atomic_json(run / "qwen_reranker_runtime.json", {"model": reranker_identity,
            "seconds_this_call": time.monotonic() - started, "oom_backoffs": reranker.oom_backoffs})
    finally:
        reranker.close()
    if report["state"] != "COMPLETE":
        raise RuntimeError("Queries checkpointed. Run all again with limits=None before exporting.")
    evidence = {"git_commit": code_commit, "dense": cm["encoder"], "second_dense": sm["encoder"],
        "document_dense": dm, "translation_llm": read_json(translation_root / "config.json")["translator"], "expansion_signature": expansions[0]["signature"],
        "reranker": reranker_identity, "index": index.manifest, "retrieval_config": asdict(policy),
        "calibration": calibration, "model_parameter_budget": budget,
        "inference_scope": "COLAB_GPU_DEV_SELECTED_ADAPTER" if calibration else "COLAB_GPU_FULL_PRETRAINED_ARCHITECTURE"}
    _, ready = finish_pilot(records, queries, catalog, report, run, contract=contract, tokenizer=tokenizer,
        evidence=evidence, reference_labels_path=root / "labels/retrieval_reference.json", work_dir=work_dir)
    diagnostics = diagnose(records, queries, index, run / "diagnostics.json")
    heldout_state = "WAITING_FOR_INDEPENDENT_REVIEWED_HELD_OUT_LABELS"
    heldout_path = root / "training/stage-a-qlora-v3-per-model-15b/heldout_evaluation.json"
    if calibration is not None and heldout_path.exists():
        heldout = read_json(heldout_path)
        if (heldout.get("state") == "FROZEN_POLICY_HELD_OUT_EVALUATED_LOCAL_PROXY"
            and heldout.get("catalog") == catalog.identity
            and heldout.get("calibration_manifest_sha256") == calibration["manifest_sha256"]
            and heldout.get("adapter_manifest_sha256") == calibration["adapter_manifest_sha256"]
            and digest_json({k:v for k,v in heldout.items() if k != "manifest_sha256"}) == heldout.get("manifest_sha256")):
            heldout_state = heldout["state"]
    status = {"master_plan": MASTER_PLAN, "architecture": config["architecture"], "query_count": len(records),
        "model_parameter_budget": budget,
        "document_dense":"EXECUTED_OR_VERIFIED_CACHE", "heldout_evaluation":heldout_state,
        "frozen_documents": len(catalog.documents), "eligible_documents": len(index.doc_ids),
        "second_dense": "EXECUTED_OR_VERIFIED_CACHE", "qwen_reranking": "EXECUTED_OR_VERIFIED_CACHE",
        "query_expansion": "VALIDATED_COMPLEX_ONLY_ADDITIVE", "medical_alias_entries": analyzer.identity["aliases"],
        "fine_tuning": "DEV_SELECTED_HELD_OUT_PENDING" if calibration else "WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS",
        "cutoff_calibration": "DEV_SELECTED" if calibration else "UNTUNED_WITHOUT_DEV_LABELS",
        "full_corpus": False, "official_btc_score": None, "quality_promoted": False}
    if calibration is not None and heldout_state == "FROZEN_POLICY_HELD_OUT_EVALUATED_LOCAL_PROXY":
        status["fine_tuning"] = "DEV_SELECTED_HELD_OUT_EVALUATED_LOCAL_PROXY"
    atomic_json(run / "full_plan_status.json", status)
    profile.finish(status["inference_scope"] if "inference_scope" in status else ready["state"])
    return {"ready": ready, "status": status, "diagnostics_path": str(run / "diagnostics.json"),
        "samples": [{"query": q, "prediction": r["prediction"]} for q, r in zip(queries[:3], records[:3], strict=True)]}
