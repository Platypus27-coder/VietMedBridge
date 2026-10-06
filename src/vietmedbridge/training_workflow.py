"""Colab supervised workflow: mine → QLoRA → dev F2 finalists → selected adapter."""
from __future__ import annotations

from pathlib import Path

from .artifacts import atomic_json, digest_json, publish_file, read_json, verify_file
from .embeddings import embed_units, embedding_matrix, unit_signature
from .query_expansion import cached_expansions, expand_queries
from .query_translation import cached_translations, translate_queries
from .qwen_models import TorchQwenEncoder, TorchQwenReranker, cached_qwen_embeddings, review_model_registry
from .reranker_training import mine_hard_negatives, train_qlora, validate_query_split
from .retrieval_data import load_catalog, load_queries
from .retrieval_eval import cutoff_sweep, evaluate_candidate_recall
from .retrieval_models import TorchDenseEncoder
from .source_parents import derive_parents
from .strong_retrieval import StrongConfig, StrongIndex, auxiliary_units, predict_strong
from .translation_model import TorchQueryTranslator
from .medical_lexical import MedicalAnalyzer


def slice_query_vectors(queries, source_queries, vectors, manifest):
    positions = {q["id"]: (idx, q) for idx, q in enumerate(source_queries)}
    selected = []
    for q in queries:
        idx, original = positions[q["id"]]
        if original != q:
            raise ValueError("Dev vector slice/query mismatch.")
        selected.append(idx)
    derived = {"units_sha256": unit_signature([{"id": q["id"], "text": q["query"]} for q in queries]),
        "input_count": len(queries), "dimension": manifest["dimension"], "encoder": manifest["encoder"],
        "state": "COMPLETE", "derived_from_manifest_sha256": manifest["manifest_sha256"]}
    derived["manifest_sha256"] = digest_json(derived)
    return vectors[selected], derived


def _selected_adapter(source, destination):
    source, destination = Path(source), Path(destination)
    old = read_json(source / "adapter_manifest.json")
    if digest_json({k: v for k, v in old.items() if k != "manifest_sha256"}) != old["manifest_sha256"]:
        raise ValueError("Checkpoint adapter manifest changed.")
    files = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if name not in old["files"]:
            raise ValueError("Checkpoint lacks a safetensors LoRA adapter.")
        verify_file(source / name, old["files"][name])
        files[name] = publish_file(source / name, destination / name)
    manifest = {"base_model": old["base_model"], "base_revision": old["base_revision"],
        "files": files, "training_contract": old["training_contract"],
        "labels_sha256": old["labels_sha256"], "fine_tuned": True,
        "selected_from_manifest_sha256": old["manifest_sha256"], "selection": "DEV_F2_SELECTED_HELD_OUT_PENDING"}
    manifest["manifest_sha256"] = digest_json(manifest)
    atomic_json(destination / "adapter_manifest.json", manifest)
    return manifest


def run_training_workflow(data_root, checkout, *, work_dir=None, run_name="stage-a-qlora-v1"):
    root, checkout = Path(data_root), Path(checkout)
    run = root / "training" / run_name
    train_path, dev_path = root / "labels/retrieval_train.json", root / "labels/retrieval_dev.json"
    if not train_path.is_file() or not dev_path.is_file():
        status = {"state": "WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS", "fine_tuned": False,
            "required_files": [str(train_path), str(dev_path)], "contest_queries_used_as_gold": False}
        atomic_json(run / "status.json", status)
        return status
    from transformers import AutoTokenizer
    from .dataset import parquet_path

    config = read_json(checkout / "configs/retrieval_full.json")
    review_model_registry(config, read_json(checkout / "configs/strong_model_manifest.json"))
    contest = load_queries(parquet_path(root, "query.parquet"), expected_count=1200)
    train_doc, dev_doc = read_json(train_path), read_json(dev_path)
    train, dev = validate_query_split(train_doc, dev_doc, contest)
    queries = [{"id": q["id"], "query": q["query"]} for q in train + dev]
    dev_queries = [{"id": q["id"], "query": q["query"]} for q in dev]
    split_contract = {"train": digest_json(train_doc), "dev": digest_json(dev_doc), "contest": digest_json(contest)}
    if (run / "split_contract.json").exists() and read_json(run / "split_contract.json") != split_contract:
        raise ValueError("Training split changed; choose a new training run.")
    atomic_json(run / "split_contract.json", split_contract)
    tokenizer = AutoTokenizer.from_pretrained(config["dense"]["model_id"], revision=config["dense"]["revision"], trust_remote_code=False)
    catalog = load_catalog(root / "processed/stage-a-data-v3-laodong", "candidate-1cd220a4be956d5a.json", tokenizer, **config["pilot_limits"])
    policy = StrongConfig(**config["retrieval"])
    catalog = derive_parents(catalog, tokenizer, (policy.parent_short_tokens, policy.parent_long_tokens))
    glossary = root / "labels/medical_aliases.json"
    analyzer = MedicalAnalyzer(segmentation=config["lexical_segmentation"], glossary_path=glossary if glossary.is_file() else None)
    query_units = [{"id": q["id"], "text": q["query"]} for q in queries]
    dense = TorchDenseEncoder(config["dense"])
    try:
        cm = embed_units(catalog.units, dense, run / "corpus_embeddings", work_dir=work_dir, **config["embedding"])
        qm = embed_units(query_units, dense, run / "query_embeddings", work_dir=work_dir, **config["embedding"])
    finally:
        dense.close()
    corpus_vectors, query_vectors = embedding_matrix(run / "corpus_embeddings", cm), embedding_matrix(run / "query_embeddings", qm)
    cached = cached_translations(queries, config["translation"], run / "translations")
    translations = cached[0] if cached is not None else None
    expansions = cached_expansions(queries, translations, config["translation"], run / "expansions", enabled=config["expansion_enabled"]) if translations is not None else None
    if expansions is None:
        translator = TorchQueryTranslator(config["translation"])
        try:
            if translations is None:
                translations, _ = translate_queries(queries, translator, run / "translations")
            expansions = expand_queries(queries, translator, run / "expansions", translations=translations, enabled=config["expansion_enabled"])
        finally:
            translator.close()
    aux = auxiliary_units(queries, expansions)
    corpus_cache = cached_qwen_embeddings(run / "qwen_corpus", catalog.units, config["second_dense"], "corpus")
    query_cache = cached_qwen_embeddings(run / "qwen_queries", aux, config["second_dense"], "query")
    if corpus_cache is None or query_cache is None:
        encoder = TorchQwenEncoder(config["second_dense"])
        try:
            sm = embed_units(catalog.units, encoder.for_role("corpus"), run / "qwen_corpus", work_dir=work_dir, **config["qwen_embedding"])
            am = embed_units(aux, encoder.for_role("query"), run / "qwen_queries", work_dir=work_dir, **config["qwen_embedding"])
        finally:
            encoder.close()
        secondary_vectors, auxiliary_vectors = embedding_matrix(run / "qwen_corpus", sm), embedding_matrix(run / "qwen_queries", am)
    else:
        secondary_vectors, sm = corpus_cache
        auxiliary_vectors, am = query_cache
    index = StrongIndex(catalog, corpus_vectors, cm, run / "index", tokenizer,
        secondary_vectors=secondary_vectors, secondary_manifest=sm, auxiliary_vectors=auxiliary_vectors,
        auxiliary_manifest=am, auxiliary_inputs=aux, queries=queries, expansions=expansions, analyzer=analyzer, work_dir=work_dir)
    model = TorchQwenReranker(config["reranker"])
    try:
        records, _ = predict_strong(index, queries, query_vectors, model, run / "mining_queries",
            query_embedding_manifest=qm, translations=translations, config=policy, batch_size=config["reranker_batch_size"])
        train_bundle = mine_hard_negatives(records, train_doc, catalog, tokenizer, contest)
        dev_bundle = mine_hard_negatives(records, dev_doc, catalog, tokenizer, contest)
        atomic_json(run / "train_mining.json", train_bundle)
        atomic_json(run / "dev_mining.json", dev_bundle)
        if train_bundle["state"] != "READY" or dev_bundle["state"] != "READY":
            status = {"state": "MINING_REQUIRES_MORE_GOLD_COVERAGE_OR_REVIEWED_NEGATIVES", "fine_tuned": False,
                "train_holds": train_bundle["holds"], "dev_holds": dev_bundle["holds"]}
            atomic_json(run / "status.json", status)
            return status
        training = train_qlora(model, train_bundle, dev_bundle, run / "checkpoints", **config["training"])
    finally:
        model.close()
    dev_vectors, dev_manifest = slice_query_vectors(dev_queries, queries, query_vectors, qm)
    translated = {q["id"]: t for q, t in zip(queries, translations, strict=True)}
    dev_translations = [translated[q["id"]] for q in dev_queries]
    finalists = sorted((run / "checkpoints").glob("checkpoint-*")) + [Path(training["adapter"])]
    trials = []
    for checkpoint in finalists:
        reranker = TorchQwenReranker(config["reranker"], adapter_path=checkpoint)
        try:
            records, report = predict_strong(index, dev_queries, dev_vectors, reranker,
                run / "finalist_evaluation" / checkpoint.name / "queries", query_embedding_manifest=dev_manifest,
                translations=dev_translations, config=policy, batch_size=config["reranker_batch_size"])
            sweep = cutoff_sweep(records, dev_doc, catalog, tokenizer, policy)
            atomic_json(run / "finalist_evaluation" / checkpoint.name / "cutoff_sweep.json", sweep)
            trials.append({"checkpoint": str(checkpoint), "best_dev_policy": sweep["best_dev_policy"],
                "prediction_signature": report["signature"], "recall": evaluate_candidate_recall(records, dev_doc["queries"], catalog, tokenizer)})
        finally:
            reranker.close()
    trials.sort(key=lambda t: (-t["best_dev_policy"]["combined_f2"], t["checkpoint"]))
    selected = _selected_adapter(trials[0]["checkpoint"], run / "selected_adapter")
    best = trials[0]["best_dev_policy"]
    from .full_plan_runtime import calibration_context
    calibration = {"state": "DEV_SELECTED_HELD_OUT_PENDING", "split": "dev", "reviewed": True,
        "catalog": catalog.identity, "adapter_manifest_sha256": selected["manifest_sha256"],
        "contest_queries_sha256": digest_json(contest), "dev_labels_sha256": digest_json(dev_doc),
        "dev_query_ids": [q["id"] for q in dev_queries], "dev_query_texts_sha256": [digest_json(q["query"]) for q in dev_queries],
        "reranker_spec": config["reranker"], "calibration_context": calibration_context(config, analyzer),
        "scorer": "PLAN_DERIVED_LOCAL_PROXY_NOT_OFFICIAL_BTC",
        "policy": {"doc_top_k": best["doc_top_k"], "chunk_top_k": best["chunk_top_k"],
            "doc_score_margin": best["score_margin"], "chunk_score_margin": best["score_margin"]}, "trials": trials}
    calibration["manifest_sha256"] = digest_json(calibration)
    atomic_json(run / "calibrated_policy.json", calibration)
    status = {"state": "TRAINED_DEV_F2_SELECTED_HELD_OUT_PENDING", "fine_tuned": True,
        "adapter": str(run / "selected_adapter"), "dev_proxy_f2": best["combined_f2"],
        "official_btc_score": None, "corpus_promoted": False}
    atomic_json(run / "status.json", status)
    return status
