"""Colab supervised workflow: mine → QLoRA → dev F2 finalists → selected adapter."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import time

from .artifacts import atomic_json, digest_json, publish_file, read_json, verify_file
from .embeddings import embed_units, embedding_matrix, unit_signature
from .document_dense import prepare_document_vectors, document_units
from .heldout import evaluate_frozen_selection, validate_heldout_split
from .model_budget import model_budget_report
from .negative_review import export_candidate_review, import_candidate_review
from .query_expansion import cached_expansions, expand_queries
from .query_translation import cached_translations, translate_queries
from .qwen_models import TorchQwenEncoder, TorchQwenReranker, cached_qwen_embeddings, review_model_registry
from .reranker_training import mine_hard_negatives, train_qlora, validate_query_split
from .retrieval_data import load_catalog, load_queries
from .retrieval_eval import cutoff_sweep, evaluate_candidate_recall
from .retrieval_diagnostics import run_dev_ablations
from .retrieval_models import TorchDenseEncoder
from .source_parents import derive_parents
from .strong_retrieval import StrongConfig, StrongIndex, auxiliary_units, predict_strong
from .translation_model import TorchQueryTranslator
from .medical_lexical import MedicalAnalyzer
from .shared_embeddings import find_embeddings
from .runtime_profile import RuntimeProfile, inference_batches
from .training_mining import cached_mining_records, retrieve_mining_candidates, unique_finalists


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


def run_training_workflow(data_root, checkout, *, work_dir=None, run_name="stage-a-qlora-v3-per-model-15b", run_ablations=False):
    root, checkout = Path(data_root), Path(checkout)
    base_run = run = root / "training" / run_name
    train_path, dev_path = root / "labels/retrieval_train.json", root / "labels/retrieval_dev.json"
    if not train_path.is_file() or not dev_path.is_file():
        status = {"state": "WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS", "fine_tuned": False,
            "required_files": [str(train_path), str(dev_path)], "contest_queries_used_as_gold": False}
        atomic_json(run / "status.json", status)
        return status
    from transformers import AutoTokenizer
    from .dataset import parquet_path

    config = read_json(checkout / "configs/retrieval_full.json")
    registry = read_json(checkout / "configs/strong_model_manifest.json")
    review_model_registry(config, registry)
    budget = model_budget_report(config, registry)
    atomic_json(run / "model_parameter_budget.json", budget)
    contest = load_queries(parquet_path(root, "query.parquet"), expected_count=1200)
    train_doc, dev_doc = read_json(train_path), read_json(dev_path)
    train, dev = validate_query_split(train_doc, dev_doc, contest)
    tokenizer = AutoTokenizer.from_pretrained(config["dense"]["model_id"], revision=config["dense"]["revision"], trust_remote_code=False)
    from .full_scale_runtime import is_large_handoff, load_scale_catalog, prepare_scale_resources
    large = is_large_handoff(root,config)
    if large:
        import tempfile
        catalog,analyzer = load_scale_catalog(root,checkout,work_dir or Path(tempfile.gettempdir()) / "vmb_training",tokenizer)
        config["scale_execution"] = read_json(checkout / "configs/retrieval_scale.json")
        training_root = base_run
        base_run = base_run / "corpora" / catalog.candidate["candidate_manifest_sha256"][:16]
    else:
        catalog = load_catalog(root / "processed/stage-a-data-v3-laodong", "candidate-1cd220a4be956d5a.json", tokenizer, **config["pilot_limits"])
    try:
        snapshot = read_json(root / "raw/snapshot.json")
        if snapshot["files"]["links_corpus.parquet"]["sha256"] != catalog.build_config["official_links_sha256"]:
            raise ValueError("Training catalog differs from official corpus snapshot.")
        policy = StrongConfig(**config["retrieval"])
        if not large:
            catalog = derive_parents(catalog, tokenizer, (policy.parent_short_tokens, policy.parent_long_tokens))
        unavailable = sorted({ref["doc_id"] for query in train+dev for ref in query["relevant_chunks"]
            if ref["doc_id"] not in catalog.documents or ref["chunk_text"] not in catalog.documents[ref["doc_id"]]["source_text"]})
        if unavailable:
            status = {"state":"WAITING_FOR_LABEL_SOURCE_RECONCILIATION","fine_tuned":False,
                "unavailable_reference_documents":unavailable,"gpu_models_loaded_this_call":0,
                "next_step":"Compose the original labeled source candidate into the cumulative corpus, or review new labels against the active corpus."}
            atomic_json(base_run / "status.json",status)
            return status
        train_doc, dev_doc, review_status = import_candidate_review(base_run, train_doc, dev_doc, catalog, tokenizer, contest)
        if review_status["state"] == "CANDIDATE_REVIEW_APPLIED":
            atomic_json(train_path, train_doc)
            atomic_json(dev_path, dev_doc)
        train, dev = validate_query_split(train_doc, dev_doc, contest)
        heldout_path = root / "labels/retrieval_heldout.json"
        heldout_doc = read_json(heldout_path) if heldout_path.exists() else None
        heldout = validate_heldout_split(heldout_doc,train_doc,dev_doc,contest,catalog) if heldout_doc is not None else []
        mining_queries = [{"id": q["id"], "query": q["query"]} for q in train + dev]
        heldout_queries = [{"id":q["id"],"query":q["query"]} for q in heldout]
        queries = mining_queries + heldout_queries
        dev_queries = [{"id": q["id"], "query": q["query"]} for q in dev]
        run = base_run / "experiments" / digest_json({"queries": queries, "config": config, "catalog": catalog.identity})[:10]
        split_contract = {"train_queries": digest_json([{"id": q["id"], "query": q["query"]} for q in train]),
            "dev_queries": digest_json(dev_queries), "heldout_queries":digest_json(heldout_queries), "contest": digest_json(contest)}
        if (run / "split_contract.json").exists() and read_json(run / "split_contract.json") != split_contract:
            raise ValueError("Training split changed; choose a new training run.")
        atomic_json(run / "split_contract.json", split_contract)
        label_version = digest_json([train_doc, dev_doc])[:10]
        atomic_json(run / "labels" / (label_version + ".json"), {"train": train_doc, "dev": dev_doc})
        profile = RuntimeProfile(base_run / "runtime_profile.json")
        # The user already paid for a full mining run. Review/replay it on CPU first.
        with profile.stage("verified_existing_mining") as stage:
            existing_records = cached_mining_records(run, mining_queries, catalog, policy)
            stage["queries_reused"] = len(existing_records) if existing_records is not None else 0
        if existing_records is not None:
            train_bundle = mine_hard_negatives(existing_records, train_doc, catalog, tokenizer, contest)
            dev_bundle = mine_hard_negatives(existing_records, dev_doc, catalog, tokenizer, contest)
            if train_bundle["holds"] or dev_bundle["holds"]:
                review_path = export_candidate_review(base_run, existing_records, train_doc, dev_doc, catalog)
                status = {"state": "MINING_REQUIRES_MORE_GOLD_COVERAGE_OR_REVIEWED_NEGATIVES", "fine_tuned": False,
                    "train_holds": train_bundle["holds"], "dev_holds": dev_bundle["holds"], "review_path": str(review_path),
                    "label_review_mode": train_doc.get("review_mode", "HUMAN_REVIEW"),
                    "gpu_models_loaded_this_call": 0, "mining_reused": True,
                    "runtime_profile": str(profile.path)}
                atomic_json(base_run / "status.json", status)
                profile.finish(status["state"])
                return status
        batches = inference_batches()
        atomic_json(base_run / "execution_policy.json", batches)
        if large:
            prepared = prepare_scale_resources(root,checkout,work_dir or Path(tempfile.gettempdir()) / "vmb_training",queries,catalog=catalog)
            if prepared["state"] != "FULL_RESOURCES_READY":
                atomic_json(base_run / "status.json",prepared)
                return prepared
            index,query_vectors,qm,translations,config = (prepared[k] for k in
                ("index","vectors","query_manifest","translations","config"))
            analyzer = catalog.base.analyzer
            atomic_json(run / "index/strong.json",index.manifest)
        else:
            glossary = root / "labels/medical_aliases.json"
            analyzer = MedicalAnalyzer(segmentation=config["lexical_segmentation"], glossary_path=glossary if glossary.is_file() else None)
            query_units = [{"id": q["id"], "text": q["query"]} for q in queries]
            corpus_cache = find_embeddings(root, catalog.units, config["dense"], family="bge", preferred=[run / "corpus_embeddings"])
            query_cache = find_embeddings(root, query_units, config["dense"], family="bge", role="query", preferred=[run / "query_embeddings"])
            with profile.stage("bge_embeddings") as stage:
                stage["corpus_reused_from"] = str(corpus_cache[2]) if corpus_cache else None
                if corpus_cache is not None:
                    corpus_vectors, cm = corpus_cache[:2]
                if query_cache is not None:
                    query_vectors, qm = query_cache[:2]
                if corpus_cache is None or query_cache is None:
                    dense = TorchDenseEncoder(config["dense"])
                    try:
                        if corpus_cache is None:
                            cm = embed_units(catalog.units, dense, run / "corpus_embeddings", work_dir=work_dir, **config["embedding"])
                            corpus_vectors = embedding_matrix(run / "corpus_embeddings", cm)
                        if query_cache is None:
                            qm = embed_units(query_units, dense, run / "query_embeddings", work_dir=work_dir, **config["embedding"])
                            query_vectors = embedding_matrix(run / "query_embeddings", qm)
                    finally:
                        dense.close()
                if cm["encoder"] != qm["encoder"]:
                    raise ValueError("Training BGE corpus/query producer runtimes differ; retain original manifests and use compatible runtimes.")
            doc_units = document_units(catalog, tokenizer, config["dense"]["max_length"])
            with profile.stage("document_embeddings") as stage:
                doc_cache = find_embeddings(root, doc_units, config["dense"], family="bge", preferred=[run / "document_embeddings"])
                if doc_cache is not None:
                    document_vectors, dm = doc_cache[:2]
                    stage["reused_from"] = str(doc_cache[2])
                else:
                    document_vectors, dm, doc_units = prepare_document_vectors(catalog, tokenizer, config["dense"],
                        run / "document_embeddings", TorchDenseEncoder, embedding=config["embedding"], work_dir=work_dir)
            cached = cached_translations(queries, config["translation"], run / "translations")
            translations = cached[0] if cached is not None else None
            expansions = cached_expansions(queries, translations, config["translation"], run / "expansions", enabled=config["expansion_enabled"]) if translations is not None else None
            if expansions is None:
                with profile.stage("query_translation_expansion"):
                    translator = TorchQueryTranslator(config["translation"])
                    try:
                        if translations is None:
                            translations, _ = translate_queries(queries, translator, run / "translations")
                        expansions = expand_queries(queries, translator, run / "expansions", translations=translations, enabled=config["expansion_enabled"])
                    finally:
                        translator.close()
            aux = auxiliary_units(queries, expansions)
            corpus_cache = find_embeddings(root, catalog.units, config["second_dense"], family="qwen", preferred=[run / "qwen_corpus"])
            query_cache = cached_qwen_embeddings(run / "qwen_queries", aux, config["second_dense"], "query")
            with profile.stage("qwen_embeddings") as stage:
                stage["corpus_reused_from"] = str(corpus_cache[2]) if corpus_cache else None
                if corpus_cache is not None:
                    secondary_vectors, sm = corpus_cache[:2]
                if query_cache is not None:
                    auxiliary_vectors, am = query_cache
                if corpus_cache is None or query_cache is None:
                    encoder = TorchQwenEncoder(config["second_dense"])
                    try:
                        options = {**config["qwen_embedding"], "batch_size": batches["embedding"]}
                        if corpus_cache is None:
                            sm = embed_units(catalog.units, encoder.for_role("corpus"), run / "qwen_corpus", work_dir=work_dir, **options)
                            secondary_vectors = embedding_matrix(run / "qwen_corpus", sm)
                        if query_cache is None:
                            am = embed_units(aux, encoder.for_role("query"), run / "qwen_queries", work_dir=work_dir, **options)
                            auxiliary_vectors = embedding_matrix(run / "qwen_queries", am)
                        stage["oom_backoffs"] = encoder.oom_backoffs
                    finally:
                        encoder.close()
            stage_started = time.perf_counter()
            index = StrongIndex(catalog, corpus_vectors, cm, run / "index", tokenizer,
                secondary_vectors=secondary_vectors, secondary_manifest=sm, auxiliary_vectors=auxiliary_vectors,
                auxiliary_manifest=am, auxiliary_inputs=aux, queries=queries, expansions=expansions, analyzer=analyzer, work_dir=work_dir,
                document_vectors=document_vectors, document_manifest=dm, document_inputs=doc_units)
            profile.record("index", stage_started)
        translated = {q["id"]: t for q, t in zip(queries, translations, strict=True)}
        mining_vectors, mining_manifest = slice_query_vectors(mining_queries,queries,query_vectors,qm)
        mining_translations = [translated[q["id"]] for q in mining_queries]
        with profile.stage("hard_negative_candidates") as stage:
            records = existing_records if existing_records is not None else retrieve_mining_candidates(
                index, mining_queries, mining_vectors, mining_translations, policy, run / "retrieval_mining")
            stage["mode"] = "VERIFIED_PREVIOUS_CASCADE" if existing_records is not None else "MULTICHANNEL_RETRIEVAL_NO_RERANK"
            train_bundle = mine_hard_negatives(records, train_doc, catalog, tokenizer, contest)
            dev_bundle = mine_hard_negatives(records, dev_doc, catalog, tokenizer, contest)
            atomic_json(run / "train_mining.json", train_bundle)
            atomic_json(run / "dev_mining.json", dev_bundle)
            if train_bundle["state"] != "READY" or dev_bundle["state"] != "READY" or train_bundle["holds"] or dev_bundle["holds"]:
                review_path = export_candidate_review(base_run, records, train_doc, dev_doc, catalog)
                status = {"state": "MINING_REQUIRES_MORE_GOLD_COVERAGE_OR_REVIEWED_NEGATIVES", "fine_tuned": False,
                    "train_holds": train_bundle["holds"], "dev_holds": dev_bundle["holds"], "review_path": str(review_path),
                    "label_review_mode": train_doc.get("review_mode", "HUMAN_REVIEW"), "mining_mode": stage["mode"],
                    "runtime_profile": str(profile.path)}
                atomic_json(base_run / "status.json", status)
                profile.finish(status["state"])
                return status
        with profile.stage("qlora_training"):
            model = TorchQwenReranker(config["reranker"])
            try:
                training = train_qlora(model, train_bundle, dev_bundle, run / "checkpoints" / label_version,
                    model_budget=budget, **config["training"])
            finally:
                model.close()
        dev_vectors, dev_manifest = slice_query_vectors(dev_queries, queries, query_vectors, qm)
        dev_translations = [translated[q["id"]] for q in dev_queries]
        finalists = unique_finalists(sorted((run / "checkpoints" / label_version).glob("checkpoint-*")) + [Path(training["adapter"])])
        trials = []
        stage_started = time.perf_counter()
        for checkpoint in finalists:
            model_budget_report(config, registry, adapter_paths=[checkpoint])
            reranker = TorchQwenReranker(config["reranker"], adapter_path=checkpoint)
            try:
                records, report = predict_strong(index, dev_queries, dev_vectors, reranker,
                    run / "finalist_evaluation" / label_version / checkpoint.name / "queries", query_embedding_manifest=dev_manifest,
                    translations=dev_translations, config=policy, batch_size=batches["reranker"])
                sweep = cutoff_sweep(records, dev_doc, catalog, tokenizer, policy)
                atomic_json(run / "finalist_evaluation" / label_version / checkpoint.name / "cutoff_sweep.json", sweep)
                trials.append({"checkpoint": str(checkpoint), "best_dev_policy": sweep["best_dev_policy"],
                    "prediction_signature": report["signature"], "recall": evaluate_candidate_recall(records, dev_doc["queries"], catalog, tokenizer)})
            finally:
                reranker.close()
        profile.record("dev_finalist_evaluation", stage_started, unique_checkpoints=len(finalists), queries=len(dev_queries))
        trials.sort(key=lambda t: (-t["best_dev_policy"]["combined_f2"], t["checkpoint"]))
        selected = _selected_adapter(trials[0]["checkpoint"], base_run / "selected_adapter")
        selected_budget = model_budget_report(config, registry, adapter_paths=[base_run / "selected_adapter"])
        atomic_json(base_run / "model_parameter_budget.json", selected_budget)
        best = trials[0]["best_dev_policy"]
        from .full_plan_runtime import calibration_context
        calibration = {"state": "DEV_SELECTED_HELD_OUT_PENDING", "split": "dev", "reviewed": True,
            "catalog": catalog.identity, "adapter_manifest_sha256": selected["manifest_sha256"],
            "contest_queries_sha256": digest_json(contest), "dev_labels_sha256": digest_json(dev_doc),
            "dev_query_ids": [q["id"] for q in dev_queries], "dev_query_texts_sha256": [digest_json(q["query"]) for q in dev_queries],
            "reranker_spec": config["reranker"], "calibration_context": calibration_context(config, analyzer),
            "scorer": "PLAN_DERIVED_LOCAL_PROXY_NOT_OFFICIAL_BTC",
            "label_review_mode": dev_doc.get("review_mode", "HUMAN_REVIEW"),
            "label_evaluation_scope": dev_doc.get("evaluation_scope", "HUMAN_REVIEWED_LOCAL_PROXY"),
            "policy": {"doc_top_k": best["doc_top_k"], "chunk_top_k": best["chunk_top_k"],
                "doc_score_margin": best["doc_score_margin"], "chunk_score_margin": best["chunk_score_margin"]}, "trials": trials}
        calibration["manifest_sha256"] = digest_json(calibration)
        atomic_json(base_run / "calibrated_policy.json", calibration)
        selected_policy = replace(policy,**calibration["policy"])
        stage_started = time.perf_counter()
        selected_model = TorchQwenReranker(config["reranker"],adapter_path=base_run / "selected_adapter")
        try:
            if run_ablations and config.get("research",{}).get("dev_ablations",True):
                ablations = run_dev_ablations(index,dev_queries,dev_vectors,dev_manifest,dev_translations,
                    selected_model,selected_policy,dev_doc,contest,run / "ablations" / label_version)
                atomic_json(base_run / "ablation_summary.json", {"split":"dev","trials":ablations,
                    "used_for_final_checkpoint_selection":False,"heldout_data_used":False})
            if heldout_doc is not None:
                vectors,manifest = slice_query_vectors(heldout_queries,queries,query_vectors,qm)
                records,_ = predict_strong(index,heldout_queries,vectors,selected_model,
                    run / "heldout" / (label_version+'-'+digest_json(heldout_doc)[:10]) / "queries",
                    query_embedding_manifest=manifest,translations=[translated[q["id"]] for q in heldout_queries],
                    config=selected_policy,batch_size=batches["reranker"])
                heldout_report = evaluate_frozen_selection(records,heldout_doc,catalog,tokenizer,calibration,base_run / "heldout_evaluation.json")
            else:
                heldout_report = {"state":"WAITING_FOR_INDEPENDENT_REVIEWED_HELD_OUT_LABELS",
                    "required_file":str(heldout_path),"used_for_checkpoint_or_cutoff_selection":False}
                atomic_json(base_run / "heldout_evaluation.json",heldout_report)
        finally:
            selected_model.close()
        profile.record("selected_adapter_heldout_and_optional_ablations", stage_started,
            heldout_queries=len(heldout_queries), ablations_requested=run_ablations)
        status = {"state": "TRAINED_DEV_F2_SELECTED_HELD_OUT_PENDING", "fine_tuned": True,
            "label_review_mode": train_doc.get("review_mode", "HUMAN_REVIEW"),
            "label_evaluation_scope": dev_doc.get("evaluation_scope", "HUMAN_REVIEWED_LOCAL_PROXY"),
            "heldout_evaluation":heldout_report,
            "model_parameter_budget": selected_budget,
            "adapter": str(base_run / "selected_adapter"), "dev_proxy_f2": best["combined_f2"],
            "official_btc_score": None, "corpus_promoted": False}
        if heldout_doc is not None:
            status["state"] = "TRAINED_DEV_SELECTED_HELD_OUT_EVALUATED_LOCAL_PROXY"
        status["research_ablations"] = "EXECUTED" if run_ablations else "DEFERRED_UNTIL_FIRST_MEASURED_RESULT"
        status["runtime_profile"] = str(profile.path)
        atomic_json(base_run / "status.json", status)
        profile.finish(status["state"])
        if large:
            pointer = {"directory":base_run.relative_to(training_root).as_posix(),
                "calibration_sha256":calibration["manifest_sha256"],"adapter_manifest_sha256":selected["manifest_sha256"]}
            pointer["manifest_sha256"] = digest_json(pointer)
            atomic_json(training_root / "active_adapter.json",pointer)
        return status
    finally:
        if large:
            catalog.close()
