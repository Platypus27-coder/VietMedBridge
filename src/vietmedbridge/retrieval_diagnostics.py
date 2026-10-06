"""CPU replay, language diagnostics and dev-only controlled ablations."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, digest_json
from .retrieval_eval import cutoff_sweep, evaluate_candidate_recall, evaluate_predictions
from .reranker_training import validate_training_labels
from .strong_retrieval import predict_strong


def diagnose(records, queries, index, output_path, *, labels=None):
    by_query = {q["id"]: q for q in queries}
    languages, branches, chunks_per_query, docs_per_query, scores = Counter(), Counter(), [], [], []
    for record in records:
        if record["query_sha256"] != digest_json(by_query[record["prediction"]["id"]]):
            raise ValueError("Diagnostic query/text mismatch.")
        prediction = record["prediction"]
        docs_per_query.append(len(prediction["relevant_docs"]))
        chunks_per_query.append(len(prediction["relevant_chunks"]))
        languages.update(index.languages[d] for d in prediction["relevant_docs"])
        for row in record["ranking"]["documents"]:
            branches.update(row["branches"].keys())
        scores.extend(r["reranker_score"] for r in record["ranking"]["children"])
    report = {"query_count": len(records), "eligible_documents": len(index.doc_ids), "held_documents": index.exclusions,
        "source_languages": dict(Counter(index.languages[d] for d in index.doc_ids)),
        "selected_document_languages": dict(languages), "candidate_branch_occurrences": dict(branches),
        "document_count_histogram": dict(Counter(docs_per_query)), "chunk_count_histogram": dict(Counter(chunks_per_query)),
        "reranker_logit_percentiles": np.percentile(scores, [0, 25, 50, 75, 100]).tolist() if scores else None,
        "scores_are_probabilities": False, "quality_state": "NOT_EVALUATED_NO_REFERENCE_LABELS"}
    if labels is not None and labels.get("reviewed") is True:
        report["evaluation"] = evaluate_predictions([r["prediction"] for r in records], labels["queries"], index.tokenizer)
        report["candidate_recall"] = evaluate_candidate_recall(records, labels["queries"], index.catalog, index.tokenizer)
        per_language = defaultdict(list)
        for q in labels["queries"]:
            for language in {index.languages.get(d, "outside_pilot") for d in q["relevant_docs"]}:
                per_language[language].append(q)
        report["per_reference_language"] = {lang: evaluate_predictions([r["prediction"] for r in records], refs, index.tokenizer)
                                             for lang, refs in per_language.items()}
        report["quality_state"] = "LOCAL_PLAN_PROXY_NOT_OFFICIAL_BTC"
    atomic_json(output_path, report)
    return report


def run_dev_ablations(index, queries, vectors, manifest, translations, reranker, config,
                     labels_document, contest_queries, output_dir):
    validate_training_labels(labels_document, contest_queries, split="dev")
    if {q["id"]: q["query"] for q in queries} != {q["id"]: q["query"] for q in labels_document["queries"]}:
        raise ValueError("Ablations must use the independent labeled dev query set.")
    variants = {"full": config, "without_second_dense": replace(config, second_dense_weight=0),
        "without_document_dense":replace(config,document_dense_weight=0),
        "without_translated_sparse": replace(config, en_weight=0, zh_weight=0),
        "without_hyde": replace(config, hyde_weight=0), "without_subqueries": replace(config, subquery_weight=0),
        "parent512": replace(config, adaptive_parents=False), "rrf_rerank_fusion": replace(config, rerank_fusion="rrf")}
    results = []
    for name, policy in variants.items():
        root = Path(output_dir) / name
        records, report = predict_strong(index, queries, vectors, reranker, root / "queries",
            query_embedding_manifest=manifest, translations=translations, config=policy)
        evaluation = diagnose(records, queries, index, root / "diagnostics.json", labels=labels_document)
        sweep = cutoff_sweep(records, labels_document, index.catalog, index.tokenizer, policy)
        atomic_json(root / "cutoff_sweep.json", sweep)
        results.append({"variant": name, "prediction_signature": report["signature"],
            "combined_f2": evaluation["evaluation"]["combined_f2"], "best_dev_policy": sweep["best_dev_policy"]})
    atomic_json(Path(output_dir) / "ablation_summary.json", {"split": "dev", "trials": results,
        "state": "REQUIRES_HELD_OUT_VALIDATION_BEFORE_PROMOTION"})
    return results
