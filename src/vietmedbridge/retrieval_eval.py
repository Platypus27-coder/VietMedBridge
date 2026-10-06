"""Local F2/LCS evaluator derived from the plan, NOT an official BTC scorer."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

from .artifacts import digest_json
from .retrieval_policy import NORMALIZATION_VERSION, lcs_length, lcs_union, score_tokens, select_parents


def f2(precision, recall):
    return 5 * precision * recall / (4 * precision + recall) if precision or recall else 0.0


def _metrics(correct, predicted, covered, references):
    precision = correct / predicted if predicted else 0.0
    recall = covered / references
    return {"precision": precision, "recall": recall, "f2": f2(precision, recall)}


def evaluate_predictions(predictions, labels, tokenizer, *, relevance_threshold=.4, dedup_threshold=.8):
    if not 0 < relevance_threshold <= 1 or not 0 < dedup_threshold <= 1:
        raise ValueError("Invalid LCS thresholds.")
    if len({p["id"] for p in predictions}) != len(predictions) or len({r["id"] for r in labels}) != len(labels):
        raise ValueError("Duplicate query ID in predictions/reference labels.")
    by_id = {p["id"]:p for p in predictions}
    rows, docs, chunks = [], [], []
    for label in labels:
        if type(label.get("id")) is not int:
            raise ValueError("Reference query IDs must be integers.")
        if label.get("relevant_docs") is not None and (not isinstance(label["relevant_docs"], list) or any(type(d) is not int for d in label["relevant_docs"])):
            raise ValueError("Reference document IDs must be integers in a list.")
        if label.get("relevant_chunks") is not None and (not isinstance(label["relevant_chunks"], list) or any(not isinstance(c, dict) or type(c.get("doc_id")) is not int or not isinstance(c.get("chunk_text"), str) or not c["chunk_text"].strip() for c in label["relevant_chunks"])):
            raise ValueError("Reference chunks need integer doc IDs and nonempty text.")
        if label["id"] not in by_id:
            raise ValueError("Reference query is absent from predictions.")
        prediction = by_id[label["id"]]
        row = {"id": label["id"]}
        reference_docs = label.get("relevant_docs")
        if reference_docs:
            expected, actual = set(reference_docs), set(prediction["relevant_docs"])
            correct = len(expected & actual)
            row["documents"] = _metrics(correct, len(actual), correct, len(expected))
            docs.append(row["documents"])
        reference_chunks = label.get("relevant_chunks")
        if reference_chunks:
            expected = [(r["doc_id"], score_tokens(tokenizer, r["chunk_text"])) for r in reference_chunks]
            if any(not tokens for _,tokens in expected):
                raise ValueError("Reference chunks must contain scoring tokens.")
            actual = []
            for chunk in prediction["relevant_chunks"]:
                tokens = score_tokens(tokenizer, chunk["chunk_text"])
                if not any(doc == chunk["doc_id"] and lcs_union(tokens, other) >= dedup_threshold for doc,other in actual):
                    actual.append((chunk["doc_id"], tokens))
            matches = [[doc == target and lcs_length(tokens, reference) >= relevance_threshold * len(reference)
                        for target,reference in expected] for doc,tokens in actual]
            correct = sum(any(match) for match in matches)
            covered = sum(any(match[idx] for match in matches) for idx in range(len(expected)))
            row["chunks"] = _metrics(correct, len(actual), covered, len(expected))
            chunks.append(row["chunks"])
        rows.append(row)
    macro = lambda values: ({key: sum(r[key] for r in values) / len(values) for key in ("precision", "recall", "f2")} if values else None)
    document_macro, chunk_macro = macro(docs), macro(chunks)
    return {"scorer": "PLAN_DERIVED_LOCAL_PROXY_NOT_OFFICIAL_BTC", "normalization": NORMALIZATION_VERSION,
            "relevance_lcs_fraction": relevance_threshold, "dedup_lcs_union": dedup_threshold,
            "labels_sha256": digest_json(labels), "predictions_sha256": digest_json(predictions),
            "document_queries": len(docs), "chunk_queries": len(chunks),
            "documents_macro": document_macro, "chunks_macro": chunk_macro,
            "combined_f2": (document_macro["f2"] + chunk_macro["f2"]) / 2 if docs and chunks else None,
            "per_query": rows}


def cutoff_sweep(records, labels_document, catalog, tokenizer, config):
    """Reuse scored candidates; select a policy ONLY on explicitly reviewed dev labels."""
    if labels_document.get("split") != "dev" or labels_document.get("reviewed") is not True:
        raise ValueError("Calibration requires reviewed dev labels; never tune on submission/test queries.")
    labels = labels_document.get("queries", [])
    if not labels:
        raise ValueError("Empty dev reference set.")
    if any(not isinstance(r.get("relevant_docs"), list) or not isinstance(r.get("relevant_chunks"), list) for r in labels):
        raise ValueError("Dev labels need explicit document and chunk reference lists.")
    scores = []
    for doc_k, chunk_k, margin in product((3, 5, 10), (2, 4, 8), (None, 1.0, 2.0, 4.0)):
        if doc_k > config.detail_doc_k:
            continue
        policy = replace(config, doc_min_k=min(config.doc_min_k, doc_k), doc_top_k=doc_k,
                         chunk_min_k=min(config.chunk_min_k, chunk_k), chunk_top_k=chunk_k,
                         doc_score_margin=margin, chunk_score_margin=margin)
        predictions = [select_parents(catalog, record["ranking"], policy, tokenizer, record["prediction"]["id"])[0] for record in records]
        report = evaluate_predictions(predictions, labels, tokenizer)
        if report["combined_f2"] is None:
            raise ValueError("Need positive document AND chunk references for combined F2.")
        scores.append({"doc_top_k": doc_k, "chunk_top_k": chunk_k, "score_margin": margin,
                       "combined_f2": report["combined_f2"],
                       "documents_macro": report["documents_macro"], "chunks_macro": report["chunks_macro"]})
    scores.sort(key=lambda r: (-r["combined_f2"], r["doc_top_k"] + r["chunk_top_k"], str(r["score_margin"])))
    return {"scorer": "PLAN_DERIVED_LOCAL_PROXY_NOT_OFFICIAL_BTC", "split": "dev",
            "labels_sha256": digest_json(labels_document), "trials": scores,
            "best_dev_policy": scores[0], "state": "DEV_CALIBRATED_REQUIRES_HELD_OUT_TEST",
            "auto_applied_to_submission": False}


def evaluate_candidate_recall(records, labels, catalog, tokenizer):
    """Separate frozen ID availability, fused doc recall and child→parent pool recall."""
    by_id = {r["prediction"]["id"]:r for r in records}
    rows, source_recalls, doc_recalls, chunk_recalls = [], [], [], []
    for label in labels:
        if label["id"] not in by_id:
            raise ValueError("Reference query is absent from scored candidates.")
        ranking = by_id[label["id"]]["ranking"]
        row = {"id": label["id"]}
        expected_docs = set(label.get("relevant_docs") or [])
        if expected_docs:
            pool = {r["doc_id"] for r in ranking["documents"]}
            row["frozen_document_id_coverage"] = len(expected_docs & set(catalog.documents)) / len(expected_docs)
            row["fused_document_pool_recall"] = len(expected_docs & pool) / len(expected_docs)
            source_recalls.append(row["frozen_document_id_coverage"])
            doc_recalls.append(row["fused_document_pool_recall"])
        expected_chunks = label.get("relevant_chunks") or []
        if expected_chunks:
            parent_ids = {catalog.children[r["child_id"]]["parent_id"] for r in ranking["children"]}
            parents = [(catalog.parents[p]["doc_id"], score_tokens(tokenizer, catalog.parents[p]["text"])) for p in parent_ids]
            covered = 0
            for reference in expected_chunks:
                tokens = score_tokens(tokenizer, reference["chunk_text"])
                if not tokens:
                    raise ValueError("Reference chunks must contain scoring tokens.")
                covered += any(doc == reference["doc_id"] and lcs_length(candidate, tokens) >= .4 * len(tokens)
                               for doc,candidate in parents)
            row["child_parent_pool_recall"] = covered / len(expected_chunks)
            chunk_recalls.append(row["child_parent_pool_recall"])
        rows.append(row)
    average = lambda values: sum(values) / len(values) if values else None
    return {"scope": "PLAN_PROXY_CANDIDATE_RECALL_BEFORE_OUTPUT_CUTOFFS",
            "frozen_document_id_coverage_macro": average(source_recalls),
            "fused_document_pool_recall_macro": average(doc_recalls),
            "child_parent_pool_recall_macro": average(chunk_recalls), "per_query": rows}
