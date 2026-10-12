"""Bounded, label-aware diagnostics over saved rankings (master plan §§10/11/34/57).

Branch metrics are conditional on the saved fused pool: older checkpoints did
not retain every hit discarded by RRF. Never report those as full-channel recall.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

from .artifacts import digest_json
from .query_expansion import is_complex
from .retrieval_eval import evaluate_predictions
from .retrieval_policy import lcs_length, score_tokens


def verify_records(records, queries):
    by_id = {q["id"]: q for q in queries}
    if len(by_id) != len(queries) or len(records) != len(queries):
        raise ValueError("Audit requires exact, unique query coverage.")
    seen, signatures = set(), set()
    for record in records:
        identifier = record["prediction"]["id"]
        if identifier not in by_id or identifier in seen:
            raise ValueError("Audit query IDs differ or are duplicated.")
        if (record.get("record_sha256") != digest_json({k: v for k, v in record.items() if k != "record_sha256"})
            or record.get("query_sha256") != digest_json(by_id[identifier])):
            raise ValueError("Audit checkpoint/query integrity mismatch.")
        seen.add(identifier)
        signatures.add(record["signature"])
        for kind, key in (("documents", "doc_id"), ("children", "child_id")):
            rows = record["ranking"][kind]
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError("Duplicate candidate in scored ranking.")
            if any(not math.isfinite(r["reranker_score"]) for r in rows):
                raise ValueError("Nonfinite candidate score.")
    if len(signatures) != 1:
        raise ValueError("Mixed prediction signatures in audit.")
    return by_id


def _percentiles(values):
    if not values:
        return None
    values = sorted(values)
    def at(fraction):
        position = (len(values) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"count": len(values), "p25": at(.25), "median": at(.5), "p75": at(.75), "p90": at(.9)}


def audit_rankings(records, queries, catalog, tokenizer, *, labels_document=None, cutoffs=(10, 20, 50, 100)):
    """No corpus scan, no model calls, and no inferred gold when labels are absent."""
    by_query = verify_records(records, queries)
    if not cutoffs or any(type(k) is not int or k < 1 for k in cutoffs):
        raise ValueError("Recall cutoffs must be positive integers.")
    report = {"schema_version": 1, "catalog": catalog.identity,
        "prediction_signature": records[0]["signature"], "queries_sha256": digest_json(queries),
        "state": "NOT_EVALUATED_NO_REFERENCE_LABELS", "official_score": None,
        "branch_scope": "OBSERVED_SAVED_FUSED_POOL_ONLY_NOT_FULL_CHANNEL_RECALL",
        "query_count": len(records), "complex_queries": sum(is_complex(q["query"]) for q in queries),
        "document_count_histogram": dict(Counter(len(r["prediction"]["relevant_docs"]) for r in records)),
        "chunk_count_histogram": dict(Counter(len(r["prediction"]["relevant_chunks"]) for r in records))}
    if labels_document is None:
        return report
    if labels_document.get("reviewed") is not True:
        raise ValueError("Diagnostics require explicitly reviewed labels.")
    labels = labels_document["queries"]
    if not labels or len({q["id"] for q in labels}) != len(labels):
        raise ValueError("Empty/duplicate diagnostic labels.")
    if any(q["id"] not in by_query or q["query"] != by_query[q["id"]]["query"] for q in labels):
        raise ValueError("Diagnostic labels belong to different queries.")
    predictions = [r["prediction"] for r in records]
    report["evaluation"] = evaluate_predictions(predictions, labels, tokenizer)
    by_id = {r["prediction"]["id"]: r for r in records}
    recalls, unique, patterns, errors = defaultdict(list), Counter(), Counter(), Counter()
    lengths, languages, complexities, details = defaultdict(list), defaultdict(list), defaultdict(list), []
    all_branches = sorted({name for r in records for d in r["ranking"]["documents"] for name in d.get("branches", {})})
    for label in labels:
        record = by_id[label["id"]]
        docs, children = record["ranking"]["documents"], record["ranking"]["children"]
        prediction = record["prediction"]
        gold = set(label.get("relevant_docs") or [])
        pool = {d["doc_id"] for d in docs}
        selected = set(prediction["relevant_docs"])
        branches = {name: {d["doc_id"]: d["branches"][name]["rank"] for d in docs if name in d.get("branches", {})}
                    for name in all_branches}
        row = {"id": label["id"], "document_errors": [], "chunk_errors": []}
        if gold:
            stages = {"fused": [d["doc_id"] for d in sorted(docs, key=lambda d: (-d["retrieval_score"], d["doc_id"]))],
                      "reranked": [d["doc_id"] for d in docs]}
            for stage, order in stages.items():
                for k in cutoffs:
                    recalls[f"{stage}@{k}"].append(len(set(order[:k]) & gold) / len(gold))
            for name, ranks in branches.items():
                for k in cutoffs:
                    recalls[f"observed_branch:{name}@{k}"].append(sum(ranks.get(d, math.inf) <= k for d in gold) / len(gold))
            for doc_id in sorted(gold):
                hit = sorted(name for name, ranks in branches.items() if doc_id in ranks)
                patterns["+".join(hit) or "no_observed_branch"] += 1
                if len(hit) == 1:
                    unique[hit[0]] += 1
                if doc_id not in catalog.documents:
                    reason = "SOURCE_UNAVAILABLE_IN_FROZEN_CANDIDATE"
                elif doc_id not in pool:
                    reason = "DOCUMENT_NOT_IN_SAVED_CANDIDATE_POOL"
                elif doc_id not in selected:
                    reason = "DOCUMENT_DROPPED_AFTER_RANKING_OR_CUTOFF"
                else:
                    continue
                errors[reason] += 1
                row["document_errors"].append({"doc_id": doc_id, "reason": reason})
        # Tokenize candidate parents once per query, not once per gold reference.
        parents = {}
        child_texts = {}
        for child_row in children:
            child = catalog.children[child_row["child_id"]]
            source = catalog.documents.get(child["doc_id"])
            if (source is None or child.get("source_text_sha256") != source.get("source_text_sha256")
                or child["text"] != source["source_text"][child["start_char"]:child["end_char"]]):
                raise ValueError("Scored child is not bound to the frozen source slice.")
            child_texts[child_row["child_id"]] = (child["doc_id"], score_tokens(tokenizer, child["text"]))
            pid = child_row.get("parent_id", child["parent_id"])
            if pid not in parents:
                parent = catalog.parents[pid]
                parents[pid] = (parent["doc_id"], score_tokens(tokenizer, parent["text"]))
        final = [(c["doc_id"], score_tokens(tokenizer, c["chunk_text"])) for c in prediction["relevant_chunks"]]
        reference_languages = set()
        for index, ref in enumerate(label.get("relevant_chunks") or []):
            tokens = score_tokens(tokenizer, ref["chunk_text"])
            if not tokens:
                raise ValueError("Empty scoring tokens in gold reference.")
            source = catalog.documents.get(ref["doc_id"])
            if source is not None and ref["chunk_text"] not in source["source_text"]:
                raise ValueError("Reviewed gold is not an exact slice of its frozen source.")
            language = source.get("language", "unknown") if source else "outside_candidate"
            reference_languages.add(language)
            lengths[language].append(len(tokens))
            lengths["all"].append(len(tokens))
            def covered(items):
                return any(d == ref["doc_id"] and lcs_length(t, tokens) >= .4 * len(tokens) for d, t in items)
            if covered(final):
                continue
            if source is None:
                reason = "SOURCE_UNAVAILABLE_IN_FROZEN_CANDIDATE"
            elif ref["doc_id"] not in pool:
                reason = "DOCUMENT_NOT_IN_SAVED_CANDIDATE_POOL"
            elif ref["doc_id"] not in selected:
                reason = "DOCUMENT_DROPPED_AFTER_RANKING_OR_CUTOFF"
            elif not covered(child_texts.values()):
                reason = "CHILD_CANDIDATE_POOL_MISS"
            elif not covered(parents.values()):
                reason = "PARENT_WINDOW_COVERAGE_MISS"
            else:
                reason = "OUTPUT_SELECTION_CUTOFF_QUOTA_OR_DEDUP"
            errors["chunk:" + reason] += 1
            row["chunk_errors"].append({"reference_index": index, "doc_id": ref["doc_id"], "reason": reason})
        for d in gold:
            source = catalog.documents.get(d)
            reference_languages.add(source.get("language", "unknown") if source else "outside_candidate")
        for language in reference_languages:
            languages[language].append(label)
        complexities["complex" if is_complex(label["query"]) else "simple"].append(label)
        details.append(row)
    report.update(state="LOCAL_PLAN_PROXY_NOT_OFFICIAL_BTC", labels_sha256=digest_json(labels_document),
        label_review_mode=labels_document.get("review_mode", "UNSPECIFIED"),
        document_recall_at_k={k: sum(v) / len(v) for k, v in sorted(recalls.items())},
        observed_unique_gold_documents=dict(unique), observed_gold_branch_patterns=dict(patterns),
        error_counts=dict(errors), per_query=details,
        reference_token_lengths={k: _percentiles(v) for k, v in lengths.items()},
        per_reference_language={k: evaluate_predictions(predictions, v, tokenizer) for k, v in languages.items()},
        per_query_complexity={k: evaluate_predictions(predictions, v, tokenizer) for k, v in complexities.items()})
    return report


def compare_evaluations(baseline, challenger, *, max_language_drop=0.0):
    """Same-label local regression gate; never a substitute for official scoring."""
    if not math.isfinite(max_language_drop) or max_language_drop < 0:
        raise ValueError("Invalid regression tolerance.")
    left, right = baseline["evaluation"], challenger["evaluation"]
    keys = ("labels_sha256", "normalization", "relevance_lcs_fraction", "dedup_lcs_union", "scorer")
    if any(left[k] != right[k] for k in keys):
        raise ValueError("Regression comparison requires identical labels and scorer.")
    if left["combined_f2"] is None or right["combined_f2"] is None:
        raise ValueError("Both document and chunk references are required.")
    regressions, deltas = [], {}
    langs = baseline["per_reference_language"]
    if set(langs) != set(challenger["per_reference_language"]):
        raise ValueError("Language buckets changed between reports.")
    for language, before in langs.items():
        after = challenger["per_reference_language"][language]
        for metric in ("documents_macro", "chunks_macro"):
            if before[metric] is not None and after[metric] is not None:
                delta = after[metric]["f2"] - before[metric]["f2"]
                deltas[f"{language}:{metric}"] = delta
                if delta < -max_language_drop:
                    regressions.append(f"{language}:{metric}")
    before = {r["id"]: r for r in left["per_query"]}
    query_regressions = sorted(r["id"] for r in right["per_query"] if any(
        k in r and k in before[r["id"]] and r[k]["f2"] < before[r["id"]][k]["f2"] for k in ("documents", "chunks")))
    delta = right["combined_f2"] - left["combined_f2"]
    return {"state": "LOCAL_REGRESSION_GATE_PASSED" if delta >= 0 and not regressions else "LOCAL_REGRESSION_GATE_FAILED",
        "combined_f2_delta": delta, "language_deltas": deltas, "language_regressions": regressions,
        "query_regressions": query_regressions, "auto_promoted": False, "official_score": None}
