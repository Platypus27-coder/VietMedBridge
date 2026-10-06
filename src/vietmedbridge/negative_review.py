"""Review retrieved candidates explicitly; unknown passages never become negatives."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from .artifacts import atomic_json, digest_json, read_json
from .reranker_training import CATEGORIES, validate_query_split
from .retrieval_policy import lcs_length, score_tokens
from .training_data import _check, _seal


def install_ai_candidate_review(upload_path, data_root, *, run_name="stage-a-qlora-v3-per-model-15b"):
    """Import a delegated AI review, preserving immutable candidates and team edits."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("Unsafe candidate-review run name.")
    base = Path(data_root) / "training" / run_name
    pointer = read_json(base / "candidate_review_current.json")
    directory = (base / pointer["directory"]).resolve()
    if not directory.is_relative_to(base.resolve()):
        raise ValueError("Candidate review pointer escapes its run.")
    target = directory / "review.json"
    existing, incoming = read_json(target), read_json(upload_path)
    sources = _check(read_json(directory / "sources.json"))
    if (incoming.get("review_mode") != "AI_ASSISTED_PILOT"
        or incoming.get("review_authorization") != "USER_DELEGATED_TO_CODEX_2026_10_06"
        or incoming.get("human_validated") is not False):
        raise ValueError("Expected a delegated AI pilot candidate review.")
    if (incoming.get("source_sha256") != sources["sha256"]
        or existing.get("source_sha256") != sources["sha256"]
        or pointer["source_sha256"] != sources["sha256"]):
        raise ValueError("Uploaded candidate review belongs to another source snapshot.")
    expected = {i["pair_id"]: i for i in sources["items"]}
    before = {i["pair_id"]: i for i in existing["items"]}
    if (len(incoming["items"]) != len(expected) or len(before) != len(expected)
        or {i["pair_id"] for i in incoming["items"]} != set(expected) or set(before) != set(expected)):
        raise ValueError("Uploaded candidate review coverage changed.")
    merged, counts = deepcopy(incoming), {}
    for item in merged["items"]:
        source, old = expected[item["pair_id"]], before[item["pair_id"]]
        if ({k: item.get(k) for k in source} != source
            or {k: old.get(k) for k in source} != source):
            raise ValueError("Uploaded review changed immutable candidate source fields.")
        decision = item["review"]
        judgment = decision.get("judgment")
        if judgment not in ("PENDING", "POSITIVE", "NEGATIVE", "SKIP"):
            raise ValueError("Unknown uploaded candidate judgment.")
        if judgment != "PENDING" and (decision.get("reviewer_type") != "AI" or not decision.get("reviewer", "").strip()):
            raise ValueError("AI candidate judgments require explicit reviewer provenance.")
        if old["review"].get("judgment") != "PENDING":
            if judgment != "PENDING" and old["review"] != decision:
                raise ValueError("Uploaded review conflicts with existing team decisions.")
            item["review"] = old["review"]
        counts[item["review"]["judgment"]] = counts.get(item["review"]["judgment"], 0) + 1
    backup = target.with_name("review.before-ai-" + digest_json(existing)[:12] + ".json")
    if not backup.exists():
        atomic_json(backup, existing)
    atomic_json(target, merged)
    return {"review_path": str(target), "backup_path": str(backup), "judgments": counts,
            "review_mode": "AI_ASSISTED_PILOT", "human_validated": False}


def export_candidate_review(run_dir, records, train_doc, dev_doc, catalog, *, top_k=100):
    labels = [train_doc, dev_doc]
    by_id = {r["prediction"]["id"]: r for r in records}
    items = []
    for document in labels:
        for query in document["queries"]:
            record = by_id[query["id"]]
            if record["query_sha256"] != digest_json({"id": query["id"], "query": query["query"]}):
                raise ValueError("Candidate review query checkpoint mismatch.")
            for rank, row in enumerate(record["ranking"]["children"][:top_k], 1):
                child = catalog.children[row["child_id"]]
                items.append({"pair_id": digest_json([query["id"], child["chunk_id"]])[:24],
                    "query_id": query["id"], "query": query["query"], "split": document["split"],
                    "child_id": child["chunk_id"], "doc_id": child["doc_id"], "text": child["text"],
                    "source_text_sha256": child["source_text_sha256"], "start_char": child["start_char"],
                    "end_char": child["end_char"], "rank": rank})
    if len({i["pair_id"] for i in items}) != len(items):
        raise ValueError("Duplicate candidate review pairs.")
    sources = _seal({"catalog": catalog.identity, "labels": labels, "items": items})
    root = Path(run_dir) / "candidate_reviews" / sources["sha256"][:10]
    source_path, review_path = root / "sources.json", root / "review.json"
    if source_path.exists() and read_json(source_path) != sources:
        raise ValueError("Candidate review sources changed.")
    atomic_json(source_path, sources)
    if not review_path.exists():
        atomic_json(review_path, {"source_sha256": sources["sha256"], "items": [i | {"review": {
            "judgment": "PENDING", "reviewer": "", "category": "", "notes": ""}} for i in items],
            "instructions": "Edit only review. POSITIVE/NEGATIVE/SKIP require reviewer name. NEGATIVE may use a reviewed category or unclassified_retrieved. Unknown passages remain PENDING; do not guess negatives."})
    atomic_json(Path(run_dir) / "candidate_review_current.json", {
        "directory": str(root.relative_to(Path(run_dir))), "source_sha256": sources["sha256"]})
    return review_path


def import_candidate_review(run_dir, train_doc, dev_doc, catalog, tokenizer, contest_queries):
    """Replay edited judgments against immutable source/label snapshots."""
    pointer = Path(run_dir) / "candidate_review_current.json"
    if not pointer.exists():
        return train_doc, dev_doc, {"state": "NO_CANDIDATE_REVIEW"}
    current = read_json(pointer)
    root = (Path(run_dir) / current["directory"]).resolve()
    if not root.is_relative_to(Path(run_dir).resolve()):
        raise ValueError("Unsafe candidate review path.")
    sources, review = _check(read_json(root / "sources.json")), read_json(root / "review.json")
    if sources["catalog"] != catalog.identity or sources["sha256"] != current["source_sha256"] or review["source_sha256"] != sources["sha256"]:
        raise ValueError("Candidate review source/catalog mismatch.")
    original_hashes = [digest_json(d) for d in sources["labels"]]
    live_hashes = [digest_json(d) for d in (train_doc, dev_doc)]
    previous = _check(read_json(root / "published.json")) if (root / "published.json").exists() else None
    known = [original_hashes] + (previous["known_label_versions"] if previous else [])
    if any(live_hashes[i] not in {version[i] for version in known} for i in range(2)):
        return train_doc, dev_doc, {"state": "STALE_CANDIDATE_REVIEW"}
    expected = {i["pair_id"]: i for i in sources["items"]}
    if len(review["items"]) != len(expected) or {i["pair_id"] for i in review["items"]} != set(expected):
        raise ValueError("Candidate review coverage changed.")
    documents = deepcopy(sources["labels"])
    by_id = {q["id"]: (d, q) for d in documents for q in d["queries"]}
    applied, audit = 0, []
    for item in review["items"]:
        source = expected[item["pair_id"]]
        if {k: item.get(k) for k in source} != source:
            raise ValueError("Candidate review modified an immutable source field.")
        child = catalog.children[item["child_id"]]
        if any(child[k] != item[k] for k in ("doc_id", "text", "source_text_sha256", "start_char", "end_char")):
            raise ValueError("Candidate review source differs from frozen catalog.")
        decision = item["review"]
        judgment = decision.get("judgment")
        if judgment == "PENDING":
            continue
        if judgment not in ("POSITIVE", "NEGATIVE", "SKIP") or not isinstance(decision.get("reviewer"), str) or not decision["reviewer"].strip():
            raise ValueError("Explicit candidate judgment and reviewer are required.")
        audit.append({"pair_id": item["pair_id"], "review": decision})
        document, query = by_id[item["query_id"]]
        if judgment == "SKIP":
            continue
        if judgment == "POSITIVE":
            source_split = document.get("source_splits", {}).get(str(child["doc_id"]))
            if source_split is not None and source_split != document["split"]:
                raise ValueError("Additional positive would cross reserved train/dev source groups.")
            ref = {"doc_id": child["doc_id"], "chunk_text": child["text"]}
            if ref not in query["relevant_chunks"]:
                query["relevant_chunks"].append(ref)
            if child["doc_id"] not in query["relevant_docs"]:
                query["relevant_docs"].append(child["doc_id"])
        else:
            category = decision.get("category") or "unclassified_retrieved"
            if category not in (*CATEGORIES, "unclassified_retrieved"):
                raise ValueError("Unknown reviewed negative category.")
            tokens = score_tokens(tokenizer, child["text"])
            if tokens and any(ref["doc_id"] == child["doc_id"] and lcs_length(tokens, score_tokens(tokenizer, ref["chunk_text"])) >= .8 * len(tokens) for ref in query["relevant_chunks"]):
                raise ValueError("Reviewed negative conflicts with a positive source reference.")
            if child["chunk_id"] not in query.setdefault("negative_child_ids", []):
                query["negative_child_ids"].append(child["chunk_id"])
            query.setdefault("negative_categories", {})[child["chunk_id"]] = category
        applied += 1
    if not audit and previous is None:
        return train_doc, dev_doc, {"state": "WAITING_FOR_CANDIDATE_REVIEW", "review_path": str(root / "review.json")}
    for document in documents:
        for query in document["queries"]:
            for child_id in query.get("negative_child_ids", []):
                child = catalog.children[child_id]
                tokens = score_tokens(tokenizer, child["text"])
                if tokens and any(ref["doc_id"] == child["doc_id"] and lcs_length(tokens, score_tokens(tokenizer, ref["chunk_text"])) >= .8 * len(tokens) for ref in query["relevant_chunks"]):
                    raise ValueError("Reviewed negative conflicts with a positive source reference.")
    review_identity = {"source_sha256": sources["sha256"], "decisions_sha256": digest_json(audit)}
    for document in documents:
        document["candidate_review"] = review_identity
    validate_query_split(*documents, contest_queries)
    result_hashes = [digest_json(d) for d in documents]
    atomic_json(root / "published.json", _seal({**review_identity, "labels_sha256": result_hashes,
        "known_label_versions": known + ([result_hashes] if result_hashes not in known else []), "audit": audit}))
    return *documents, {"state": "CANDIDATE_REVIEW_APPLIED", "judgments": applied, "review_path": str(root / "review.json")}
