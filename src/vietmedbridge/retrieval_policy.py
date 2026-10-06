"""Shared source-preserving selection and plan-derived token comparisons."""
from __future__ import annotations

import html
import math
import re
import unicodedata
from collections import Counter


NORMALIZATION_VERSION = "plan-nfkc-html-case-punctuation-space-v1"


def scoring_text(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                                        "–": "-", "—": "-", "−": "-"}))
    return re.sub(r"\s+", " ", text).strip()


def score_tokens(tokenizer, text):
    return tokenizer(scoring_text(text), add_special_tokens=False, truncation=False,
                     verbose=False)["input_ids"]


def lcs_length(left, right):
    """Exact bit-parallel LCS, including token order and repeated tokens."""
    masks = {}
    for position, token in enumerate(right):
        masks[token] = masks.get(token, 0) | (1 << position)
    state = 0
    for token in left:
        union = state | masks.get(token, 0)
        state = union & ~(union - ((state << 1) | 1))
    return state.bit_count()


def lcs_union(left, right):
    common = lcs_length(left, right)
    union = len(left) + len(right) - common
    return common / union if union else 1.0


def cutoff(rows, *, minimum, maximum, margin=None, floor=None, score_key="reranker_score"):
    """Keep the minimum, then apply optional calibrated raw-score conditions."""
    if type(minimum) is not int or type(maximum) is not int or not 0 <= minimum <= maximum:
        raise ValueError("Invalid minimum/maximum cutoff.")
    if margin is not None and (not math.isfinite(margin) or margin < 0):
        raise ValueError("Score margin must be finite and nonnegative.")
    if floor is not None and not math.isfinite(floor):
        raise ValueError("Score floor must be finite.")
    if not rows:
        return []
    best = max(row[score_key] for row in rows)
    result = []
    for row in rows:
        score = row[score_key]
        if len(result) < minimum or ((floor is None or score >= floor)
                                    and (margin is None or score >= best - margin)):
            result.append(row)
        if len(result) == maximum:
            break
    return result


def select_parents(catalog, ranking, config, tokenizer, query_id):
    documents = cutoff(ranking["documents"], minimum=config.doc_min_k, maximum=config.doc_top_k,
                       margin=config.doc_score_margin, floor=config.doc_score_floor)
    doc_ids = [row["doc_id"] for row in documents]
    allowed = set(doc_ids)
    candidates, tokens, seen, per_doc = [], [], set(), Counter()
    for row in ranking["children"]:
        child = catalog.children[row["child_id"]]
        parent = catalog.parents[child["parent_id"]]
        doc_id = child["doc_id"]
        if doc_id not in allowed or parent["chunk_id"] in seen or per_doc[doc_id] >= config.max_chunks_per_doc:
            continue
        parent_tokens = score_tokens(tokenizer, parent["text"])
        if any(old_doc == doc_id and lcs_union(old_tokens, parent_tokens) >= config.dedup_threshold
               for old_doc, old_tokens in tokens):
            continue
        seen.add(parent["chunk_id"])
        per_doc[doc_id] += 1
        tokens.append((doc_id, parent_tokens))
        candidates.append({**row, "parent_id": parent["chunk_id"], "doc_id": doc_id})
    selected = cutoff(candidates, minimum=config.chunk_min_k, maximum=config.chunk_top_k,
                      margin=config.chunk_score_margin, floor=config.chunk_score_floor)
    chunks, provenance = [], []
    for row in selected:
        parent = catalog.parents[row["parent_id"]]
        chunks.append({"doc_id": parent["doc_id"], "chunk_text": parent["text"]})
        provenance.append({k: parent[k] for k in ("doc_id", "start_char", "end_char", "source_text_sha256")}
                          | {"parent_id": parent["chunk_id"], "anchor_child_id": row["child_id"],
                             "reranker_score": row["reranker_score"]})
    return {"id": query_id, "relevant_docs": doc_ids, "relevant_chunks": chunks}, provenance
