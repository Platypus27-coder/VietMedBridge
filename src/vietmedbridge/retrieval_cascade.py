"""Bounded document → child → source-parent cascade, with auditable scores."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .embeddings import unit_signature, validate_vectors
from .quality import error_page_reason, has_encoded_payload
from .representation import build_sparse_text
from .retrieval import HybridIndex, lexical_tokens
from .retrieval_policy import cutoff, select_parents
from .text import _redirected_to_homepage, language_hint


@dataclass(frozen=True)
class CascadeConfig:
    doc_candidate_k: int = 200
    sparse_top_k: int = 200
    detail_doc_k: int = 30
    children_per_doc: int = 5
    summary_children: int = 2
    rrf_k: int = 60
    dense_weight: float = 1.0
    vi_weight: float = .8
    en_weight: float = .6
    zh_weight: float = .6
    rerank_fusion: str = "reranker"
    retrieval_share: float = .5
    doc_min_k: int = 3
    doc_top_k: int = 10
    chunk_min_k: int = 2
    chunk_top_k: int = 8
    max_chunks_per_doc: int = 2
    dedup_threshold: float = .8
    doc_score_margin: float | None = None
    chunk_score_margin: float | None = None
    doc_score_floor: float | None = None
    chunk_score_floor: float | None = None
    window_overlap_tokens: int = 32

    def validate(self):
        positive = ("doc_candidate_k", "sparse_top_k", "detail_doc_k", "children_per_doc",
                    "summary_children", "rrf_k", "doc_top_k", "chunk_top_k", "max_chunks_per_doc")
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in positive):
            raise ValueError("Cascade candidate limits must be positive integers.")
        if not self.doc_min_k <= self.doc_top_k <= self.detail_doc_k <= self.doc_candidate_k or not 0 <= self.chunk_min_k <= self.chunk_top_k:
            raise ValueError("Invalid cascade stage/cutoff limits.")
        if self.rerank_fusion not in ("reranker", "rrf"):
            raise ValueError("Unknown reranker fusion policy.")
        if not 0 <= self.retrieval_share <= 1 or not 0 < self.dedup_threshold <= 1:
            raise ValueError("Invalid fusion/dedup parameter.")
        if any(not math.isfinite(getattr(self, key)) or getattr(self, key) < 0
               for key in ("dense_weight", "vi_weight", "en_weight", "zh_weight")) or self.dense_weight <= 0:
            raise ValueError("The original dense branch must remain enabled.")
        if type(self.window_overlap_tokens) is not int or self.window_overlap_tokens < 0:
            raise ValueError("Invalid MaxP window overlap.")
        cutoff([], minimum=self.doc_min_k, maximum=self.doc_top_k,
               margin=self.doc_score_margin, floor=self.doc_score_floor)
        cutoff([], minimum=self.chunk_min_k, maximum=self.chunk_top_k,
               margin=self.chunk_score_margin, floor=self.chunk_score_floor)


class PostingBM25:
    """Lucene-style positive IDF and sparse postings; no all-row query scan."""
    VERSION = "postings-bm25-lucene-idf-k1-1.5-b-.75-v1"

    def __init__(self, texts):
        self.lengths = np.array([len(lexical_tokens(text)) for text in texts], dtype=np.float64)
        self.n = len(texts)
        average = float(self.lengths.mean()) if self.n else 1.0
        self.norm = 1.5 * (.25 + .75 * self.lengths / max(average, 1.0))
        postings = defaultdict(list)
        for idx, text in enumerate(texts):
            for term, frequency in Counter(lexical_tokens(text)).items():
                postings[term].append((idx, frequency))
        self.postings = {term: (np.array([i for i, _ in rows], dtype=np.int64),
                               np.array([f for _, f in rows], dtype=np.float64))
                         for term, rows in postings.items()}

    def scores(self, text):
        scores = np.zeros(self.n, dtype=np.float64)
        for term in set(lexical_tokens(text)):
            if term not in self.postings:
                continue
            indices, frequencies = self.postings[term]
            idf = math.log1p((self.n - len(indices) + .5) / (len(indices) + .5))
            scores[indices] += idf * frequencies * 2.5 / (frequencies + self.norm[indices])
        return scores


def weighted_rrf(orders, weights, offset):
    result = defaultdict(float)
    for name, order in orders.items():
        if len(set(order)) != len(order):
            raise ValueError("A retrieval branch contains duplicate IDs.")
        for rank, identifier in enumerate(order, 1):
            result[identifier] += weights[name] / (offset + rank)
    return dict(result)


def _prefix(tokenizer, text, budget):
    if budget <= 0:
        return ""
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)["offset_mapping"]
    return text if len(offsets) <= budget else text[:offsets[budget-1][1]]


def pair_budget(tokenizer, query, max_length):
    query_length = len(tokenizer(query, add_special_tokens=False, truncation=False,
                                 verbose=False)["input_ids"])
    # BGE's XLM-R pair has four special tokens; query text is never shortened.
    budget = max_length - query_length - tokenizer.num_special_tokens_to_add(pair=True)
    if budget < 8:
        raise ValueError("Query leaves too little passage budget for reranking.")
    return budget


def document_representation(catalog, doc_id, child_ids, tokenizer, budget):
    title = catalog.documents[doc_id].get("title", "")
    title = _prefix(tokenizer, title, min(32, budget // 6))
    title_tokens = len(tokenizer(title, add_special_tokens=False, verbose=False)["input_ids"])
    # Reserve separators; allocate equal space so the second hit survives truncation.
    per_hit = max(1, (budget - title_tokens - 8) // max(1, len(child_ids)))
    texts = [_prefix(tokenizer, catalog.children[identifier]["text"], per_hit) for identifier in child_ids]
    result = "\n\n".join(part for part in [title, *texts] if part)
    return _prefix(tokenizer, result, budget)


def passage_windows(tokenizer, text, budget, overlap):
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)["offset_mapping"]
    if not offsets:
        raise ValueError("Empty child representation.")
    if len(offsets) <= budget:
        return [{"text": text, "start_char": 0, "end_char": len(text)}]
    # MaxP retains tail evidence for long query/child pairs instead of head truncation.
    stride = max(1, budget - min(overlap, budget // 2))
    result = []
    for start in range(0, len(offsets), stride):
        end = min(len(offsets), start + budget)
        a, b = offsets[start][0], offsets[end-1][1]
        result.append({"text": text[a:b], "start_char": a, "end_char": b})
        if end == len(offsets):
            break
    return result


class CascadeIndex:
    def __init__(self, catalog, vectors, embedding_manifest, index_dir, tokenizer, *, work_dir=None):
        base = HybridIndex(catalog, vectors, embedding_manifest, index_dir, work_dir=work_dir)
        self.catalog, self.tokenizer, self.dense = catalog, tokenizer, base.dense
        self.encoder = embedding_manifest["encoder"]
        self.unit_for_child = {child: idx for idx, unit in enumerate(catalog.units)
                               for child in catalog.aliases[unit["id"]]}
        self.doc_children = defaultdict(list)
        for child_id, child in catalog.children.items():
            self.doc_children[child["doc_id"]].append(child_id)
        self.exclusions, self.languages = {}, {}
        for doc_id, doc in catalog.documents.items():
            reason = error_page_reason(doc.get("title", ""), doc["source_text"])
            if doc.get("url") and doc.get("final_url") and _redirected_to_homepage(doc["url"], doc["final_url"]):
                reason = "ARTICLE_REDIRECTED_TO_HOMEPAGE"
            if has_encoded_payload(doc["source_text"]):
                reason = "ENCODED_PAYLOAD_REVIEW"
            if reason:
                self.exclusions[doc_id] = reason
            # Recompute from content; stale declared language never controls branch routing.
            self.languages[doc_id] = language_hint(doc["source_text"])[0]
        self.doc_ids = sorted(set(self.doc_children) - set(self.exclusions))
        if not self.doc_ids:
            raise ValueError("No eligible documents after source quarantine.")
        self.sparse = {}
        for language in ("vi", "en", "zh"):
            ids = [d for d in self.doc_ids if self.languages[d] in (language, "unknown")]
            texts = [catalog.documents[d].get("title", "") + "\n" + catalog.documents[d]["source_text"] for d in ids]
            self.sparse[language] = (ids, PostingBM25(texts))
        self.child_sparse = {d: PostingBM25([self.child_text(c) for c in self.doc_children[d]]) for d in self.doc_ids}
        identity = {"base_index_signature": base.manifest["signature"], "encoder": self.encoder,
            "documents_sha256": digest_json(catalog.documents), "children_sha256": digest_json(catalog.children),
            "languages": {str(d):lang for d,lang in self.languages.items()},
            "quarantine": {str(d):reason for d,reason in self.exclusions.items()}, "sparse_policy": PostingBM25.VERSION,
            "tokenizer": catalog.candidate.get("golden", {}).get("tokenizer", {}),
            "tokenizer_class": type(tokenizer).__name__, "code_sha256": sha256_file(Path(__file__))}
        saved = {**identity, "signature": digest_json(identity)}
        saved["manifest_sha256"] = digest_json(saved)
        path = Path(index_dir) / "cascade.json"
        if path.exists() and digest_json(read_json(path)) != digest_json(saved):
            raise ValueError("Cascade source/index policy changed; use a new retrieval run.")
        atomic_json(path, saved)
        atomic_json(Path(index_dir) / "quarantine.json", {"excluded_documents": self.exclusions,
            "action": "RETRIEVAL_HOLD_SOURCE_AND_OFFICIAL_IDS_PRESERVED",
            "eligible_documents": len(self.doc_ids), "frozen_documents": len(catalog.documents)})
        self.manifest = saved

    def child_text(self, child_id):
        child = self.catalog.children[child_id]
        return build_sparse_text(child, title=self.catalog.documents[child["doc_id"]].get("title", ""),
                                 heading=child.get("heading", ""))

    def document_candidates(self, query, vector, variants, config):
        qv = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        validate_vectors(qv, 1, self.dense.d)
        values, indices = self.dense.search(qv, self.dense.ntotal)
        unit_scores = {int(i): float(s) for i, s in zip(indices[0], values[0], strict=True) if i >= 0}
        dense_scores = {d: max(unit_scores[self.unit_for_child[c]] for c in self.doc_children[d]) for d in self.doc_ids}
        orders = {"dense": sorted(self.doc_ids, key=lambda d: (-dense_scores[d], d))[:config.doc_candidate_k]}
        branch_scores = {"dense": dense_scores}
        for language, text in (("vi", query), ("en", variants.get("query_en")), ("zh", variants.get("query_zh"))):
            ids, bm25 = self.sparse[language]
            if not text or not ids:
                continue
            scores = bm25.scores(text)
            order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], ids[i]))[:config.sparse_top_k]
            orders[language] = [ids[i] for i in order]
            branch_scores[language] = {ids[i]: float(scores[i]) for i in order}
        weights = {"dense": config.dense_weight, "vi": config.vi_weight,
                   "en": config.en_weight, "zh": config.zh_weight}
        fused = weighted_rrf(orders, weights, config.rrf_k)
        doc_ids = sorted(fused, key=lambda d: (-fused[d], d))[:config.doc_candidate_k]
        ranks = {name: {d:r for r,d in enumerate(order, 1)} for name,order in orders.items()}
        rows = [{"doc_id": d, "retrieval_score": fused[d], "branches": {
            name: {"rank": ranks[name][d], "score": branch_scores[name][d]}
            for name in orders if d in ranks[name]}} for d in doc_ids]
        return rows, unit_scores

    def child_candidates(self, doc_id, query, variants, unit_scores, config, limit):
        ids = self.doc_children[doc_id]
        dense_order = sorted(ids, key=lambda c: (-unit_scores[self.unit_for_child[c]], c))
        queries = [query] + [variants["query_" + self.languages[doc_id]]] if self.languages[doc_id] in ("en", "zh") and variants.get("query_" + self.languages[doc_id]) else [query]
        bm25 = self.child_sparse[doc_id]
        sparse_scores = np.maximum.reduce([bm25.scores(text) for text in queries])
        sparse_order = [ids[i] for i in sorted((i for i,s in enumerate(sparse_scores) if s > 0), key=lambda i: (-sparse_scores[i], ids[i]))]
        fused = weighted_rrf({"dense": dense_order, "sparse": sparse_order},
                             {"dense": 1.0, "sparse": .8}, config.rrf_k)
        global_rank = {idx:rank for rank,idx in enumerate(unit_scores, 1)}
        for c in fused:
            fused[c] += .5 / (config.rrf_k + global_rank[self.unit_for_child[c]])
        return [{"child_id": c, "doc_id": doc_id, "retrieval_score": fused[c],
                 "global_dense_rank": global_rank[self.unit_for_child[c]],
                 "dense_score": unit_scores[self.unit_for_child[c]],
                 "sparse_score": float(sparse_scores[ids.index(c)])}
                for c in sorted(fused, key=lambda c: (-fused[c], c))[:limit]]


def _score_checkpoint(path, pairs, reranker, signature, batch_size):
    identity = {"signature": signature, "pairs_sha256": digest_json(pairs), "reranker": reranker.identity}
    if path.exists():
        saved = read_json(path)
        if any(saved.get(k) != v for k,v in identity.items()) or saved.get("pairs") != pairs or digest_json({k:v for k,v in saved.items() if k != "record_sha256"}) != saved["record_sha256"]:
            raise ValueError("Candidate/score checkpoint integrity mismatch.")
    else:
        scores = np.asarray(reranker.score([(p["query"], p["text"]) for p in pairs], batch_size=batch_size), dtype=np.float32)
        if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
            raise ValueError("Each candidate pair needs exactly one finite reranker score.")
        saved = {**identity, "pairs": pairs, "scores": scores.tolist()}
        saved["record_sha256"] = digest_json(saved)
        atomic_json(path, saved)
    if len(saved["scores"]) != len(pairs) or not all(type(s) in (int, float) and math.isfinite(s) for s in saved["scores"]):
        raise ValueError("Incomplete/nonfinite candidate score checkpoint.")
    return saved


def _rank(rows, config, id_key):
    by_model = sorted(rows, key=lambda r: (-r["reranker_score"], -r["retrieval_score"], r[id_key]))
    if config.rerank_fusion == "reranker":
        return by_model
    by_retrieval = sorted(rows, key=lambda r: (-r["retrieval_score"], r[id_key]))
    scores = weighted_rrf({"retrieval": [r[id_key] for r in by_retrieval],
                           "reranker": [r[id_key] for r in by_model]},
                          {"retrieval": config.retrieval_share, "reranker": 1-config.retrieval_share}, config.rrf_k)
    return sorted(rows, key=lambda r: (-scores[r[id_key]], -r["retrieval_score"], r[id_key]))


def predict_cascade(index, queries, vectors, reranker, output_dir, *, query_embedding_manifest,
                    translations, translation_signature=None, config=None, batch_size=16, max_new_queries=None):
    config = config or CascadeConfig()
    config.validate()
    if len({q["id"] for q in queries}) != len(queries) or any(type(q["id"]) is not int or not q["query"].strip() for q in queries):
        raise ValueError("Queries need unique integer IDs and nonempty text.")
    if max_new_queries is not None and (type(max_new_queries) is not int or max_new_queries < 0):
        raise ValueError("Invalid query limit.")
    validate_vectors(vectors, len(queries), index.dense.d)
    qm = query_embedding_manifest
    if qm.get("state") != "COMPLETE" or qm["units_sha256"] != unit_signature([{"id":q["id"], "text":q["query"]} for q in queries]) or digest_json({k:v for k,v in qm.items() if k != "manifest_sha256"}) != qm["manifest_sha256"] or qm["encoder"] != index.encoder:
        raise ValueError("Query embedding manifest/order/model mismatch.")
    if len(translations) > len(queries):
        raise ValueError("Unexpected extra query translations.")
    translation_signature = translation_signature or (translations[0]["signature"] if translations else None)
    if not translation_signature:
        raise ValueError("Translation contract signature is required.")
    from .query_translation import validate_variants
    for query, record in zip(queries[:len(translations)], translations, strict=True):
        if record["signature"] != translation_signature or record["query_sha256"] != digest_json(query) or digest_json({k:v for k,v in record.items() if k != "record_sha256"}) != record["record_sha256"] or record["variants"] != validate_variants(query["query"], record["raw"]):
            raise ValueError("Translation checkpoint/query/validation mismatch.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"index_signature": index.manifest["signature"], "queries_sha256": digest_json(queries),
        "query_embedding_manifest_sha256": qm["manifest_sha256"],
        "query_vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "translation_signature": translation_signature,
        "reranker": reranker.identity, "config": asdict(config), "selection_version": "document-child-parent-cascade-v2",
        "code_sha256": sha256_file(Path(__file__)),
        "selection_code_sha256": sha256_file(Path(__file__).with_name("retrieval_policy.py"))}
    signature = digest_json(identity)
    contract = {**identity, "signature": signature}
    if (root / "config.json").exists() and read_json(root / "config.json") != contract:
        raise ValueError("Cascade query/model/selection changed; use a new retrieval run.")
    atomic_json(root / "config.json", contract)
    records, written = [], 0
    for query, vector, translation in tqdm(zip(queries[:len(translations)], vectors[:len(translations)], translations, strict=True), total=len(translations), desc="Cascade checkpoints"):
        path = root / f"query-{query['id']}.done.json"
        if path.exists():
            saved = read_json(path)
            if saved["signature"] != signature or saved["query_sha256"] != digest_json(query) or saved["translation_record_sha256"] != translation["record_sha256"] or digest_json({k:v for k,v in saved.items() if k != "record_sha256"}) != saved["record_sha256"]:
                raise ValueError("Cascade query checkpoint integrity mismatch.")
            # A completed selection still requires both auditable scored stages.
            for stage, checksum in saved["score_checkpoints"].items():
                evidence = read_json(root / f"query-{query['id']}-{stage}.done.json")
                if evidence["record_sha256"] != checksum or digest_json({k:v for k,v in evidence.items() if k != "record_sha256"}) != checksum:
                    raise ValueError("Cascade score evidence missing/changed.")
        else:
            if max_new_queries is not None and written >= max_new_queries:
                break
            variants = translation["variants"]
            documents, dense_scores = index.document_candidates(query["query"], vector, variants, config)
            budget = pair_budget(index.tokenizer, query["query"], reranker.identity["max_length"])
            pairs = []
            for row in documents:
                hits = index.child_candidates(row["doc_id"], query["query"], variants, dense_scores, config, config.summary_children)
                child_ids = [r["child_id"] for r in hits]
                pairs.append({"doc_id": row["doc_id"], "query": query["query"], "anchor_children": child_ids,
                              "text": document_representation(index.catalog, row["doc_id"], child_ids, index.tokenizer, budget)})
            doc_stage = _score_checkpoint(root / f"query-{query['id']}-documents.done.json", pairs, reranker, signature, batch_size)
            documents = _rank([row | {"reranker_score": score} for row,score in zip(documents, doc_stage["scores"], strict=True)], config, "doc_id")
            children = [row for doc in documents[:config.detail_doc_k]
                        for row in index.child_candidates(doc["doc_id"], query["query"], variants, dense_scores, config, config.children_per_doc)]
            pairs = []
            for row in children:
                text = index.child_text(row["child_id"])
                for window in passage_windows(index.tokenizer, text, budget, config.window_overlap_tokens):
                    pairs.append({"child_id": row["child_id"], "query": query["query"], **window})
            child_stage = _score_checkpoint(root / f"query-{query['id']}-children.done.json", pairs, reranker, signature, batch_size)
            best_scores = defaultdict(lambda: -float("inf"))
            for pair, score in zip(pairs, child_stage["scores"], strict=True):
                best_scores[pair["child_id"]] = max(best_scores[pair["child_id"]], score)
            children = _rank([row | {"reranker_score": best_scores[row["child_id"]]} for row in children], config, "child_id")
            ranking = {"documents": documents, "children": children}
            prediction, provenance = select_parents(index.catalog, ranking, config, index.tokenizer, query["id"])
            saved = {"signature": signature, "query_sha256": digest_json(query), "prediction": prediction,
                     "provenance": provenance, "ranking": ranking,
                     "translation_record_sha256": translation["record_sha256"],
                     "score_checkpoints": {"documents": doc_stage["record_sha256"], "children": child_stage["record_sha256"]}}
            saved["record_sha256"] = digest_json(saved)
            atomic_json(path, saved)
            written += 1
        records.append(saved)
    report = {"signature": signature, "completed_queries": len(records), "requested_queries": len(queries),
              "state": "COMPLETE" if len(records) == len(queries) else "IN_PROGRESS",
              "evaluation": "NOT_EVALUATED_NO_REFERENCE_LABELS", "scope": "plan-aligned pretrained pilot; quality not yet measured"}
    atomic_json(root / "predictions.json", report)
    return records, report
