"""Bounded pilot BM25 + exact dense retrieval, RRF and source-parent selection."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, sha256_file, verify_file
from .embeddings import unit_signature, validate_vectors
from .text import normalize_for_retrieval

LEXICAL_VERSION = "unicode-words-cjk-unigrams-bigrams-v1"


def lexical_tokens(text):
    pieces = re.findall(r"[\u3400-\u9fff]+|[^\W_\u3400-\u9fff]+", normalize_for_retrieval(text).casefold())
    result = []
    for piece in pieces:
        if "\u3400" <= piece[0] <= "\u9fff":
            result.extend(piece)
            result.extend(piece[i:i+2] for i in range(len(piece)-1))
        else:
            result.append(piece)
    return result


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 100
    sparse_top_k: int = 100
    rrf_k: int = 60
    rerank_top_k: int = 40
    doc_top_k: int = 10
    chunk_top_k: int = 8
    max_chunks_per_doc: int = 2

    def validate(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Retrieval limits must be positive integers.")


class HybridIndex:
    def __init__(self, catalog, vectors, embedding_manifest, index_dir, *, work_dir=None):
        import faiss
        import importlib.metadata
        from rank_bm25 import BM25Okapi

        validate_vectors(vectors, len(catalog.units), embedding_manifest["dimension"])
        if embedding_manifest.get("state") != "COMPLETE" or digest_json({k: v for k, v in embedding_manifest.items() if k != "manifest_sha256"}) != embedding_manifest["manifest_sha256"]:
            raise ValueError("Index requires a verified complete embedding manifest.")
        if embedding_manifest["units_sha256"] != unit_signature(catalog.units):
            raise ValueError("Vectors belong to another representation ordering.")
        self.catalog = catalog
        self.tokens = [lexical_tokens(u.get("sparse_text", u["text"])) for u in catalog.units]
        if not any(self.tokens):
            raise ValueError("No lexical tokens to index.")
        self.bm25 = BM25Okapi(self.tokens)
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)
        identity = {"catalog": catalog.identity, "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
                    "encoder": embedding_manifest["encoder"],
                    "sparse_inputs_sha256": digest_json([u.get("sparse_text", u["text"]) for u in catalog.units]),
                    "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
                    "lexical_version": LEXICAL_VERSION, "rank_bm25": importlib.metadata.version("rank-bm25"),
                    "faiss": faiss.__version__, "dense_index": "IndexFlatIP", "dimension": vectors.shape[1]}
        signature = digest_json(identity)
        manifest_path = root / "index.json"
        if manifest_path.exists():
            saved = read_json(manifest_path)
            if saved["signature"] != signature or any(saved.get(k) != v for k, v in identity.items()) or digest_json({k: v for k, v in saved.items() if k != "manifest_sha256"}) != saved["manifest_sha256"]:
                raise ValueError("Index input/policy changed. Use a new retrieval run.")
            verify_file(root / "dense.faiss", saved["dense_sha256"])
            self.dense = faiss.read_index(str(root / "dense.faiss"))
        else:
            self.dense = faiss.IndexFlatIP(vectors.shape[1])
            self.dense.add(vectors)
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / "dense.faiss"
                faiss.write_index(self.dense, str(local))
                checksum = publish_file(local, root / local.name)
            saved = {**identity, "signature": signature, "dense_sha256": checksum, "rows": len(catalog.units),
                     "scope": "pilot exact search; not a full-scale ANN benchmark"}
            saved["manifest_sha256"] = digest_json(saved)
            atomic_json(manifest_path, saved)
        if self.dense.ntotal != len(catalog.units) or self.dense.d != vectors.shape[1]:
            raise ValueError("FAISS row/dimension mapping mismatch.")
        self.manifest = saved

    def candidates(self, query, query_vector, config):
        config.validate()
        qv = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        validate_vectors(qv, 1, self.dense.d)
        _, indices = self.dense.search(qv, min(config.dense_top_k, self.dense.ntotal))
        dense_ids = [int(i) for i in indices[0] if i >= 0]
        terms = lexical_tokens(query)
        scores = self.bm25.get_scores(terms)
        # Keep genuine lexical matches, including negative BM25 IDF on tiny fixtures.
        matches = [i for i, freq in enumerate(self.bm25.doc_freqs) if any(t in freq for t in terms)]
        sparse_ids = sorted(matches, key=lambda i: (-float(scores[i]), self.catalog.units[i]["id"]))[:config.sparse_top_k]
        fused = defaultdict(float)
        for ranks in (dense_ids, sparse_ids):
            for rank, idx in enumerate(ranks, start=1):
                fused[idx] += 1.0 / (config.rrf_k + rank)
        return sorted(fused, key=lambda i: (-fused[i], self.catalog.units[i]["id"]))[:config.rerank_top_k]


def _select(catalog, indices, scores, config, query_id):
    expanded = []
    for idx, score in zip(indices, scores, strict=True):
        for child_id in catalog.aliases[catalog.units[idx]["id"]]:
            child = catalog.children[child_id]
            expanded.append((float(score), child_id, child["doc_id"], child["parent_id"]))
    expanded.sort(key=lambda row: (-row[0], row[2], row[1]))
    document_scores = {}
    for score, _, doc_id, _ in expanded:
        document_scores[doc_id] = max(score, document_scores.get(doc_id, -float("inf")))
    docs = sorted(document_scores, key=lambda d: (-document_scores[d], d))[:config.doc_top_k]
    chunks, provenance, seen, per_doc = [], [], set(), Counter()
    for score, child_id, doc_id, parent_id in expanded:
        if doc_id not in docs or parent_id in seen or per_doc[doc_id] >= config.max_chunks_per_doc:
            continue
        parent = catalog.parents[parent_id]
        # Suppress near-identical overlapping source parents within the SAME document.
        redundant = False
        for selected in provenance:
            if selected["doc_id"] != doc_id:
                continue
            intersection = max(0, min(selected["end_char"], parent["end_char"]) - max(selected["start_char"], parent["start_char"]))
            union = max(selected["end_char"], parent["end_char"]) - min(selected["start_char"], parent["start_char"])
            redundant |= intersection / union >= .8
        if redundant:
            continue
        seen.add(parent_id)
        per_doc[doc_id] += 1
        chunks.append({"doc_id": doc_id, "chunk_text": parent["text"]})
        provenance.append({"doc_id": doc_id, "parent_id": parent_id, "anchor_child_id": child_id,
                           "start_char": parent["start_char"], "end_char": parent["end_char"],
                           "source_text_sha256": parent["source_text_sha256"], "reranker_score": score})
        if len(chunks) >= config.chunk_top_k:
            break
    return {"id": query_id, "relevant_docs": docs, "relevant_chunks": chunks}, provenance


def predict_queries(index, queries, query_vectors, reranker, output_dir, *, config=None,
                    query_embedding_manifest, batch_size=16, max_new_queries=None):
    config = config or RetrievalConfig()
    config.validate()
    if len({q["id"] for q in queries}) != len(queries) or any(type(q["id"]) is not int or not q["query"].strip() for q in queries):
        raise ValueError("Queries need unique integer IDs and nonempty text.")
    validate_vectors(query_vectors, len(queries), index.dense.d)
    query_units = [{"id": q["id"], "text": q["query"]} for q in queries]
    qm = query_embedding_manifest
    if qm.get("state") != "COMPLETE" or qm["units_sha256"] != unit_signature(query_units) or digest_json({k: v for k, v in qm.items() if k != "manifest_sha256"}) != qm["manifest_sha256"]:
        raise ValueError("Query vectors need a complete manifest in official query order.")
    if qm["encoder"] != index.manifest["encoder"]:
        raise ValueError("Queries and corpus must use the same embedding model/policy.")
    if max_new_queries is not None and max_new_queries < 0:
        raise ValueError("max_new_queries must be nonnegative or None.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"index_signature": index.manifest["signature"], "queries_sha256": digest_json(queries),
                "query_embedding_manifest_sha256": qm["manifest_sha256"],
                "query_vectors_sha256": hashlib.sha256(query_vectors.tobytes()).hexdigest(),
                "reranker": reranker.identity, "config": asdict(config), "selection_version": "rrf-child-parent-v1",
                "retrieval_code_sha256": sha256_file(Path(__file__))}
    signature = digest_json(identity)
    config_path = root / "config.json"
    if config_path.exists() and read_json(config_path) != {**identity, "signature": signature}:
        raise ValueError("Query/model/selection policy changed. Use a new retrieval run name.")
    if not config_path.exists():
        atomic_json(config_path, {**identity, "signature": signature})
    records, written = [], 0
    for q, vector in tqdm(zip(queries, query_vectors, strict=True), total=len(queries), desc="Query checkpoints"):
        path = root / f"query-{q['id']}.done.json"
        if path.exists():
            saved = read_json(path)
            if saved["signature"] != signature or saved["query_sha256"] != digest_json(q) or digest_json({k:v for k,v in saved.items() if k != "record_sha256"}) != saved["record_sha256"]:
                raise ValueError("Query checkpoint changed or belongs to another input.")
        else:
            if max_new_queries is not None and written >= max_new_queries:
                break
            indices = index.candidates(q["query"], vector, config)
            scores = np.asarray(reranker.score([(q["query"], index.catalog.units[i]["text"]) for i in indices], batch_size=batch_size), dtype=np.float32)
            if scores.shape != (len(indices),) or not np.isfinite(scores).all():
                raise ValueError("Invalid reranker score shape/values.")
            prediction, provenance = _select(index.catalog, indices, scores, config, q["id"])
            saved = {"signature": signature, "query_sha256": digest_json(q), "prediction": prediction,
                     "provenance": provenance, "reranked_representations": len(indices)}
            saved["record_sha256"] = digest_json(saved)
            atomic_json(path, saved)
            written += 1
        records.append(saved)
    report = {"signature": signature, "completed_queries": len(records), "requested_queries": len(queries),
              "state": "COMPLETE" if len(records) == len(queries) else "IN_PROGRESS",
              "evaluation": "NOT_EVALUATED_NO_REFERENCE_LABELS", "scope": "pilot predictions, not a passed relevance gate"}
    atomic_json(root / "predictions.json", report)
    return records, report
