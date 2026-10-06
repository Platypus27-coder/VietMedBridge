"""Full pretrained architecture: dual dense, medical sparse, expansion and Qwen cascade."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .embeddings import unit_signature, validate_vectors
from .medical_lexical import MedicalAnalyzer
from .query_expansion import is_complex
from .retrieval_cascade import (CascadeConfig, CascadeIndex, PostingBM25, _rank, _score_checkpoint,
    document_representation, pair_budget, passage_windows, weighted_rrf)
from .retrieval_policy import select_parents


@dataclass(frozen=True)
class StrongConfig(CascadeConfig):
    document_dense_weight: float = .6
    second_dense_weight: float = .8
    subquery_weight: float = .3
    hyde_weight: float = .3
    auxiliary_rerank_weight: float = 0.0
    parent_short_tokens: int = 512
    parent_long_tokens: int = 640
    adaptive_parents: bool = True

    def validate(self):
        super().validate()
        if any(not np.isfinite(getattr(self, k)) or getattr(self, k) < 0 for k in
               ("document_dense_weight", "second_dense_weight", "subquery_weight", "hyde_weight", "auxiliary_rerank_weight")):
            raise ValueError("Invalid strong-branch weight.")
        if (type(self.parent_short_tokens) is not int or type(self.parent_long_tokens) is not int
            or self.parent_short_tokens < 180 or self.parent_long_tokens < self.parent_short_tokens):
            raise ValueError("Invalid adaptive parent budgets.")


def auxiliary_units(queries, expansions):
    units, seen = [], set()
    for query, expansion in zip(queries, expansions, strict=True):
        if expansion["query_sha256"] != digest_json(query) or digest_json({k: v for k, v in expansion.items() if k != "record_sha256"}) != expansion["record_sha256"]:
            raise ValueError("Expansion/query integrity mismatch.")
        texts = [query["query"], *expansion["variants"]["subqueries"]]
        if expansion["variants"]["hyde_en"]:
            texts.append(expansion["variants"]["hyde_en"])
        for text in texts:
            identifier = digest_json(text)
            if identifier not in seen:
                units.append({"id": identifier, "text": text})
                seen.add(identifier)
    return units


class _AnalyzedBM25:
    def __init__(self, texts, analyzer, language):
        self.analyzer, self.language = analyzer, language
        self.index = PostingBM25([self._text(t) for t in texts])

    def _text(self, text):
        return " ".join("t" + hashlib.sha256(t.encode()).hexdigest() for t in self.analyzer.analyze(text, self.language))

    def scores(self, text):
        return self.index.scores(self._text(text))


class UnitScores(dict):
    pass


class StrongIndex(CascadeIndex):
    def __init__(self, catalog, vectors, embedding_manifest, index_dir, tokenizer, *,
                 secondary_vectors, secondary_manifest, auxiliary_vectors,
                 auxiliary_manifest, auxiliary_inputs, queries, expansions,
                 analyzer=None, work_dir=None, document_vectors=None, document_manifest=None, document_inputs=None):
        import faiss

        super().__init__(catalog, vectors, embedding_manifest, index_dir, tokenizer, work_dir=work_dir)
        if auxiliary_inputs != auxiliary_units(queries, expansions):
            raise ValueError("Auxiliary query vectors must cover the exact validated expansion inputs.")
        self.analyzer = analyzer or MedicalAnalyzer()
        self.document_dense = None
        document_evidence = None
        if document_vectors is not None:
            dm = document_manifest
            validate_vectors(document_vectors, len(document_inputs), self.dense.d)
            if (dm["state"] != "COMPLETE" or dm["dimension"] != self.dense.d
                or dm["units_sha256"] != unit_signature(document_inputs)
                or digest_json({k:v for k,v in dm.items() if k != "manifest_sha256"}) != dm["manifest_sha256"]):
                raise ValueError("Document embedding manifest/input mismatch.")
            keys = ("model_id", "revision", "max_length", "pooling", "dimension", "query_instruction",
                    "truncation", "precision", "device_class", "inference_code_sha256", "model")
            if any(dm["encoder"].get(k) != self.encoder.get(k) for k in keys):
                raise ValueError("Document and child BGE embedding policies differ.")
            self.document_dense_ids = [int(u["id"]) for u in document_inputs]
            if len(set(self.document_dense_ids)) != len(self.document_dense_ids) or set(self.document_dense_ids) != set(self.doc_ids):
                raise ValueError("Document embeddings must cover all eligible source IDs exactly once.")
            self.document_dense = faiss.IndexFlatIP(self.dense.d)
            self.document_dense.add(document_vectors)
            document_evidence = {"manifest_sha256": dm["manifest_sha256"],
                "vectors_sha256": hashlib.sha256(document_vectors.tobytes()).hexdigest()}
        elif document_manifest is not None or document_inputs is not None:
            raise ValueError("Incomplete document dense inputs.")
        for values, manifest, inputs, role in ((secondary_vectors, secondary_manifest, catalog.units, "corpus"),
                                               (auxiliary_vectors, auxiliary_manifest, auxiliary_inputs, "query")):
            validate_vectors(values, len(inputs), manifest["dimension"])
            if manifest["state"] != "COMPLETE" or manifest["units_sha256"] != unit_signature(inputs) or digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
                raise ValueError("Secondary embedding manifest/input mismatch.")
            if manifest["encoder"].get("input_role") != role:
                raise ValueError("Secondary encoder role mismatch.")
        if {k: v for k, v in secondary_manifest["encoder"].items() if k != "input_role"} != {k: v for k, v in auxiliary_manifest["encoder"].items() if k != "input_role"}:
            raise ValueError("Secondary corpus/query model policies differ.")
        self.secondary = faiss.IndexFlatIP(secondary_manifest["dimension"])
        self.secondary.add(secondary_vectors)
        self.auxiliary = {u["id"]: v for u, v in zip(auxiliary_inputs, auxiliary_vectors, strict=True)}
        self.expansions = {q["query"]: e["variants"] for q, e in zip(queries, expansions, strict=True)}
        self.fields = {}
        boosts = {"title": 2.0, "headings": 1.5, "body": 1.0, "aliases": 1.5}
        for language in ("vi", "en", "zh"):
            ids = [d for d in self.doc_ids if self.languages[d] in (language, "unknown")]
            self.fields[language] = (ids, {field: (boost, _AnalyzedBM25([
                self._field(d, field) for d in ids], self.analyzer, language)) for field, boost in boosts.items()})
        self.local_sparse = {d: _AnalyzedBM25([self.child_text(c) for c in self.doc_children[d]], self.analyzer, self.languages[d]) for d in self.doc_ids}
        identity = {"cascade_signature": self.manifest["signature"], "secondary": secondary_manifest["manifest_sha256"],
            "auxiliary": auxiliary_manifest["manifest_sha256"], "expansions_sha256": digest_json(expansions),
            "secondary_vectors_sha256": hashlib.sha256(secondary_vectors.tobytes()).hexdigest(),
            "auxiliary_vectors_sha256": hashlib.sha256(auxiliary_vectors.tobytes()).hexdigest(),
            "catalog": catalog.identity, "analyzer": self.analyzer.identity, "field_boosts": boosts,
            "document_dense": document_evidence,
            "code_sha256": sha256_file(Path(__file__))}
        self.manifest = identity | {"signature": digest_json(identity)}
        self.manifest["manifest_sha256"] = digest_json(self.manifest)
        path = Path(index_dir) / "strong.json"
        if path.exists() and read_json(path) != self.manifest:
            raise ValueError("Strong index inputs/policy changed; choose a new run.")
        atomic_json(path, self.manifest)

    def _field(self, doc, field):
        value = self.catalog.documents[doc]
        if field == "title":
            return value.get("title", "")
        if field == "headings":
            return "\n".join(dict.fromkeys(self.catalog.children[c].get("heading", "") for c in self.doc_children[doc]))
        if field == "aliases":
            return self.analyzer.aliases(value["source_text"])
        return value["source_text"]

    def _secondary_scores(self, text):
        vector = self.auxiliary.get(digest_json(text))
        if vector is None:
            raise ValueError("Missing secondary query/expansion vector.")
        values, indices = self.secondary.search(np.asarray(vector, np.float32).reshape(1, -1), self.secondary.ntotal)
        return {int(i): float(s) for i, s in zip(indices[0], values[0], strict=True) if i >= 0}

    def document_candidates(self, query, vector, variants, config):
        _, primary = super().document_candidates(query, vector, variants, config)
        scores = UnitScores(primary)
        scores.secondary = self._secondary_scores(query)
        orders, branch_scores, weights = {}, {}, {}

        def dense_leg(name, units, weight):
            if weight <= 0:
                return
            values = {d: max(units[self.unit_for_child[c]] for c in self.doc_children[d]) for d in self.doc_ids}
            orders[name] = sorted(values, key=lambda d: (-values[d], d))[:config.doc_candidate_k]
            branch_scores[name], weights[name] = values, weight

        dense_leg("dense", primary, config.dense_weight)
        dense_leg("dense_qwen", scores.secondary, config.second_dense_weight)
        if self.document_dense is not None and config.document_dense_weight:
            values, positions = self.document_dense.search(np.asarray(vector,np.float32).reshape(1,-1), self.document_dense.ntotal)
            values = {self.document_dense_ids[int(i)]:float(v) for i,v in zip(positions[0],values[0],strict=True) if i >= 0}
            orders["dense_document"] = sorted(values,key=lambda d:(-values[d],d))[:config.doc_candidate_k]
            branch_scores["dense_document"],weights["dense_document"] = values,config.document_dense_weight
        expansion = self.expansions[query]
        for idx, text in enumerate(expansion["subqueries"]):
            dense_leg(f"subquery_{idx}", self._secondary_scores(text), config.subquery_weight)
        if expansion["hyde_en"]:
            dense_leg("hyde", self._secondary_scores(expansion["hyde_en"]), config.hyde_weight)
        for language, text in (("vi", query), ("en", variants.get("query_en")), ("zh", variants.get("query_zh"))):
            ids, fields = self.fields[language]
            if not text or not ids:
                continue
            values = sum((boost * index.scores(text) for boost, index in fields.values()), np.zeros(len(ids)))
            order = sorted((i for i, s in enumerate(values) if s > 0), key=lambda i: (-values[i], ids[i]))[:config.sparse_top_k]
            orders[language] = [ids[i] for i in order]
            branch_scores[language] = {ids[i]: float(values[i]) for i in order}
            weights[language] = getattr(config, language + "_weight")
        fused = weighted_rrf(orders, weights, config.rrf_k)
        ids = sorted(fused, key=lambda d: (-fused[d], d))[:config.doc_candidate_k]
        ranks = {name: {d: r for r, d in enumerate(order, 1)} for name, order in orders.items()}
        return [{"doc_id": d, "retrieval_score": fused[d], "branches": {name: {"rank": ranks[name][d],
                    "score": branch_scores[name][d]} for name in orders if d in ranks[name]}} for d in ids], scores

    def child_candidates(self, doc_id, query, variants, unit_scores, config, limit):
        ids = self.doc_children[doc_id]
        order = lambda values: sorted(ids, key=lambda c: (-values[self.unit_for_child[c]], c))
        scores = self.local_sparse[doc_id].scores(query)
        language = self.languages[doc_id]
        if variants.get("query_" + language):
            scores = np.maximum(scores, self.local_sparse[doc_id].scores(variants["query_" + language]))
        sparse = [ids[i] for i in sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], ids[i]))]
        orders, weights = {"dense": order(unit_scores), "sparse": sparse}, {"dense": 1., "sparse": .8}
        if config.second_dense_weight:
            orders["dense_qwen"] = order(unit_scores.secondary)
            weights["dense_qwen"] = config.second_dense_weight
        fused = weighted_rrf(orders, weights, config.rrf_k)
        global_rank = {idx: rank for rank, idx in enumerate(unit_scores, 1)}
        return [{"child_id": c, "doc_id": doc_id, "retrieval_score": fused[c],
            "global_dense_rank": global_rank[self.unit_for_child[c]],
            "dense_score": unit_scores[self.unit_for_child[c]], "sparse_score": float(scores[ids.index(c)])}
            for c in sorted(fused, key=lambda c: (-fused[c], c))[:limit]]


def _windows(reranker, tokenizer, query, text, overlap):
    if hasattr(reranker, "windows"):
        return reranker.windows(query, text, overlap)
    return passage_windows(tokenizer, text, pair_budget(tokenizer, query, reranker.identity["max_length"]), overlap)


def _maxp_rows(rows, pairs, scores, id_key, config):
    best = defaultdict(dict)
    for pair, score in zip(pairs, scores, strict=True):
        key, leg = pair[id_key], pair["score_leg"]
        best[key][leg] = max(best[key].get(leg, -float("inf")), score)
    return _rank([row | {"reranker_score": best[row[id_key]]["original"]
                  + config.auxiliary_rerank_weight * best[row[id_key]].get("translated", 0.)} for row in rows], config, id_key)


def predict_strong(index, queries, vectors, reranker, output_dir, *, query_embedding_manifest,
                   translations, config=None, batch_size=2, max_new_queries=None):
    config = config or StrongConfig()
    config.validate()
    if len({q["id"] for q in queries}) != len(queries) or any(type(q["id"]) is not int or not isinstance(q.get("query"), str) or not q["query"].strip() for q in queries):
        raise ValueError("Queries need unique integer IDs and nonempty text.")
    if max_new_queries is not None and (type(max_new_queries) is not int or max_new_queries < 0):
        raise ValueError("Invalid query limit.")
    validate_vectors(vectors, len(queries), index.dense.d)
    qm = query_embedding_manifest
    if (qm["state"] != "COMPLETE" or qm["units_sha256"] != unit_signature([{"id": q["id"], "text": q["query"]} for q in queries])
        or qm["encoder"] != index.encoder or digest_json({k: v for k, v in qm.items() if k != "manifest_sha256"}) != qm["manifest_sha256"]):
        raise ValueError("Primary query embeddings mismatch.")
    from .query_translation import validate_variants
    for query, translation in zip(queries, translations, strict=True):
        if (translation["query_sha256"] != digest_json(query) or translation["variants"] != validate_variants(query["query"], translation["raw"])
            or digest_json({k: v for k, v in translation.items() if k != "record_sha256"}) != translation["record_sha256"]):
            raise ValueError("Translation evidence mismatch.")
    identity = {"index": index.manifest["signature"], "queries_sha256": digest_json(queries),
        "query_embeddings": qm["manifest_sha256"], "query_vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "translations_sha256": digest_json(translations), "reranker": reranker.identity, "config": asdict(config),
        "code_sha256": sha256_file(Path(__file__)), "selection_code_sha256": sha256_file(Path(__file__).with_name("retrieval_policy.py"))}
    signature = digest_json(identity)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    contract = identity | {"signature": signature}
    if (root / "config.json").exists() and read_json(root / "config.json") != contract:
        raise ValueError("Strong model/policy/query changed; use a new run.")
    atomic_json(root / "config.json", contract)
    records, written = [], 0
    for query, vector, translation in tqdm(zip(queries, vectors, translations, strict=True), total=len(queries), desc="Strong cascade"):
        path = root / f"query-{query['id']}.done.json"
        if path.exists():
            saved = read_json(path)
            if saved["signature"] != signature or saved["query_sha256"] != digest_json(query) or digest_json({k: v for k, v in saved.items() if k != "record_sha256"}) != saved["record_sha256"]:
                raise ValueError("Strong query checkpoint integrity mismatch.")
            for stage, checksum in saved["score_checkpoints"].items():
                evidence = read_json(root / f"query-{query['id']}-{stage}.done.json")
                if (evidence["record_sha256"] != checksum or digest_json({k: v for k, v in evidence.items() if k != "record_sha256"}) != checksum
                    or evidence["signature"] != signature or evidence["reranker"] != reranker.identity
                    or evidence["pairs_sha256"] != digest_json(evidence["pairs"])
                    or len(evidence["scores"]) != len(evidence["pairs"]) or not np.isfinite(evidence["scores"]).all()):
                    raise ValueError("Strong scored-stage evidence mismatch.")
        else:
            if max_new_queries is not None and written >= max_new_queries:
                break
            variants = translation["variants"]
            documents, unit_scores = index.document_candidates(query["query"], vector, variants, config)
            pairs = []
            for row in documents:
                hits = index.child_candidates(row["doc_id"], query["query"], variants, unit_scores, config, config.summary_children)
                summary_tokenizer = getattr(reranker, "tokenizer", index.tokenizer)
                if hasattr(reranker, "spec"):
                    from .qwen_models import pack_pair
                    budget = reranker.spec["max_length"] - len(pack_pair(summary_tokenizer, query["query"], "", reranker.spec["max_length"])) - 8
                else:
                    budget = pair_budget(summary_tokenizer, query["query"], reranker.identity["max_length"])
                text = document_representation(index.catalog, row["doc_id"], [h["child_id"] for h in hits], summary_tokenizer, budget)
                legs = [("original", query["query"])]
                translated = variants.get("query_" + index.languages[row["doc_id"]])
                if config.auxiliary_rerank_weight and translated:
                    legs.append(("translated", translated))
                for leg, text_query in legs:
                    for window in _windows(reranker, index.tokenizer, text_query, text, config.window_overlap_tokens):
                        pairs.append({"doc_id": row["doc_id"], "query": text_query, "score_leg": leg, **window})
            doc_stage = _score_checkpoint(root / f"query-{query['id']}-documents.done.json", pairs, reranker, signature, batch_size)
            documents = _maxp_rows(documents, pairs, doc_stage["scores"], "doc_id", config)
            children = [row for d in documents[:config.detail_doc_k] for row in index.child_candidates(d["doc_id"], query["query"], variants, unit_scores, config, config.children_per_doc)]
            pairs = []
            for row in children:
                legs = [("original", query["query"])]
                translated = variants.get("query_" + index.languages[row["doc_id"]])
                if config.auxiliary_rerank_weight and translated:
                    legs.append(("translated", translated))
                for leg, text_query in legs:
                    for window in _windows(reranker, index.tokenizer, text_query, index.child_text(row["child_id"]), config.window_overlap_tokens):
                        pairs.append({"child_id": row["child_id"], "query": text_query, "score_leg": leg, **window})
            child_stage = _score_checkpoint(root / f"query-{query['id']}-children.done.json", pairs, reranker, signature, batch_size)
            children = _maxp_rows(children, pairs, child_stage["scores"], "child_id", config)
            budget = config.parent_long_tokens if config.adaptive_parents and is_complex(query["query"]) else config.parent_short_tokens
            for row in children:
                row["parent_id"] = index.catalog.children[row["child_id"]].get("source_parent_alternatives", {}).get(str(budget), index.catalog.children[row["child_id"]]["parent_id"])
            ranking = {"documents": documents, "children": children}
            prediction, provenance = select_parents(index.catalog, ranking, config, index.tokenizer, query["id"])
            saved = {"signature": signature, "query_sha256": digest_json(query), "prediction": prediction,
                "provenance": provenance, "ranking": ranking, "translation_record_sha256": translation["record_sha256"],
                "score_checkpoints": {"documents": doc_stage["record_sha256"], "children": child_stage["record_sha256"]}}
            saved["record_sha256"] = digest_json(saved)
            atomic_json(path, saved)
            written += 1
        records.append(saved)
    report = {"signature": signature, "completed_queries": len(records), "requested_queries": len(queries),
        "state": "COMPLETE" if len(records) == len(queries) else "IN_PROGRESS",
        "evaluation": "NOT_EVALUATED_NO_REFERENCE_LABELS", "scope": "STRONG_ARCHITECTURE_PARTIAL_CORPUS"}
    atomic_json(root / "predictions.json", report)
    return records, report
