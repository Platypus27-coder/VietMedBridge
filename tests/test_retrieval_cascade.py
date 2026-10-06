"""Real CPU indices and plan contracts; no downloaded weights/GPU inference."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import re

import numpy as np
import pytest

from test_retrieval import Encoder, Reranker, catalog
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.embeddings import embed_units, embedding_matrix
from vietmedbridge.query_translation import cached_translations, translate_queries, validate_variants
from vietmedbridge.retrieval_cache import reuse_bge_cache
from vietmedbridge.retrieval_cascade import (CascadeConfig, CascadeIndex, PostingBM25,
    document_representation, passage_windows, predict_cascade, weighted_rrf)
from vietmedbridge.retrieval_eval import cutoff_sweep, evaluate_candidate_recall, evaluate_predictions, f2
from vietmedbridge.retrieval_policy import lcs_length, lcs_union, score_tokens, select_parents
from vietmedbridge.submission import export_submission


class Tokenizer:
    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        return {"input_ids": [hashlib.sha256(m.group().encode()).hexdigest() for m in matches],
                "offset_mapping": [(m.start(), m.end()) for m in matches]}

    def num_special_tokens_to_add(self, *, pair):
        return 4 if pair else 2


class Translator:
    identity = {"model": "offline-test-double", "precision": "test"}

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def translate(self, query, prompt):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated disconnect")
        return json.dumps({"original_vi":query, "entities":[], "constraints":[],
                           "query_en":query, "query_zh":"病毒感染 " + query}, ensure_ascii=False)


class CascadeReranker(Reranker):
    identity = {**Reranker.identity, "max_length": 512}

    def __init__(self, fail_at=None):
        super().__init__()
        self.fail_at = fail_at

    def score(self, pairs, *, batch_size):
        if self.calls + 1 == self.fail_at:
            self.calls += 1
            raise RuntimeError("simulated disconnect")
        return super().score(pairs, batch_size=batch_size)


def pipeline(tmp_path, catalog):
    encoder = Encoder()
    cm = embed_units(catalog.units, encoder, tmp_path / "corpus", part_size=1)
    vectors = embedding_matrix(tmp_path / "corpus", cm)
    index = CascadeIndex(catalog, vectors, cm, tmp_path / "index", Tokenizer())
    queries = [{"id": 123, "query": "HbA1c điều trị"}, {"id": 900, "query": "病毒感染"}]
    qm = embed_units([{"id":q["id"],"text":q["query"]} for q in queries], encoder,
                     tmp_path / "qvectors", part_size=1)
    qv = embedding_matrix(tmp_path / "qvectors", qm)
    translations, _ = translate_queries(queries, Translator(), tmp_path / "translations")
    config = CascadeConfig(doc_candidate_k=3, sparse_top_k=3, detail_doc_k=3,
        doc_top_k=3, doc_min_k=1, chunk_top_k=3, chunk_min_k=1, children_per_doc=1)
    return index, queries, qm, qv, translations, config


def test_postings_positive_bm25_no_zero_matches():
    bm25 = PostingBM25(["HbA1c", "HbA1c diabetes", "病毒感染"])
    scores = bm25.scores("HbA1c")
    assert scores[0] > scores[1] > scores[2] == 0
    assert not bm25.scores("SCC").any()
    assert PostingBM25([]).scores("query").size == 0


def test_weighted_rrf_uses_ranks_and_deduplicates_documents():
    scores = weighted_rrf({"dense":[1,2], "vi":[2]}, {"dense":1, "vi":.8}, 60)
    assert scores[2] == pytest.approx(1/62 + .8/61)
    assert scores[2] > scores[1]
    with pytest.raises(ValueError, match="duplicate"):
        weighted_rrf({"dense":[1,1]}, {"dense":1}, 60)


def test_translation_disables_only_invalid_branch_and_preserves_query():
    query = "HbA1c > 7 ở người mang thai không dùng thuốc"
    value = {"original_vi":query, "entities":["HbA1c"], "constraints":["mang thai", "không"],
             "query_en":"HbA1c > 7 during pregnancy without medicine",
             "query_zh":"妊娠期间 HbA1c > 8 未用药"}
    variants = validate_variants(query, value)
    assert variants["original_vi"] == query and variants["query_en"] == value["query_en"]
    assert variants["query_zh"] is None and "NUMBER_CHANGED_OR_ADDED" in variants["rejections"]["zh"]
    for original, translated in (("Yqh+", "Yqh"), ("PDW", "PCT")):
        broken = {"original_vi":original, "entities":[], "constraints":[],
                  "query_en":translated, "query_zh":"检测 " + translated}
        assert validate_variants(original, broken)["query_en"] is None
    assert validate_variants(query, "bad JSON")["query_en"] is None


def test_translation_resume_and_tamper(tmp_path):
    queries = [{"id":1,"query":"HbA1c"}, {"id":2,"query":"病毒感染"}]
    with pytest.raises(RuntimeError, match="disconnect"):
        translate_queries(queries, Translator(fail_at=2), tmp_path)
    translator = Translator()
    records, report = translate_queries(queries, translator, tmp_path)
    assert report["state"] == "COMPLETE" and translator.calls == 1
    assert cached_translations(queries, {}, tmp_path) == (records, report)
    record = read_json(tmp_path / "query-1.done.json")
    record["variants"]["original_vi"] = "changed"
    atomic_json(tmp_path / "query-1.done.json", record)
    with pytest.raises(ValueError, match="integrity"):
        translate_queries(queries, translator, tmp_path)
    with pytest.raises(ValueError, match="integrity"):
        cached_translations(queries, {}, tmp_path)


def test_cascade_document_aliases_and_stage_resume(tmp_path, catalog):
    index, queries, qm, qv, translations, config = pipeline(tmp_path, catalog)
    rows, _ = index.document_candidates(queries[0]["query"], qv[0], translations[0]["variants"], config)
    assert {row["doc_id"] for row in rows} == {583, 620, 991}
    assert len(rows) == len({row["doc_id"] for row in rows})
    failed = CascadeReranker(fail_at=2)
    with pytest.raises(RuntimeError, match="disconnect"):
        predict_cascade(index, queries, qv, failed, tmp_path / "pred", query_embedding_manifest=qm,
                        translations=translations, config=config)
    assert (tmp_path / "pred/query-123-documents.done.json").exists()
    fresh = CascadeReranker()
    records, report = predict_cascade(index, queries, qv, fresh, tmp_path / "pred", query_embedding_manifest=qm,
                        translations=translations, config=config)
    assert report["state"] == "COMPLETE" and fresh.calls == 3  # cached doc stage; then 1+2 stages
    assert {583,620} <= set(records[0]["prediction"]["relevant_docs"])
    assert {583,620} <= {c["doc_id"] for c in records[0]["prediction"]["relevant_chunks"]}
    export_submission(records, queries, catalog, report, tmp_path / "submission", evidence={}, expected_count=2)
    resumed = CascadeReranker()
    predict_cascade(index, queries, qv, resumed, tmp_path / "pred", query_embedding_manifest=qm,
                    translations=translations, config=config)
    assert resumed.calls == 0
    stage = tmp_path / "pred/query-123-documents.done.json"
    value = read_json(stage)
    value["scores"][0] += 1
    atomic_json(stage, value)
    with pytest.raises(ValueError, match="evidence"):
        predict_cascade(index, queries, qv, resumed, tmp_path / "pred", query_embedding_manifest=qm,
                        translations=translations, config=config)


def test_cascade_policy_change_rejected_and_quota(tmp_path, catalog):
    index, queries, qm, qv, translations, config = pipeline(tmp_path, catalog)
    records, report = predict_cascade(index, queries, qv, CascadeReranker(), tmp_path / "pred",
        query_embedding_manifest=qm, translations=translations, config=config, max_new_queries=1)
    assert len(records) == 1 and report["state"] == "IN_PROGRESS"
    ranking = records[0]["ranking"]
    assert all(sum(c["doc_id"] == d for c in ranking["children"]) <= config.children_per_doc for d in catalog.documents)
    with pytest.raises(ValueError, match="selection changed"):
        predict_cascade(index, queries, qv, CascadeReranker(), tmp_path / "pred", query_embedding_manifest=qm,
                        translations=translations, config=replace(config, vi_weight=.9))


def test_partial_translation_canary_resumes_with_same_signature(tmp_path, catalog):
    index, queries, qm, qv, translations, config = pipeline(tmp_path, catalog)
    _, partial = predict_cascade(index, queries, qv, CascadeReranker(), tmp_path / "pred",
        query_embedding_manifest=qm, translations=translations[:1], config=config)
    resumed = CascadeReranker()
    records, complete = predict_cascade(index, queries, qv, resumed, tmp_path / "pred",
        query_embedding_manifest=qm, translations=translations, config=config)
    assert partial["state"] == "IN_PROGRESS" and complete["state"] == "COMPLETE"
    assert partial["signature"] == complete["signature"] and resumed.calls == 2
    labels = {"split":"dev", "reviewed":True, "queries":[{"id":123,
        "relevant_docs":[583], "relevant_chunks":[{"doc_id":583,"chunk_text":catalog.parents["parent-583"]["text"]}]}]}
    sweep = cutoff_sweep(records, labels, catalog, index.tokenizer, config)
    assert len(sweep["trials"]) == 12 and sweep["best_dev_policy"]["combined_f2"] > 0
    assert not sweep["auto_applied_to_submission"]
    recall = evaluate_candidate_recall(records, labels["queries"], catalog, index.tokenizer)
    assert recall["frozen_document_id_coverage_macro"] == recall["fused_document_pool_recall_macro"] == recall["child_parent_pool_recall_macro"] == 1


def test_source_quarantine_keeps_frozen_ids(tmp_path, catalog):
    catalog.documents[991].update(url="https://example.org/article", final_url="https://example.org/")
    index, queries, qm, qv, translations, config = pipeline(tmp_path, catalog)
    assert 991 in catalog.documents and 991 not in index.doc_ids
    assert index.exclusions[991] == "ARTICLE_REDIRECTED_TO_HOMEPAGE"
    rows, _ = index.document_candidates(queries[0]["query"], qv[0], translations[0]["variants"], config)
    assert 991 not in {row["doc_id"] for row in rows}
    assert read_json(tmp_path / "index/quarantine.json")["frozen_documents"] == 3


def test_index_reopens_after_json_roundtrip_with_different_length_ids(tmp_path, catalog):
    catalog.documents[10000] = {**catalog.documents[583], "doc_id":10000}
    catalog.parents["parent-10000"] = {**catalog.parents["parent-583"], "doc_id":10000,"chunk_id":"parent-10000"}
    catalog.children["child-10000"] = {**catalog.children["child-583"], "doc_id":10000,"chunk_id":"child-10000","parent_id":"parent-10000"}
    catalog.aliases["a"].append("child-10000")
    first, *_ = pipeline(tmp_path, catalog)
    encoder = Encoder()
    cm = read_json(tmp_path / "corpus/embeddings.json")
    second = CascadeIndex(catalog, embedding_matrix(tmp_path / "corpus", cm), cm,
                          tmp_path / "index", Tokenizer())
    assert first.manifest == second.manifest
    assert first.manifest == read_json(tmp_path / "index/cascade.json")


def test_content_language_routing_ignores_bad_declared_label(tmp_path, catalog):
    catalog.documents[991]["source_text"] = "病毒感染病毒感染病毒感染病毒感染病毒感染病毒感染病毒感染病毒感染" * 5
    catalog.documents[991]["language"] = "en"
    index, *_ = pipeline(tmp_path, catalog)
    assert index.languages[991] == "zh"
    assert 991 in index.sparse["zh"][0] and 991 not in index.sparse["en"][0]


def test_document_two_hits_and_maxp_tail(catalog):
    second = deepcopy(catalog.children["child-583"])
    second["text"] = "tail evidence " * 30
    catalog.children["second"] = second
    catalog.documents[583]["title"] = "title " * 100
    tokenizer = Tokenizer()
    text = document_representation(catalog, 583, ["child-583", "second"], tokenizer, 40)
    assert "HbA1c" in text and "tail evidence" in text
    assert len(tokenizer(text)["input_ids"]) <= 40
    windows = passage_windows(tokenizer, "head " * 50 + "tail evidence", 20, 5)
    assert "tail evidence" in windows[-1]["text"]
    assert all(len(tokenizer(w["text"])["input_ids"]) <= 20 for w in windows)


def test_bit_lcs_matches_dynamic_programming():
    rng = np.random.default_rng(1234)
    for _ in range(100):
        left, right = rng.integers(0,4,20).tolist(), rng.integers(0,4,25).tolist()
        previous = [0] * (len(right) + 1)
        for x in left:
            row = [0]
            for j,y in enumerate(right,1):
                row.append(previous[j-1] + 1 if x == y else max(previous[j],row[-1]))
            previous = row
        assert lcs_length(left,right) == previous[-1]
    assert lcs_union([1,2],[1,2]) == 1


def test_parent_dedup_same_doc_uses_tokens_not_offsets(catalog):
    original = catalog.parents["parent-583"]
    twin = {**original, "chunk_id":"twin", "start_char":500, "end_char":500 + len(original["text"])}
    catalog.parents["twin"] = twin
    catalog.children["twin-anchor"] = {**catalog.children["child-583"], "parent_id":"twin"}
    ranking = {"documents":[{"doc_id":583,"reranker_score":5},{"doc_id":620,"reranker_score":4}],
               "children":[{"child_id":c,"reranker_score":s} for c,s in (("child-583",5),("twin-anchor",4),("child-620",3))]}
    prediction,_ = select_parents(catalog, ranking, CascadeConfig(), Tokenizer(), 1)
    assert [r["doc_id"] for r in prediction["relevant_chunks"]] == [583,620]


def test_f2_macro_per_query_and_same_doc_lcs():
    labels = [{"id":1,"relevant_docs":[1],"relevant_chunks":[{"doc_id":1,"chunk_text":"a b c d e"}]},
              {"id":2,"relevant_docs":[2],"relevant_chunks":[{"doc_id":2,"chunk_text":"f g h i j"}]},
              {"id":3,"relevant_docs":[],"relevant_chunks":[]}]
    predictions = [{"id":1,"relevant_docs":[1,9],"relevant_chunks":[{"doc_id":1,"chunk_text":"a b"}, {"doc_id":1,"chunk_text":"a b"}]},
                   {"id":2,"relevant_docs":[],"relevant_chunks":[{"doc_id":99,"chunk_text":"f g h i j"}]},
                   {"id":3,"relevant_docs":[9],"relevant_chunks":[]}]
    report = evaluate_predictions(predictions, labels, Tokenizer())
    assert report["document_queries"] == report["chunk_queries"] == 2
    assert report["documents_macro"]["f2"] == pytest.approx(f2(.5,1) / 2)
    assert report["chunks_macro"]["f2"] == .5
    assert report["per_query"][0]["chunks"]["precision"] == 1  # duplicates merged
    assert report["per_query"][1]["chunks"]["recall"] == 0  # wrong doc never matches


def test_cutoff_sweep_forbids_test_labels(tmp_path, catalog):
    with pytest.raises(ValueError, match="reviewed dev"):
        cutoff_sweep([], {"split":"test", "reviewed":True}, catalog, Tokenizer(), CascadeConfig())


def test_verified_vector_cache_reuse_and_corruption(tmp_path):
    spec = {"model_id":"BAAI/bge-m3", "revision":"a"*40, "max_length":512}
    encoder = Encoder()
    encoder.identity = {**encoder.identity, **spec, "pooling":"cls-l2-v1", "truncation":False,
                        "query_instruction":None,"precision":"torch.float16","device_class":"cuda",
                        "inference_code_sha256":sha256_file(__import__("vietmedbridge.retrieval_models", fromlist=["_"]).__file__)}
    units = [{"id":1,"text":"HbA1c"}]
    original = embed_units(units, encoder, tmp_path, part_size=1)
    vectors, manifest = reuse_bge_cache(tmp_path, units, spec, part_size=1)
    assert manifest == original and vectors.shape == (1,3)
    assert encoder.calls == 1
    with pytest.raises(ValueError, match="inputs/order"):
        reuse_bge_cache(tmp_path, [{"id":1,"text":"changed"}], spec, part_size=1)
    (tmp_path / manifest["parts"][0]["path"]).write_bytes(b"broken")
    with pytest.raises(ValueError):
        reuse_bge_cache(tmp_path, units, spec, part_size=1)
