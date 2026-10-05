"""Checkpoint interruption, real CPU ranking and source-safe submission contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from types import SimpleNamespace

import numpy as np
import pytest

from vietmedbridge.artifacts import atomic_json, digest_json, read_json
from vietmedbridge.embeddings import embed_units, embedding_matrix
from vietmedbridge.retrieval import HybridIndex, RetrievalConfig, lexical_tokens, predict_queries
from vietmedbridge.retrieval_data import Catalog, load_catalog
from vietmedbridge.submission import export_submission, validate_submission


class Encoder:
    dimension = 3
    identity = {"model": "offline-test-double", "pooling": "normalized", "dimension": 3}

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def encode(self, texts, *, batch_size):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated disconnect")
        return np.array([[1, 0, 0] if "HbA1c" in text else [0, 1, 0] for text in texts], dtype=np.float32)


class Reranker:
    identity = {"model": "offline-test-double", "fine_tuned": False}

    def __init__(self):
        self.calls = 0

    def score(self, pairs, *, batch_size):
        self.calls += 1
        return np.array([10 if "HbA1c" in p else 2 for q, p in pairs], dtype=np.float32)


@pytest.fixture
def catalog():
    documents, children, parents = {}, {}, {}
    alpha = "HbA1c điều trị diabetes ở người lớn.\n\nKết quả được theo dõi."
    beta = "病毒感染 cần xét nghiệm virus.\n\nĐọc kết quả xét nghiệm."
    aliases = {"a": [], "b": []}
    for doc_id, text, unit in ((583, alpha, "a"), (620, alpha, "a"), (991, beta, "b")):
        checksum = hashlib.sha256(text.encode()).hexdigest()
        documents[doc_id] = {"doc_id": doc_id, "source_text": text, "source_text_sha256": checksum}
        parent_id, child_id = f"parent-{doc_id}", f"child-{doc_id}"
        parent = {"chunk_id": parent_id, "doc_id": doc_id, "start_char": 0, "end_char": len(text),
                  "source_text_sha256": checksum, "text": text}
        parents[parent_id] = parent
        end = text.index("\n\n")
        children[child_id] = {**parent, "chunk_id": child_id, "parent_id": parent_id,
                              "text": text[:end], "end_char": end}
        aliases[unit].append(child_id)
    return Catalog({"snapshot_sha256": "fixture", "candidate_manifest_sha256": "fixture"},
                   documents, children, parents,
                   [{"id": "a", "text": alpha.split("\n\n")[0]},
                    {"id": "b", "text": beta.split("\n\n")[0]}], aliases)


def pipeline(tmp_path, catalog):
    encoder = Encoder()
    cm = embed_units(catalog.units, encoder, tmp_path / "corpus", part_size=1, work_dir=tmp_path)
    index = HybridIndex(catalog, embedding_matrix(tmp_path / "corpus", cm), cm,
                        tmp_path / "index", work_dir=tmp_path)
    queries = [{"id": 19, "query": "HbA1c điều trị"}, {"id": 1037, "query": "病毒感染"}]
    units = [{"id": q["id"], "text": q["query"]} for q in queries]
    qm = embed_units(units, encoder, tmp_path / "qvec", part_size=1, work_dir=tmp_path)
    vectors = embedding_matrix(tmp_path / "qvec", qm)
    return index, queries, qm, vectors


def test_embedding_disconnect_resume_and_corruption(tmp_path):
    units = [{"id": n, "text": f"HbA1c {n}"} for n in range(5)]
    encoder = Encoder(fail_at=2)
    with pytest.raises(RuntimeError, match="disconnect"):
        embed_units(units, encoder, tmp_path, part_size=2, work_dir=tmp_path)
    assert len(list(tmp_path.glob("*.done.json"))) == 1
    # A file published before its final marker is not a committed checkpoint.
    (tmp_path / "part-0000000002-00002.npy").write_bytes(b"incomplete")
    resumed = Encoder()
    manifest = embed_units(units, resumed, tmp_path, part_size=2, batch_size=1, work_dir=tmp_path)
    assert resumed.calls == 2 and manifest["state"] == "COMPLETE"
    assert embedding_matrix(tmp_path, manifest).shape == (5, 3)
    repeat = Encoder()
    assert embed_units(units, repeat, tmp_path, part_size=2, work_dir=tmp_path) == manifest
    assert repeat.calls == 0
    with pytest.raises(ValueError, match="changed"):
        embed_units(list(reversed(units)), Encoder(), tmp_path, part_size=2)
    (tmp_path / manifest["parts"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed artifact"):
        embed_units(units, Encoder(), tmp_path, part_size=2)


def test_partial_embeddings_cannot_be_indexed(tmp_path):
    units = [{"id": n, "text": f"item {n}"} for n in range(3)]
    manifest = embed_units(units, Encoder(), tmp_path, part_size=1, max_new_parts=1, work_dir=tmp_path)
    assert manifest["state"] == "IN_PROGRESS"
    with pytest.raises(ValueError, match="Resume all"):
        embedding_matrix(tmp_path, manifest)


def test_oom_batch_backoff_preserves_order_and_next_part_limit():
    from vietmedbridge.retrieval_models import _TorchInference

    class OOM(Exception):
        pass

    model = _TorchInference()
    attempts, cleared = [], []
    model.torch = SimpleNamespace(OutOfMemoryError=OOM,
        cuda=SimpleNamespace(empty_cache=lambda: cleared.append(True)))
    model.device, model.oom_backoffs = "cuda", 0
    def compute(items):
        attempts.append(len(items))
        if len(items) > 2:
            raise OOM("simulated CUDA OOM")
        return np.array(items, dtype=np.float32)
    model._batch = compute
    assert model._batches(list(range(5)), 8).tolist() == list(range(5))
    assert model.oom_backoffs == len(cleared) == 2
    attempts.clear()
    assert model._batches([5, 6, 7], 8).tolist() == [5, 6, 7]
    assert attempts == [2, 1]  # A new part does not repeat known unsafe batches.
    model.device = "cpu"
    model.safe_batch_size = 8
    with pytest.raises(OOM):
        model._batches(list(range(5)), 8)


def test_real_hybrid_index_aliases_and_independent_output_limits(tmp_path, catalog):
    index, queries, qm, vectors = pipeline(tmp_path, catalog)
    assert {"病", "毒", "病毒", "HbA1c".casefold()} <= set(lexical_tokens("病毒 HbA1c"))
    config = RetrievalConfig(doc_top_k=3, chunk_top_k=1)
    assert index.candidates("HbA1c", vectors[0], config)[0] == 0
    records, report = predict_queries(index, queries, vectors, Reranker(), tmp_path / "queries",
        config=config, query_embedding_manifest=qm)
    p = records[0]["prediction"]
    assert p["relevant_docs"][:2] == [583, 620]  # Dedup must not discard official aliases.
    assert len(p["relevant_docs"]) == 3 and len(p["relevant_chunks"]) == 1
    assert p["relevant_chunks"][0]["chunk_text"] == catalog.documents[583]["source_text"]
    assert "Kết quả" in p["relevant_chunks"][0]["chunk_text"]  # Parent expansion.
    validated = validate_submission(records, queries, catalog, report, expected_count=2)
    assert validated[0] == p
    cm = read_json(tmp_path / "corpus/embeddings.json")
    reused = HybridIndex(catalog, embedding_matrix(tmp_path / "corpus", cm), cm, tmp_path / "index")
    assert reused.manifest == index.manifest
    with pytest.raises(ValueError, match="ordering"):
        other = deepcopy(catalog)
        other.units.reverse()
        HybridIndex(other, embedding_matrix(tmp_path / "corpus", cm), cm, tmp_path / "other-index")


def test_query_resume_and_reordered_vectors_rejected(tmp_path, catalog):
    index, queries, qm, vectors = pipeline(tmp_path, catalog)
    first = Reranker()
    records, report = predict_queries(index, queries, vectors, first, tmp_path / "pred",
        query_embedding_manifest=qm, max_new_queries=1)
    assert first.calls == 1 and report["state"] == "IN_PROGRESS"
    with pytest.raises(ValueError, match="Finish every"):
        validate_submission(records, queries, catalog, report, expected_count=2)
    reranker = Reranker()
    records, report = predict_queries(index, queries, vectors, reranker, tmp_path / "pred",
        query_embedding_manifest=qm)
    assert reranker.calls == 1 and report["state"] == "COMPLETE"
    cached = Reranker()
    predict_queries(index, queries, vectors, cached, tmp_path / "pred", query_embedding_manifest=qm)
    assert cached.calls == 0
    with pytest.raises(ValueError, match="official query order"):
        predict_queries(index, list(reversed(queries)), vectors[::-1], Reranker(), tmp_path / "bad", query_embedding_manifest=qm)
    altered = Reranker()
    altered.identity = {"model": "another model"}
    with pytest.raises(ValueError, match="policy changed"):
        predict_queries(index, queries, vectors, altered, tmp_path / "pred", query_embedding_manifest=qm)
    checkpoint = tmp_path / "pred/query-19.done.json"
    saved = read_json(checkpoint)
    saved["prediction"]["relevant_docs"] = [123]
    atomic_json(checkpoint, saved)
    with pytest.raises(ValueError, match="checkpoint changed"):
        predict_queries(index, queries, vectors, Reranker(), tmp_path / "pred", query_embedding_manifest=qm)


def test_full_1200_query_zip_and_source_mutation_rejected(tmp_path, catalog):
    index, queries, qm, vectors = pipeline(tmp_path, catalog)
    original, _ = predict_queries(index, queries, vectors, Reranker(), tmp_path / "pred", query_embedding_manifest=qm)
    queries = [{"id": 7 + n * 23, "query": f"HbA1c query {n}"} for n in range(1200)]
    records = []
    for q in queries:
        row = deepcopy(original[0])
        row["prediction"]["id"] = q["id"]
        row["query_sha256"] = digest_json(q)
        row["record_sha256"] = digest_json({k: v for k, v in row.items() if k != "record_sha256"})
        records.append(row)
    report = {"state": "COMPLETE", "signature": records[0]["signature"], "completed_queries": 1200}
    export = export_submission(records, queries, catalog, report, tmp_path / "submission",
                               evidence={"test": "offline"}, work_dir=tmp_path)
    assert export["query_count"] == 1200 and export["fine_tuned"] is False
    with ZipFile(tmp_path / "submission/submission.zip") as archive:
        assert archive.namelist() == ["results.json"]
        data = json.loads(archive.read("results.json"))
        assert len(data) == 1200 and [p["id"] for p in data] == [q["id"] for q in queries]
        assert set(data[0]["relevant_chunks"][0]) == {"doc_id", "chunk_text"}
    records[0]["prediction"]["relevant_chunks"][0]["chunk_text"] += " invented text"
    records[0]["record_sha256"] = digest_json({k: v for k, v in records[0].items() if k != "record_sha256"})
    with pytest.raises(ValueError, match="exact frozen source"):
        validate_submission(records, queries, catalog, report)
    records[0]["prediction"]["relevant_docs"] = [True]
    records[0]["record_sha256"] = digest_json({k: v for k, v in records[0].items() if k != "record_sha256"})
    with pytest.raises(ValueError, match="official integer IDs"):
        validate_submission(records, queries, catalog, report)


def test_frozen_catalog_reader_and_artifact_checksum(tmp_path):
    from test_supplement import make_build, Tokenizer, SPEC, FIXTURES
    from vietmedbridge.golden import run_golden_suite
    from vietmedbridge.health import health_report, freeze_candidate

    links, raw, root, _ = make_build(tmp_path)
    health = health_report(root, official_links=links, crawl_dir=raw, work_dir=tmp_path)
    candidate = freeze_candidate(root, health, golden_report=run_golden_suite(FIXTURES, Tokenizer(), SPEC))
    filename = f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json"
    catalog = load_catalog(root, filename, Tokenizer())
    assert set(catalog.documents) == {500, 537}
    assert sum(map(len, catalog.aliases.values())) == len(catalog.children)
    with pytest.raises(ValueError, match="memory limit"):
        load_catalog(root, filename, Tokenizer(), max_documents=1)
    with pytest.raises(ValueError, match="escapes"):
        load_catalog(root, "../outside.json", Tokenizer())
    file = root / candidate["parts"][0]["files"]["children"]["path"]
    file.write_bytes(b"broken")
    with pytest.raises(ValueError, match="changed artifact"):
        load_catalog(root, filename, Tokenizer())
