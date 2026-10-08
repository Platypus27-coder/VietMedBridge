"""Reuse immutable artifacts, fail on corruption and stop review before GPU work."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.embeddings import embed_units
from vietmedbridge.shared_embeddings import find_embeddings
from vietmedbridge.training_mining import cached_mining_records, retrieve_mining_candidates, unique_finalists
from vietmedbridge.strong_retrieval import StrongConfig
from vietmedbridge.negative_review import install_ai_candidate_review
from vietmedbridge.training_data import _seal
from vietmedbridge import retrieval_models
from vietmedbridge.runtime_profile import inference_batches


class Encoder:
    dimension = 2
    def __init__(self, spec):
        self.identity = {**spec, "dimension": 2, "pooling": "cls-l2-v1", "truncation": False,
            "query_instruction": None, "precision": "torch.float16", "device_class": "cuda",
            "inference_code_sha256": sha256_file(retrieval_models.__file__)}
    def encode(self, texts, **kwargs):
        return np.tile(np.array([[1., 0.]], np.float32), (len(texts), 1))


@pytest.mark.parametrize("memory_gib,batch", [(15, 4), (24, 8), (40, 16), (80, 16)])
def test_gpu_inference_batch_policy(memory_gib, batch):
    policy = inference_batches(memory_gib * 2**30)
    assert policy["embedding"] == policy["reranker"] == batch
    assert policy["oom_policy"] == "halve-and-retry"


def test_shared_corpus_from_04_and_training_keeps_original_producer_and_partition(tmp_path):
    units = [{"id": "a", "text": "source A"}, {"id": "b", "text": "source B"}]
    spec = {"model_id": "CPU-double", "revision": "fixed", "max_length": 512}
    original = tmp_path / "retrieval/04/corpus_embeddings"
    manifest = embed_units(units, Encoder(spec), original, part_size=1, work_dir=tmp_path / "work")
    found = find_embeddings(tmp_path, units, spec, family="bge")
    assert found[1] == manifest and found[2] == original.resolve()
    assert found[1]["part_size"] == 1
    assert not (tmp_path / "training/05/corpus_embeddings").exists()
    # A different query set doesn't invalidate an unchanged corpus producer.
    assert find_embeddings(tmp_path, units, spec, family="bge")[1] == manifest
    assert find_embeddings(tmp_path, units[::-1], spec, family="bge") is None
    assert find_embeddings(tmp_path, units, {**spec, "revision": "other"}, family="bge") is None
    marker = original / "part-0000000000-00001.done.json"
    value = read_json(marker); value["sha256"] = "tampered"; atomic_json(marker, value)
    with pytest.raises(ValueError, match="completion marker"):
        find_embeddings(tmp_path, units, spec, family="bge")


def test_matching_shared_vector_corruption_is_not_hidden(tmp_path):
    units, spec = [{"id": "a", "text": "A"}], {"model_id": "CPU", "revision": "fixed"}
    original = tmp_path / "training/05/experiments/one/corpus_embeddings"
    embed_units(units, Encoder(spec), original, work_dir=tmp_path / "work")
    (original / "part-0000000000-00001.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        find_embeddings(tmp_path, units, spec, family="bge")


def legacy_mining(tmp_path):
    queries = [{"id": 123, "query": "Source question?"}]
    catalog = SimpleNamespace(identity={"candidate": "fixed"}, children={"child": {"doc_id": 1}})
    policy = StrongConfig()
    index = {"catalog": catalog.identity, "signature": "index"}
    index["manifest_sha256"] = digest_json(index)
    atomic_json(tmp_path / "index/strong.json", index)
    contract = {"index": "index", "queries_sha256": digest_json(queries), "config": asdict(policy),
                "reranker": {"model_id": "CPU-test"},
                "code_sha256": sha256_file(Path(__file__).parents[1] / "src/vietmedbridge/strong_retrieval.py"),
                "selection_code_sha256": sha256_file(Path(__file__).parents[1] / "src/vietmedbridge/retrieval_policy.py")}
    contract["signature"] = digest_json(contract)
    atomic_json(tmp_path / "mining_queries/config.json", contract)
    evidence = {"signature": contract["signature"], "reranker": contract["reranker"], "pairs": [], "scores": [], "pairs_sha256": digest_json([])}
    evidence["record_sha256"] = digest_json(evidence)
    atomic_json(tmp_path / "mining_queries/query-123-children.done.json", evidence)
    record = {"signature": contract["signature"], "query_sha256": digest_json(queries[0]), "prediction": {"id": 123},
              "ranking": {"children": [{"child_id": "child", "doc_id": 1}]}, "score_checkpoints": {"children": evidence["record_sha256"]}}
    record["record_sha256"] = digest_json(record)
    atomic_json(tmp_path / "mining_queries/query-123.done.json", record)
    return queries, catalog, policy, record


def test_paid_mining_replays_without_model_and_checks_scores_and_query(tmp_path):
    queries, catalog, policy, record = legacy_mining(tmp_path)
    assert cached_mining_records(tmp_path, queries, catalog, policy) == [record]
    with pytest.raises(ValueError, match="source/query/policy"):
        cached_mining_records(tmp_path, [{"id": 123, "query": "Changed"}], catalog, policy)
    path = tmp_path / "mining_queries/query-123-children.done.json"
    value = read_json(path); value["scores"] = [0.1]; atomic_json(path, value)
    with pytest.raises(ValueError, match="scored-stage"):
        cached_mining_records(tmp_path, queries, catalog, policy)


def test_first_mining_uses_wide_retrieval_and_does_not_invent_labels(tmp_path):
    class Index:
        manifest = {"signature": "CPU-index"}
        def document_candidates(self, *args):
            return [{"doc_id": 1}, {"doc_id": 2}], {}
        def child_candidates(self, doc, *args):
            return [{"doc_id": doc, "child_id": f"child-{doc}"}]
    queries = [{"id": 3, "query": "Question?"}]
    records = retrieve_mining_candidates(Index(), queries, [[1, 0]], [{"variants": {}}], StrongConfig(), tmp_path)
    assert {r["doc_id"] for r in records[0]["ranking"]["children"]} == {1, 2}
    assert records[0]["scope"] == "MULTICHANNEL_RETRIEVAL_NO_RERANK"
    assert all("label" not in r for r in records[0]["ranking"]["children"])
    assert retrieve_mining_candidates(Index(), queries, [[1, 0]], [{"variants": {}}], StrongConfig(), tmp_path) == records


def test_duplicate_final_adapter_is_evaluated_once(tmp_path):
    paths = [tmp_path / "epoch1", tmp_path / "epoch2", tmp_path / "final"]
    for path, weight in zip(paths, ("A", "B", "B")):
        path.mkdir()
        (path / "adapter_model.safetensors").write_text(weight)
        (path / "adapter_config.json").write_text("C")
        manifest = {"base_model": "M", "base_revision": "R", "files": {name: sha256_file(path / name)
            for name in ("adapter_model.safetensors", "adapter_config.json")}}
        manifest["manifest_sha256"] = digest_json(manifest)
        atomic_json(path / "adapter_manifest.json", manifest)
    assert unique_finalists(paths) == paths[:2]
    (paths[-1] / "adapter_model.safetensors").write_text("changed")
    with pytest.raises(ValueError):
        unique_finalists(paths)


def test_shared_qwen_preserves_role_instruction_and_original_manifest(tmp_path):
    from vietmedbridge import qwen_models
    units = [{"id": "source", "text": "Vietnamese source text"}]
    spec = {"model_id": qwen_models.EMBEDDING_ID, "revision": qwen_models.EMBEDDING_REVISION,
            "max_length": 1536, "quantization": "nf4"}
    class Secondary:
        dimension = qwen_models.EMBEDDING_DIMENSION
        identity = {**spec, "dimension": qwen_models.EMBEDDING_DIMENSION, "input_role": "corpus",
            "query_instruction": qwen_models.QUERY_INSTRUCTION, "fine_tuned": False,
            "pooling": "last-attended-token-l2", "truncation": False, "role": "second_dense",
            "precision": "torch.float16", "device_class": "cuda",
            "inference_code_sha256": sha256_file(qwen_models.__file__)}
        def encode(self, texts, **kwargs):
            result = np.zeros((len(texts), self.dimension), np.float32)
            result[:, 0] = 1
            return result
    producer = tmp_path / "retrieval/04/qwen_corpus"
    manifest = embed_units(units, Secondary(), producer, work_dir=tmp_path / "work")
    reused = find_embeddings(tmp_path, units, spec, family="qwen")
    assert reused[1] == manifest and reused[2] == producer.resolve()
    assert find_embeddings(tmp_path, units, spec, family="qwen", role="query") is None
    marker = next(producer.glob("*.done.json"))
    value = read_json(marker); value["sha256"] = "changed"; atomic_json(marker, value)
    with pytest.raises(ValueError, match="completion marker"):
        find_embeddings(tmp_path, units, spec, family="qwen")


def review_fixture(tmp_path,scoped=False):
    base = tmp_path / "training/fixture"
    if scoped:
        base = base / "corpora/new-corpus"
    item = {"pair_id": "pair", "query_id": 1, "child_id": "child", "text": "Original source"}
    sources = _seal({"items": [item]})
    directory = base / "candidate_reviews/source"
    atomic_json(directory / "sources.json", sources)
    atomic_json(base / "candidate_review_current.json", {"directory": "candidate_reviews/source", "source_sha256": sources["sha256"]})
    original = {"source_sha256": sources["sha256"], "items": [item | {"review": {"judgment": "PENDING"}}]}
    atomic_json(directory / "review.json", original)
    incoming = {**deepcopy(original), "review_mode": "AI_ASSISTED_PILOT", "human_validated": False,
                "review_authorization": "USER_DELEGATED_TO_CODEX_2026_10_06"}
    incoming["items"][0]["review"] = {"judgment": "NEGATIVE", "reviewer": "Codex-AI", "reviewer_type": "AI"}
    upload = tmp_path / "upload.json"; atomic_json(upload, incoming)
    return upload, directory, incoming


@pytest.mark.parametrize("scoped",[False,True])
def test_candidate_upload_checks_sources_backs_up_and_preserves_team(tmp_path,scoped):
    upload, directory, incoming = review_fixture(tmp_path,scoped)
    result = install_ai_candidate_review(upload, tmp_path, run_name="fixture")
    assert result["judgments"] == {"NEGATIVE": 1} and Path(result["backup_path"]).is_file()
    existing = read_json(directory / "review.json")
    existing["items"][0]["review"] = {"judgment": "POSITIVE", "reviewer": "Team"}
    atomic_json(directory / "review.json", existing)
    with pytest.raises(ValueError, match="conflicts"):
        install_ai_candidate_review(upload, tmp_path, run_name="fixture")
    incoming["items"][0]["review"] = {"judgment": "PENDING"}; atomic_json(upload, incoming)
    install_ai_candidate_review(upload, tmp_path, run_name="fixture")
    assert read_json(directory / "review.json")["items"][0]["review"]["reviewer"] == "Team"


def test_candidate_upload_cannot_edit_source_text(tmp_path):
    upload, directory, incoming = review_fixture(tmp_path)
    incoming["items"][0]["text"] = "changed"; atomic_json(upload, incoming)
    with pytest.raises(ValueError, match="immutable"):
        install_ai_candidate_review(upload, tmp_path, run_name="fixture")
