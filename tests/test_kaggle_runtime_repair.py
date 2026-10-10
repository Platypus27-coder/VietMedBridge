"""Run the shipped notebook's repair against real portable files/checkpoints."""
import ast
import json
from pathlib import Path
import time

import pytest

from test_portable_embeddings import external, frozen, job, FakeEncoder, REPO, RUNTIME  # noqa: F401
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.content_embeddings import seal
from vietmedbridge.portable_embeddings import (
    _copy_checked, cuda_embedding_runtime, export_kaggle_job, finalize_result,
    hydrate_job, install_result, run_worker, runtime_install_commands, validate_job,
)


@pytest.fixture
def repair():
    path = REPO / "notebooks/04_kaggle_embedding_worker.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    tree = ast.parse(next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "repair_cpu_export_job")
    namespace = {name: globals()[name] for name in (
        "Path", "validate_job", "read_json", "runtime_install_commands", "digest_json", "_copy_checked", "atomic_json")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["repair_cpu_export_job"]


def cpu_bundle(tmp_path, job, **updates):
    original = seal({k: v for k, v in job.items() if k != "manifest_sha256"}
                    | {"runtime": {**job["runtime"], "torch": "2.11.0+cpu"}} | updates)
    atomic_json(tmp_path / "bundle/job.json", original)
    return original


def test_cpu_export_host_does_not_determine_default_gpu_runtime(tmp_path, job, monkeypatch):
    from vietmedbridge import portable_embeddings
    monkeypatch.setattr(portable_embeddings, "embedding_runtime", lambda: {**RUNTIME, "torch": "2.11.0+cpu"})
    exported = export_kaggle_job(tmp_path, REPO, tmp_path / "default-gpu", code_commit="a" * 40)
    assert exported["runtime"] == read_json(REPO / "configs/kaggle_runtime.json")
    assert exported["candidate"] == job["candidate"] and exported["inputs"] == job["inputs"]
    assert exported["files"] == job["files"]


@pytest.mark.parametrize("torch_version", ["2.11.0+cpu", "2.8.0", "2.8;bad"])
def test_invalid_export_target_stops_before_reading_corpus(tmp_path, torch_version):
    with pytest.raises(ValueError, match="CUDA|runtime lock"):
        export_kaggle_job(tmp_path, REPO, tmp_path / "output", code_commit="a" * 40,
                          runtime={**RUNTIME, "torch": torch_version})
    assert not (tmp_path / "output").exists()


def test_existing_cuda_producer_lock_is_preserved():
    assert cuda_embedding_runtime(REPO, RUNTIME) == RUNTIME


def test_repair_is_deterministic_preserves_all_inputs_and_source_job(tmp_path, job, repair):
    original = cpu_bundle(tmp_path, job)
    source = tmp_path / "bundle"
    before = {p.name: sha256_file(p) for p in source.iterdir()}
    directory, updated = repair(source, REPO, tmp_path / "account-0")
    second_directory, second = repair(source, REPO, tmp_path / "account-1")
    assert directory != source and second_directory != directory
    assert updated == second and updated["manifest_sha256"] != original["manifest_sha256"]
    assert updated["runtime"] == cuda_embedding_runtime(REPO)
    assert updated["runtime_repair"]["source_job"] == original["manifest_sha256"]
    assert updated["runtime_repair"]["source_runtime"] == original["runtime"]
    for key in original.keys() - {"runtime", "manifest_sha256"}:
        assert updated[key] == original[key]
    assert {p.name: sha256_file(p) for p in source.iterdir()} == before
    validate_job(updated, REPO, runtime=cuda_embedding_runtime(REPO))
    for entry in updated["files"]:
        assert sha256_file(directory / entry["asset"]) == entry["sha256"]
    assert repair(source, REPO, tmp_path / "account-0") == (directory, updated)
    assert repair(directory, REPO, tmp_path / "account-0") == (directory, updated)


def test_cuda_job_is_not_rewritten(tmp_path, job, repair):
    directory, updated = repair(tmp_path / "bundle", REPO, tmp_path / "kaggle-work")
    assert directory == tmp_path / "bundle" and updated == job
    assert not (tmp_path / "kaggle-work").exists()


@pytest.mark.parametrize("seed_mode", ["listed", "unlisted"])
def test_repair_refuses_to_relabel_paid_seed_vectors(tmp_path, job, repair, seed_mode):
    updates = {"seeds": ["model_cache/producer/block.done.json"]} if seed_mode == "listed" else {
        "files": job["files"] + [{"path": "model_cache/producer/block.npy", "asset": "asset-extra.npy",
                                  "sha256": "a" * 64, "bytes": 4}]}
    cpu_bundle(tmp_path, job, **updates)
    with pytest.raises(ValueError, match="seed vectors"):
        repair(tmp_path / "bundle", REPO, tmp_path / "kaggle-work")
    assert not (tmp_path / "kaggle-work").exists()


def test_invalid_original_seal_stops_before_copying(tmp_path, job, repair):
    original = cpu_bundle(tmp_path, job)
    atomic_json(tmp_path / "bundle/job.json", {**original, "input_count": -1})
    with pytest.raises(ValueError, match="metadata changed"):
        repair(tmp_path / "bundle", REPO, tmp_path / "kaggle-work")
    assert not (tmp_path / "kaggle-work").exists()


def test_corrupt_original_payload_cannot_publish_repaired_job(tmp_path, job, repair):
    original = cpu_bundle(tmp_path, job)
    (tmp_path / "bundle" / original["files"][-1]["asset"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed artifact"):
        repair(tmp_path / "bundle", REPO, tmp_path / "work")
    assert not list((tmp_path / "work").glob("*/job.json"))


def test_repair_refuses_conflicting_local_header(tmp_path, job, repair):
    cpu_bundle(tmp_path, job)
    directory, updated = repair(tmp_path / "bundle", REPO, tmp_path / "work")
    atomic_json(directory / "job.json", {**updated, "input_count": -1})
    with pytest.raises(ValueError, match="Local CUDA job differs"):
        repair(tmp_path / "bundle", REPO, tmp_path / "work")


def test_repaired_job_roundtrips_into_native_colab_cache(tmp_path, job, frozen, repair, monkeypatch):
    import test_portable_embeddings
    cpu_bundle(tmp_path, job)
    directory, updated = repair(tmp_path / "bundle", REPO, tmp_path / "work")
    monkeypatch.setattr(test_portable_embeddings, "RUNTIME", updated["runtime"])
    hydrate_job(directory, tmp_path / "scratch", REPO)
    run_worker(directory, tmp_path / "scratch", tmp_path / "output", REPO,
               worker_id=0, workers=1, deadline=time.time() + 60, encoder_factory=FakeEncoder)
    finalize_result(tmp_path / "output", updated)
    result = install_result(tmp_path / "output", tmp_path, REPO)
    assert result["corpus_embeddings_complete"]
    assert result["parts"] == 2 * len(frozen[2]["parts"])
    assert read_json(tmp_path / "retrieval_embedding_runtime_lock.json")["runtime"] == updated["runtime"]
