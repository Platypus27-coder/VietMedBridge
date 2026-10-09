"""Real disk transfers and native checkpoint reuse, with explicit fake GPU models."""
from pathlib import Path
import hashlib
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from test_external_import import external  # noqa: F401
from test_scale_baseline import frozen  # noqa: F401
from vietmedbridge.artifacts import atomic_json, read_json, sha256_file
from vietmedbridge.content_embeddings import ContentVectorParts, content_embeddings, seal
from vietmedbridge.portable_embeddings import (
    MODEL_FILES, carry_forward, export_kaggle_job, finalize_result, hydrate_job,
    install_result, relative_path, resources_relative, run_worker, runtime_install_commands,
    validate_job,
)
from vietmedbridge.qwen_models import QUERY_INSTRUCTION

REPO = Path(__file__).resolve().parents[1]
RUNTIME = {"torch": "2.8.0+cu128", "transformers": "4.57.1", "bitsandbytes": "0.49.2"}


class FakeEncoder:
    """Deterministic vectors; identity mimics the contract, never GPU evidence."""
    calls = 0

    def __init__(self, spec):
        self.dimension = 1024 if spec["model_id"] == "BAAI/bge-m3" else 4096
        qwen = self.dimension == 4096
        self.identity = {**spec, "dimension": self.dimension, "device_class": "cuda",
            "precision": "torch.float16", "truncation": False,
            "pooling": "last-attended-token-l2" if qwen else "cls-l2-v1",
            "query_instruction": QUERY_INSTRUCTION if qwen else None,
            "torch": RUNTIME["torch"], "transformers": RUNTIME["transformers"],
            "inference_code_sha256": sha256_file(REPO / "src/vietmedbridge" / MODEL_FILES[qwen])}
        if qwen:
            self.identity.update(input_role="corpus", role="second_dense", fine_tuned=False,
                                 bitsandbytes=RUNTIME["bitsandbytes"])

    def encode(self, texts, **_):
        FakeEncoder.calls += len(texts)
        result = np.zeros((len(texts), self.dimension), np.float32)
        for i, text in enumerate(texts):
            result[i, :4] = [1 + v for v in hashlib.sha256(text.encode()).digest()[:4]]
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    def close(self):
        pass


@pytest.fixture
def job(tmp_path, frozen):
    _, candidate, _ = frozen
    ready = tmp_path / "retrieval/cpu_preparation" / (candidate["candidate_manifest_sha256"][:16] + ".json")
    atomic_json(ready, {"state": "CPU_PREPARATION_COMPLETE", "candidate_manifest_sha256": candidate["candidate_manifest_sha256"]})
    value = export_kaggle_job(tmp_path, REPO, tmp_path / "bundle", code_commit="a" * 40, runtime=RUNTIME)
    FakeEncoder.calls = 0
    return value


def worker(tmp_path, *, output="output", worker_id=0, workers=1, **kwargs):
    run_worker(tmp_path / "bundle", tmp_path / "scratch", tmp_path / output, REPO,
               worker_id=worker_id, workers=workers, deadline=time.time() + 60,
               encoder_factory=FakeEncoder, **kwargs)


def test_roundtrip_two_gpus_then_three_colab_workers_no_reencoding(tmp_path, frozen, job):
    build, _, inputs = frozen
    assert len(inputs["parts"]) >= 2
    hydrate_job(tmp_path / "bundle", tmp_path / "scratch", REPO)
    worker(tmp_path, worker_id=0, workers=2)
    worker(tmp_path, worker_id=1, workers=2)
    finalize_result(tmp_path / "output", job)
    report = install_result(tmp_path / "output", tmp_path, REPO)
    assert read_json(tmp_path / "retrieval_embedding_runtime_lock.json")["runtime"] == RUNTIME
    assert len(read_json(tmp_path / "retrieval_code_lock.json")["git_commit"]) == 40
    assert report["bge_parts"] == report["qwen_parts"] == len(inputs["parts"])
    assert install_result(tmp_path / "output", tmp_path, REPO, update_lock=False)["parts"] == report["parts"]
    for family, spec in (("bge", "dense"), ("qwen", "second_dense")):
        encoder = FakeEncoder(job["config"][spec])
        view = tmp_path / resources_relative(frozen[1], job["config"]) / family
        before = FakeEncoder.calls
        for worker_id in range(3):
            content_embeddings(tmp_path, build / "index_inputs", inputs, encoder, view,
                work_dir=tmp_path / "local", worker_id=worker_id, workers=3)
        manifest = content_embeddings(tmp_path, build / "index_inputs", inputs, encoder, view,
            work_dir=tmp_path / "local", encode_missing=False)
        assert manifest["state"] == "COMPLETE" and FakeEncoder.calls == before
        store = ContentVectorParts(tmp_path, view, manifest, tmp_path / "read-cache")
        source = pq.read_table(build / "index_inputs" / inputs["parts"][0]["path"], columns=["text"]).column("text").to_pylist()
        np.testing.assert_allclose(store.part(0), encoder.encode(source), atol=1e-6)


def test_bounded_partial_and_saved_output_resume_carries_previous_parts(tmp_path, frozen, job):
    hydrate_job(tmp_path / "bundle", tmp_path / "scratch", REPO)
    worker(tmp_path, workers=2, max_output_bytes=1)  # output budget stops before inference
    assert FakeEncoder.calls == 0
    assert not list((tmp_path / "output/receipts").glob("*.json"))
    worker(tmp_path, worker_id=0, workers=2)
    finalize_result(tmp_path / "output", job)
    partial = install_result(tmp_path / "output", tmp_path, REPO, update_lock=False)
    assert 0 < partial["bge_parts"] < len(frozen[2]["parts"])
    # A new session only has the original private dataset and saved Outputs.
    shutil.rmtree(tmp_path / "scratch")
    hydrate_job(tmp_path / "bundle", tmp_path / "scratch", REPO)
    install_result(tmp_path / "output", tmp_path / "scratch", REPO, update_lock=False)
    carry_forward(tmp_path / "output", tmp_path / "next-output")
    worker(tmp_path, output="next-output", worker_id=1, workers=2)
    finalize_result(tmp_path / "next-output", job)
    report = install_result(tmp_path / "next-output", tmp_path, REPO, update_lock=False)
    assert report["bge_parts"] == report["qwen_parts"] == len(frozen[2]["parts"])


def test_source_binding_tamper_rejected_before_any_canonical_write(tmp_path, job):
    hydrate_job(tmp_path / "bundle", tmp_path / "scratch", REPO)
    worker(tmp_path)
    output = tmp_path / "output"
    finalize_result(output, job)
    receipt_path = next((output / "receipts").glob("bge-*.json"))
    receipt = read_json(receipt_path)
    marker_entry = next(f for f in receipt["files"] if f["path"].endswith(".done.json") and f["path"].startswith("model_cache/"))
    marker_path = output / "data" / marker_entry["path"]
    marker = read_json(marker_path)
    marker["keys"][0] = "0" * 64
    atomic_json(marker_path, seal({k: v for k, v in marker.items() if k != "manifest_sha256"}))
    marker_entry["sha256"] = sha256_file(marker_path)
    atomic_json(receipt_path, seal({k: v for k, v in receipt.items() if k != "manifest_sha256"}))
    finalize_result(output, job)
    with pytest.raises(ValueError, match="text/vector row binding"):
        install_result(output, tmp_path, REPO, update_lock=False)
    assert not (tmp_path / job["resources"]).exists()
    assert not (tmp_path / "model_cache").exists()


def test_vector_corruption_rejected_and_cpu_gate(tmp_path, frozen, job):
    hydrate_job(tmp_path / "bundle", tmp_path / "scratch", REPO)
    worker(tmp_path)
    finalize_result(tmp_path / "output", job)
    next((tmp_path / "output/data/model_cache").glob("*/*.npy")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed artifact"):
        install_result(tmp_path / "output", tmp_path, REPO, update_lock=False)
    ready = tmp_path / "retrieval/cpu_preparation" / (frozen[1]["candidate_manifest_sha256"][:16] + ".json")
    atomic_json(ready, {"state": "RUNNING"})
    with pytest.raises(ValueError, match="Finish notebook 04"):
        export_kaggle_job(tmp_path, REPO, tmp_path / "another", code_commit="a" * 40, runtime=RUNTIME)


@pytest.mark.parametrize("path", ["../file", "/file", "x/../file", "C:/file", "x\\file", "x//file"])
def test_portable_paths_are_platform_independent(path):
    with pytest.raises(ValueError, match="canonical relative"):
        relative_path(path)


def test_runtime_and_source_identity_are_pinned(tmp_path, job, monkeypatch):
    with pytest.raises(ValueError, match="runtime"):
        validate_job(job, REPO, runtime={**RUNTIME, "torch": "2.7.0+cu126"})
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "different")
    commands = runtime_install_commands(RUNTIME)
    assert "torch==2.8.0+cu128" in commands[0]
    assert "https://download.pytorch.org/whl/cu128" in commands[0]
    assert "transformers==4.57.1" in commands[-1]
    with pytest.raises(ValueError, match="runtime lock"):
        runtime_install_commands({**RUNTIME, "torch": "2.8;bad"})


def test_native_callback_can_republish_after_interruption(tmp_path, frozen, job):
    build, _, inputs = frozen
    encoder = FakeEncoder(job["config"]["dense"])
    view = tmp_path / "callback-view"
    def interrupted(_):
        raise RuntimeError("copy interrupted after native map commit")
    with pytest.raises(RuntimeError, match="copy interrupted"):
        content_embeddings(tmp_path, build / "index_inputs", inputs, encoder, view, work_dir=tmp_path / "local", on_part=interrupted)
    before = FakeEncoder.calls
    published = []
    content_embeddings(tmp_path, build / "index_inputs", inputs, encoder, view,
        work_dir=tmp_path / "local", max_new_parts=0, on_part=published.append)
    assert len(published) == 1 and FakeEncoder.calls == before


def test_export_existing_blocks_seeds_gpu_without_reencoding(tmp_path, frozen, job):
    hydrate_job(tmp_path / "bundle", tmp_path / "scratch", REPO)
    worker(tmp_path)
    finalize_result(tmp_path / "output", job)
    install_result(tmp_path / "output", tmp_path, REPO, update_lock=False)
    second = export_kaggle_job(tmp_path, REPO, tmp_path / "seed-bundle", code_commit="a" * 40, runtime=RUNTIME)
    assert second["seeds"]
    hydrate_job(tmp_path / "seed-bundle", tmp_path / "seed-scratch", REPO)
    before = FakeEncoder.calls
    run_worker(tmp_path / "seed-bundle", tmp_path / "seed-scratch", tmp_path / "seed-output", REPO,
        worker_id=0, workers=1, deadline=time.time() + 60, encoder_factory=FakeEncoder)
    assert FakeEncoder.calls == before
    finalize_result(tmp_path / "seed-output", second)
    assert install_result(tmp_path / "seed-output", tmp_path, REPO, update_lock=False)["parts"] == 2 * len(frozen[2]["parts"])


def test_launcher_uses_distinct_visible_gpus_and_cli_arguments(tmp_path, job, monkeypatch):
    from vietmedbridge import portable_embeddings as runtime
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    monkeypatch.setattr(runtime, "embedding_runtime", lambda: RUNTIME)
    launched = []
    class Process:
        def __init__(self, command, env):
            self.command, self.env = command, env
            self.returncode = 0
            launched.append(self)
        def wait(self, **kwargs):
            assert len(launched) == 2  # both launch before either is awaited
            return 0
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    result = runtime.launch_kaggle(tmp_path / "bundle", REPO, output=tmp_path / "launched",
        scratch=tmp_path / "scratch", session_started=time.time(), hours=1)
    assert [p.env["CUDA_VISIBLE_DEVICES"] for p in launched] == ["0", "1"]
    # Python -m module contributes sys.argv[0]; CLI receives 9 arguments in all.
    assert all(len(p.command[2:]) == 9 for p in launched)
    assert result["processes"] == [0, 0]


def test_hung_gpu_worker_is_stopped_before_save_deadline(tmp_path, job, monkeypatch):
    from vietmedbridge import portable_embeddings as runtime
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    monkeypatch.setattr(runtime, "embedding_runtime", lambda: RUNTIME)
    class Process:
        returncode = None
        def __init__(self, command, env):
            pass
        def wait(self, timeout):
            if self.returncode is None:
                raise runtime.subprocess.TimeoutExpired("fake GPU", timeout)
            return self.returncode
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    result = runtime.launch_kaggle(tmp_path / "bundle", REPO, output=tmp_path / "timeout-output",
        scratch=tmp_path / "scratch", session_started=time.time(), hours=1)
    assert result["processes"] == [-15, -15]
    assert result["state"] == "WORKER_FAILED_WITH_CHECKPOINTS"
