"""Several bounded rounds converge without re-encoding or carrying old vector bytes."""
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

from test_portable_embeddings import parallel_outputs, FakeEncoder, REPO  # noqa: F401
from vietmedbridge.artifacts import atomic_json, read_json
from vietmedbridge.content_embeddings import seal
from vietmedbridge.kaggle_continuation import assignment_progress, export_continuation, find_imported_job, load_plan, upload_continuation
from vietmedbridge.portable_embeddings import (
    embedding_coverage, finalize_result, hydrate_job, install_result, kaggle_assignment, run_worker,
)


@pytest.fixture
def partial(tmp_path, parallel_outputs):
    job, build, inputs = parallel_outputs
    # Model the production situation: both accounts completed BGE but only some Qwen.
    for account in range(2):
        output = tmp_path / f"output-{account}"
        result = read_json(output / "result-manifest.json")
        qwen = [item for item in result["receipts"] if "qwen-" in item["path"]][:1]
        result["receipts"] = [item for item in result["receipts"] if "bge-" in item["path"]] + qwen
        atomic_json(output / "result-manifest.json", seal({k: v for k, v in result.items() if k != "manifest_sha256"}))
        report = install_result(output, tmp_path, REPO, update_lock=False)
    assert report["coverage"]["bge"]["complete"]
    assert not report["coverage"]["qwen"]["complete"]
    FakeEncoder.calls = 0
    return job, build, inputs


def test_repeated_capacity_bounded_rounds_finish_all_missing_qwen_once(tmp_path, partial):
    job, _, inputs = partial
    before = embedding_coverage(tmp_path, job)
    expected_rows = inputs["input_count"] - before["qwen"]["input_rows"]
    seen = set()
    def qwen_only(spec):
        assert spec["model_id"] != "BAAI/bge-m3", "BGE must not even be loaded"
        return FakeEncoder(spec)
    for round_id in range(10):
        if embedding_coverage(tmp_path, job)["qwen"]["complete"]:
            break
        directory, plan = export_continuation(tmp_path, REPO, tmp_path / "continuation-inputs", job,
                                               worker_code_commit="c" * 40, round_output_bytes=35_000)
        assert not plan["scheduled"]["bge"]
        assert not any(entry["path"].startswith("model_cache/") for entry in job["files"])
        assert all(item["scheduled_bytes_with_metadata_allowance"] <= 35_000 for item in plan["round_estimates"])
        for account in range(2):
            scratch = tmp_path / f"round-{round_id}-account-{account}"
            output = tmp_path / f"round-output-{round_id}-{account}"
            hydrate_job(directory, scratch, REPO)
            assignment = kaggle_assignment(accounts=2, account_id=account, gpu_count=2)
            atomic_json(output / "assignment.json", assignment)
            shutil.copyfile(directory / "continuation-plan.json", output / "continuation-plan.json")
            for worker in assignment["worker_ids"]:
                run_worker(directory, scratch, output, REPO, worker_id=worker, workers=4,
                           deadline=time.time() + 60, encoder_factory=qwen_only)
            result = finalize_result(output, job, processes=[0, 0], inputs=inputs)
            assert result["state"] in {"ROUND_ASSIGNMENT_COMPLETE", "ACCOUNT_ASSIGNMENT_COMPLETE"}
            assert all(value["round_complete"] for value in result["account_coverage"].values())
            receipts = [read_json(output / entry["path"]) for entry in result["receipts"]]
            for receipt in receipts:
                assert receipt["family"] == "qwen"
                start = receipt["map"]["start"]
                assert start not in seen and start not in plan["completed"]["qwen"]
                seen.add(start)
            report = install_result(output, tmp_path, REPO, update_lock=False)
    else:
        pytest.fail("Continuation rounds failed to converge")
    assert round_id > 1, "The fixture must exercise repeated delta rounds"
    assert report["corpus_embeddings_complete"]
    assert FakeEncoder.calls == expected_rows
    assert len(seen) == len(inputs["parts"]) - before["qwen"]["completed_parts"]


def test_two_account_checkpoint_continues_on_one_account_without_reencoding(tmp_path, partial):
    original, _, inputs = partial
    before = embedding_coverage(tmp_path, original)
    directory, plan = export_continuation(tmp_path, REPO, tmp_path / "one-account", original,
                                           worker_code_commit="c" * 40, accounts=1, gpu_count=2)
    next_job = read_json(directory / "job.json")
    assert original["team"] == {"accounts": 2, "gpu_count": 2}
    assert next_job["team"] == {"accounts": 1, "gpu_count": 2}
    assert next_job["manifest_sha256"] != original["manifest_sha256"]
    assert {k: v for k, v in next_job.items() if k not in {"team", "manifest_sha256"}} == {
        k: v for k, v in original.items() if k not in {"team", "manifest_sha256"}}
    assert not plan["scheduled"]["bge"]
    assert set(plan["scheduled"]["qwen"]) == {
        part["start"] for part in inputs["parts"]} - set(plan["completed"]["qwen"])
    assert find_imported_job(tmp_path, REPO, accounts=1, gpu_count=2) == next_job
    assert embedding_coverage(tmp_path, next_job) == before

    scratch, output = tmp_path / "one-scratch", tmp_path / "one-output"
    hydrate_job(directory, scratch, REPO)
    assignment = kaggle_assignment(accounts=1, account_id=0, gpu_count=2)
    atomic_json(output / "assignment.json", assignment)
    shutil.copyfile(directory / "continuation-plan.json", output / "continuation-plan.json")
    def qwen_only(spec):
        assert spec["model_id"] != "BAAI/bge-m3"
        return FakeEncoder(spec)
    for worker_id in assignment["worker_ids"]:
        run_worker(directory, scratch, output, REPO, worker_id=worker_id, workers=2,
                   deadline=time.time() + 60, encoder_factory=qwen_only)
    result = finalize_result(output, next_job, processes=[0, 0], inputs=inputs)
    assert result["state"] == "ACCOUNT_ASSIGNMENT_COMPLETE"
    report = install_result(output, tmp_path, REPO, update_lock=False)
    assert report["corpus_embeddings_complete"]
    assert FakeEncoder.calls == inputs["input_count"] - before["qwen"]["input_rows"]


def test_one_account_checkpoint_switches_back_to_four_gpus_without_duplicate_work(tmp_path, partial):
    original, _, inputs = partial
    one_dir, _ = export_continuation(tmp_path, REPO, tmp_path / "one", original,
                                    worker_code_commit="c" * 40, accounts=1, gpu_count=2)
    one_job = read_json(one_dir / "job.json")
    before = embedding_coverage(tmp_path, one_job)
    directory, plan = export_continuation(tmp_path, REPO, tmp_path / "two", one_job,
                                         worker_code_commit="c" * 40, accounts=2, gpu_count=2)
    job = read_json(directory / "job.json")
    assert job["team"] == {"accounts": 2, "gpu_count": 2}
    for field in ("candidate", "inputs", "runtime", "model_code", "config", "resources"):
        assert job[field] == one_job[field]
    assert not plan["scheduled"]["bge"]
    scheduled = set(plan["scheduled"]["qwen"])
    seen = set()
    FakeEncoder.calls = 0
    def qwen_only(spec):
        assert spec["model_id"] != "BAAI/bge-m3"
        return FakeEncoder(spec)
    for account_id in range(2):
        scratch, output = tmp_path / f"s{account_id}", tmp_path / f"o{account_id}"
        hydrate_job(directory, scratch, REPO)
        assignment = kaggle_assignment(accounts=2, account_id=account_id, gpu_count=2)
        assert assignment["worker_ids"] == [2 * account_id, 2 * account_id + 1]
        atomic_json(output / "assignment.json", assignment)
        shutil.copyfile(directory / "continuation-plan.json", output / "continuation-plan.json")
        for worker_id in assignment["worker_ids"]:
            run_worker(directory, scratch, output, REPO, worker_id=worker_id, workers=4,
                       deadline=time.time() + 60, encoder_factory=qwen_only)
        result = finalize_result(output, job, processes=[0, 0], inputs=inputs)
        for entry in result["receipts"]:
            receipt = read_json(output / entry["path"])
            assert receipt["family"] == "qwen"
            start = receipt["map"]["start"]
            assert start in scheduled and start not in seen
            seen.add(start)
        report = install_result(output, tmp_path, REPO, update_lock=False)
    assert seen == scheduled
    assert report["corpus_embeddings_complete"]
    assert FakeEncoder.calls == inputs["input_count"] - before["qwen"]["input_rows"]


def test_continuation_rejects_wrong_job_invalid_selection_and_broken_drive_map(tmp_path, partial):
    job, _, inputs = partial
    directory, plan = export_continuation(tmp_path, REPO, tmp_path / "continuation", job, worker_code_commit="c" * 40)
    for update, message in [
        ({"job": "f" * 64}, "another job"),
        ({"completed": {"bge": [True], "qwen": []}}, "invalid completed"),
        ({"scheduled": {"bge": [inputs['parts'][0]['start']], "qwen": []}}, "invalid scheduled"),
        ({"scheduled": {"bge": [], "qwen": [-1]}}, "invalid scheduled"),
    ]:
        atomic_json(directory / "continuation-plan.json", seal({k: v for k, v in plan.items() if k != "manifest_sha256"} | update))
        with pytest.raises(ValueError, match=message):
            load_plan(directory, job, inputs)
    view = tmp_path / job["resources"] / "bge"
    next(view.glob("map-*.parquet")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="changed artifact"):
        export_continuation(tmp_path, REPO, tmp_path / "invalid-export", job, worker_code_commit="c" * 40)
    assert not (tmp_path / "invalid-export").exists()


@pytest.mark.parametrize("limit", [0, -1, True, 17_000_000_000])
def test_invalid_round_capacity_rejected(tmp_path, partial, limit):
    with pytest.raises(ValueError, match="round output budget"):
        export_continuation(tmp_path, REPO, tmp_path / "invalid-export", partial[0], round_output_bytes=limit)


def test_deadline_reports_reason_without_loading_a_model(tmp_path, partial):
    job, _, inputs = partial
    directory, plan = export_continuation(tmp_path, REPO, tmp_path / "continuation", job, worker_code_commit="c" * 40)
    hydrate_job(directory, tmp_path / "fresh", REPO)
    def must_not_load(_):
        pytest.fail("Expired session must not load a model")
    run_worker(directory, tmp_path / "fresh", tmp_path / "expired", REPO, worker_id=0, workers=4,
               deadline=0, encoder_factory=must_not_load)
    status = read_json(tmp_path / "expired/workers/worker-0.json")
    assert status["families"]["bge"]["stop_reason"] == "ALREADY_COMPLETE"
    assert status["families"]["qwen"]["stop_reason"] == "SESSION_TIME_LIMIT"
    progress = assignment_progress(inputs, kaggle_assignment(accounts=2, account_id=0, gpu_count=2), plan, [])
    assert progress["bge"]["complete"]
    assert not progress["qwen"]["round_complete"]


def test_output_capacity_reports_reason_without_loading_model(tmp_path, partial):
    job, _, _ = partial
    directory, _ = export_continuation(tmp_path, REPO, tmp_path / "continuation", job, worker_code_commit="c" * 40)
    hydrate_job(directory, tmp_path / "fresh", REPO)
    def must_not_load(_):
        pytest.fail("Full output must not load a model")
    run_worker(directory, tmp_path / "fresh", tmp_path / "full", REPO, worker_id=0, workers=4,
               deadline=time.time() + 60, max_output_bytes=1, encoder_factory=must_not_load)
    assert read_json(tmp_path / "full/workers/worker-0.json")["families"]["qwen"]["stop_reason"] == "OUTPUT_SIZE_LIMIT"


def test_model_loading_failure_is_reported(tmp_path, partial):
    directory, _ = export_continuation(tmp_path, REPO, tmp_path / "continuation", partial[0], worker_code_commit="c" * 40)
    hydrate_job(directory, tmp_path / "fresh", REPO)
    def failed(_):
        raise RuntimeError("test model load failed")
    with pytest.raises(RuntimeError, match="model load failed"):
        run_worker(directory, tmp_path / "fresh", tmp_path / "failed", REPO, worker_id=0, workers=4,
                   deadline=time.time() + 60, encoder_factory=failed)
    assert read_json(tmp_path / "failed/workers/worker-0.json")["families"]["qwen"]["stop_reason"] == "WORKER_ERROR"


def test_discovery_reuses_imported_input_job_instead_of_exporting_vector_cache(tmp_path, partial):
    assert find_imported_job(tmp_path, REPO) == partial[0]
    # A candidate imported earlier must not affect the active corpus.
    atomic_json(tmp_path / "retrieval/portable_imports/unrelated/job.json",
                {"candidate": "f" * 64, "inputs": "f" * 64})
    assert find_imported_job(tmp_path, REPO) == partial[0]


def test_continuation_upload_reuses_private_dataset_pointer(tmp_path, partial, monkeypatch):
    targets = {}
    def upload(handle, path):
        assert not targets
        targets[handle] = Path(path)
    def download(handle, *, path, force_download):
        assert force_download
        return str(targets[handle] / path)
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_upload=upload, dataset_download=download))
    args = (tmp_path, REPO, tmp_path / "upload", partial[0])
    first = upload_continuation(*args, owner="test-owner", worker_code_commit="c" * 40)
    second = upload_continuation(*args, owner="test-owner", worker_code_commit="c" * 40)
    assert first == second and len(targets) == 1
    assert not list(next(iter(targets.values())).glob("*.npy"))
