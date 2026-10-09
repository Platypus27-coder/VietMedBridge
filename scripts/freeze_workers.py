"""Checkpoint and verify disjoint Notebook 03 freeze workers."""
from __future__ import annotations

from pathlib import Path

from vietmedbridge.artifacts import atomic_json, digest_json, read_json


FREEZE_WORKER_SCHEMA = 1


def freeze_batch_signature(build_refs, *, code_sha256: str, team_size: int) -> str:
    """Bind worker manifests to one ordered corpus/build set and code revision."""
    if type(team_size) is not int or team_size < 1:
        raise ValueError("Freeze team size must be a positive integer.")
    builds = []
    seen = set()
    for item in build_refs:
        run = item.get("build_run")
        snapshot = item.get("snapshot_sha256")
        source = item.get("source_path")
        if (not isinstance(run, str) or not run or Path(run).name != run
            or run in {".", ".."} or run in seen):
            raise ValueError("Freeze batch must contain unique build_run values.")
        if not isinstance(snapshot, str) or not snapshot:
            raise ValueError(f"Freeze batch is missing snapshot_sha256 for {run}.")
        seen.add(run)
        builds.append({"build_run": run, "source_path": source, "snapshot_sha256": snapshot})
    if not builds:
        raise ValueError("Freeze batch is empty.")
    return digest_json({"workflow": "parallel-freeze-v1", "code_sha256": code_sha256,
                        "team_size": team_size, "builds": builds})


def partition_freeze_builds(build_refs, *, team_size: int, worker_id: int) -> list[dict]:
    """Deterministically distribute builds, keeping each build on one worker."""
    if type(team_size) is not int or team_size < 1:
        raise ValueError("Freeze team size must be a positive integer.")
    if type(worker_id) is not int or not 0 <= worker_id < team_size:
        raise ValueError("Freeze worker ID must be in [0, team_size).")
    return [item for index, item in enumerate(build_refs) if index % team_size == worker_id]


def freeze_worker_manifest_path(data_root, *, batch_signature: str, worker_id: int) -> Path:
    if not isinstance(batch_signature, str) or len(batch_signature) < 16:
        raise ValueError("Invalid freeze batch signature.")
    if type(worker_id) is not int or worker_id < 0:
        raise ValueError("Invalid freeze worker ID.")
    return Path(data_root) / "freeze_workers" / batch_signature[:16] / f"worker-{worker_id}.json"


def write_freeze_worker_manifest(path, *, batch_signature: str, code_sha256: str,
                                 team_size: int, worker_id: int, assigned_build_runs: list[str],
                                 completed: dict[str, dict], state: str) -> dict:
    """Atomically checkpoint a worker's finished per-build artifacts."""
    if state not in {"RUNNING", "COMPLETE"}:
        raise ValueError("Freeze worker state must be RUNNING or COMPLETE.")
    if len(set(assigned_build_runs)) != len(assigned_build_runs):
        raise ValueError("Freeze worker has duplicate assigned builds.")
    if not set(completed).issubset(set(assigned_build_runs)):
        raise ValueError("Freeze worker completed a build it was not assigned.")
    if state == "COMPLETE" and set(completed) != set(assigned_build_runs):
        raise ValueError("Cannot complete a worker before all assigned builds are frozen.")
    payload = {
        "schema_version": FREEZE_WORKER_SCHEMA,
        "batch_signature": batch_signature,
        "code_sha256": code_sha256,
        "team_size": team_size,
        "worker_id": worker_id,
        "state": state,
        "assigned_build_runs": assigned_build_runs,
        "completed": completed,
    }
    payload["manifest_sha256"] = digest_json(payload)
    atomic_json(path, payload)
    return payload


def verify_freeze_workers(data_root, build_refs, *, batch_signature: str,
                          code_sha256: str, team_size: int) -> list[dict]:
    """Verify complete worker coverage and the candidate/index handoff manifests."""
    records: dict[str, dict] = {}
    for worker_id in range(team_size):
        path = freeze_worker_manifest_path(data_root, batch_signature=batch_signature,
                                           worker_id=worker_id)
        if not path.is_file():
            raise FileNotFoundError(f"Freeze worker manifest missing: {path}")
        worker = read_json(path)
        payload = {key: value for key, value in worker.items() if key != "manifest_sha256"}
        expected = partition_freeze_builds(build_refs, team_size=team_size, worker_id=worker_id)
        expected_runs = [item["build_run"] for item in expected]
        if (worker.get("schema_version") != FREEZE_WORKER_SCHEMA
            or digest_json(payload) != worker.get("manifest_sha256")
            or worker.get("batch_signature") != batch_signature
            or worker.get("code_sha256") != code_sha256
            or worker.get("team_size") != team_size
            or worker.get("worker_id") != worker_id
            or worker.get("state") != "COMPLETE"
            or worker.get("assigned_build_runs") != expected_runs
            or set(worker.get("completed", {})) != set(expected_runs)):
            raise ValueError(f"Freeze worker manifest is stale or incomplete: {path.name}")
        for run_name, record in worker["completed"].items():
            if record.get("state") != "COMPLETE" or run_name in records:
                raise ValueError(f"Duplicate or incomplete freeze output: {run_name}")
            if Path(run_name).name != run_name or run_name in {".", ".."}:
                raise ValueError(f"Invalid build run in freeze worker output: {run_name}")
            ref = next(item for item in build_refs if item["build_run"] == run_name)
            build_dir = Path(data_root) / "processed" / run_name
            build = read_json(build_dir / "build.json")
            if (not build.get("selected_range_complete")
                or build.get("snapshot_sha256") != ref.get("snapshot_sha256")
                or record.get("snapshot_sha256") != build.get("snapshot_sha256")):
                raise ValueError(f"Freeze worker build snapshot mismatch: {run_name}")
            candidate_name = record.get("candidate_name")
            if (not isinstance(candidate_name, str) or not candidate_name
                or Path(candidate_name).name != candidate_name or candidate_name in {".", ".."}):
                raise ValueError(f"Invalid candidate path in freeze worker output: {run_name}")
            candidate_path = build_dir / candidate_name
            if candidate_path is None or not candidate_path.is_file():
                raise FileNotFoundError(f"Frozen candidate missing for {run_name}.")
            candidate = read_json(candidate_path)
            candidate_sha = digest_json({key: value for key, value in candidate.items()
                                         if key != "candidate_manifest_sha256"})
            if (candidate.get("state") != "FROZEN_CANDIDATE"
                or candidate.get("candidate_manifest_sha256") != candidate_sha
                or candidate_sha != record.get("candidate_manifest_sha256")
                or candidate.get("snapshot_sha256") != build.get("snapshot_sha256")
                or candidate.get("golden", {}).get("code_sha256") != code_sha256):
                raise ValueError(f"Frozen candidate integrity mismatch: {run_name}")
            inputs = read_json(build_dir / "index_inputs" / "units.json")
            inputs_sha = digest_json({key: value for key, value in inputs.items()
                                      if key != "manifest_sha256"})
            if (inputs.get("state") != "COMPLETE"
                or inputs.get("candidate_manifest_sha256") != candidate_sha
                or inputs.get("manifest_sha256") != inputs_sha
                or inputs_sha != record.get("index_inputs_manifest_sha256")):
                raise ValueError(f"Index-input manifest integrity mismatch: {run_name}")
            records[run_name] = {**record, "candidate": candidate, "inputs": inputs}
    expected_runs = [item["build_run"] for item in build_refs]
    if set(records) != set(expected_runs):
        missing = sorted(set(expected_runs) - set(records))
        extra = sorted(set(records) - set(expected_runs))
        raise ValueError(f"Freeze workers do not cover the batch: missing={missing}, extra={extra}")
    return [records[run] for run in expected_runs]
