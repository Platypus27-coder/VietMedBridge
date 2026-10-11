"""Continue from verified Drive maps without copying old vectors into Kaggle Outputs."""
from __future__ import annotations

from pathlib import Path
import math
import re
import subprocess
import time

from .artifacts import atomic_json, read_json
from .content_embeddings import checked, seal
from .scale_benchmark import load_handoff
from .scale_vectors import bound


def find_imported_job(data_root, checkout):
    """Reuse the original small bundle instead of exporting the growing vector cache."""
    from .portable_embeddings import validate_job
    root = Path(data_root)
    _, candidate, inputs = load_handoff(root)
    jobs = []
    for path in (root / "retrieval/portable_imports").glob("*/job.json"):
        job = read_json(path)
        if job["candidate"] == candidate["candidate_manifest_sha256"] and job["inputs"] == inputs["manifest_sha256"]:
            jobs.append(validate_job(job, checkout))
    return min(jobs, key=lambda job: sum(entry["bytes"] for entry in job["files"])) if jobs else None


def load_plan(directory, job, inputs):
    path = Path(directory) / "continuation-plan.json"
    if not path.exists():
        return None
    plan = checked(read_json(path))
    if (inputs["manifest_sha256"] != job["inputs"] or inputs["input_count"] != job["input_count"]
        or plan.get("schema") != "portable-continuation-v1" or plan.get("job") != job["manifest_sha256"]
        or plan.get("inputs") != inputs["manifest_sha256"] or plan.get("candidate") != job["candidate"]
        or plan.get("team") != job.get("team")
        or not re.fullmatch(r"[0-9a-f]{40}", plan.get("worker_code_commit", ""))
        or set(plan.get("completed", {})) != {"bge", "qwen"}
        or set(plan.get("scheduled", {})) != {"bge", "qwen"}):
        raise ValueError("Continuation plan belongs to another job/input/team.")
    starts = {part["start"] for part in inputs["parts"]}
    for completed in plan["completed"].values():
        if (not isinstance(completed, list) or any(type(start) is not int for start in completed)
            or len(completed) != len(set(completed)) or not set(completed) <= starts):
            raise ValueError("Continuation plan contains invalid completed parts.")
    for family, scheduled in plan["scheduled"].items():
        if (not isinstance(scheduled, list) or any(type(start) is not int for start in scheduled)
            or len(scheduled) != len(set(scheduled)) or not set(scheduled) <= starts
            or set(scheduled) & set(plan["completed"][family])):
            raise ValueError("Continuation plan contains invalid scheduled parts.")
    return plan


def export_continuation(data_root, checkout, destination, job, *, worker_code_commit=None,
                        round_output_bytes=16_000_000_000):
    """Keep the original job/model identities; export inputs and a small skip plan only."""
    from .portable_embeddings import _copy_checked, embedding_coverage, validate_job
    root, checkout = Path(data_root), Path(checkout)
    job = validate_job(job, checkout)
    _, candidate, inputs = load_handoff(root)
    if candidate["candidate_manifest_sha256"] != job["candidate"] or inputs["manifest_sha256"] != job["inputs"]:
        raise ValueError("Continuation belongs to another frozen corpus.")
    coverage = embedding_coverage(root, job)  # verifies config, input binding and map bytes
    if all(value["complete"] for value in coverage.values()):
        raise ValueError("Both corpus embedding families are already complete; continue with the coordinator.")
    atomic_json(root / "retrieval/portable_imports" / job["manifest_sha256"][:16] / "job.json", job)
    if worker_code_commit is None:
        worker_code_commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", worker_code_commit):
        raise ValueError("Pin the continuation worker's full Git commit.")
    completed = {family: [part["start"] for part in inputs["parts"] if (
        root / job["resources"] / family / f"map-{part['start']:012d}.done.json").exists()]
        for family in ("bge", "qwen")}
    if type(round_output_bytes) is not int or not 0 < round_output_bytes <= 16_000_000_000:
        raise ValueError("Continuation round output budget must be a positive integer up to 16 GB.")
    team = job.get("team") or {"accounts": 1, "gpu_count": 2}
    scheduled = {family: [] for family in completed}
    used, remaining_bytes = [0] * team["accounts"], [0] * team["accounts"]
    for family, dimension in (("bge", 1024), ("qwen", 4096)):
        previous = set(completed[family])
        for i, part in enumerate(inputs["parts"]):
            if part["start"] in previous:
                continue
            account = (i % (team["accounts"] * team["gpu_count"])) // team["gpu_count"]
            size = part["rows"] * (dimension * 4 + 512)
            if size > round_output_bytes:
                raise ValueError("One embedding part exceeds the continuation round budget.")
            remaining_bytes[account] += size
            if used[account] + size <= round_output_bytes:
                scheduled[family].append(part["start"])
                used[account] += size
    plan = seal({"schema": "portable-continuation-v1", "job": job["manifest_sha256"],
        "candidate": job["candidate"], "inputs": job["inputs"], "team": job.get("team"),
        "worker_code_commit": worker_code_commit, "completed": completed, "coverage": coverage,
        "scheduled": scheduled, "round_output_bytes": round_output_bytes,
        "round_estimates": [{"account_id": i, "scheduled_bytes_with_metadata_allowance": used[i],
            "remaining_bytes_with_metadata_allowance": remaining_bytes[i],
            "minimum_rounds_by_capacity": math.ceil(remaining_bytes[i] / round_output_bytes)}
            for i in range(team["accounts"])],
        "capacity_note": "Capacity estimates only; session time or runtime failures can require further rounds."})
    # A new plan gets its own folder, so a rerun cannot overwrite another round.
    target = Path(destination) / plan["manifest_sha256"][:16]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "job.json").exists() and read_json(target / "job.json") != job:
        raise ValueError("Continuation folder belongs to another job.")
    for entry in job["files"]:
        _copy_checked(bound(root, entry["path"]), target / entry["asset"], entry["sha256"])
    atomic_json(target / "job.json", job)
    atomic_json(target / "continuation-plan.json", plan)
    load_plan(target, job, inputs)
    return target, plan


def assignment_progress(inputs, assignment, plan, receipts):
    """Separate account completion from completion of all accounts and all families."""
    assigned = {part["start"] for i, part in enumerate(inputs["parts"])
                if i % assignment["workers"] in assignment["worker_ids"]}
    current = {family: set() for family in ("bge", "qwen")}
    for receipt in receipts:
        family, start = receipt["family"], receipt["map"]["start"]
        if family not in current or start not in assigned or start in current[family]:
            raise ValueError("Continuation receipt is duplicate or outside the account assignment.")
        current[family].add(start)
    result = {}
    for family in current:
        previous = set(plan["completed"][family]) if plan is not None else set()
        finished = (previous | current[family]) & assigned
        scheduled = set(plan["scheduled"][family]) & assigned if plan is not None else assigned - previous
        result[family] = {"assigned_parts": len(assigned), "previously_completed_parts": len(previous & assigned),
            "saved_parts_this_output": len(current[family]), "remaining_parts": len(assigned - finished),
            "scheduled_parts_this_round": len(scheduled), "remaining_parts_this_round": len(scheduled - finished),
            "round_complete": scheduled <= finished, "complete": finished == assigned}
    return result


def upload_continuation(data_root, checkout, destination, job, *, owner, worker_code_commit=None):
    """Publish one private input bundle per plan, reusing the upload on retries."""
    import kagglehub
    if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner.")
    target, plan = export_continuation(data_root, checkout, destination, job,
                                       worker_code_commit=worker_code_commit)
    pointer = Path(data_root) / "retrieval/kaggle_exports" / (
        owner + "-continue-" + plan["manifest_sha256"][:16] + ".json")
    if pointer.exists():
        saved = read_json(pointer)
        if saved.get("plan") != plan["manifest_sha256"] or saved.get("job") != job["manifest_sha256"]:
            raise ValueError("Saved continuation upload differs.")
        handle = saved["dataset_handle"]
    else:
        handle = owner + "/vmb-continue-" + plan["manifest_sha256"][:12] + "-" + str(time.time_ns())
        kagglehub.dataset_upload(handle, str(target))
        atomic_json(pointer, {"dataset_handle": handle, "plan": plan["manifest_sha256"],
                              "job": job["manifest_sha256"], "state": "UPLOADED_VERIFY_PENDING"})
    for attempt in range(6):
        try:
            remote_plan = Path(kagglehub.dataset_download(handle, path="continuation-plan.json", force_download=True))
            remote_job = Path(kagglehub.dataset_download(handle, path="job.json", force_download=True))
            break
        except Exception:
            if attempt == 5:
                raise RuntimeError("Dataset is processing; rerun to verify this upload without creating another.") from None
            time.sleep(min(30, 5 * (attempt + 1)))
    if read_json(remote_plan) != plan or read_json(remote_job) != job:
        raise ValueError("Uploaded continuation job/plan differs; do not run it.")
    atomic_json(pointer, {"dataset_handle": handle, "plan": plan["manifest_sha256"],
                          "job": job["manifest_sha256"], "state": "VERIFIED"})
    print("Continuation dataset:", "https://www.kaggle.com/datasets/" + handle, flush=True)
    for family, value in plan["coverage"].items():
        print(f"{family}: verified on Drive {value['completed_parts']}/{value['total_parts']}; "
              f"remaining {value['total_parts'] - value['completed_parts']}", flush=True)
    print("Round capacity plan:", plan["round_estimates"], flush=True)
    print("Kaggle: Add Input ONLY this continuation dataset; use the updated worker notebook. "
          "Keep account IDs 0/1 and T4 x2. Old Outputs stay on Drive; do not attach them.", flush=True)
    return {"dataset_handle": handle, "plan": plan["manifest_sha256"], "coverage": plan["coverage"]}
