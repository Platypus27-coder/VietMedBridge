"""Embedding-only Kaggle handoff; source, models and native Colab maps stay bound.

Input bundles contain model inputs/cache only. Each GPU publishes a receipt AFTER
its vector blocks and map have reached /kaggle/working. A normal, bounded Kaggle
Save & Run All saves these outputs; this is not a remote live Drive checkpoint.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, utc_now, verify_file
from .content_embeddings import ContentBlocks, checked, content_embeddings, key, seal
from .embeddings import validate_vectors
from .scale_benchmark import load_handoff
from .scale_vectors import bound

MODEL_FILES = ("retrieval_models.py", "qwen_models.py")
REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"


def kaggle_assignment(*, accounts, account_id, gpu_count):
    if (type(accounts) is not int or accounts < 1 or type(account_id) is not int
        or not 0 <= account_id < accounts or type(gpu_count) is not int or gpu_count < 1):
        raise ValueError("Invalid Kaggle account/GPU assignment.")
    return {"accounts": accounts, "account_id": account_id, "gpu_count": gpu_count,
            "workers": accounts * gpu_count,
            "worker_ids": list(range(account_id * gpu_count, (account_id + 1) * gpu_count))}


def relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
        or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
        or PurePosixPath(value).as_posix() != value):
        raise ValueError("Portable artifact must use a canonical relative path.")
    return value


def validate_encoder(job, family, encoder, dimension):
    from .qwen_models import QUERY_INSTRUCTION
    spec = job["config"]["dense" if family == "bge" else "second_dense"]
    expected = {**spec, "dimension": 1024 if family == "bge" else 4096,
        "precision": "torch.float16", "device_class": "cuda", "truncation": False,
        "pooling": "cls-l2-v1" if family == "bge" else "last-attended-token-l2",
        "query_instruction": None if family == "bge" else QUERY_INSTRUCTION,
        "inference_code_sha256": job["model_code"][MODEL_FILES[family == "qwen"]],
        "torch": job["runtime"]["torch"], "transformers": job["runtime"]["transformers"]}
    if family == "qwen":
        expected.update(input_role="corpus", role="second_dense", fine_tuned=False,
                        bitsandbytes=job["runtime"]["bitsandbytes"])
    if dimension != expected["dimension"] or any(encoder.get(k) != v for k, v in expected.items()):
        raise ValueError("Portable embedding model/runtime/pooling/role differs.")


def resources_relative(candidate, config):
    return "retrieval/full_resources/" + candidate["candidate_manifest_sha256"][:16] + "/" + digest_json({
        "dense": config["dense"], "second_dense": config["second_dense"],
        "document_builder": "document-title-bounded-opening-v1"})[:12]


def embedding_runtime():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")}


def runtime_install_commands(runtime):
    if set(runtime) != {"torch", "transformers", "bitsandbytes"} or any(
        not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", v) for v in runtime.values()
    ):
        raise ValueError("Invalid portable embedding runtime lock.")
    def installed(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    commands = []
    if installed("torch") != runtime["torch"]:
        command = [sys.executable, "-m", "pip", "install", "torch==" + runtime["torch"]]
        suffix = re.search(r"\+(cu\d+|cpu)$", runtime["torch"])
        if suffix:
            command += ["--index-url", "https://download.pytorch.org/whl/" + suffix[1]]
        commands.append(command)
        # These text encoders do not use vision/audio. Preinstalled extensions
        # compiled for another torch can break transformers imports after pinning.
        commands.append([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"])
    packages = [name + "==" + runtime[name] for name in ("transformers", "bitsandbytes")
                if installed(name) != runtime[name]]
    if packages:
        commands.append([sys.executable, "-m", "pip", "install", *packages])
    return commands


def _copy_checked(source, destination, checksum):
    verify_file(source, checksum)
    if Path(destination).exists():
        verify_file(destination, checksum)
    elif publish_file(source, destination) != checksum:
        raise ValueError("Portable artifact copy changed.")


def export_kaggle_job(data_root, checkout, destination, *, code_commit, runtime=None, accounts=1, gpu_count=2):
    """CPU only; never loads model weights or republishes the source corpus."""
    from types import SimpleNamespace
    from .full_scale_runtime import baseline_seed_sources
    kaggle_assignment(accounts=accounts, account_id=0, gpu_count=gpu_count)
    root, checkout, target = Path(data_root), Path(checkout), Path(destination)
    build, candidate, inputs = load_handoff(root)
    ready = root / "retrieval/cpu_preparation" / (candidate["candidate_manifest_sha256"][:16] + ".json")
    if not ready.is_file() or read_json(ready).get("state") != "CPU_PREPARATION_COMPLETE":
        raise ValueError("Finish notebook 04 PREPARE_ONLY=True on CPU before exporting to Kaggle.")
    if read_json(ready).get("candidate_manifest_sha256") != candidate["candidate_manifest_sha256"]:
        raise ValueError("CPU preparation belongs to another candidate.")
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("Pin the full Git commit for the Kaggle worker.")
    config = read_json(checkout / "configs/retrieval_full.json")
    runtime = runtime or embedding_runtime()
    resources = resources_relative(candidate, config)
    contract = {"config": config, "runtime": runtime,
                "model_code": {n: sha256_file(checkout / "src/vietmedbridge" / n) for n in MODEL_FILES}}
    for family in ("bge", "qwen"):
        view_config = root / resources / family / "config.json"
        if view_config.exists():
            producer = read_json(view_config)
            validate_encoder(contract, family, producer["encoder"], producer["dimension"])
    # Register valid paid-for legacy vectors. This does not re-encode or copy them.
    for source_root, source_inputs, vectors in baseline_seed_sources(root, candidate):
        saved = read_json(vectors / "config.json")
        encoder = saved.get("encoder", {})
        if (any(encoder.get(k) != v for k, v in config["dense"].items())
            or any(encoder.get(k) != runtime[k] for k in ("torch", "transformers"))
            or encoder.get("inference_code_sha256") != sha256_file(checkout / "src/vietmedbridge/retrieval_models.py")):
            continue
        blocks = ContentBlocks(root, SimpleNamespace(identity=encoder, dimension=saved["dimension"]), target.parent / "seed-work")
        try:
            blocks.seed(source_root, source_inputs, vectors)
        finally:
            blocks.close()
    logical = {}
    def add(relative, checksum=None):
        relative_path(relative)
        path = bound(root, relative)
        logical[relative] = {"path": relative, "sha256": checksum or sha256_file(path),
                             "bytes": path.stat().st_size,
                             "asset": "asset-" + digest_json(relative)[:24] + path.suffix}
    active = read_json(root / "active_data_candidate.json")
    add("active_data_candidate.json")
    add(f"processed/{build.name}/{active['candidate_name']}")
    add(f"processed/{build.name}/index_inputs/units.json")
    for part in inputs["parts"]:
        add(f"processed/{build.name}/index_inputs/{part['path']}", part["sha256"])
    seeds = []
    for marker in sorted((root / "model_cache").glob("*/*.done.json")):
        block = checked(read_json(marker))
        encoder = block["identity"]["encoder"]
        family = next((family for family, name in (("bge", "dense"), ("qwen", "second_dense"))
                       if all(encoder.get(k) == v for k, v in config[name].items())), None)
        if family is None or any(encoder.get(k) != runtime[k] for k in ("torch", "transformers")):
            continue
        model_file = MODEL_FILES[family == "qwen"]
        if encoder.get("inference_code_sha256") != sha256_file(checkout / "src/vietmedbridge" / model_file):
            continue
        if family == "qwen" and (encoder.get("input_role") != "corpus" or encoder.get("fine_tuned") is not False
                                 or encoder.get("bitsandbytes") != runtime["bitsandbytes"]):
            continue
        try:
            validate_encoder(contract, family, encoder, block["identity"]["dimension"])
        except ValueError:
            continue
        relative = marker.relative_to(root).as_posix()
        add(relative)
        add(block["path"], block["sha256"])
        seeds.append(relative)
    # Content blocks seed existing work; mappings are regenerated from those
    # blocks without GPU inference. Never bundle a view from another producer.
    job = seal({"schema": "portable-embeddings-v1", "repo_url": REPO_URL, "code_commit": code_commit,
        "team": {"accounts": accounts, "gpu_count": gpu_count},
        "build_run": build.name, "candidate": candidate["candidate_manifest_sha256"],
        "inputs": inputs["manifest_sha256"], "input_count": inputs["input_count"], "config": config,
        "runtime": runtime, "resources": resources, "seeds": seeds,
        "model_code": {n: sha256_file(checkout / "src/vietmedbridge" / n) for n in MODEL_FILES},
        "files": sorted(logical.values(), key=lambda f: f["path"])})
    target.mkdir(parents=True, exist_ok=True)
    if (target / "job.json").exists() and read_json(target / "job.json") != job:
        raise ValueError("Export folder belongs to another job; choose a new local export folder.")
    for index, entry in enumerate(job["files"], 1):
        _copy_checked(bound(root, entry["path"]), target / entry["asset"], entry["sha256"])
        if index % 100 == 0:
            print(f"Kaggle inputs: {index}/{len(job['files'])} files", flush=True)
    atomic_json(target / "job.json", job)
    return job


def validate_job(job, checkout, *, runtime=None):
    checked(job)
    if (job["schema"] != "portable-embeddings-v1" or job["repo_url"] != REPO_URL
        or not re.fullmatch(r"[0-9a-f]{40}", job["code_commit"])):
        raise ValueError("Invalid Kaggle job contract.")
    for name, checksum in job["model_code"].items():
        if name not in MODEL_FILES:
            raise ValueError("Unexpected model source.")
        verify_file(Path(checkout) / "src/vietmedbridge" / name, checksum)
    if set(job["model_code"]) != set(MODEL_FILES):
        raise ValueError("Missing model source identity.")
    if runtime is not None and runtime != job["runtime"]:
        raise ValueError("Kaggle/Colab torch, transformers or bitsandbytes differ; install the exported runtime lock.")
    if job.get("team") is not None:
        kaggle_assignment(accounts=job["team"]["accounts"], account_id=0, gpu_count=job["team"]["gpu_count"])
    seen, assets = set(), set()
    for entry in job["files"]:
        relative_path(entry["path"])
        relative_path(entry["asset"])
        if (entry["path"] in seen or entry["asset"] in assets
            or Path(entry["asset"]).name != entry["asset"] or not entry["asset"].startswith("asset-")):
            raise ValueError("Invalid or duplicate portable input path.")
        seen.add(entry["path"])
        assets.add(entry["asset"])
    return job


def hydrate_job(job_dir, data_root, checkout):
    job_dir, root = Path(job_dir), Path(data_root)
    job = validate_job(read_json(job_dir / "job.json"), checkout)
    for entry in job["files"]:
        _copy_checked(job_dir / entry["asset"], bound(root, entry["path"]), entry["sha256"])
    _, candidate, inputs = load_handoff(root)
    if (candidate["candidate_manifest_sha256"] != job["candidate"] or inputs["manifest_sha256"] != job["inputs"]
        or resources_relative(candidate, job["config"]) != job["resources"]):
        raise ValueError("Kaggle input binding changed.")
    return job


def _entry(root, path):
    return {"path": Path(path).relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size}


def publish_receipt(root, output, job, family, saved, encoder, *, state=None):
    root, output = Path(root), Path(output)
    receipt_path = output / "receipts" / f"{family}-{saved['start']:012d}.json"
    if receipt_path.exists():
        old = checked(read_json(receipt_path))
        if old["job"] != job["manifest_sha256"] or old["map"] != saved or old["encoder"] != encoder.identity:
            raise ValueError("Existing portable receipt differs.")
        return
    view = root / job["resources"] / family
    mapping = view / saved["path"]
    verify_file(mapping, saved["sha256"])
    refs = pq.read_table(mapping).to_pylist()
    # Index marker PATHS once; do not rebuild a million-row SQLite cache for
    # each receipt. Multiple keys can legitimately yield identical vectors.
    state = state if state is not None else {}
    seen_markers, markers = state.setdefault("seen", set()), state.setdefault("markers", {})
    identity = {"encoder": encoder.identity, "dimension": encoder.dimension, "format": "content-float32-l2-v1"}
    directory = root / "model_cache" / digest_json(identity)[:24]
    for marker in directory.glob("*.done.json"):
        if marker not in seen_markers:
            block = checked(read_json(marker))
            if block["identity"] != identity:
                raise ValueError("Content marker producer differs.")
            markers.setdefault(block["path"], []).append(marker)
            seen_markers.add(marker)
    seed_files = {f["path"]: f["sha256"] for f in job["files"]}
    selected, dependencies = {}, {}
    entries = state.setdefault("entries", {})
    for vector_path in sorted({ref["path"] for ref in refs}):
        vector, block_markers = bound(root, vector_path), markers.get(vector_path)
        if not block_markers:
            raise ValueError("A vector has no immutable content marker.")
        for path in (vector, *block_markers):
            relative = path.relative_to(root).as_posix()
            if relative in seed_files:
                dependencies[relative] = seed_files[relative]
            else:
                if relative not in entries:
                    entries[relative] = _entry(root, path)
                selected[relative] = entries[relative]
    marker = view / f"map-{saved['start']:012d}.done.json"
    for path in (mapping, marker):
        selected[path.relative_to(root).as_posix()] = _entry(root, path)
    # Publish data before the completion receipt; an interruption leaves no false success.
    copied = state.setdefault("copied", set())
    for entry in selected.values():
        pair = (entry["path"], entry["sha256"])
        if pair not in copied:
            _copy_checked(bound(root, entry["path"]), bound(output / "data", entry["path"]), entry["sha256"])
            copied.add(pair)
    receipt = seal({"job": job["manifest_sha256"], "family": family, "encoder": encoder.identity,
        "dimension": encoder.dimension, "map": saved, "files": sorted(selected.values(), key=lambda x: x["path"]),
        "dependencies": dependencies})
    atomic_json(receipt_path, receipt)


def finalize_result(output, job, *, processes=None):
    output = Path(output)
    if (output / "job.json").exists() and read_json(output / "job.json") != job:
        raise ValueError("Kaggle output directory belongs to another job.")
    receipts = [_entry(output, p) for p in sorted((output / "receipts").glob("*.json"))]
    atomic_json(output / "job.json", job)
    result = seal({"schema": "portable-result-v1", "job": job["manifest_sha256"],
        "state": "WORKER_FAILED_WITH_CHECKPOINTS" if any(processes or []) else "CHECKPOINTED_PARTIAL", "receipts": receipts,
        "assignment": read_json(output / "assignment.json") if (output / "assignment.json").exists() else None,
        "processes": processes or [], "updated_at": utc_now(),
        "scope": "Completed corpus embedding parts only; no query inference or submission"})
    atomic_json(output / "result-manifest.json", result)
    return result


def carry_forward(result_dir, output):
    """A new saved version must include earlier session outputs, not just refs."""
    source, target = Path(result_dir), Path(output)
    result = checked(read_json(source / "result-manifest.json"))
    for entry in result["receipts"]:
        receipt = checked(read_json(bound(source, entry["path"])))
        for file in receipt["files"]:
            relative_path(file["path"])
            _copy_checked(bound(source / "data", file["path"]), bound(target / "data", file["path"]), file["sha256"])
        _copy_checked(bound(source, entry["path"]), bound(target, entry["path"]), entry["sha256"])


def run_worker(job_dir, scratch, output, checkout, *, worker_id, workers, deadline,
               max_output_bytes=18_000_000_000, encoder_factory=None):
    """CUDA_VISIBLE_DEVICES is set by the launcher BEFORE this process imports torch."""
    from .retrieval_models import TorchDenseEncoder
    from .qwen_models import TorchQwenEncoder
    from .runtime_profile import inference_batches
    job_dir, scratch, output = Path(job_dir), Path(scratch), Path(output)
    job = validate_job(read_json(job_dir / "job.json"), checkout,
                       runtime=embedding_runtime() if encoder_factory is None else None)
    build, _, inputs = load_handoff(scratch)
    local_workers = job.get("team", {}).get("gpu_count", workers)
    for family, spec_name in (("bge", "dense"), ("qwen", "second_dense")):
        if time.time() >= deadline:
            break
        assigned = [p for i, p in enumerate(inputs["parts"]) if i % workers == worker_id]
        if all((scratch / job["resources"] / family / f"map-{p['start']:012d}.done.json").exists()
               and (output / "receipts" / f"{family}-{p['start']:012d}.json").exists() for p in assigned):
            continue
        used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if used + local_workers * max(1_000_000, max((p["rows"] for p in assigned), default=0) *
                                ((1024 if family == "bge" else 4096) * 4 + 512)) > max_output_bytes:
            print(f"Worker {worker_id}: output budget reached before loading {family}.", flush=True)
            continue
        factory = encoder_factory or (TorchDenseEncoder if family == "bge" else TorchQwenEncoder)
        print(f"Worker {worker_id}/{workers}: {family}, {len(assigned)} assigned input parts.", flush=True)
        model = factory(job["config"][spec_name])
        encoder = model.for_role("corpus") if family == "qwen" and hasattr(model, "for_role") else model
        try:
            validate_encoder(job, family, encoder.identity, encoder.dimension)
            publishing = {}
            def stop(part):
                used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                reserve = max(1_000_000, part["rows"] * encoder.dimension * 4 + part["rows"] * 512)
                # Reserve for both processes that can pass this check concurrently.
                return time.time() >= deadline or used + local_workers * reserve > max_output_bytes
            content_embeddings(scratch, build / "index_inputs", inputs, encoder,
                scratch / job["resources"] / family, work_dir=scratch / f"work-{worker_id}",
                worker_id=worker_id, workers=workers,
                batch_size=32 if family == "bge" else (inference_batches()["embedding"] if encoder_factory is None else 2),
                should_stop=stop, on_part=lambda saved: publish_receipt(scratch, output, job, family, saved, encoder, state=publishing))
        finally:
            model.close()


def launch_kaggle(job_dir, checkout, *, output, scratch, session_started, hours=10.0, gpu_count=None,
                  accounts=1, account_id=0):
    import torch
    if not 0 < hours <= 10:
        raise ValueError("Use at most 10 hours, leaving time for setup and a normally saved Kaggle version.")
    count = torch.cuda.device_count() if gpu_count is None else gpu_count
    if type(count) is not int or not 1 <= count <= torch.cuda.device_count():
        raise ValueError("No requested CUDA devices available.")
    if accounts != 1 and gpu_count is None:
        raise ValueError("Parallel accounts must set the same explicit GPU count; select T4 x2 on each account.")
    assignment = kaggle_assignment(accounts=accounts, account_id=account_id, gpu_count=count)
    output, scratch = Path(output), Path(scratch)
    assignment_path = output / "assignment.json"
    if assignment_path.exists() and read_json(assignment_path) != assignment:
        raise ValueError("Kaggle output belongs to another account/team assignment. Keep the saved assignment to resume.")
    job = validate_job(read_json(Path(job_dir) / "job.json"), checkout, runtime=embedding_runtime())
    if job.get("team") is not None and job["team"] != {"accounts": accounts, "gpu_count": count}:
        raise ValueError("Kaggle account/GPU count differs from the shared exported job. Use its saved team settings.")
    if accounts > 1 and job.get("team") is None:
        raise ValueError("Parallel Kaggle accounts need a shared job exported by the updated notebook.")
    hydrate_job(job_dir, scratch, checkout)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(assignment_path, assignment)
    print(f"Kaggle account {account_id}/{accounts}: local GPUs {count}, global workers {assignment['worker_ids']}"
          f"/{assignment['workers']}; deadline {hours:g} hours from first cell.", flush=True)
    finalize_result(output, job)
    # Saved outputs attached as Add Input allow another bounded Kaggle session.
    for previous in sorted(Path("/kaggle/input").glob("**/result-manifest.json")):
        previous_result = read_json(previous)
        if previous_result.get("job") == job["manifest_sha256"]:
            if previous_result.get("assignment") != assignment and not (
                accounts == 1 and previous_result.get("assignment") is None
            ):
                print(f"Skipping saved Outputs for another account/assignment: {previous.parent.name}", flush=True)
                continue
            install_result(previous.parent, scratch, checkout, update_lock=False)
            carry_forward(previous.parent, output)
    finalize_result(output, job)
    processes = []
    def stop_processes():
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
    try:
        for device_id, worker_id in enumerate(assignment["worker_ids"]):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device_id)
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            command = [sys.executable, "-m", "vietmedbridge.portable_embeddings", "worker", str(job_dir),
                str(scratch), str(output), str(checkout), str(worker_id), str(assignment["workers"]), str(session_started + hours * 3600)]
            processes.append(subprocess.Popen(command, env=env))
        # Allow a part to drain, then stop hung workers in time to save Outputs.
        forced_finish = session_started + (hours + 0.75) * 3600
        statuses = [p.wait(timeout=max(1, forced_finish - time.time())) for p in processes]
    except subprocess.TimeoutExpired:
        stop_processes()
        statuses = [p.returncode for p in processes]
    except BaseException:
        stop_processes()
        finalize_result(output, job, processes=[p.returncode for p in processes])
        raise
    return finalize_result(output, job, processes=statuses)


def install_result(result_dir, data_root, checkout, *, update_lock=True):
    """Validate EVERY receipt and byte before publishing into native Colab caches."""
    result_dir, root, checkout = Path(result_dir), Path(data_root), Path(checkout)
    job = validate_job(read_json(result_dir / "job.json"), checkout)
    result = checked(read_json(result_dir / "result-manifest.json"))
    if result.get("schema") != "portable-result-v1" or result["job"] != job["manifest_sha256"]:
        raise ValueError("Kaggle output/job binding changed.")
    build, candidate, inputs = load_handoff(root)
    if (candidate["candidate_manifest_sha256"] != job["candidate"] or inputs["manifest_sha256"] != job["inputs"]
        or resources_relative(candidate, job["config"]) != job["resources"]):
        raise ValueError("Kaggle result belongs to another frozen corpus/config.")
    if read_json(checkout / "configs/retrieval_full.json") != job["config"]:
        raise ValueError("Kaggle and Colab model/retrieval configuration differ.")
    runtime_path = root / "retrieval_embedding_runtime_lock.json"
    if update_lock and runtime_path.exists() and checked(read_json(runtime_path))["runtime"] != job["runtime"]:
        raise ValueError("Existing embedding runtime differs; do not mix vector producers.")
    source_parts = {p["start"]: p for p in inputs["parts"]}
    source_indexes = {p["start"]: i for i, p in enumerate(inputs["parts"])}
    assignment = result.get("assignment")
    if assignment is not None and assignment != kaggle_assignment(
        accounts=assignment["accounts"], account_id=assignment["account_id"], gpu_count=assignment["gpu_count"]
    ):
        raise ValueError("Portable result account assignment differs.")
    if assignment is not None and job.get("team") is not None and job["team"] != {
        "accounts": assignment["accounts"], "gpu_count": assignment["gpu_count"]
    }:
        raise ValueError("Portable result team differs from its shared job.")
    seed_files = {f["path"]: f["sha256"] for f in job["files"]}
    copies, seen, verified, validated_blocks, view_configs = {}, set(), set(), {}, {}
    def verify_once(path, checksum):
        pair = (Path(path).resolve(), checksum)
        if pair not in verified:
            verify_file(path, checksum)
            verified.add(pair)
    for entry in result["receipts"]:
        relative_path(entry["path"])
        if not re.fullmatch(r"receipts/(bge|qwen)-[0-9]{12}\.json", entry["path"]):
            raise ValueError("Invalid receipt path.")
        receipt_path = bound(result_dir, entry["path"])
        verify_file(receipt_path, entry["sha256"])
        receipt = checked(read_json(receipt_path))
        family, saved, encoder = receipt["family"], checked(receipt["map"]), receipt["encoder"]
        if family not in ("bge", "qwen") or receipt["job"] != job["manifest_sha256"]:
            raise ValueError("Invalid portable receipt family/job.")
        part = source_parts.get(saved["start"])
        view = job["resources"] + "/" + family
        spec = job["config"]["dense" if family == "bge" else "second_dense"]
        validate_encoder(job, family, encoder, receipt["dimension"])
        if saved["path"] != f"map-{saved['start']:012d}.parquet":
            raise ValueError("Invalid portable mapping filename.")
        identity = {"input_manifest_sha256": inputs["manifest_sha256"], "input_count": inputs["input_count"],
                    "encoder": encoder, "dimension": receipt["dimension"], "format": "content-reference-v1"}
        if assignment is not None and (part is None or source_indexes[saved["start"]] % assignment["workers"] not in assignment["worker_ids"]):
            raise ValueError("Portable receipt is outside this account's assigned input parts.")
        if family in view_configs and view_configs[family] != identity:
            raise ValueError("Portable receipts mix model producers.")
        view_configs[family] = identity
        existing_config = root / view / "config.json"
        if existing_config.exists() and read_json(existing_config) != identity:
            raise ValueError("Existing Colab vector view has another input/model producer.")
        if (part is None or saved["rows"] != part["rows"] or saved["input_sha256"] != part["sha256"]
            or saved["signature"] != digest_json(identity) or any(encoder.get(k) != v for k, v in spec.items())
            or any(encoder.get(k) != v for k, v in job["runtime"].items() if k != "bitsandbytes" or family == "qwen")
            or encoder.get("inference_code_sha256") != job["model_code"][MODEL_FILES[family == "qwen"]]
            or (family, saved["start"]) in seen):
            raise ValueError("Portable map/input/model identity differs.")
        seen.add((family, saved["start"]))
        cache_identity = {"encoder": encoder, "dimension": receipt["dimension"], "format": "content-float32-l2-v1"}
        cache_prefix = "model_cache/" + digest_json(cache_identity)[:24] + "/"
        file_entries = {f["path"]: f for f in receipt["files"]}
        if len(file_entries) != len(receipt["files"]):
            raise ValueError("Duplicate portable output file.")
        for relative, item in file_entries.items():
            relative_path(relative)
            if not (relative.startswith(cache_prefix) or relative in (
                view + "/" + saved["path"], view + f"/map-{saved['start']:012d}.done.json")):
                raise ValueError("Portable output escapes its model cache/view.")
            path = bound(result_dir / "data", relative)
            verify_once(path, item["sha256"])
            destination = bound(root, relative)
            if destination.exists():
                verify_once(destination, item["sha256"])
            if relative in copies and copies[relative][1] != item["sha256"]:
                raise ValueError("Conflicting portable file hashes.")
            copies[relative] = (path, item["sha256"])
        for relative, checksum in receipt["dependencies"].items():
            relative_path(relative)
            if seed_files.get(relative) != checksum:
                raise ValueError("Unbound seed dependency.")
            verify_once(bound(root, relative), checksum)
        map_path = bound(result_dir / "data", view + "/" + saved["path"])
        verify_file(map_path, saved["sha256"])
        marker = read_json(bound(result_dir / "data", view + f"/map-{saved['start']:012d}.done.json"))
        if marker != saved:
            raise ValueError("Portable mapping completion marker differs.")
        verify_file(bound(build / "index_inputs", part["path"]), part["sha256"])
        texts = pq.read_table(build / "index_inputs" / part["path"], columns=["text"]).column("text").to_pylist()
        refs = pq.read_table(map_path).to_pylist()
        if len(refs) != len(texts):
            raise ValueError("Portable mapping row count differs.")
        block_paths = set()
        for relative in (*file_entries, *receipt["dependencies"]):
            if relative.startswith(cache_prefix) and relative.endswith(".done.json"):
                block_paths.add(relative)
        blocks = {}
        for relative in block_paths:
            block_file = bound(result_dir / "data", relative) if relative in file_entries else bound(root, relative)
            pair = (block_file, file_entries[relative]["sha256"] if relative in file_entries else receipt["dependencies"][relative])
            if pair in validated_blocks:
                block = validated_blocks[pair]
                blocks.setdefault(block["path"], []).append(block)
                continue
            block = checked(read_json(block_file))
            relative_path(block["path"])
            if block["identity"] != cache_identity or len(block["keys"]) != block["rows"]:
                raise ValueError("Portable block identity differs.")
            vector_file = bound(result_dir / "data", block["path"]) if block["path"] in file_entries else bound(root, block["path"])
            if block["path"] not in file_entries and receipt["dependencies"].get(block["path"]) != block["sha256"]:
                raise ValueError("Vector is not bound to a published or existing seed artifact.")
            verify_once(vector_file, block["sha256"])
            validate_vectors(np.load(vector_file, allow_pickle=False, mmap_mode="r"), block["rows"], receipt["dimension"])
            validated_blocks[pair] = block
            blocks.setdefault(block["path"], []).append(block)
        for text, ref in zip(texts, refs, strict=True):
            if type(ref["row"]) is not int or not any(
                ref["sha256"] == block["sha256"] and 0 <= ref["row"] < block["rows"]
                and block["keys"][ref["row"]] == key(text) for block in blocks.get(ref["path"], [])
            ):
                raise ValueError("Portable source text/vector row binding differs.")
    # Completion markers go last. Interrupted import can be safely repeated.
    for relative in sorted(copies, key=lambda p: (p.endswith(".done.json"), p)):
        path, checksum = copies[relative]
        _copy_checked(path, bound(root, relative), checksum)
    for family, identity in view_configs.items():
        atomic_json(root / job["resources"] / family / "config.json", identity)
    if update_lock:
        atomic_json(runtime_path, seal({"runtime": job["runtime"], "model_code": job["model_code"]}))
        lock_path = root / "retrieval_code_lock.json"
        lock = read_json(lock_path) if lock_path.exists() else {"repo_url": REPO_URL, "pipeline_api": 2,
            "workflow_api": "full-master-plan-strong-v10-resume-cpu-preparation"}
        lock["git_commit"] = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        atomic_json(lock_path, lock)
    report = {"state": "KAGGLE_CHECKPOINTS_IMPORTED", "parts": len(seen),
              "bge_parts": sum(f == "bge" for f, _ in seen), "qwen_parts": sum(f == "qwen" for f, _ in seen),
              "corpus_complete": False, "job": job["manifest_sha256"], "updated_at": utc_now()}
    report["assignment"] = assignment
    report["coverage"] = embedding_coverage(root, job)
    report["corpus_embeddings_complete"] = all(value["complete"] for value in report["coverage"].values())
    report["scope"] = "Corpus child embeddings only; coordinator query/document/search/reranker stages remain separate."
    if assignment is not None:
        atomic_json(root / "retrieval/portable_imports" / job["manifest_sha256"][:16]
                    / f"account-{assignment['account_id']}.json", report)
    atomic_json(root / "retrieval/portable_imports" / (job["manifest_sha256"][:16] + ".json"), report)
    return report


def embedding_coverage(data_root, job):
    """Count verified native maps after sequential account imports; never a URL score."""
    root = Path(data_root)
    _, _, inputs = load_handoff(root)
    coverage = {}
    for family in ("bge", "qwen"):
        view = root / job["resources"] / family
        config_path = view / "config.json"
        completed, rows = 0, 0
        if config_path.exists():
            config = read_json(config_path)
            validate_encoder(job, family, config["encoder"], config["dimension"])
            if config["input_manifest_sha256"] != inputs["manifest_sha256"] or config["input_count"] != inputs["input_count"]:
                raise ValueError("Imported embedding view belongs to another corpus.")
            signature = digest_json(config)
            for source in inputs["parts"]:
                marker = view / f"map-{source['start']:012d}.done.json"
                if not marker.exists():
                    continue
                saved = checked(read_json(marker))
                if any(saved.get(k) != v for k, v in {
                    "signature": signature, "start": source["start"], "rows": source["rows"],
                    "input_sha256": source["sha256"], "path": f"map-{source['start']:012d}.parquet"
                }.items()):
                    raise ValueError("Imported embedding coverage has a mismatched checkpoint.")
                verify_file(view / saved["path"], saved["sha256"])
                completed += 1
                rows += source["rows"]
        coverage[family] = {"completed_parts": completed, "total_parts": len(inputs["parts"]),
                            "input_rows": rows, "total_input_rows": inputs["input_count"],
                            "complete": completed == len(inputs["parts"])}
    return coverage


if __name__ == "__main__":
    if len(sys.argv) != 9 or sys.argv[1] != "worker":
        raise SystemExit("Use the Kaggle notebook launcher.")
    run_worker(*sys.argv[2:6], worker_id=int(sys.argv[6]), workers=int(sys.argv[7]), deadline=float(sys.argv[8]))
