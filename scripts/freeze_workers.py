"""Checkpoint and verify disjoint Notebook 03 freeze workers."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from vietmedbridge.artifacts import atomic_json, digest_json, local_workspace, read_json, sha256_file, verify_file


FREEZE_WORKER_SCHEMA = 1


def _source_locator_key(source):
    """Only remove Colab's Drive mount prefix, never match archive basenames."""
    if not isinstance(source, str) or not source:
        return None
    parts = PurePosixPath(source).parts
    if ".." in parts:
        raise ValueError("Source locator cannot contain parent traversal.")
    if parts[:3] == ("/", "content", "drive"):
        if len(parts) > 4 and parts[3] == "MyDrive":
            return "colab-drive:" + "/".join(parts[4:])
        if len(parts) > 5 and parts[3] == ".shortcut-targets-by-id":
            return "colab-drive:" + "/".join(parts[5:])
    return source


def _verify_source_alias(item, current_path, data_root, identities):
    """Bind an alias to the completed build's original Notebook 02 descriptor.

    Use the importer's size/mtime identity for tar files and manifest/archive
    hashes for split archives. This does not reread or certify raw payloads.
    """
    from vietmedbridge.external_import import _source_identity

    run = item.get("build_run")
    if (not isinstance(run, str) or not run or Path(run).name != run
        or run in {".", ".."}):
        raise ValueError("Invalid source alias build_run.")
    if data_root is None:
        raise ValueError("Drive source alias requires DATA_ROOT to verify the original build.")
    root = Path(data_root) / "processed" / run
    build = read_json(root / "build.json")
    config = read_json(root / "config.json")
    metadata = read_json(root / "external_source/source.json")
    if (not build.get("selected_range_complete")
        or build.get("snapshot_sha256") != item.get("snapshot_sha256")
        or build.get("snapshot_sha256") != digest_json({"signature": build["signature"], "parts": build["parts"]})
        or config.get("signature") != build["signature"]
        or config["signature"] != digest_json({k: v for k, v in config.items() if k != "signature"})
        or build.get("source_provenance") != metadata
        or metadata.get("input_signature") != digest_json(metadata["files"])
        or config.get("source_input_signature") != metadata["input_signature"]
        or _source_locator_key(metadata["source"].get("path")) != _source_locator_key(current_path)):
        raise ValueError(f"Drive source alias/build provenance mismatch: {run}")
    if current_path not in identities:
        identities[current_path] = _source_identity(current_path)
    original = {k: v for k, v in metadata["source"].items() if k != "path"}
    current = {k: v for k, v in identities[current_path].items() if k != "path"}
    if original != current:
        raise ValueError(f"Archive source identity changed across Drive aliases: {current_path}. "
                         "Check that this account uses the same shared Drive source.")


def collect_completed_build_refs(expected_paths, prior_batch, workers, *, data_root=None):
    """Collect all discovered sources, accepting verified Colab Drive aliases."""
    expected_paths = [str(path) for path in expected_paths]
    expected = set(expected_paths)
    by_key = {}
    for path in expected_paths:
        key = _source_locator_key(path)
        if key in by_key:
            raise ValueError(f"Ambiguous/duplicate discovered source: {path}")
        by_key[key] = path
    identities = {}

    def match(item):
        source = item.get("source_path")
        current = by_key.get(_source_locator_key(source))
        if current is not None and source != current:
            _verify_source_alias(item, current, data_root, identities)
        return current

    gathered = {}

    def remember(item, source):
        if source in gathered and any(gathered[source].get(key) != item.get(key)
                                     for key in ("build_run", "snapshot_sha256")):
            raise ValueError(f"Hai worker đã tạo build không khớp cho cùng archive: {source}")
        gathered[source] = {**gathered.get(source, {}), **item, "source_path": source}

    for item in prior_batch.get("builds", []):
        if item.get("state") in {"COMPLETE", "FROZEN"}:
            source = match(item)
            if source is not None:
                remember(item, source)
    for name, worker in workers:
        items = [(item, source) for item in worker.get("builds", [])
                 if (source := by_key.get(_source_locator_key(item.get("source_path")))) is not None]
        if not items:
            continue
        if worker.get("state") != "COMPLETE":
            raise RuntimeError(f"Worker chưa hoàn tất: {name} ({worker.get('state')})")
        for item, source in items:
            if item.get("state") not in {"COMPLETE", "FROZEN"}:
                raise RuntimeError(f"Build worker chưa hoàn tất: {item.get('source_path')}")
            match(item)
            # Preserve a prior frozen candidate reference when the worker only
            # records its original Notebook 02 completion.
            remember(item, source)
    if set(gathered) != expected:
        known = [item.get("source_path") for item in prior_batch.get("builds", [])]
        known += [item.get("source_path") for _, worker in workers for item in worker.get("builds", [])]
        raise ValueError(f"Worker coverage chưa khớp batch; missing={sorted(expected-set(gathered))}, "
                         f"extra={sorted(set(gathered)-expected)}; manifests={len(workers)}, "
                         f"known_sources={known[:3]}. Check DATA_ROOT points to the team's shared data.")
    return [gathered[path] for path in expected_paths]


def _read_candidate(build_dir, name, build, config):
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid frozen candidate filename.")
    candidate = read_json(build_dir / name)
    sha = digest_json({key: value for key, value in candidate.items()
                       if key != "candidate_manifest_sha256"})
    if (candidate.get("state") != "FROZEN_CANDIDATE"
        or not candidate.get("selected_range_complete")
        or candidate.get("candidate_manifest_sha256") != sha
        or candidate.get("snapshot_sha256") != build.get("snapshot_sha256")
        or any(candidate.get(key) != build.get(key) for key in (
            "signature", "parts", "counts", "chunking", "origin_corpus_sha256", "schema_version"))
        or not candidate.get("golden", {}).get("passed")
        or candidate["golden"].get("tokenizer") != config["tokenizer"]
        or candidate["golden"].get("chunking") != config["chunks"]
        or candidate.get("health", {}).get("snapshot_sha256") != build.get("snapshot_sha256")
        or not candidate.get("health", {}).get("integrity", {}).get("passed")
        or candidate["health"]["integrity"].get("official_membership") != "VERIFIED"):
        raise ValueError(f"Frozen candidate integrity/policy mismatch: {name}")
    return candidate


def freeze_build_for_notebook(item, *, data_root, official_links, tokenizer,
                              golden_report, work_dir, audit_size):
    """Keep a cached candidate immutable; store current revalidation separately."""
    from vietmedbridge.health import freeze_candidate, health_report
    from vietmedbridge.index_inputs import prepare_index_inputs

    root = Path(item["build_dir"])
    build, config = item["build"], item["config"]
    candidate, candidate_name = None, None
    cached_config = root / "index_inputs" / "config.json"
    if cached_config.is_file():
        bound_sha = read_json(cached_config).get("candidate_manifest_sha256")
        if not isinstance(bound_sha, str) or len(bound_sha) != 64:
            raise ValueError("Index input cache has an invalid candidate binding.")
        candidate_name = f"candidate-{bound_sha[:16]}.json"
        candidate = _read_candidate(root, candidate_name, build, config)
        if candidate["candidate_manifest_sha256"] != bound_sha:
            raise ValueError("Index input cache candidate binding changed.")
    else:
        preferred = item.get("batch_item", {}).get("candidate_name")
        names = ([preferred] if preferred else []) + [p.name for p in sorted(root.glob("candidate-*.json"))]
        for name in dict.fromkeys(names):
            if not (root / name).is_file():
                continue
            saved = read_json(root / name)
            if saved.get("snapshot_sha256") != build["snapshot_sha256"]:
                continue
            candidate = _read_candidate(root, name, build, config)
            candidate_name = name
            break
    validation = candidate
    if candidate is None or candidate["golden"].get("code_sha256") != golden_report["code_sha256"]:
        validation = None
        for path in sorted(root.glob("candidate-*.json")):
            saved = read_json(path)
            if (saved.get("snapshot_sha256") == build["snapshot_sha256"]
                and saved.get("golden", {}).get("code_sha256") == golden_report["code_sha256"]):
                validation = _read_candidate(root, path.name, build, config)
                break
        if validation is None:
            crawl_dir = Path(data_root) / "crawl" / build["crawl_run"] if build.get("crawl_run") else None
            health = health_report(root, official_links=official_links, crawl_dir=crawl_dir,
                                   work_dir=work_dir, audit_size=audit_size)
            validation = freeze_candidate(root, health, golden_report=golden_report)
    if candidate is None:
        candidate = validation
        candidate_name = f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json"
    inputs = prepare_index_inputs(root, candidate_name, tokenizer, work_dir=work_dir, part_size=4096)
    record = {"state": "COMPLETE", "snapshot_sha256": build["snapshot_sha256"],
              "candidate_name": candidate_name,
              "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
              "validation_candidate_name": f"candidate-{validation['candidate_manifest_sha256'][:16]}.json",
              "validation_candidate_manifest_sha256": validation["candidate_manifest_sha256"],
              "index_inputs_manifest_sha256": inputs["manifest_sha256"]}
    return candidate, inputs, record


def freeze_batch_signature(build_refs, *, code_sha256: str, team_size: int) -> str:
    """Bind worker manifests to one ordered corpus/build set and code revision."""
    if type(team_size) is not int or team_size < 1:
        raise ValueError("Freeze team size must be a positive integer.")
    builds = []
    seen = set()
    for item in build_refs:
        run = item.get("build_run")
        snapshot = item.get("snapshot_sha256")
        if (not isinstance(run, str) or not run or Path(run).name != run
            or run in {".", ".."} or run in seen):
            raise ValueError("Freeze batch must contain unique build_run values.")
        if not isinstance(snapshot, str) or not snapshot:
            raise ValueError(f"Freeze batch is missing snapshot_sha256 for {run}.")
        seen.add(run)
        # A path is a runtime locator, not corpus identity. The immutable build
        # snapshot binds the producer policy and every processed shard checksum.
        builds.append({"build_run": run, "snapshot_sha256": snapshot})
    if not builds:
        raise ValueError("Freeze batch is empty.")
    return digest_json({"workflow": "parallel-freeze-v3-drive-aliases", "code_sha256": code_sha256,
                        "orchestrator_sha256": sha256_file(Path(__file__)),
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


def verify_freeze_record(data_root, run_name, ref, record, code_sha256):
    import pyarrow.parquet as pq
    from vietmedbridge.validation import artifact_paths

    root = Path(data_root) / "processed" / run_name
    build, config = read_json(root / "build.json"), read_json(root / "config.json")
    if (not build.get("selected_range_complete")
        or build.get("snapshot_sha256") != ref.get("snapshot_sha256")
        or record.get("snapshot_sha256") != build.get("snapshot_sha256")
        or build.get("snapshot_sha256") != digest_json({"signature": build["signature"], "parts": build["parts"]})
        or config.get("signature") != build["signature"]
        or config["signature"] != digest_json({k: v for k, v in config.items() if k != "signature"})):
        raise ValueError(f"Freeze worker build snapshot mismatch: {run_name}")
    candidate = _read_candidate(root, record.get("candidate_name"), build, config)
    if candidate["candidate_manifest_sha256"] != record.get("candidate_manifest_sha256"):
        raise ValueError(f"Frozen candidate integrity mismatch: {run_name}")
    validation = _read_candidate(root, record.get("validation_candidate_name", record["candidate_name"]), build, config)
    validation_sha = record.get("validation_candidate_manifest_sha256", record["candidate_manifest_sha256"])
    if (validation["candidate_manifest_sha256"] != validation_sha
        or validation["golden"].get("code_sha256") != code_sha256):
        raise ValueError(f"Current freeze validation is missing/stale: {run_name}")
    counts = build["counts"]
    if (counts.get("input_records") != build.get("requested_input_records")
        or counts["input_records"] != counts.get("documents", 0) + counts.get("failures", 0)
        or validation["health"]["integrity"].get("checked_input_records") != counts["input_records"]):
        raise ValueError(f"Input/outcome counts mismatch: {run_name}")
    for kind in ("documents", "children", "parents", "sections", "failures", "ledger"):
        artifact_paths(root, build, kind)
    for entry in validation["health"].get("files", {}).values():
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Health artifact escapes its build directory.")
        verify_file(path, entry["sha256"])

    inputs_root = root / "index_inputs"
    inputs, policy = read_json(inputs_root / "units.json"), read_json(inputs_root / "config.json")
    inputs_sha = digest_json({key: value for key, value in inputs.items() if key != "manifest_sha256"})
    expected_policy = {"candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
        "tokenizer": candidate["golden"]["tokenizer"], "builder": inputs.get("builder"),
        "dense_max_tokens": candidate["chunking"]["dense_max_tokens"], "part_size": inputs.get("part_size")}
    if (inputs.get("state") != "COMPLETE" or policy != expected_policy
        or any(inputs.get(key) != value for key, value in policy.items())
        or inputs.get("signature") != digest_json(policy)
        or inputs.get("manifest_sha256") != inputs_sha
        or inputs_sha != record.get("index_inputs_manifest_sha256")
        or inputs.get("input_count") != validation["health"]["chunks"]["distinct_representations"]):
        raise ValueError(f"Index-input manifest integrity mismatch: {run_name}")
    cursor = 0
    for part in inputs.get("parts", []):
        path = (inputs_root / part["path"]).resolve()
        if not path.is_relative_to(inputs_root.resolve()):
            raise ValueError("Index input part escapes its directory.")
        if (part.get("signature") != inputs["signature"] or part.get("start") != cursor
            or part.get("rows") != min(policy["part_size"], inputs["input_count"] - cursor)
            or part["rows"] <= 0):
            raise ValueError(f"Index-input part range mismatch: {run_name}")
        verify_file(path, part["sha256"])
        if pq.ParquetFile(path).metadata.num_rows != part["rows"]:
            raise ValueError(f"Index-input actual row count mismatch: {run_name}")
        cursor += part["rows"]
    if cursor != inputs["input_count"]:
        raise ValueError(f"Index-input part coverage incomplete: {run_name}")
    return {**record, "candidate": candidate, "inputs": inputs, "validation": validation}


def verify_freeze_workers(data_root, build_refs, *, batch_signature: str,
                          code_sha256: str, team_size: int) -> list[dict]:
    """Verify complete worker coverage and the candidate/index handoff manifests."""
    records: dict[str, dict] = {}
    for worker_id in range(team_size):
        path = freeze_worker_manifest_path(data_root, batch_signature=batch_signature,
                                           worker_id=worker_id)
        expected = partition_freeze_builds(build_refs, team_size=team_size, worker_id=worker_id)
        expected_runs = [item["build_run"] for item in expected]
        reused = False
        if not path.is_file():
            # Source locator/orchestrator updates need not discard completed
            # receipts. Bind every assigned snapshot and verify actual artifacts
            # below, instead of trusting another batch signature on its own.
            for previous_path in sorted((Path(data_root) / "freeze_workers").glob(f"*/worker-{worker_id}.json")):
                previous = read_json(previous_path)
                if (previous.get("schema_version") == FREEZE_WORKER_SCHEMA
                    and previous.get("code_sha256") == code_sha256
                    and previous.get("worker_id") == worker_id
                    and previous.get("team_size") == team_size
                    and previous.get("state") == "COMPLETE"
                    and previous.get("assigned_build_runs") == expected_runs
                    and set(previous.get("completed", {})) == set(expected_runs)
                    and all(previous["completed"][item["build_run"]].get("snapshot_sha256")
                            == item["snapshot_sha256"] for item in expected)):
                    path, reused = previous_path, True
                    break
            if not reused:
                raise FileNotFoundError(f"Freeze worker manifest missing: {path}")
        worker = read_json(path)
        payload = {key: value for key, value in worker.items() if key != "manifest_sha256"}
        if (worker.get("schema_version") != FREEZE_WORKER_SCHEMA
            or digest_json(payload) != worker.get("manifest_sha256")
            or (not reused and worker.get("batch_signature") != batch_signature)
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
            records[run_name] = verify_freeze_record(data_root, run_name, ref, record, code_sha256)
        if reused:
            print(f"Reused verified freeze checkpoint: worker-{worker_id} ({path.parent.name})", flush=True)
    expected_runs = [item["build_run"] for item in build_refs]
    if set(records) != set(expected_runs):
        missing = sorted(set(expected_runs) - set(records))
        extra = sorted(set(records) - set(expected_runs))
        raise ValueError(f"Freeze workers do not cover the batch: missing={missing}, extra={extra}")
    return [records[run] for run in expected_runs]


def audit_freeze_batch(data_root, build_refs, records, *, work_dir=None):
    """Reconcile all selected outcomes and distinguish records from unique IDs."""
    from vietmedbridge.dataset import _connection

    if len(build_refs) != len(records):
        raise ValueError("Batch audit is missing freeze records.")
    totals, ledgers, source_gaps, sources = {}, [], 0, []
    for ref, record in zip(build_refs, records):
        candidate = record["candidate"]
        counts = candidate["counts"]
        if (candidate["snapshot_sha256"] != ref["snapshot_sha256"]
            or counts["input_records"] != candidate["requested_input_records"]
            or counts["input_records"] != counts.get("documents", 0) + counts.get("failures", 0)):
            raise ValueError(f"Batch input/outcome count mismatch: {ref['build_run']}")
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        root = Path(data_root) / "processed" / ref["build_run"]
        ledgers.extend(str(root / part["files"]["ledger"]["path"]) for part in candidate["parts"])
        source_gaps += candidate.get("source_audit", {}).get("coverage_gaps", {}).get("official_ids", 0)
        sources.append({"build_run": ref["build_run"], "source_path": ref.get("source_path"),
            "snapshot_sha256": candidate["snapshot_sha256"], "input_records": counts["input_records"],
            "documents": counts["documents"], "failures": counts["failures"]})
    with local_workspace(work_dir) as temporary:
        with _connection(temporary) as con:
            con.read_parquet(ledgers).create_view("batch_ledger")
            actual, unique = con.execute("SELECT count(*), count(DISTINCT doc_id) FROM batch_ledger").fetchone()
    if actual != totals["input_records"]:
        raise ValueError("Batch ledger record count does not match the complete build set.")
    report = {"state": "BATCH_COVERAGE_VERIFIED", "archives": len(build_refs),
        "counts": totals, "sources": sources, "input_records": actual, "unique_official_ids": unique,
        "duplicate_official_ids_across_builds": actual - unique,
        "unaccounted_input_records": actual - totals["documents"] - totals["failures"],
        "source_coverage_gap_records": source_gaps,
        "all_selected_archives_frozen": True,
        "quality_scope": "Integrity and outcome coverage; relevance and human source QA are not certified"}
    report["report_sha256"] = digest_json(report)
    atomic_json(Path(data_root) / "reports" / "freeze_batch_coverage.json", report)
    return report
