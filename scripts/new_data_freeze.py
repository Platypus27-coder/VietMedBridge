"""Freeze completed intake builds without touching the active submission root.

This adapter is embedded in the notebook, outside the pinned producer package.
Completed builds resume from sealed JSON. Payload checks happen on first use;
unfinished builds retain per-part input checkpoints on Drive.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import shutil
from uuid import uuid4

from vietmedbridge.artifacts import (atomic_json, code_fingerprint, digest_json,
    publish_file, read_json, utc_now)

PARENT_CAMPAIGN = "btc-new-71be0e12-20261019-v1"
EXTRA_CAMPAIGN = "btc-new-71be0e12-00008-20261020-v1"
KINDS = ("documents", "children", "parents", "sections", "failures", "ledger")


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Artifact path escapes its owning directory.")
    return path


def checked(value, field="manifest_sha256"):
    if value.get(field) != digest_json({k: v for k, v in value.items() if k != field}):
        raise ValueError("Checkpoint JSON checksum changed.")
    return value


def seal(path, value):
    result = dict(value)
    result["manifest_sha256"] = digest_json(result)
    atomic_json(path, result)
    return result


def collect_intake_builds(project):
    """Read ten receipts/configs, never TARs, mappings, or processed payloads."""
    project = Path(project).resolve()
    parent = project / "data_staging" / PARENT_CAMPAIGN
    extra = project / "data_staging" / EXTRA_CAMPAIGN
    report = read_json(parent / "full_intake_status.json")
    scope = read_json(parent / "full_intake_scope.json")
    if (report.get("state") != "FULL_INTAKE_COMPLETE"
            or report.get("scope_sha256") != digest_json(scope)
            or report.get("completed_sources") != 9 or report.get("total_sources") != 9
            or report.get("pending_sources")):
        raise ValueError("The parent nine-source intake is incomplete or changed.")
    snapshot = scope["base_plan"]["required_snapshot_id"]
    groups = [(parent, scope["base_plan"]["sources"]),
              (parent / "additional-frontiers", scope["additional_sources"]),
              (parent / "retained-recovery", [scope["retained"]]),
              (extra, read_json(extra / "intake_plan.json")["sources"])]
    result, parent_counts = [], {}
    for root, expected in groups:
        plan = read_json(root / "intake_plan.json")
        status = read_json(root / "intake_status.json")
        if (status.get("state") != "INTAKE_COMPLETE"
                or status.get("plan_sha256") != digest_json(plan)
                or status.get("completed_sources") != len(expected)
                or status.get("total_sources") != len(expected)
                or len(plan["sources"]) != len(expected)):
            raise ValueError(f"Intake phase is incomplete or changed: {root.name}")
        # Retained's portable adapter adds frontier/binding fields to the plan;
        # the audited identity remains the same archive and BTC source.
        for source, reference in zip(plan["sources"], expected):
            keys = ("job_id", "snapshot_id", "drive_file_id", "archive", "mode")
            if any(source.get(k) != reference.get(k) for k in keys):
                raise ValueError("Phase source order/archive identity changed.")
            run = source["job_id"]
            if source.get("snapshot_id") != snapshot or source.get("mode") != "NEW_SNAPSHOT_SOURCE":
                raise ValueError("Source is outside the pinned new BTC snapshot.")
            receipt = read_json(inside(root, f"sources/{run}/complete.json"))
            build_dir = inside(root, f"processed/{run}")
            build, config = read_json(build_dir / "build.json"), read_json(build_dir / "config.json")
            if (receipt.get("state") != "COMPLETE" or receipt.get("source") != source
                    or receipt.get("plan_sha256") != digest_json(plan)
                    or receipt.get("build_run") != run
                    or receipt.get("build_snapshot_sha256") != build.get("snapshot_sha256")
                    or not build.get("selected_range_complete")
                    or build.get("snapshot_sha256") != digest_json({"signature": build["signature"], "parts": build["parts"]})
                    or config.get("signature") != build.get("signature")
                    or config["signature"] != digest_json({k: v for k, v in config.items() if k != "signature"})
                    or receipt.get("data_counts") != build.get("counts")):
                raise ValueError(f"Receipt/build/config identity changed: {run}")
            counts = build["counts"]
            if (counts.get("input_records") != build.get("requested_input_records")
                    or counts["input_records"] != counts.get("documents", 0) + counts.get("failures", 0)
                    or receipt["counts"].get("ADDED_VALID_DOCUMENT", 0) != counts.get("documents", 0)
                    or receipt["counts"].get("PIPELINE_FAILURE", 0) != counts.get("failures", 0)):
                raise ValueError(f"Input/outcome coverage changed: {run}")
            if root != extra:
                for key, count in receipt["counts"].items():
                    parent_counts[key] = parent_counts.get(key, 0) + count
                ref = next((b for b in report["builds"] if b["build_run"] == run), {})
                if ref.get("snapshot_sha256") != build["snapshot_sha256"]:
                    raise ValueError(f"Parent report/build binding changed: {run}")
            result.append({"build_run": run, "build_dir": build_dir.relative_to(project).as_posix(),
                "snapshot_sha256": build["snapshot_sha256"], "config_sha256": digest_json(config),
                "receipt_sha256": digest_json(receipt), "counts": counts,
                "tokenizer": config["tokenizer"], "chunking": config["chunks"],
                "producer_code_sha256": config["code_sha256"],
                "official_links_sha256": config["official_links_sha256"]})
    expected_runs = {f"new-{i:05d}" for i in range(9)} | {scope["retained"]["job_id"]}
    if (len(result) != 10 or {r["build_run"] for r in result} != expected_runs
            or parent_counts != report["counts"]):
        raise ValueError("Expected exactly the completed nine sources plus new-00008.")
    first = result[0]
    for item in result:
        for key in ("tokenizer", "chunking", "producer_code_sha256", "official_links_sha256"):
            if item[key] != first[key]:
                raise ValueError(f"Mixed producer/tokenizer/official-links contract: {key}")
    return result


def _stat(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def copy_verified(source, destination, checksum):
    """One streaming read from Drive; same-VM copies reuse a local stat proof."""
    source, destination = Path(source), Path(destination)
    proof = destination.with_name(destination.name + ".verified.json")
    if destination.is_file() and proof.is_file():
        saved = checked(read_json(proof))
        if saved["sha256"] == checksum and saved["stat"] == _stat(destination):
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + uuid4().hex + ".partial")
    try:
        digest = hashlib.sha256()
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            for block in iter(lambda: incoming.read(4 * 1024**2), b""):
                digest.update(block)
                outgoing.write(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Source payload checksum changed: {source.name}")
        os.replace(temporary, destination)
        seal(proof, {"sha256": checksum, "stat": _stat(destination)})
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@contextmanager
def bounded_freeze_sql(work, memory_mb):
    from vietmedbridge import health, validation
    from vietmedbridge.dataset import _connection
    if type(memory_mb) is not int or not 64 <= memory_mb <= 2048:
        raise ValueError("SQL RAM budget must be 64..2048 MiB.")
    originals = health._connection, validation._connection

    def connect(temp_dir, memory_limit=None):
        con = _connection(temp_dir, memory_limit=memory_limit or f"{memory_mb}MiB")
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        return con

    health._connection = validation._connection = connect
    try:
        yield
    finally:
        health._connection, validation._connection = originals


class NewDataFreeze:
    def __init__(self, project, work_dir):
        self.project, self.work = Path(project).resolve(), Path(work_dir).resolve()
        if (self.work.is_relative_to(self.project) or self.project.is_relative_to(self.work)
                or self.work.is_relative_to(Path("/content/drive"))):
            raise ValueError("Freeze scratch must be separate from Drive/project data.")
        self.builds = collect_intake_builds(self.project)
        self.root = self.project / "data_staging" / PARENT_CAMPAIGN / "freeze_batch_v1"
        self.contract = {"schema_version": 1, "workflow": "NEW_BTC_INTAKE_FREEZE_V1",
            "producer_code_sha256": code_fingerprint(), "sources": self.builds,
            "input_part_size": 4096, "payload_verification": "SHA256_ON_USE"}
        if self.contract["producer_code_sha256"] != self.builds[0]["producer_code_sha256"]:
            raise ValueError("Load the intake producer commit before freezing; do not rewrite its code lock.")
        contract_path = self.root / "contract.json"
        if contract_path.is_file() and checked(read_json(contract_path)) != dict(
                self.contract, manifest_sha256=digest_json(self.contract)):
            raise ValueError("Frozen batch source/code policy changed; retain its checkpoints.")
        self.binding = seal(contract_path, self.contract)["manifest_sha256"]
        self.work.mkdir(parents=True, exist_ok=True)

    def _candidate(self, item, record):
        root = inside(self.project, item["build_dir"])
        candidate = checked(read_json(inside(root, record["candidate_name"])), "candidate_manifest_sha256")
        if (record["contract_sha256"] != self.binding
                or record["snapshot_sha256"] != item["snapshot_sha256"]
                or candidate["candidate_manifest_sha256"] != record["candidate_manifest_sha256"]
                or candidate.get("snapshot_sha256") != item["snapshot_sha256"]
                or candidate.get("state") != "FROZEN_CANDIDATE"
                or candidate.get("counts") != item["counts"]
                or not candidate.get("health", {}).get("integrity", {}).get("passed")
                or candidate["health"]["integrity"].get("official_membership") != "VERIFIED"
                or candidate["health"]["integrity"].get("checked_input_records") != item["counts"]["input_records"]
                or not candidate.get("golden", {}).get("passed")
                or candidate["golden"].get("code_sha256") != self.contract["producer_code_sha256"]
                or candidate["golden"].get("tokenizer") != item["tokenizer"]
                or candidate["golden"].get("chunking") != item["chunking"]):
            raise ValueError("Frozen candidate/receipt contract changed.")
        return candidate

    def completed(self, item):
        path = self.root / "builds" / item["build_run"] / "complete.json"
        if not path.is_file():
            return None
        record = checked(read_json(path))
        candidate = self._candidate(item, record)
        inputs = checked(read_json(inside(self.project, item["build_dir"]) / "index_inputs/units.json"))
        if (record.get("state") != "COMPLETE" or inputs.get("state") != "COMPLETE"
                or inputs["manifest_sha256"] != record["index_inputs_manifest_sha256"]
                or inputs["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]
                or inputs["input_count"] != record["unique_dense_inputs"]):
            raise ValueError("Completed model-input manifest changed.")
        cursor = 0
        for part in inputs["parts"]:
            inside(Path("inputs"), part["path"])
            if (part["start"] != cursor or not 0 < part["rows"] <= 4096
                    or part["signature"] != inputs["signature"]):
                raise ValueError("Completed input part coverage changed.")
            cursor += part["rows"]
        if cursor != inputs["input_count"]:
            raise ValueError("Completed input row coverage changed.")
        return record

    def _stage(self, item, *, inputs_only=False):
        source = inside(self.project, item["build_dir"])
        local = inside(self.work, "staged/" + item["snapshot_sha256"])
        build = read_json(source / "build.json")
        kinds = ("documents", "children") if inputs_only else KINDS
        entries = [part["files"][kind] for part in build["parts"] for kind in kinds]
        missing = sum(inside(source, e["path"]).stat().st_size for e in entries
                      if not inside(local, e["path"]).is_file())
        if shutil.disk_usage(self.work).free < 2 * missing + 1024**3:
            raise OSError("Colab disk lacks room for this one-build staging and SQL spill; preserve Drive checkpoints.")
        atomic_json(local / "build.json", build)
        atomic_json(local / "config.json", read_json(source / "config.json"))
        for index, entry in enumerate(entries, 1):
            copy_verified(inside(source, entry["path"]), inside(local, entry["path"]), entry["sha256"])
            if index % 12 == 0 or index == len(entries):
                print(f"Freeze staging {item['build_run']}: {index}/{len(entries)} payloads | bounded to one build", flush=True)
        return local

    def _input_local(self, item):
        """A frozen build needs metadata only until InputSpool requests a part."""
        source = inside(self.project, item["build_dir"])
        local = inside(self.work, "staged/" + item["snapshot_sha256"])
        atomic_json(local / "build.json", read_json(source / "build.json"))
        atomic_json(local / "config.json", read_json(source / "config.json"))
        return local

    def _restore_input_spool(self, item, candidate, state_dir):
        """Restore one committed SQLite checkpoint, not the processed Parquets."""
        inputs = inside(self.project, item["build_dir"]) / "index_inputs"
        if (inputs / "units.json").is_file():
            return
        cursor = inputs / "resume.json"
        if cursor.is_file():
            saved = checked(read_json(cursor))
            if saved.get("cursor") == candidate["health"]["chunks"]["distinct_representations"]:
                return  # Native consumer can seal the completed part manifest directly.
        path = state_dir / "input_spool.json"
        if not path.is_file():
            return  # Older checkpoint: native staging commits the spool once.
        saved = checked(read_json(path))
        policy = read_json(inputs / "config.json")
        if (saved["contract_sha256"] != self.binding
                or saved["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]
                or saved["input_signature"] != digest_json(policy)):
            raise ValueError("Input SQL checkpoint binding changed.")
        directory = inside(self.work, saved["directory"])
        if not directory.is_relative_to(self.work / "index_spools"):
            raise ValueError("Input SQL checkpoint directory escapes the spool root.")
        destination = directory / "representatives.sqlite"
        if not destination.is_file():
            print(f"RESTORE_INPUT_SQL {item['build_run']} | one saved payload; no corpus restaging", flush=True)
            copy_verified(inside(state_dir, saved["path"]), destination, saved["sha256"])

    @contextmanager
    def _durable_input_spool(self, item, candidate, state_dir):
        """Seal the full/tail spool before native code emits its first input part."""
        from vietmedbridge import index_inputs
        native = index_inputs.InputSpool
        owner = self
        checkpoint = state_dir / "input_spool.json"

        class DurableSpool(native):
            def stage(spool):
                count = super().stage()
                saved = checked(read_json(checkpoint)) if checkpoint.is_file() else None
                directory = spool.directory.resolve().relative_to(owner.work).as_posix()
                if saved is not None:
                    if (saved["contract_sha256"] != owner.binding
                            or saved["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]
                            or saved["directory"] != directory or saved["representatives"] != count):
                        raise ValueError("Completed input SQL checkpoint changed.")
                else:
                    spool.con.commit()  # DELETE journal; this connection is the only writer.
                    policy = read_json(inside(owner.project, item["build_dir"]) / "index_inputs/config.json")
                    relative = "input_spool/representatives.sqlite"
                    print(f"SAVE_INPUT_SQL {item['build_run']} | {count:,} representatives to Drive", flush=True)
                    checksum = publish_file(spool.path, inside(state_dir, relative))
                    seal(checkpoint, {"contract_sha256": owner.binding,
                        "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
                        "input_signature": digest_json(policy), "directory": directory,
                        "path": relative, "sha256": checksum, "representatives": count,
                        "minimum_id": spool.minimum_id})
                    print(f"INPUT_SQL_SAVED {item['build_run']} | future VM restores this checkpoint", flush=True)
                return count

        index_inputs.InputSpool = DurableSpool
        try:
            yield
        finally:
            index_inputs.InputSpool = native

    def run(self, *, load_tokenizer, load_links, load_golden, memory_mb=1024):
        from vietmedbridge.health import health_report, freeze_candidate
        from vietmedbridge.index_inputs import prepare_index_inputs
        for index, item in enumerate(self.builds, 1):
            self.phase = f"{index}/10 {item['build_run']}: receipt"
            if self.completed(item) is not None:
                print(f"SKIP_FROZEN {index}/10 {item['build_run']} | sealed JSON only; no payload scan", flush=True)
                continue
            source = inside(self.project, item["build_dir"])
            state_dir = self.root / "builds" / item["build_run"]
            frozen_path = state_dir / "frozen.json"
            frozen = checked(read_json(frozen_path)) if frozen_path.is_file() else None
            if frozen:
                candidate = self._candidate(item, frozen)
            self.phase = f"{index}/10 {item['build_run']}: local staging"
            local = self._input_local(item) if frozen else self._stage(item)
            def get_tokenizer():
                tokenizer, spec = load_tokenizer(item)
                if spec != item["tokenizer"]:
                    raise ValueError("Loaded tokenizer differs from the producer.")
                return tokenizer, spec
            if frozen is None:
                tokenizer, spec = get_tokenizer()
            else:
                # Native input resume does not need a tokenizer if all text
                # parts are already saved. Materialize only for missing text.
                class LazyTokenizer:
                    def __call__(self, *args, **kwargs):
                        return get_tokenizer()[0](*args, **kwargs)
                    def __getattr__(self, name):
                        return getattr(get_tokenizer()[0], name)
                tokenizer = LazyTokenizer()
            if frozen is None:
                links = load_links(item)
                golden = load_golden(item, tokenizer, spec)
                print(f"FREEZE_VALIDATE {index}/10 {item['build_run']} | first full integrity pass on local staging", flush=True)
                self.phase = f"{index}/10 {item['build_run']}: integrity/health on local files"
                with bounded_freeze_sql(self.work, memory_mb):
                    health = health_report(local, official_links=links, work_dir=self.work, audit_size=20)
                    candidate = freeze_candidate(local, health, golden_report=golden)
                for entry in health["files"].values():
                    if publish_file(inside(local, entry["path"]), inside(source, entry["path"])) != entry["sha256"]:
                        raise ValueError("Published health payload differs from validated local source.")
                atomic_json(source / "reports" / item["snapshot_sha256"][:16] / "health.json", health)
                name = "candidate-" + candidate["candidate_manifest_sha256"][:16] + ".json"
                atomic_json(source / name, candidate)
                frozen = seal(frozen_path, {"state": "FROZEN", "contract_sha256": self.binding,
                    "snapshot_sha256": item["snapshot_sha256"], "candidate_name": name,
                    "candidate_manifest_sha256": candidate["candidate_manifest_sha256"]})
            atomic_json(local / frozen["candidate_name"], candidate)
            print(f"FREEZE_INPUTS {index}/10 {item['build_run']} | resume missing input parts", flush=True)
            self.phase = f"{index}/10 {item['build_run']}: input parts"
            self._restore_input_spool(item, candidate, state_dir)
            def load_source_part(entry):
                copy_verified(inside(source, entry["path"]), inside(local, entry["path"]), entry["sha256"])
            with self._durable_input_spool(item, candidate, state_dir):
                inputs = prepare_index_inputs(local, frozen["candidate_name"], tokenizer,
                    work_dir=self.work, checkpoint_root=source / "index_inputs",
                    resume_policy="verify_on_use", source_loader=load_source_part)
            record = {k: v for k, v in frozen.items() if k != "manifest_sha256"}
            record.update(state="COMPLETE", build_run=item["build_run"], build_dir=item["build_dir"],
                index_inputs_manifest_sha256=inputs["manifest_sha256"], unique_dense_inputs=inputs["input_count"],
                input_parts=len(inputs["parts"]), counts=item["counts"], updated_at=utc_now())
            seal(state_dir / "complete.json", record)
            self.completed(item)
            # Only this owned local staging copy is disposable after its Drive receipt.
            if local.is_relative_to(self.work / "staged") and not local.is_relative_to(self.project):
                shutil.rmtree(local)
            print(f"FREEZE_SAVED {index}/10 {item['build_run']} | checkpoint on Drive", flush=True)
        records = [self.completed(item) for item in self.builds]
        report_path = self.root / "freeze_batch.json"
        if report_path.is_file():
            old = checked(read_json(report_path))
            if old.get("contract_sha256") != self.binding or old.get("builds") != records:
                raise ValueError("Completed freeze handoff changed.")
            return old
        return seal(report_path, {"state": "NEW_DATA_INPUTS_READY", "contract_sha256": self.binding,
            "completed_builds": len(records), "total_builds": 10, "builds": records,
            "documents": sum(r["counts"].get("documents", 0) for r in records),
            "children": sum(r["counts"].get("children", 0) for r in records),
            "input_parts": sum(r["input_parts"] for r in records),
            "unique_dense_inputs_per_build_sum": sum(r["unique_dense_inputs"] for r in records),
            "embedding_complete": False, "submission_candidate_changed": False,
            "payload_verification": "SHA256_ON_USE", "updated_at": utc_now(),
            "scope": "Ten new-source builds frozen with model inputs; cross-build hashes may overlap. Embedding/index/union remain separate."})
