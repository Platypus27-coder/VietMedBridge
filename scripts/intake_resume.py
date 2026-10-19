"""Manifest-first intake controller, embedded into the Colab notebook.

Kept outside src/ deliberately: changing resume orchestration must not change
the pinned producer fingerprint or invalidate existing document/chunk shards.
Completed receipts are NOT a fresh payload-integrity audit: only metadata is
read for status; mapping payloads are SHA-checked when actually used.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import re
import tempfile

from vietmedbridge import isolated_intake as intake_core
from vietmedbridge.artifacts import (atomic_json, code_fingerprint, digest_json,
    read_json, runtime_versions, sha256_file, utc_now, verify_file)
from vietmedbridge.dataset import _connection


def intake_memory_mb():
    """At most 2 GiB and a quarter of available RAM; no reservation or GPU use."""
    available = []
    try:
        match = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        if match:
            available.append(int(match[1])*1024)
    except OSError:
        pass
    for limit, used in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            available.append(max(0, int(Path(limit).read_text())-int(Path(used).read_text())))
        except (OSError, ValueError):
            pass
    return min(2048, max(64, min(available)//(4*1024**2))) if available else 512


def bounded_intake_parquet(con, query, destination, work_dir):
    """Flush wide text every 2,048 rows, then atomically publish the full file."""
    with tempfile.TemporaryDirectory(prefix="intake-table-", dir=work_dir) as temporary:
        path = Path(temporary)/"table.parquet"
        con.sql(query).write_parquet(str(path), compression="zstd", row_group_size=2048)
        return intake_core.publish_file(path, destination)


@contextmanager
def bounded_intake_sql(memory_mb=None):
    """Scoped I/O adapter for the single intake controller, including old locks.

    The pinned producer files, policies and completed selections stay untouched.
    Only this module's selection/outcome SQL and Parquet writer are adapted;
    normal importer/chunk shards still use their original producer.
    """
    budget = intake_memory_mb() if memory_mb is None else memory_mb
    if type(budget) is not int or not 64 <= budget <= 2048:
        raise ValueError("Intake SQL budget must be 64..2048 MiB.")
    original_connection, original_parquet = intake_core._connection, intake_core._parquet

    def connect(temp_dir, memory_limit=None):
        con = _connection(temp_dir, memory_limit=memory_limit or f"{budget}MiB")
        try:
            con.execute("SET threads=1")
            con.execute("SET preserve_insertion_order=false")
        except BaseException:
            con.close()
            raise
        return con

    intake_core._connection, intake_core._parquet = connect, bounded_intake_parquet
    print(f"Intake SQL: RAM limit {budget} MiB | 1 thread | local spill | Parquet groups 2,048 rows", flush=True)
    try:
        yield
    finally:
        intake_core._connection, intake_core._parquet = original_connection, original_parquet


class IntakeResume:
    def __init__(self, plan, *, root, work_dir, tokenizer_spec, chunk_config, shard_size=2048,
                 prior_mappings=()):
        self.plan, self.root, self.work = plan, Path(root), Path(work_dir)
        self.spec, self.config, self.shard_size = tokenizer_spec, chunk_config, shard_size
        self.plan_sha = digest_json(plan)
        self.prior_mappings = [(Path(path), checksum) for path, checksum in prior_mappings]
        if [checksum for _, checksum in self.prior_mappings] != plan.get("prior_mapping_sha256", []):
            raise ValueError("Prior intake mappings differ from the pinned continuation plan.")
        if any((p/"active_data_candidate.json").exists() for p in (self.root, *self.root.parents)):
            raise ValueError("Intake must stay outside the active submission data root.")
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", s["job_id"]) for s in plan["sources"]):
            raise ValueError("Invalid intake job ID.")
        intake_core.pin_plan(self.root, plan)
        self.work.mkdir(parents=True, exist_ok=True)
        self.contract = {"tokenizer": tokenizer_spec, "chunking": asdict(chunk_config),
            "shard_size": shard_size, "code_sha256": code_fingerprint(), "runtime": runtime_versions()}
        contract = self.root/"intake_contract.json"
        if contract.exists() and read_json(contract) != self.contract:
            raise ValueError("Pinned intake code/runtime/chunk policy changed; do not discard checkpoints.")
        if not contract.exists():
            atomic_json(contract, self.contract)
        self.refresh()

    def _file(self, root, item):
        path = intake_core._inside(root, item["path"])
        stat = path.stat()  # Missing/changed files fail; never silently reprocess a completed source.
        if stat.st_size <= 0 or ("bytes" in item and stat.st_size != item["bytes"]):
            raise ValueError(f"Checkpoint artifact size changed: {path}")
        return path

    def refresh(self):
        self.completed = []
        gap = False
        for source in self.plan["sources"]:
            target = self.root/"sources"/source["job_id"]
            receipt = target/"complete.json"
            if not receipt.exists():
                gap = True
                continue
            if gap:
                raise ValueError("Out-of-order complete receipt; preserve files and inspect source lineage.")
            saved = read_json(receipt)
            if (saved.get("state") != "COMPLETE" or saved["source"] != source
                    or saved["plan_sha256"] != self.plan_sha):
                raise ValueError("Completed source/plan binding changed.")
            for kind, name, sha_key in (("added_ids", "added_ids.parquet", "added_ids_sha256"),
                                       ("outcomes", "source_outcomes.parquet", "outcomes_sha256")):
                item = saved.get("files", {}).get(kind, {"path": name, "sha256": saved[sha_key]})
                if item["sha256"] != saved[sha_key]:
                    raise ValueError("Receipt payload binding changed.")
                self._file(target, item)
            self.completed.append(saved)

    def status(self):
        counts = Counter()
        for saved in self.completed:
            counts.update(saved["counts"])
        return {"state": "INTAKE_COMPLETE" if len(self.completed) == len(self.plan["sources"]) else "INTAKE_IN_PROGRESS",
            "plan_sha256": self.plan_sha, "completed_sources": len(self.completed),
            "total_sources": len(self.plan["sources"]), "counts": dict(counts),
            "builds": [{"build_run": s["build_run"], "snapshot_sha256": s["build_snapshot_sha256"],
                "snapshot_id": s["source"]["snapshot_id"]} for s in self.completed if s["build_run"]],
            "resume_policy": "MANIFEST_FIRST_MAPPING_SHA_CHECKED_ON_USE",
            "scope": "CPU import, quality and exact-span chunks only; not embedded/frozen/promoted or submitted",
            "submission_candidate_changed": False, "updated_at": utc_now()}

    def _selection(self, source, target, existing):
        marker = target/"selection/selection.json"
        if not marker.exists():
            return None
        saved = read_json(marker)
        metadata = read_json(target/"metadata/metadata.json")
        expected = {"metadata": metadata,
            "links_sha256": self.plan["snapshots"][source["snapshot_id"]]["sha256"],
            "existing": [{"sha256": checksum} for _, checksum in existing]}
        if (saved["identity"] != expected or metadata["source"] != source
                or metadata["files"]["frontier"]["sha256"] != source["frontier_sha256"]
                or sum(saved["counts"].values()) != saved["official_ids"]):
            raise ValueError("Saved selection binding changed; no automatic rebuild.")
        for item in saved["files"].values():
            self._file(target/"selection", item)
        print(f"REUSE_SELECTION {source['job_id']} | no archive/baseline rescan", flush=True)
        return saved, metadata

    def _validated_build(self, source, selection, build_dir):
        path = build_dir/"build.json"
        if not path.exists():
            return None
        build = read_json(path)
        if not build.get("selected_range_complete"):
            return None
        if build.get("state") != "DATA_VALIDATED" or not build.get("integrity", {}).get("passed"):
            raise ValueError("Complete build has not passed integrity; retain and inspect it.")
        config = read_json(build_dir/"config.json")
        queued = selection["counts"].get("QUEUED_FOR_PIPELINE", 0)
        files = {kind: {"name": Path(intake_core.MEMBERS[kind]).name,
            "sha256": selection["files"][kind]["sha256"],
            "bytes": self._file(self.root/"sources"/source["job_id"]/"selection", selection["files"][kind]).stat().st_size}
            for kind in intake_core.MEMBERS}
        if (config["signature"] != digest_json({k: v for k, v in config.items() if k != "signature"})
                or config["signature"] != build["signature"]
                or config["source_input_signature"] != digest_json(files)
                or config["external_snapshot_id"] != source.get("original_snapshot_id", source["snapshot_id"])
                or config["official_links_sha256"] != self.plan["snapshots"][source["snapshot_id"]]["sha256"]
                or config["tokenizer"] != self.spec or config["chunks"] != asdict(self.config)
                or config["code_sha256"] != self.contract["code_sha256"]
                or config["runtime"] != self.contract["runtime"] or config["shard_size"] != self.shard_size
                or build["requested_input_records"] != queued or build["counts"]["input_records"] != queued
                or build["snapshot_sha256"] != digest_json({"signature": build["signature"], "parts": build["parts"]})):
            raise ValueError("Validated build identity differs from the saved selection/policy.")
        for part in build["parts"]:
            for kind, item in part["files"].items():
                path = self._file(build_dir, item)
                if kind in ("documents", "failures"):
                    # Only these payloads are about to be used to finish the
                    # ID/outcome mapping. Child/parent chunks stay deferred to freeze.
                    verify_file(path, item["sha256"])
        print(f"REUSE_VALIDATED_BUILD {source['job_id']} | no re-chunk or full validation replay", flush=True)
        return build

    @bounded_intake_sql()
    def run(self, *, load_reader, load_links, load_baseline, load_tokenizer, load_metadata=None,
            select_metadata=None, source_completed=None):
        self.refresh()
        for saved in self.completed:
            print(f"SKIP_COMPLETED_SOURCE {saved['source']['job_id']} | receipt only; payload checked on use", flush=True)
        if len(self.completed) == len(self.plan["sources"]):
            result = self.status()
            atomic_json(self.root/"intake_status.json", result)
            return result
        # Paths/checksums suffice until a NEW selection really needs a mapping.
        existing = [(None, self.plan["baseline"]["sha256"])] + self.prior_mappings + [
            (self.root/"sources"/s["source"]["job_id"]/"added_ids.parquet", s["added_ids_sha256"])
            for s in self.completed]
        for source in self.plan["sources"][len(self.completed):]:
            job = source["job_id"]
            target, build_dir = self.root/"sources"/job, self.root/"processed"/job
            print(f"CONTINUE_SOURCE {job} | {len(self.completed)}/{len(self.plan['sources'])} sources complete", flush=True)
            cached = self._selection(source, target, existing)
            if cached is None:
                metadata = (load_metadata(source, target/"metadata") if load_metadata is not None
                            else intake_core.stage_metadata(load_reader(), source, target/"metadata"))
                local_maps = [load_baseline() if path is None else intake_core._local_mapping(path, checksum, self.work)
                              for path, checksum in existing]
                for path, (_, checksum) in zip(local_maps, existing):
                    verify_file(path, checksum)
                links = load_links(source["snapshot_id"])
                selection = (select_metadata(metadata, links, local_maps, target/"selection") if select_metadata is not None
                    else intake_core.select_delta(target/"metadata", metadata, links, local_maps, target/"selection", self.work))
            else:
                selection, metadata = cached
            build = None
            if selection["counts"].get("QUEUED_FOR_PIPELINE", 0):
                build = self._validated_build(source, selection, build_dir)
                if build is None:
                    markers = sorted((build_dir/"parts").glob("*.done.json"))
                    committed = [read_json(path) for path in markers]
                    print(f"RESUME_PARTS {job} | {len(committed)} committed shards, "
                          f"{sum(p['records'] for p in committed):,} IDs; verify current source only", flush=True)
                    staged = intake_core.restore_import_source(target/"selection", selection, self.work/"sources"/job)
                    for kind in intake_core.MEMBERS:
                        item = selection["files"][kind]
                        verify_file(staged/item["path"], item["sha256"])
                    tokenizer, actual_spec = load_tokenizer()
                    if actual_spec != self.spec:
                        raise ValueError("Loaded tokenizer differs from the pinned contract.")
                    build = intake_core.import_external_corpus(staged, load_links(source["snapshot_id"]),
                        self.root/"processed", tokenizer, self.spec, run_name=job, config=self.config,
                        work_dir=self.work, shard_size=self.shard_size)
                    if not build["selected_range_complete"] or not build["integrity"]["passed"]:
                        raise ValueError("Intake build is not complete and valid.")
                atomic_json(build_dir/"intake_provenance.json", {"source": source,
                    "original_metadata": metadata["files"], "plan_sha256": self.plan_sha,
                    "membership_snapshot_id": source["snapshot_id"],
                    "original_snapshot_id": source.get("original_snapshot_id", source["snapshot_id"]),
                    "portable_origin": metadata.get("portable_origin"),
                    "raw_locator": "Drive file ID -> optional inner_tar -> suffix after # in document.raw_file",
                    "raw_verification": "METADATA_ONLY_RAW_PAYLOAD_NOT_REPLAYED"})
            entry = selection["files"]["selection"]
            verify_file(target/"selection"/entry["path"], entry["sha256"])
            counts = intake_core._write_outcomes(target, target/"selection", selection, build_dir, build, self.work)
            with _connection(self.work) as con:
                if build:
                    con.read_parquet([str(build_dir/p["files"]["documents"]["path"]) for p in build["parts"]]).create_view("added")
                    query = "SELECT doc_id,url FROM added"
                else:
                    query = "SELECT NULL::BIGINT AS doc_id,NULL::VARCHAR AS url WHERE false"
                intake_core._parquet(con, query, target/"added_ids.parquet", self.work)
            files = {kind: {"path": name, "sha256": sha256_file(target/name), "bytes": (target/name).stat().st_size}
                     for kind, name in (("added_ids", "added_ids.parquet"), ("outcomes", "source_outcomes.parquet"))}
            saved = {"source": source, "plan_sha256": self.plan_sha, "state": "COMPLETE", "counts": counts,
                "build_run": job if build else None, "build_snapshot_sha256": build["snapshot_sha256"] if build else None,
                "data_counts": build["counts"] if build else {}, "files": files,
                "added_ids_sha256": files["added_ids"]["sha256"], "outcomes_sha256": files["outcomes"]["sha256"],
                "updated_at": utc_now()}
            atomic_json(target/"complete.json", saved)  # last: durable commit of the whole source
            self.completed.append(saved)
            existing.append((target/"added_ids.parquet", saved["added_ids_sha256"]))
            result = self.status()
            atomic_json(self.root/"intake_status.json", result)
            print(f"SOURCE_CHECKPOINT_SAVED {job} | {counts}", flush=True)
            if source_completed is not None:
                source_completed()
        return result
