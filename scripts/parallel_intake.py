"""Four independent CPU workers with immutable source ownership and a reducer.

Legacy COMPLETE sources are references, not copied payloads. At most the one
legacy in-progress source is resumed in place; all untouched sources get their
own new controller root. Stop the serial notebook before starting these workers.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from vietmedbridge import isolated_intake as intake_core
from vietmedbridge.artifacts import (atomic_json, code_fingerprint, digest_json,
    read_json, runtime_versions, sha256_file, utc_now, verify_file)


class ParallelIntake:
    OWNERS = (0, 0, 1, 1, 2, 0, 2, 3, 3)

    def __init__(self, scope, *, root, work_dir, tokenizer_spec, chunk_config,
                 controller_type, retained_type, sql_context, parquet_writer,
                 worker_id=0, team_size=4, shard_size=2048):
        if type(team_size) is not int or team_size != 4 or type(worker_id) is not int or not 0 <= worker_id < 4:
            raise ValueError("Dùng đúng TEAM_SIZE=4 và WORKER_ID=0,1,2,3; mỗi ID một phiên.")
        self.scope, self.root, self.work = scope, Path(root), Path(work_dir)
        if any((p/"active_data_candidate.json").exists() for p in (self.root, *self.root.parents)):
            raise ValueError("Parallel intake must stay outside the active submission data root.")
        self.sources = scope["base_plan"]["sources"]+scope["additional_sources"]+[scope["retained"]]
        required = scope["base_plan"]["required_snapshot_id"]
        if (not 1 <= len(self.sources) <= 9 or len({s["job_id"] for s in self.sources}) != len(self.sources)
                or any(s["snapshot_id"] != required or s["mode"] != "NEW_SNAPSHOT_SOURCE" for s in self.sources)
                or digest_json(scope["base_plan"]) != scope["base_plan_sha256"]):
            raise ValueError("Parallel sources must match the audited new BTC scope.")
        self.worker_id, self.parallel = worker_id, self.root/"parallel_cpu4"
        self.controller_type, self.retained_type = controller_type, retained_type
        self.sql_context, self.parquet_writer = sql_context, parquet_writer
        self.options = dict(work_dir=self.work, tokenizer_spec=tokenizer_spec,
                            chunk_config=chunk_config, shard_size=shard_size)
        self.owners = {s["job_id"]: self.OWNERS[i] for i, s in enumerate(self.sources)}
        self.contract = {"schema_version": 1, "scope_sha256": digest_json(scope), "team_size": 4,
            "owners": self.owners, "baseline_sha256": scope["base_plan"]["baseline"]["sha256"],
            "tokenizer": tokenizer_spec, "chunking": asdict(chunk_config), "shard_size": shard_size,
            "code_sha256": code_fingerprint(), "runtime": runtime_versions(),
            "dedup_policy": "FROZEN_BASELINE_PER_SOURCE_CROSS_SOURCE_IDS_CHECKED_AT_COLLECT"}
        marker = self.parallel/"parallel_plan.json"
        if marker.exists() and read_json(marker) != self.contract:
            raise ValueError("Parallel scope/team/runtime changed; do not overwrite existing checkpoints.")
        if not marker.exists():
            atomic_json(marker, self.contract)  # identical immutable content on all four workers
        self.plan_sha = digest_json(self.contract)

    def _path(self, relative):
        return intake_core._inside(self.root, relative)

    def _old_location(self, job):
        for root in (self.root, self.root/"additional-frontiers", self.root/"retained-recovery"):
            marker = root/"intake_plan.json"
            if marker.exists():
                plan = read_json(marker)
                source = next((s for s in plan["sources"] if s["job_id"] == job), None)
                if source:
                    self._check_plan(plan)
                    expected = next(s for s in self.sources if s["job_id"] == job)
                    self._check_source(source, expected)
                    return root, plan, source
        return None

    def _check_source(self, actual, expected):
        if expected.get("format") == "RETAINED_PORTABLE_V1":
            if (actual.get("portable_source_sha256") != digest_json(expected)
                    or any(actual.get(k) != v for k, v in expected.items() if k != "members")):
                raise ValueError("Retained source differs from the audited portable source.")
        elif actual != expected:
            raise ValueError("Source differs from the audited frontier.")

    def _check_plan(self, plan):
        base = self.scope["base_plan"]
        if (plan["baseline"] != base["baseline"] or plan["snapshots"] != base["snapshots"]
                or plan.get("required_snapshot_id") != base["required_snapshot_id"]):
            raise ValueError("Controller corpus/baseline differs from the pinned parallel scope.")
        for source in plan["sources"]:
            expected = next((s for s in self.sources if s["job_id"] == source["job_id"]), None)
            if expected is None:
                raise ValueError("Controller contains an unaudited source.")
            self._check_source(source, expected)

    def _receipt(self, root, plan, source):
        marker = root/"sources"/source["job_id"]/"complete.json"
        saved = read_json(marker)
        if saved["state"] != "COMPLETE" or saved["source"] != source or saved["plan_sha256"] != digest_json(plan):
            raise ValueError("Completed source binding changed.")
        for kind, name, sha_key in (("added_ids", "added_ids.parquet", "added_ids_sha256"),
                                   ("outcomes", "source_outcomes.parquet", "outcomes_sha256")):
            item = saved.get("files", {}).get(kind, {"path": name, "sha256": saved[sha_key]})
            path = intake_core._inside(marker.parent, item["path"])
            if (item["sha256"] != saved[sha_key] or not path.is_file() or path.stat().st_size <= 0
                    or ("bytes" in item and path.stat().st_size != item["bytes"])):
                raise ValueError("Completed source artifact is missing or changed.")
        return saved, marker

    def _ref(self, job):
        marker = self.parallel/"results"/(job+".json")
        if not marker.exists():
            return None
        ref = read_json(marker)
        if (ref["parallel_plan_sha256"] != self.plan_sha or ref["job_id"] != job
                or ref["worker_id"] != self.owners[job]):
            raise ValueError("Parallel result owner/scope changed.")
        root = self._path(ref["controller_root"])
        plan = read_json(root/"intake_plan.json")
        self._check_plan(plan)
        source = next(s for s in plan["sources"] if s["job_id"] == job)
        saved, receipt = self._receipt(root, plan, source)
        if sha256_file(receipt) != ref["receipt_sha256"]:
            raise ValueError("Committed parallel receipt changed.")
        return ref, saved, receipt

    def status(self):
        jobs = [job for job, owner in self.owners.items() if owner == self.worker_id]
        refs = [self._ref(job) for job in jobs]
        completed = [item[0]["job_id"] for item in refs if item is not None]
        return {"state": "CPU_WORKER_COMPLETE" if len(completed) == len(jobs) else "CPU_WORKER_PENDING",
            "worker_id": self.worker_id, "team_size": 4, "assigned_sources": jobs,
            "completed_sources": len(completed), "completed_jobs": completed,
            "total_sources": len(jobs), "pending_sources": [job for job in jobs if job not in completed],
            "parallel_plan_sha256": self.plan_sha, "submission_candidate_changed": False}

    def _dependencies(self, root, plan):
        expected = plan.get("prior_mapping_sha256", [])
        if not expected:
            return []
        refs = []
        for previous in (self.root, self.root/"additional-frontiers"):
            if previous == root or not (previous/"intake_plan.json").exists():
                break
            prior = read_json(previous/"intake_plan.json")
            self._check_plan(prior)
            for source in prior["sources"]:
                saved, receipt = self._receipt(previous, prior, source)
                refs.append((receipt.parent/"added_ids.parquet", saved["added_ids_sha256"]))
        if [checksum for _, checksum in refs] != expected:
            raise ValueError("Legacy in-progress source dependency mappings changed.")
        return refs

    def _bind(self, source):
        job = source["job_id"]
        marker = self.parallel/"bindings"/(job+".json")
        if marker.exists():
            saved = read_json(marker)
            if saved["parallel_plan_sha256"] != self.plan_sha or saved["worker_id"] != self.worker_id:
                raise ValueError("Job already belongs to another parallel worker or scope.")
            return saved
        old = self._old_location(job)
        kind, root = "NEW_SOURCE", self.parallel/"jobs"/job
        if old:
            old_root, _, _ = old
            if ((old_root/"sources"/job).exists() or (old_root/"processed"/job).exists()):
                kind, root = "LEGACY_SOURCE", old_root
        saved = {"parallel_plan_sha256": self.plan_sha, "worker_id": self.worker_id,
                 "job_id": job, "kind": kind, "controller_root": str(root.relative_to(self.root))}
        atomic_json(marker, saved)  # only the statically assigned owner writes this job
        return saved

    def _adapter(self, source, root):
        return self.retained_type(source, root=root/"sources"/source["job_id"]/"metadata",
            work_dir=self.work, parquet_writer=self.parquet_writer)

    def run(self, **loaders):
        print(f"CPU worker {self.worker_id}/4 | {self.status()['assigned_sources']}", flush=True)
        for source in self.sources:
            job = source["job_id"]
            if self.owners[job] != self.worker_id:
                continue
            if self._ref(job) is not None:
                print(f"SKIP_PARALLEL_COMPLETE {job} | receipt only", flush=True)
                continue
            binding = self._bind(source)
            root = self._path(binding["controller_root"])
            if binding["kind"] == "LEGACY_SOURCE":
                plan = read_json(root/"intake_plan.json")
                self._check_plan(plan)
                actual = next(s for s in plan["sources"] if s["job_id"] == job)
                prior = self._dependencies(root, plan)
            else:
                prior = []
                marker = root/"intake_plan.json"
                if marker.exists():
                    plan = read_json(marker)
                    self._check_plan(plan)
                    actual = plan["sources"][0]
                else:
                    actual = source
                    if source.get("format") == "RETAINED_PORTABLE_V1":
                        with self.sql_context():
                            actual = self._adapter(source, root).prepare(loaders["load_reader"]())["source"]
                    plan = {**self.scope["base_plan"], "sources": [actual], "campaign": "parallel-"+job,
                        "parallel_plan_sha256": self.plan_sha, "prior_mapping_sha256": []}
            receipt = root/"sources"/job/"complete.json"
            if not receipt.exists():
                controller = self.controller_type(plan, root=root, prior_mappings=prior, **self.options)
                pending = [s for s in plan["sources"] if s["job_id"] not in {v["source"]["job_id"] for v in controller.completed}]
                if not pending or pending[0]["job_id"] != job:
                    raise ValueError("Legacy resume is not this worker's source; stop serial intake before using parallel workers.")
                options = {}
                if source.get("format") == "RETAINED_PORTABLE_V1":
                    adapter = self._adapter(source, root)
                    options = {"load_metadata": lambda s, target: adapter.prepare(loaders["load_reader"]()),
                               "select_metadata": lambda m, links, maps, out: adapter.select(loaders["load_reader"](), m, links, maps, out)}
                controller.run(**loaders, **options, max_new_sources=1)
            saved, receipt = self._receipt(root, plan, actual)
            ref = {"parallel_plan_sha256": self.plan_sha, "job_id": job, "worker_id": self.worker_id,
                "controller_root": str(root.relative_to(self.root)), "receipt_sha256": sha256_file(receipt)}
            atomic_json(self.parallel/"results"/(job+".json"), ref)
            status = self.status()
            atomic_json(self.parallel/"workers"/(f"worker-{self.worker_id}.json"), status)
            print(f"PARALLEL_SOURCE_SAVED worker={self.worker_id} {job} | {saved['counts']}", flush=True)
        status = self.status()
        atomic_json(self.parallel/"workers"/(f"worker-{self.worker_id}.json"), status)
        return status

    def collect(self):
        refs = [self._ref(source["job_id"]) for source in self.sources]
        pending = [s["job_id"] for s, ref in zip(self.sources, refs) if ref is None]
        if pending:
            return {"state": "WAITING_FOR_CPU_WORKERS", "completed_sources": len(refs)-len(pending),
                    "total_sources": len(refs), "pending_sources": pending}
        cache = self.parallel/"collection.json"
        identity = {"parallel_plan_sha256": self.plan_sha,
                    "receipts": [ref[0] for ref in refs]}
        if cache.exists():
            saved = read_json(cache)
            if saved["identity"] != identity:
                raise ValueError("Collected source receipts changed; preserve the previous collection.")
            print("REUSED_PARALLEL_COLLECTION | no payload replay", flush=True)
            return saved["result"]
        counts, builds, paths = Counter(), [], []
        for ref, receipt, marker in refs:
            for kind, name, sha_key in (("added_ids", "added_ids.parquet", "added_ids_sha256"),
                                       ("outcomes", "source_outcomes.parquet", "outcomes_sha256")):
                item = receipt.get("files", {}).get(kind, {"path": name, "sha256": receipt[sha_key]})
                path = intake_core._inside(marker.parent, item["path"])
                verify_file(path, item["sha256"])
                if kind == "outcomes":
                    paths.append(intake_core._local_mapping(path, item["sha256"], self.work))
            counts.update(receipt["counts"])
            if receipt["build_run"]:
                builds.append({"build_run": receipt["build_run"], "snapshot_sha256": receipt["build_snapshot_sha256"],
                    "snapshot_id": receipt["source"]["snapshot_id"], "phase_root": ref["controller_root"]})
        with self.sql_context():
            with intake_core._connection(self.work) as con:
                con.read_parquet([str(p) for p in paths], union_by_name=True).create_view("outcomes")
                rows, unique_ids, unfinished = con.sql("""SELECT count(*),count(DISTINCT doc_id),
                    count(*) FILTER (WHERE state NOT IN ('ADDED_VALID_DOCUMENT','REUSED_EXISTING_DOCUMENT','PIPELINE_FAILURE')
                    OR state IS NULL) FROM outcomes""").fetchone()
                expected = self.scope["audited_coverage"].get("union_native_ids")
                if rows != unique_ids or unfinished or sum(counts.values()) != rows or (expected is not None and rows != expected):
                    raise ValueError(f"Parallel coverage failed: rows={rows}, unique_ids={unique_ids}, unfinished={unfinished}, expected={expected}")
                actual_counts = dict(con.sql("SELECT state,count(*) FROM outcomes GROUP BY 1").fetchall())
                if actual_counts != dict(counts):
                    raise ValueError("Parallel source counters differ from the actual ID outcomes.")
        result = {"state": "FULL_INTAKE_COMPLETE", "completed_sources": len(refs), "total_sources": len(refs),
            "counts": dict(counts), "builds": builds, "audited_coverage": self.scope["audited_coverage"],
            "coverage": {"passed": True, "input_native_ids": rows, "unique_native_ids": unique_ids},
            "scope_sha256": digest_json(self.scope), "parallel_plan_sha256": self.plan_sha,
            "submission_candidate_changed": False, "updated_at": utc_now(),
            "scope": "CPU import/quality/chunks; not embedded, indexed, promoted or submitted"}
        atomic_json(cache, {"identity": identity, "result": result})
        atomic_json(self.root/"full_intake_status.json", result)
        return result
