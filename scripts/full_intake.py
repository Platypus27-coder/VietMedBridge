"""Run the complete audited folder selection without rewriting the original plan."""
from collections import Counter
from pathlib import Path

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, utc_now


class FullIntake:
    def __init__(self, scope, *, root, work_dir, tokenizer_spec, chunk_config,
                 controller_type, retained_type, sql_context, parquet_writer, shard_size=2048):
        self.scope, self.root, self.work = scope, Path(root), Path(work_dir)
        if any((p/"active_data_candidate.json").exists() for p in (self.root, *self.root.parents)):
            raise ValueError("Full intake must stay outside the active submission data root.")
        self.controller_type, self.retained_type = controller_type, retained_type
        self.sql_context, self.parquet_writer = sql_context, parquet_writer
        self.options = dict(work_dir=self.work, tokenizer_spec=tokenizer_spec,
                            chunk_config=chunk_config, shard_size=shard_size)
        base = scope["base_plan"]
        required = base["required_snapshot_id"]
        sources = base["sources"]+scope["additional_sources"]+[scope["retained"]]
        if (scope["base_plan_sha256"] != digest_json(base) or set(base["snapshots"]) != {required}
                or not sources or any(s["snapshot_id"] != required or s["mode"] != "NEW_SNAPSHOT_SOURCE" for s in sources)
                or len({s["job_id"] for s in sources}) != len(sources)):
            raise ValueError("Full intake must contain only audited new BTC membership sources.")
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root/"full_intake_scope.json"
        if marker.exists() and read_json(marker) != scope:
            raise ValueError("Audited folder scope changed; preserve checkpoints and audit the additions.")
        if not marker.exists():
            atomic_json(marker, scope)
        self.base = controller_type(base, root=self.root, **self.options)
        self.frontiers = self.retained = None
        self.extension_root = self.root/"additional-frontiers"
        self.retained_root = self.root/"retained-recovery"
        if (self.extension_root/"intake_plan.json").exists():
            if len(self.base.completed) != len(base["sources"]):
                raise ValueError("Continuation exists before all base sources completed.")
            prior = self._mappings(self.base)
            plan = self._plan(scope["additional_sources"], prior)
            self.frontiers = controller_type(plan, root=self.extension_root, prior_mappings=prior, **self.options)
        if (self.retained_root/"intake_plan.json").exists():
            if self.frontiers is None or len(self.frontiers.completed) != len(scope["additional_sources"]):
                raise ValueError("Retained continuation exists before the additional frontiers completed.")
            prior = self._mappings(self.base, self.frontiers)
            plan = read_json(self.retained_root/"intake_plan.json")
            if (len(plan["sources"]) != 1 or plan["sources"][0].get("portable_source_sha256") != digest_json(scope["retained"])
                    or plan != self._plan(plan["sources"], prior)):
                raise ValueError("Retained continuation source/dependencies changed.")
            self.retained = controller_type(plan, root=self.retained_root, prior_mappings=prior, **self.options)

    def _mappings(self, *controllers):
        return [(controller.root/"sources"/saved["source"]["job_id"]/"added_ids.parquet", saved["added_ids_sha256"])
                for controller in controllers for saved in controller.completed]

    def _plan(self, sources, prior):
        base = self.scope["base_plan"]
        return {**base, "campaign": "continuation-"+digest_json(sources)[:16], "sources": sources,
            "parent_plan_sha256": self.scope["base_plan_sha256"],
            "full_scope_sha256": digest_json(self.scope),
            "prior_mapping_sha256": [checksum for _, checksum in prior]}

    def _status(self):
        controllers = [self.base]
        if self.frontiers is not None:
            controllers.append(self.frontiers)
        if self.retained is not None:
            controllers.append(self.retained)
        counts, builds, completed_jobs = Counter(), [], []
        for controller in controllers:
            controller.refresh()
            phase = controller.status()
            counts.update(phase["counts"])
            builds.extend({**build, "phase_root": str(controller.root.relative_to(self.root))} for build in phase["builds"])
            completed_jobs.extend(s["source"]["job_id"] for s in controller.completed)
        jobs = [s["job_id"] for s in self.scope["base_plan"]["sources"]+self.scope["additional_sources"]+[self.scope["retained"]]]
        result = {"state": "FULL_INTAKE_COMPLETE" if set(completed_jobs) == set(jobs) else "FULL_INTAKE_IN_PROGRESS",
            "completed_sources": len(completed_jobs), "total_sources": len(jobs), "counts": dict(counts),
            "pending_sources": [job for job in jobs if job not in completed_jobs], "builds": builds,
            "scope_sha256": digest_json(self.scope), "audited_coverage": self.scope["audited_coverage"],
            "submission_candidate_changed": False, "updated_at": utc_now(),
            "scope": "CPU import/quality/chunks for all audited new BTC sources; not embedding, index, or submission",
            "counts_unit": "native ID outcomes; ADDED/REUSED/FAILURE are distinct, not all successful documents"}
        atomic_json(self.root/"full_intake_status.json", result)
        print(f"FULL_INTAKE_PROGRESS {len(completed_jobs)}/{len(jobs)} sources complete", flush=True)
        return result

    def status(self):
        self._serial_guard()
        return self._status()

    def _serial_guard(self):
        if (self.root/"parallel_cpu4/parallel_plan.json").exists():
            raise ValueError("Lượt này đã chuyển sang 4 CPU workers. Mở notebook worker 0–3 hoặc chọn RUN_MODE='collect'; không chạy lại sequential.")

    def run(self, **loaders):
        self._serial_guard()
        print("FULL_INTAKE: original frontiers -> additional frontiers -> retained recovery", flush=True)
        self.base.run(**loaders, source_completed=self._status)
        self._status()
        prior = self._mappings(self.base)
        plan = self._plan(self.scope["additional_sources"], prior)
        self.frontiers = self.controller_type(plan, root=self.extension_root, prior_mappings=prior, **self.options)
        self.frontiers.run(**loaders, source_completed=self._status)
        self._status()
        prior = self._mappings(self.base, self.frontiers)
        adapter = self.retained_type(self.scope["retained"],
            root=self.retained_root/"sources"/self.scope["retained"]["job_id"]/"metadata",
            work_dir=self.work, parquet_writer=self.parquet_writer)
        marker = self.retained_root/"intake_plan.json"
        if marker.exists():
            plan = read_json(marker)
            sources = plan["sources"]
            if (len(sources) != 1 or sources[0].get("portable_source_sha256") != digest_json(self.scope["retained"])
                    or plan != self._plan(sources, prior)):
                raise ValueError("Retained continuation source/dependencies changed.")
        else:
            with self.sql_context():
                metadata = adapter.prepare(loaders["load_reader"]())
            plan = self._plan([metadata["source"]], prior)
        self.retained = self.controller_type(plan, root=self.retained_root, prior_mappings=prior, **self.options)

        def metadata_loader(source, target):
            if Path(target) != adapter.root:
                raise ValueError("Unexpected retained metadata directory.")
            metadata = adapter.prepare(loaders["load_reader"]())
            if metadata["source"] != source:
                raise ValueError("Retained conversion identity changed.")
            return metadata

        def selector(metadata, links, maps, output):
            return adapter.select(loaders["load_reader"](), metadata, links, maps, output)

        self.retained.run(**loaders, load_metadata=metadata_loader, select_metadata=selector,
                          source_completed=self._status)
        return self._status()
