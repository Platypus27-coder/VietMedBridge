import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freeze_workers import (
    freeze_batch_signature,
    freeze_worker_manifest_path,
    partition_freeze_builds,
    verify_freeze_workers,
    write_freeze_worker_manifest,
    collect_completed_build_refs,
    audit_freeze_batch,
)


def _prepare_build(root, ref, code_sha):
    build_dir = root / "processed" / ref["build_run"]
    build_dir.mkdir(parents=True)
    doc_id = ref.get("doc_id", int(ref["build_run"].split("-")[-1]))
    config = {"tokenizer": {"model_id": "fixture", "revision": "v1"},
              "chunks": {"dense_max_tokens": 512}, "schema_version": "data-v2"}
    config["signature"] = digest_json(config)
    atomic_json(build_dir / "config.json", config)
    files = {}
    for kind in ("documents", "children", "parents", "sections", "failures", "ledger"):
        path = build_dir / f"{kind}.parquet"
        ids = [doc_id, doc_id + 100000] if kind == "ledger" else [doc_id]
        pq.write_table(pa.table({"doc_id": ids}), path)
        files[kind] = {"path": path.name, "sha256": sha256_file(path)}
    parts = [{"files": files}]
    snapshot = digest_json({"signature": config["signature"], "parts": parts})
    ref["snapshot_sha256"] = snapshot
    counts = {"input_records": 2, "documents": 1, "failures": 1,
              "children": 1, "parents": 1, "sections": 1}
    build = {"selected_range_complete": True, "snapshot_sha256": snapshot,
             "signature": config["signature"], "parts": parts, "counts": counts,
             "requested_input_records": 2, "schema_version": "data-v2", "chunking": config["chunks"]}
    atomic_json(build_dir / "build.json", build)

    candidate = {**build,
        "state": "FROZEN_CANDIDATE",
        "selected_range_complete": True,
        "snapshot_sha256": ref["snapshot_sha256"],
        "golden": {"code_sha256": code_sha, "passed": True,
                   "tokenizer": config["tokenizer"], "chunking": config["chunks"]},
        "health": {"snapshot_sha256": snapshot,
                   "integrity": {"passed": True, "official_membership": "VERIFIED", "checked_input_records": 2},
                   "chunks": {"distinct_representations": 1}, "files": {}},
    }
    candidate_sha = digest_json(candidate)
    candidate["candidate_manifest_sha256"] = candidate_sha
    candidate_name = f"candidate-{candidate_sha[:16]}.json"
    atomic_json(build_dir / candidate_name, candidate)

    policy = {"candidate_manifest_sha256": candidate_sha, "tokenizer": config["tokenizer"],
              "builder": "fixture", "dense_max_tokens": 512, "part_size": 4096}
    atomic_json(build_dir / "index_inputs/config.json", policy)
    unit_path = build_dir / "index_inputs/units-000000000000.parquet"
    pq.write_table(pa.table({"id": ["representation"], "text": ["text"]}), unit_path)
    signature = digest_json(policy)
    part = {"signature": signature, "start": 0, "rows": 1,
            "path": unit_path.name, "sha256": sha256_file(unit_path)}
    inputs = {**policy, "state": "COMPLETE", "signature": signature, "input_count": 1, "parts": [part]}
    inputs_sha = digest_json(inputs)
    inputs["manifest_sha256"] = inputs_sha
    atomic_json(build_dir / "index_inputs" / "units.json", inputs)
    return {
        "state": "COMPLETE",
        "snapshot_sha256": ref["snapshot_sha256"],
        "candidate_name": candidate_name,
        "candidate_manifest_sha256": candidate_sha,
        "validation_candidate_name": candidate_name,
        "validation_candidate_manifest_sha256": candidate_sha,
        "index_inputs_manifest_sha256": inputs_sha,
    }


def _write_workers(root, refs, completed, *, code_sha="code", team_size=2):
    signature = freeze_batch_signature(refs, code_sha256=code_sha, team_size=team_size)
    for worker_id in range(team_size):
        assigned = partition_freeze_builds(refs, team_size=team_size, worker_id=worker_id)
        runs = [item["build_run"] for item in assigned]
        write_freeze_worker_manifest(
            freeze_worker_manifest_path(root, batch_signature=signature, worker_id=worker_id),
            batch_signature=signature, code_sha256=code_sha, team_size=team_size, worker_id=worker_id,
            assigned_build_runs=runs, completed={run: completed[run] for run in runs}, state="COMPLETE")
    return signature


def _prepare_alias_source(root, ref, identity):
    metadata = {"source": {"path": ref["source_path"], **identity},
                "files": {"dataset": {"name": "dataset_manifest.json", "bytes": 2, "sha256": "a" * 64}}}
    metadata["input_signature"] = digest_json(metadata["files"])
    config = {"source_input_signature": metadata["input_signature"], "schema_version": "data-v2"}
    config["signature"] = digest_json(config)
    build = {"selected_range_complete": True, "signature": config["signature"],
             "parts": [], "source_provenance": metadata}
    build["snapshot_sha256"] = digest_json({"signature": build["signature"], "parts": build["parts"]})
    ref["snapshot_sha256"] = build["snapshot_sha256"]
    directory = root / "processed" / ref["build_run"]
    atomic_json(directory / "config.json", config)
    atomic_json(directory / "build.json", build)
    atomic_json(directory / "external_source/source.json", metadata)


class FreezeWorkerTests(unittest.TestCase):
    def test_seven_sources_across_drive_aliases_preserve_builds_and_worker_signature(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            old = "/content/drive/.shortcut-targets-by-id/shared-id/VietMedBridge/data/incoming/"
            current = "/content/drive/MyDrive/VietMedBridge/data/incoming/"
            suffixes = ["vibiomir_shard_00000.tar.parts"] + [
                f"team-batch/vibiomir_shard_{i:05d}.tar" for i in range(1, 7)]
            refs, identities = [], {}
            for i, suffix in enumerate(suffixes):
                identity = ({"is_directory": True, "archive_sha256": "b" * 64, "manifest_sha256": "c" * 64}
                            if i == 0 else {"is_directory": False, "bytes": 3000 + i, "mtime_ns": 1000 + i})
                ref = {"build_run": f"build-{i}", "state": "FROZEN" if i == 0 else "COMPLETE",
                       "source_path": old + suffix}
                _prepare_alias_source(root, ref, identity)
                refs.append(ref)
                identities[current + suffix] = {"path": current + suffix, **identity}
            refs[0]["candidate_name"] = "old-candidate.json"
            prior = {"builds": refs[:1]}
            workers = [(f"worker-{i}", {"state": "COMPLETE", "builds": [refs[i]]}) for i in range(1, 7)]
            expected = [current + suffix for suffix in suffixes]
            with patch("vietmedbridge.external_import._source_identity", side_effect=identities.__getitem__):
                gathered = collect_completed_build_refs(expected, prior, workers, data_root=root)
            self.assertEqual([item["source_path"] for item in gathered], expected)
            self.assertEqual([item["build_run"] for item in gathered], [f"build-{i}" for i in range(7)])
            self.assertEqual(gathered[0]["candidate_name"], "old-candidate.json")
            self.assertEqual(prior["builds"][0]["source_path"], old + suffixes[0])
            self.assertEqual(freeze_batch_signature(refs, code_sha256="code", team_size=3),
                             freeze_batch_signature(gathered, code_sha256="code", team_size=3))
            self.assertEqual([item["build_run"] for item in partition_freeze_builds(
                gathered, team_size=3, worker_id=1)], ["build-1", "build-4"])

    def test_drive_alias_rejects_changed_source_and_mismatched_snapshot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            old = "/content/drive/.shortcut-targets-by-id/shared/VietMedBridge/data/incoming/a.tar"
            current = "/content/drive/MyDrive/VietMedBridge/data/incoming/a.tar"
            identity = {"is_directory": False, "bytes": 100, "mtime_ns": 12345}
            ref = {"source_path": old, "state": "COMPLETE", "build_run": "build-0"}
            _prepare_alias_source(root, ref, identity)
            for field in ("bytes", "mtime_ns"):
                with self.subTest(field=field), patch("vietmedbridge.external_import._source_identity",
                    return_value={"path": current, **identity, field: 999}):
                    with self.assertRaisesRegex(ValueError, "source identity changed"):
                        collect_completed_build_refs([current], {"builds": [ref]}, [], data_root=root)
            changed = {**ref, "snapshot_sha256": "another-snapshot"}
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                collect_completed_build_refs([current], {"builds": [changed]}, [], data_root=root)
            with self.assertRaisesRegex(ValueError, "requires DATA_ROOT"):
                collect_completed_build_refs([current], {"builds": [ref]}, [])

    def test_source_matching_never_uses_basename_and_rejects_duplicate_aliases(self):
        current = "/content/drive/MyDrive/VietMedBridge/data/incoming/batch-one/a.tar"
        other = "/content/drive/.shortcut-targets-by-id/shared/VietMedBridge/data/incoming/batch-two/a.tar"
        ref = {"source_path": other, "state": "COMPLETE", "build_run": "build-0"}
        with self.assertRaisesRegex(ValueError, "missing="):
            collect_completed_build_refs([current], {"builds": [ref]}, [])
        alias = current.replace("MyDrive", ".shortcut-targets-by-id/shared")
        with self.assertRaisesRegex(ValueError, "Ambiguous/duplicate"):
            collect_completed_build_refs([current, alias], {}, [])

    def test_worker_batch_identity_changes_with_snapshot_code_and_order(self):
        refs = [{"build_run": f"build-{i}", "snapshot_sha256": f"snapshot-{i}"} for i in range(2)]
        original = freeze_batch_signature(refs, code_sha256="code", team_size=3)
        for changed, code, size in (([{**refs[0], "snapshot_sha256": "new"}, refs[1]], "code", 3),
                                    (list(reversed(refs)), "code", 3), (refs, "new-code", 3), (refs, "code", 2)):
            self.assertNotEqual(original, freeze_batch_signature(changed, code_sha256=code, team_size=size))

    def test_source_collection_retains_frozen_archive_zero_and_rejects_missing(self):
        expected = [f"/incoming/archive-{i}" for i in range(7)]
        prior = {"builds": [{"source_path": expected[0], "state": "FROZEN",
                             "build_run": "build-0", "snapshot_sha256": "snapshot-0"}]}
        workers = [(f"worker-{i}", {"state": "COMPLETE", "builds": [
            {"source_path": expected[i], "state": "COMPLETE", "build_run": f"build-{i}",
             "snapshot_sha256": f"snapshot-{i}"},
            {"source_path": "/incoming/other-batch", "state": "COMPLETE"}]}) for i in range(1, 7)]
        self.assertEqual(len(collect_completed_build_refs(expected, prior, workers)), 7)
        with self.assertRaisesRegex(ValueError, "missing=.*archive-6"):
            collect_completed_build_refs(expected, prior, workers[:-1])

    def test_coordinator_checks_actual_source_and_index_files(self):
        for kind in ("documents", "index"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
                root = Path(directory)
                refs = [{"build_run": "build-0", "snapshot_sha256": "pending"}]
                completed = {"build-0": _prepare_build(root, refs[0], "code")}
                signature = _write_workers(root, refs, completed, team_size=1)
                path = root / "processed/build-0" / ("documents.parquet" if kind == "documents"
                                                       else "index_inputs/units-000000000000.parquet")
                with path.open("ab") as stream:
                    stream.write(b"corruption")
                with self.assertRaisesRegex(ValueError, "changed artifact"):
                    verify_freeze_workers(root, refs, batch_signature=signature, code_sha256="code", team_size=1)

    def test_coordinator_reuses_finished_workers_from_previous_orchestrator(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            refs = [{"build_run": f"build-{i}"} for i in range(7)]
            completed = {item["build_run"]: _prepare_build(root, item, "code") for item in refs}
            current = freeze_batch_signature(refs, code_sha256="code", team_size=3)
            for worker_id in range(3):
                signature = current if worker_id == 1 else "old-signature-" + str(worker_id) * 32
                assigned = [item["build_run"] for item in partition_freeze_builds(refs, team_size=3, worker_id=worker_id)]
                write_freeze_worker_manifest(
                    freeze_worker_manifest_path(root, batch_signature=signature, worker_id=worker_id),
                    batch_signature=signature, code_sha256="code", team_size=3, worker_id=worker_id,
                    assigned_build_runs=assigned, completed={run: completed[run] for run in assigned}, state="COMPLETE")
            verified = verify_freeze_workers(root, refs, batch_signature=current, code_sha256="code", team_size=3)
            self.assertEqual(len(verified), 7)
            self.assertFalse(freeze_worker_manifest_path(root, batch_signature=current, worker_id=0).exists())
            # Reuse never bypasses physical artifact validation.
            with (root / "processed/build-3/documents.parquet").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                verify_freeze_workers(root, refs, batch_signature=current, code_sha256="code", team_size=3)

    def test_coordinator_rejects_old_receipts_with_changed_snapshot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            refs = [{"build_run": "build-0"}]
            record = _prepare_build(root, refs[0], "code")
            signature = "old-signature-" + "0" * 32
            write_freeze_worker_manifest(
                freeze_worker_manifest_path(root, batch_signature=signature, worker_id=0),
                batch_signature=signature, code_sha256="code", team_size=1, worker_id=0,
                assigned_build_runs=["build-0"], completed={"build-0": {**record, "snapshot_sha256": "other"}},
                state="COMPLETE")
            with self.assertRaises(FileNotFoundError):
                verify_freeze_workers(root, refs, batch_signature="new-signature-" + "1" * 32,
                                     code_sha256="code", team_size=1)

    def test_coordinator_rejects_missing_worker_and_stale_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            refs = [{"build_run": f"build-{i}", "snapshot_sha256": "pending"} for i in range(2)]
            completed = {item["build_run"]: _prepare_build(root, item, "old-code") for item in refs}
            signature = _write_workers(root, refs, completed, code_sha="current-code")
            with self.assertRaisesRegex(ValueError, "validation.*stale"):
                verify_freeze_workers(root, refs, batch_signature=signature, code_sha256="current-code", team_size=2)
            path = freeze_worker_manifest_path(root, batch_signature=signature, worker_id=0)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_freeze_workers(root, refs, batch_signature=signature, code_sha256="current-code", team_size=2)

    def test_batch_report_preserves_failures_and_reports_duplicate_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            refs = [{"build_run": f"build-{i}", "snapshot_sha256": "pending", "doc_id": 42} for i in range(2)]
            completed = {item["build_run"]: _prepare_build(root, item, "code") for item in refs}
            signature = _write_workers(root, refs, completed)
            verified = verify_freeze_workers(root, refs, batch_signature=signature, code_sha256="code", team_size=2)
            report = audit_freeze_batch(root, refs, verified, work_dir=root / "work")
            self.assertEqual(report["input_records"], 4)
            self.assertEqual(report["unique_official_ids"], 2)
            self.assertEqual(report["duplicate_official_ids_across_builds"], 2)
            self.assertEqual(report["counts"]["failures"], 2)
            self.assertEqual(report["unaccounted_input_records"], 0)
            self.assertTrue((root / "reports/freeze_batch_coverage.json").is_file())

    def test_partition_assigns_each_build_once_and_balances_workers(self):
        refs = [{"build_run": f"build-{i}", "snapshot_sha256": f"snapshot-{i}"} for i in range(7)]

        assignments = [partition_freeze_builds(refs, team_size=3, worker_id=i) for i in range(3)]

        self.assertEqual([[item["build_run"] for item in group] for group in assignments], [
            ["build-0", "build-3", "build-6"],
            ["build-1", "build-4"],
            ["build-2", "build-5"],
        ])
        self.assertEqual(sum(map(len, assignments)), len(refs))

    def test_coordinator_verifies_all_worker_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            tmp_path = Path(directory)
            refs = [{"build_run": f"build-{i}", "snapshot_sha256": f"snapshot-{i}"} for i in range(4)]
            code_sha = "code-sha"
            completed = {item["build_run"]: _prepare_build(tmp_path, item, code_sha) for item in refs}
            signature = _write_workers(tmp_path, refs, completed, code_sha=code_sha)

            verified = verify_freeze_workers(
                tmp_path, refs, batch_signature=signature, code_sha256=code_sha, team_size=2
            )

            self.assertEqual([item["candidate"]["snapshot_sha256"] for item in verified], [
                item["snapshot_sha256"] for item in refs
            ])

    def test_coordinator_rejects_incomplete_worker_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            tmp_path = Path(directory)
            refs = [{"build_run": f"build-{i}", "snapshot_sha256": f"snapshot-{i}"} for i in range(3)]
            signature = freeze_batch_signature(refs, code_sha256="code", team_size=2)
            worker_id = 0
            assigned = partition_freeze_builds(refs, team_size=2, worker_id=worker_id)
            completed = {
                item["build_run"]: _prepare_build(tmp_path, item, "code")
                for item in assigned[:1]
            }
            write_freeze_worker_manifest(
                freeze_worker_manifest_path(tmp_path, batch_signature=signature, worker_id=worker_id),
                batch_signature=signature,
                code_sha256="code",
                team_size=2,
                worker_id=worker_id,
                assigned_build_runs=[item["build_run"] for item in assigned],
                completed=completed,
                state="RUNNING",
            )

            with self.assertRaisesRegex(ValueError, "stale or incomplete"):
                verify_freeze_workers(
                    tmp_path, refs, batch_signature=signature, code_sha256="code", team_size=2
                )


if __name__ == "__main__":
    unittest.main()
