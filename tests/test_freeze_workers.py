import tempfile
import unittest
from pathlib import Path
import sys

from vietmedbridge.artifacts import atomic_json, digest_json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freeze_workers import (
    freeze_batch_signature,
    freeze_worker_manifest_path,
    partition_freeze_builds,
    verify_freeze_workers,
    write_freeze_worker_manifest,
)


def _prepare_build(root, ref, code_sha):
    build_dir = root / "processed" / ref["build_run"]
    build_dir.mkdir(parents=True)
    build = {"selected_range_complete": True, "snapshot_sha256": ref["snapshot_sha256"]}
    atomic_json(build_dir / "build.json", build)

    candidate = {
        "state": "FROZEN_CANDIDATE",
        "selected_range_complete": True,
        "snapshot_sha256": ref["snapshot_sha256"],
        "golden": {"code_sha256": code_sha},
        "counts": {"documents": 1, "children": 1},
    }
    candidate_sha = digest_json(candidate)
    candidate["candidate_manifest_sha256"] = candidate_sha
    candidate_name = "candidate-test.json"
    atomic_json(build_dir / candidate_name, candidate)

    inputs = {"state": "COMPLETE", "candidate_manifest_sha256": candidate_sha, "input_count": 1}
    inputs_sha = digest_json(inputs)
    inputs["manifest_sha256"] = inputs_sha
    atomic_json(build_dir / "index_inputs" / "units.json", inputs)
    return {
        "state": "COMPLETE",
        "snapshot_sha256": ref["snapshot_sha256"],
        "candidate_name": candidate_name,
        "candidate_manifest_sha256": candidate_sha,
        "index_inputs_manifest_sha256": inputs_sha,
    }


class FreezeWorkerTests(unittest.TestCase):
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
            signature = freeze_batch_signature(refs, code_sha256=code_sha, team_size=2)

            for worker_id in range(2):
                assigned = partition_freeze_builds(refs, team_size=2, worker_id=worker_id)
                completed = {
                    item["build_run"]: _prepare_build(tmp_path, item, code_sha)
                    for item in assigned
                }
                write_freeze_worker_manifest(
                    freeze_worker_manifest_path(tmp_path, batch_signature=signature, worker_id=worker_id),
                    batch_signature=signature,
                    code_sha256=code_sha,
                    team_size=2,
                    worker_id=worker_id,
                    assigned_build_runs=[item["build_run"] for item in assigned],
                    completed=completed,
                    state="COMPLETE",
                )

            verified = verify_freeze_workers(
                tmp_path, refs, batch_signature=signature, code_sha256=code_sha, team_size=2
            )

            self.assertEqual([item["candidate"]["snapshot_sha256"] for item in verified], [
                "snapshot-0", "snapshot-1", "snapshot-2", "snapshot-3"
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
