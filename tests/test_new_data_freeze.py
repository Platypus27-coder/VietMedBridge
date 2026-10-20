"""Real producer artifacts exercise the isolated CPU freeze handoff and resume."""
import ast
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil

import nbformat
import pytest

from scripts.new_data_freeze import (NewDataFreeze, collect_intake_builds, copy_verified,
    PARENT_CAMPAIGN, EXTRA_CAMPAIGN, checked, seal)
from vietmedbridge.artifacts import atomic_json, code_fingerprint, digest_json, read_json, sha256_file
from vietmedbridge.external_import import import_external_corpus
from test_external_import import external, CharacterTokenizer, SPEC, CONFIG


@pytest.fixture
def campaign(tmp_path, external):
    project = tmp_path / "project"
    parent = project / "data_staging" / PARENT_CAMPAIGN
    extra = project / "data_staging" / EXTRA_CAMPAIGN
    native = tmp_path / "native"
    build = import_external_corpus(external[0], external[1], native, CharacterTokenizer(), SPEC,
        run_name="fixture", config=CONFIG, shard_size=2, work_dir=tmp_path / "import-work")
    groups = [
        (parent, [f"new-{i:05d}" for i in range(6)]),
        (parent / "additional-frontiers", ["new-00006", "new-00007"]),
        (parent / "retained-recovery", ["retained-test"]),
        (extra, ["new-00008"]),
    ]
    counts, refs, sources_by_root = {}, [], {}
    for root, runs in groups:
        sources = [{"job_id": run, "snapshot_id": "team-test", "mode": "NEW_SNAPSHOT_SOURCE",
                    "drive_file_id": run, "archive": run + ".tar"} for run in runs]
        if runs == ["retained-test"]:
            sources[0]["portable_source_sha256"] = "a" * 64
        plan = {"sources": sources}
        atomic_json(root / "intake_plan.json", plan)
        atomic_json(root / "intake_status.json", {"state": "INTAKE_COMPLETE", "plan_sha256": digest_json(plan),
            "completed_sources": len(runs), "total_sources": len(runs)})
        sources_by_root[root] = sources
        for source in sources:
            run = source["job_id"]
            shutil.copytree(native / "fixture", root / "processed" / run)
            outcome = {"ADDED_VALID_DOCUMENT": build["counts"]["documents"],
                "PIPELINE_FAILURE": build["counts"]["failures"], "REUSED_EXISTING_DOCUMENT": 0}
            receipt = {"state": "COMPLETE", "source": source, "plan_sha256": digest_json(plan),
                "build_run": run, "build_snapshot_sha256": build["snapshot_sha256"],
                "counts": outcome, "data_counts": build["counts"]}
            atomic_json(root / "sources" / run / "complete.json", receipt)
            if root != extra:
                for key, value in outcome.items():
                    counts[key] = counts.get(key, 0) + value
                refs.append({"build_run": run, "snapshot_sha256": build["snapshot_sha256"]})
    # The portable retained source legitimately adds derived fields to its plan.
    retained = {k: v for k, v in sources_by_root[parent / "retained-recovery"][0].items()
                if k != "portable_source_sha256"}
    retained["members"] = {"metadata": "not a source-plan field"}
    scope = {"base_plan": {"sources": sources_by_root[parent], "required_snapshot_id": "team-test"},
             "additional_sources": sources_by_root[parent / "additional-frontiers"], "retained": retained}
    atomic_json(parent / "full_intake_scope.json", scope)
    atomic_json(parent / "full_intake_status.json", {"state": "FULL_INTAKE_COMPLETE", "scope_sha256": digest_json(scope),
        "completed_sources": 9, "total_sources": 9, "pending_sources": [], "builds": refs, "counts": counts})
    return project, tmp_path / "freeze-work", external[1]


def loaders(links):
    return dict(load_tokenizer=lambda item: (CharacterTokenizer(), SPEC), load_links=lambda item: links,
        load_golden=lambda item, tokenizer, spec: {"passed": True, "code_sha256": code_fingerprint(),
            "tokenizer": spec, "chunking": asdict(CONFIG)}, memory_mb=128)


def protected_files(project):
    return {p.relative_to(project).as_posix(): sha256_file(p) for p in project.rglob("*.parquet")}


def test_real_ten_build_freeze_then_cold_resume_does_no_payload_or_model_work(campaign, monkeypatch):
    project, work, links = campaign
    active = project / "data" / "active_data_candidate.json"
    atomic_json(active, {"must": "remain frozen for ongoing submission"})
    active_before = active.read_bytes()
    original_payloads = protected_files(project)
    freezer = NewDataFreeze(project, work)
    report = freezer.run(**loaders(links))
    assert report["state"] == "NEW_DATA_INPUTS_READY" and report["completed_builds"] == 10
    assert report["documents"] == 20 and report["embedding_complete"] is False
    assert len(list(freezer.root.glob("builds/*/input_spool.json"))) == 10
    assert active.read_bytes() == active_before
    for rel, checksum in original_payloads.items():
        assert sha256_file(project / rel) == checksum
    assert not list((work / "staged").glob("*/build.json"))
    def fail(*args, **kwargs):
        pytest.fail("Completed freeze re-opened corpus, weights, or tokenizer")
    shutil.rmtree(work)
    resumed = NewDataFreeze(project, work)
    monkeypatch.setattr(resumed, "_stage", fail)
    assert resumed.run(load_tokenizer=fail, load_links=fail, load_golden=fail) == report
    assert active.read_bytes() == active_before


def test_resume_after_candidate_commit_reuses_candidate_and_input_parts(campaign, monkeypatch):
    project, work, links = campaign
    first = NewDataFreeze(project, work)
    report = first.run(**loaders(links))
    item = first.builds[0]
    folder = first.root / "builds" / item["build_run"]
    frozen_before = (folder / "frozen.json").read_bytes()
    (folder / "complete.json").unlink()
    (first.root / "freeze_batch.json").unlink()
    inputs = project / item["build_dir"] / "index_inputs"
    part_before = {p.name: sha256_file(p) for p in inputs.glob("*.parquet")}
    (inputs / "units.json").unlink()  # Crash after the input parts, before sealing handoff.
    def fail(*a, **k):
        pytest.fail("Frozen candidate was validated/frozen again")
    monkeypatch.setattr("vietmedbridge.health.health_report", fail)
    shutil.rmtree(work)  # A new VM; completed input parts need no source/SQL restore.
    kwargs = loaders(links)
    kwargs.update(load_tokenizer=fail, load_links=fail, load_golden=fail)
    resumed = NewDataFreeze(project, work)
    new = resumed.run(**kwargs)
    assert new["documents"] == report["documents"]
    assert (folder / "frozen.json").read_bytes() == frozen_before
    assert {p.name: sha256_file(p) for p in inputs.glob("*.parquet")} == part_before


def test_cold_sql_checkpoint_restores_committed_source_rows_without_parquet_reads(campaign):
    from vietmedbridge.input_spool import InputSpool
    project, work, links = campaign
    freezer = NewDataFreeze(project, work)
    freezer.run(**loaders(links))
    item = freezer.builds[0]
    state = freezer.root / "builds" / item["build_run"]
    frozen = checked(read_json(state / "frozen.json"))
    candidate = freezer._candidate(item, frozen)
    inputs = project / item["build_dir"] / "index_inputs"
    (inputs / "units.json").unlink()
    # Model an interruption just after the SQL snapshot, before any input part.
    cursor = read_json(inputs / "resume.json")
    cursor.pop("manifest_sha256")
    cursor.update(parts=[], cursor=0, last_id="", spool_base_id=None, spool_base_rows=0)
    seal(inputs / "resume.json", cursor)
    shutil.rmtree(work)
    reopened = NewDataFreeze(project, work)
    local = reopened._input_local(item)  # Contains JSON only, no source Parquet.
    reopened._restore_input_spool(item, candidate, state)
    saved = checked(read_json(state / "input_spool.json"))
    def fail(*a, **k):
        pytest.fail("Restored SQL checkpoint re-read a processed source payload")
    spool = InputSpool(local, candidate, work / saved["directory"], saved["input_signature"], source_loader=fail)
    try:
        assert spool.stage() == saved["representatives"]
        assert spool.rows_after("", 1)
        assert not list(local.rglob("*.parquet"))
    finally:
        spool.close()


@pytest.mark.parametrize("fault", ["missing_source", "counts", "config", "producer"])
def test_source_contract_fails_before_any_heavy_stage(campaign, fault):
    project, work, _ = campaign
    parent = project / "data_staging" / PARENT_CAMPAIGN
    if fault == "missing_source":
        (project / "data_staging" / EXTRA_CAMPAIGN / "sources/new-00008/complete.json").unlink()
    elif fault == "counts":
        path = parent / "full_intake_status.json"
        obj = read_json(path); obj["counts"]["ADDED_VALID_DOCUMENT"] -= 1; atomic_json(path, obj)
    else:
        path = parent / "processed/new-00000/config.json"
        obj = read_json(path); obj["code_sha256"] = "changed"; atomic_json(path, obj)
    with pytest.raises((ValueError, FileNotFoundError)):
        NewDataFreeze(project, work)
    assert not work.exists()


def test_completed_inputs_manifest_tampering_is_not_silently_rebuilt(campaign):
    project, work, links = campaign
    freezer = NewDataFreeze(project, work)
    freezer.run(**loaders(links))
    item = freezer.builds[0]
    path = project / item["build_dir"] / "index_inputs/units.json"
    obj = read_json(path); obj["input_count"] += 1; atomic_json(path, obj)
    with pytest.raises(ValueError, match="checksum"):
        freezer.completed(item)


def test_local_copy_proof_reuses_verified_bytes_and_rejects_corruption(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "local/payload"
    data = b"verified content"
    source.write_bytes(data)
    checksum = hashlib.sha256(data).hexdigest()
    copy_verified(source, destination, checksum)
    source.unlink()
    assert copy_verified(source, destination, checksum).read_bytes() == data
    destination.write_bytes(b"tampered")
    with pytest.raises(FileNotFoundError):
        copy_verified(source, destination, checksum)
    source.write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="checksum"):
        copy_verified(source, destination, checksum)
    assert not list(destination.parent.glob("*.partial"))


def test_notebook_matches_generator_and_pins_intake_without_active_pointer_writes():
    from scripts.write_new_data_freeze_notebook import notebook, ROOT
    generated = notebook()
    saved = nbformat.read(ROOT / "notebooks/03_colab_freeze_new_data.ipynb", as_version=4)
    assert [c.source for c in saved.cells] == [c.source for c in generated.cells]
    code = "\n".join(c.source for c in generated.cells if c.cell_type == "code")
    for cell in generated.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert "publish_data_handoff(" not in code
    assert "build_corpus(" not in code and "import_external_corpus(" not in code
    assert "lock[\"commit\"]" in code and "resume_policy=\"verify_on_use\"" in code
    assert "threading.Thread" in code and "freeze_batch.json" in code
