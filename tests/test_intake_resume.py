import shutil
from pathlib import Path

import pytest

from test_external_import import external, CharacterTokenizer, SPEC, CONFIG
from test_isolated_intake import setup_intake
from vietmedbridge.artifacts import read_json, sha256_file
from vietmedbridge.isolated_intake import run_intake
from scripts.intake_resume import IntakeResume
from vietmedbridge import isolated_intake as core


def controller(plan, kwargs):
    return IntakeResume(plan, root=kwargs["root"], work_dir=kwargs["work_dir"],
        tokenizer_spec=SPEC, chunk_config=CONFIG, shard_size=1)


def loaders(kwargs):
    return dict(load_reader=lambda: kwargs["reader"], load_links=lambda s: kwargs["links_by_snapshot"][s],
        load_baseline=lambda: kwargs["baseline_aliases"], load_tokenizer=lambda: (CharacterTokenizer(), SPEC))


def forbidden(*args, **kwargs):
    pytest.fail("Completed or already-selected source restored an unneeded resource")


def test_complete_resume_reads_no_parquet_or_source_and_needs_no_resources(tmp_path, external, monkeypatch):
    plan, kwargs, protected, before = setup_intake(tmp_path, external)
    result = controller(plan, kwargs).run(**loaders(kwargs))
    assert result["counts"] == {"ADDED_VALID_DOCUMENT": 2, "PIPELINE_FAILURE": 4, "REUSED_EXISTING_DOCUMENT": 2}
    original = Path.open

    def guard(path, *args, **kw):
        if path.suffix == ".parquet":
            pytest.fail("Read completed Parquet on status-only resume")
        return original(path, *args, **kw)

    monkeypatch.setattr(Path, "open", guard)
    resumed = controller(plan, kwargs).run(**{k: forbidden for k in loaders(kwargs)})
    assert resumed["state"] == "INTAKE_COMPLETE"
    assert resumed["counts"] == result["counts"]
    assert before == {str(p.relative_to(protected)): p.read_bytes() for p in protected.rglob("*") if p.is_file()}


def test_existing_deployed_receipts_remain_compatible_without_rehash(tmp_path, external, monkeypatch):
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    original = run_intake(plan, **kwargs)
    def reject(*args, **kw):
        pytest.fail("Rehashed an existing completed payload")
    monkeypatch.setattr(core, "verify_file", reject)
    monkeypatch.setattr(core, "_local_mapping", reject)
    result = controller(plan, kwargs).run(**{k: forbidden for k in loaders(kwargs)})
    assert result["counts"] == original["counts"]


@pytest.mark.parametrize("legacy_first", [False, True])
def test_resume_partial_source_after_local_loss_does_not_reload_baseline_or_rechunk_prefix(tmp_path, external, monkeypatch, legacy_first):
    import vietmedbridge.external_import as importer
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    original, calls = importer.chunk_source, []
    def stop(doc, *args, **kw):
        calls.append(doc["doc_id"])
        if doc["doc_id"] == 921:
            raise RuntimeError("interrupted")
        return original(doc, *args, **kw)
    monkeypatch.setattr(importer, "chunk_source", stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        if legacy_first:
            run_intake(plan, **kwargs)
        else:
            controller(plan, kwargs).run(**loaders(kwargs))
    assert calls == [583, 921]
    checkpoint = kwargs["root"]/"processed/new-00000/parts/part-000000.done.json"
    old = checkpoint.read_bytes()
    shutil.rmtree(kwargs["work_dir"])
    calls.clear()
    def count(doc, *args, **kw):
        calls.append(doc["doc_id"])
        return original(doc, *args, **kw)
    monkeypatch.setattr(importer, "chunk_source", count)
    resources = loaders(kwargs)
    resources.update(load_baseline=forbidden, load_reader=forbidden)
    result = controller(plan, kwargs).run(**resources)
    assert result["state"] == "INTAKE_COMPLETE"
    assert calls == [921]
    assert checkpoint.read_bytes() == old


def test_completed_prefix_not_loaded_when_next_source_has_saved_selection(tmp_path, external, monkeypatch):
    plan, kwargs, _, _ = setup_intake(tmp_path, external)
    original = core.import_external_corpus
    def interrupt(*args, **kw):
        if kw["run_name"] == "legacy-00014":
            raise RuntimeError("next source interrupted")
        return original(*args, **kw)
    monkeypatch.setattr(core, "import_external_corpus", interrupt)
    with pytest.raises(RuntimeError, match="next source"):
        controller(plan, kwargs).run(**loaders(kwargs))
    first = kwargs["root"]/"sources/new-00000"
    monkeypatch.setattr(core, "import_external_corpus", original)
    resources = loaders(kwargs)
    resources.update(load_baseline=forbidden, load_reader=forbidden)
    old_open = Path.open
    def guard(path, *args, **kw):
        if path.suffix == ".parquet" and path.is_relative_to(first):
            pytest.fail("Read completed predecessor payload unnecessarily")
        return old_open(path, *args, **kw)
    monkeypatch.setattr(Path, "open", guard)
    assert controller(plan, kwargs).run(**resources)["state"] == "INTAKE_COMPLETE"


def test_validated_build_finalize_does_not_restore_tokenizer_reimport_or_revalidate(tmp_path, external, monkeypatch):
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    original = core._write_outcomes
    def stop(*args, **kw):
        raise RuntimeError("stopped after validated build")
    monkeypatch.setattr(core, "_write_outcomes", stop)
    with pytest.raises(RuntimeError, match="validated build"):
        controller(plan, kwargs).run(**loaders(kwargs))
    build = kwargs["root"]/"processed/new-00000/build.json"
    before = sha256_file(build)
    monkeypatch.setattr(core, "_write_outcomes", original)
    monkeypatch.setattr(core, "import_external_corpus", forbidden)
    result = controller(plan, kwargs).run(**{k: forbidden for k in loaders(kwargs)})
    assert result["state"] == "INTAKE_COMPLETE"
    assert sha256_file(build) == before


def test_missing_completed_payload_fails_without_rebuild(tmp_path, external):
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    controller(plan, kwargs).run(**loaders(kwargs))
    (kwargs["root"]/"sources/new-00000/added_ids.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        controller(plan, kwargs)


def test_completed_payload_hash_is_checked_when_needed_for_a_new_selection(tmp_path, external, monkeypatch):
    plan, kwargs, _, _ = setup_intake(tmp_path, external)
    # Stop after committing source one and before selecting source two.
    original = core.stage_metadata
    def stop(reader, source, target):
        if source["job_id"] == "legacy-00014":
            raise RuntimeError("new selection not started")
        return original(reader, source, target)
    monkeypatch.setattr(core, "stage_metadata", stop)
    with pytest.raises(RuntimeError, match="not started"):
        controller(plan, kwargs).run(**loaders(kwargs))
    mapping = kwargs["root"]/"sources/new-00000/added_ids.parquet"
    data = bytearray(mapping.read_bytes())
    data[20] ^= 1  # same size: cheap status check is deliberately NOT a SHA audit
    mapping.write_bytes(data)
    monkeypatch.setattr(core, "stage_metadata", original)
    with pytest.raises(ValueError, match="mapping changed"):
        controller(plan, kwargs).run(**loaders(kwargs))


def test_resume_controller_does_not_modify_pinned_producer_identity():
    root = Path(__file__).resolve().parents[1]
    notebook = __import__("json").loads((root/"notebooks/02_colab_import_new_data.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert (root/"scripts/intake_resume.py").read_text(encoding="utf-8").strip() in text
    assert "CONTROLLER.run(" in text and "from vietmedbridge.isolated_intake import run_intake" not in text
    assert "LAZY_RESOURCES" in text
