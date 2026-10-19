import ast
import copy
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_external_import import external, CharacterTokenizer, SPEC, CONFIG, write_parquet
from vietmedbridge.artifacts import atomic_json, read_json, sha256_file
from vietmedbridge.external_import import MEMBERS
from vietmedbridge.isolated_intake import (
    DriveRange, Window, intake_root, pin_plan, run_intake, stage_metadata,
)


class LocalReader:
    def __init__(self, files):
        self.files, self.opens = files, 0

    def metadata(self, file_id):
        path = self.files[file_id]
        return {"id": file_id, "name": path.name, "size": str(path.stat().st_size),
                "modifiedTime": "fixed", "md5Checksum": hashlib.md5(path.read_bytes()).hexdigest()}

    def open(self, file_id, metadata):
        self.opens += 1
        return self.files[file_id].open("rb")


def make_tar(directory, target):
    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as archive:
        for name in MEMBERS.values():
            path = directory/name
            member = archive.gettarinfo(path, arcname=name)
            member.pax_headers["vibiomir.sha256"] = sha256_file(path)
            with path.open("rb") as stream:
                archive.addfile(member, stream)
    return target


def make_source(tmp_path, external, job, snapshot="team-test", nested=False):
    directory = tmp_path/(job+"-source")
    shutil.copytree(external[0], directory)
    descriptor = read_json(directory/MEMBERS["dataset"])
    descriptor["snapshot_id"] = snapshot
    atomic_json(directory/MEMBERS["dataset"], descriptor)
    for kind in ("crawl", "extracted"):
        rows = pq.read_table(directory/MEMBERS[kind]).to_pylist()
        for row in rows:
            row["snapshot_id"] = snapshot
        write_parquet(directory/MEMBERS[kind], rows)
    archive = make_tar(directory, tmp_path/(job+".tar"))
    inner = None
    if nested:
        inner = archive.name
        outer = tmp_path/(job+"-outer.tar")
        with tarfile.open(outer, "w") as stream:
            stream.add(archive, arcname=inner)
        archive = outer
    return archive, {"job_id": job, "drive_file_id": job, "archive": archive.name,
        "archive_bytes": archive.stat().st_size, "inner_tar": inner,
        "frontier": MEMBERS["frontier"], "frontier_sha256": sha256_file(directory/MEMBERS["frontier"]),
        "snapshot_id": snapshot}


def test_flat_nested_hash_snapshot_and_cache(tmp_path, external):
    for nested in (False, True):
        path, source = make_source(tmp_path, external, "nested" if nested else "flat", nested=nested)
        reader, target = LocalReader({source["job_id"]: path}), tmp_path/source["job_id"]
        saved = stage_metadata(reader, source, target)
        assert set(saved["files"]) == set(MEMBERS)
        assert not list(target.glob("*.tar"))
        assert reader.opens == 1
        assert stage_metadata(reader, source, target) == saved
        assert reader.opens == 1
        (target/MEMBERS["extracted"]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="changed artifact"):
            stage_metadata(reader, source, target)
    changed = copy.deepcopy(source)
    changed["frontier_sha256"] = "0"*64
    with pytest.raises(ValueError, match="SHA-256"):
        stage_metadata(reader, changed, tmp_path/"bad-frontier")
    changed = copy.deepcopy(source)
    changed["snapshot_id"] = "wrong"
    with pytest.raises(ValueError, match="snapshot"):
        stage_metadata(reader, changed, tmp_path/"bad-snapshot")


def setup_intake(tmp_path, external, *, two=True):
    protected = tmp_path/"project/data"
    protected.mkdir(parents=True)
    for name in ("active_data_candidate.json", "active_data_batch.json", "raw/snapshot.json",
                 "data_processing_code_lock.json", "retrieval_embedding_runtime_lock.json",
                 "retrieval/full_resources/bge/map.json"):
        atomic_json(protected/name, {"preserve": name})
    before = {str(p.relative_to(protected)): p.read_bytes() for p in protected.rglob("*") if p.is_file()}
    baseline = tmp_path/"baseline.parquet"
    # Only one alias already exists; both source aliases must survive processing.
    write_parquet(baseline, [{"doc_id": 583, "url": "https://example.org/article"}])
    first, source1 = make_source(tmp_path, external, "new-00000", nested=True)
    files, sources = {"new-00000": first}, [source1]
    if two:
        second, source2 = make_source(tmp_path, external, "legacy-00014", snapshot="old-test")
        files["legacy-00014"] = second
        sources.append(source2)
    plan = {"sources": sources, "baseline": {"sha256": sha256_file(baseline)},
        "snapshots": {s["snapshot_id"]: {"sha256": sha256_file(external[1])} for s in sources}}
    kwargs = dict(root=intake_root(protected, "test-campaign"), reader=LocalReader(files),
        links_by_snapshot={s["snapshot_id"]: external[1] for s in sources}, baseline_aliases=baseline,
        tokenizer=CharacterTokenizer(), tokenizer_spec=SPEC, chunk_config=CONFIG,
        work_dir=tmp_path/"work", shard_size=1)
    return plan, kwargs, protected, before


def test_full_two_snapshot_import_aliases_failures_isolation_and_completed_resume(tmp_path, external, monkeypatch):
    plan, kwargs, protected, before = setup_intake(tmp_path, external)
    result = run_intake(plan, **kwargs)
    assert result["state"] == "INTAKE_COMPLETE"
    assert result["counts"] == {"ADDED_VALID_DOCUMENT": 2, "PIPELINE_FAILURE": 4, "REUSED_EXISTING_DOCUMENT": 2}
    root = kwargs["root"]
    for job in ("new-00000", "legacy-00014"):
        outcomes = pq.read_table(root/"sources"/job/"source_outcomes.parquet").to_pylist()
        assert {r["doc_id"] for r in outcomes} == {583, 921, 4000, 9999}
        build = read_json(root/"processed"/job/"build.json")
        assert build["integrity"]["passed"]
    assert before == {str(p.relative_to(protected)): p.read_bytes() for p in protected.rglob("*") if p.is_file()}
    assert not (root/"active_data_candidate.json").exists()
    # A complete source must not rescan the archive or invoke the importer.
    monkeypatch.setattr("vietmedbridge.isolated_intake.import_external_corpus", lambda *a, **k: pytest.fail("reimported"))
    monkeypatch.setattr(kwargs["reader"], "metadata", lambda *a: pytest.fail("rescanned archive"))
    resumed = run_intake(plan, **kwargs)
    assert resumed["counts"] == result["counts"]


def test_interrupted_import_resumes_saved_shard_after_local_scratch_loss(tmp_path, external, monkeypatch):
    import vietmedbridge.external_import as importer
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    original = importer.chunk_source
    calls = []

    def interrupt(doc, *args, **kw):
        calls.append(doc["doc_id"])
        if doc["doc_id"] == 921:
            raise RuntimeError("session interrupted")
        return original(doc, *args, **kw)

    monkeypatch.setattr(importer, "chunk_source", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_intake(plan, **kwargs)
    assert calls == [583, 921]
    marker = kwargs["root"]/"processed/new-00000/parts/part-000000.done.json"
    checkpoint = marker.read_bytes()
    shutil.rmtree(kwargs["work_dir"])
    calls.clear()

    def count(doc, *args, **kw):
        calls.append(doc["doc_id"])
        return original(doc, *args, **kw)

    monkeypatch.setattr(importer, "chunk_source", count)
    result = run_intake(plan, **kwargs)
    assert result["state"] == "INTAKE_COMPLETE"
    assert calls == [921]
    assert marker.read_bytes() == checkpoint
    runtime = read_json(kwargs["root"]/"processed/new-00000/import_runtime.json")
    assert runtime["reused_shards"] == 1


def test_resume_after_selection_file_published_before_its_marker(tmp_path, external, monkeypatch):
    import vietmedbridge.isolated_intake as intake
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    original = intake._parquet
    interrupted = False

    def stop_after_publish(con, query, destination, work):
        nonlocal interrupted
        result = original(con, query, destination, work)
        if not interrupted:
            interrupted = True
            raise RuntimeError("stopped before selection marker")
        return result

    monkeypatch.setattr(intake, "_parquet", stop_after_publish)
    with pytest.raises(RuntimeError, match="selection marker"):
        run_intake(plan, **kwargs)
    assert not (kwargs["root"]/"sources/new-00000/selection/selection.json").exists()
    assert run_intake(plan, **kwargs)["state"] == "INTAKE_COMPLETE"


def test_bad_binding_fails_before_filter_can_hide_foreign_or_corrupt_records(tmp_path, external):
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    # A changed official snapshot cannot be quietly used with an old source.
    altered = tmp_path/"altered.parquet"
    pq.write_table(pa.table({"id": [583], "url": ["https://other.org/"]}), altered)
    kwargs["links_by_snapshot"]["team-test"] = altered
    with pytest.raises(ValueError, match="changed artifact"):
        run_intake(plan, **kwargs)
    assert not (kwargs["root"]/"processed").exists()


def test_reuse_only_source_preserves_every_id_without_empty_build(tmp_path, external, monkeypatch):
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    links = pq.read_table(external[1]).to_pylist()
    write_parquet(kwargs["baseline_aliases"], [{"doc_id": r["id"], "url": r["url"]} for r in links])
    plan["baseline"]["sha256"] = sha256_file(kwargs["baseline_aliases"])
    monkeypatch.setattr("vietmedbridge.isolated_intake.import_external_corpus", lambda *a, **k: pytest.fail("reimported reused source"))
    result = run_intake(plan, **kwargs)
    assert result["counts"] == {"REUSED_EXISTING_DOCUMENT": 4}
    assert result["builds"] == []
    assert not (kwargs["root"]/"processed").exists()


def test_native_id_conflict_is_not_treated_as_a_duplicate(tmp_path, external):
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    write_parquet(kwargs["baseline_aliases"], [{"doc_id": 583, "url": "https://other.org/article"}])
    plan["baseline"]["sha256"] = sha256_file(kwargs["baseline_aliases"])
    with pytest.raises(ValueError, match="Native ID/URL conflict"):
        run_intake(plan, **kwargs)
    assert not (kwargs["root"]/"processed").exists()


def test_full_audit_runs_even_if_all_frontier_urls_would_be_reused(tmp_path, external):
    plan, kwargs, _, _ = setup_intake(tmp_path, external, two=False)
    # Add a foreign crawl record to an otherwise valid, fully reusable source.
    folder = tmp_path/"new-00000-source"
    rows = pq.read_table(folder/MEMBERS["crawl"]).to_pylist()
    rows.append(dict(rows[0], crawl_url_id="foreign", fetch_url="https://outside.org/"))
    write_parquet(folder/MEMBERS["crawl"], rows)
    source = plan["sources"][0]
    archive = make_tar(folder, tmp_path/"corrupt.tar")
    source.update(archive=archive.name, archive_bytes=archive.stat().st_size, inner_tar=None)
    kwargs["reader"].files[source["drive_file_id"]] = archive
    links = pq.read_table(external[1]).to_pylist()
    write_parquet(kwargs["baseline_aliases"], [{"doc_id": r["id"], "url": r["url"]} for r in links])
    plan["baseline"]["sha256"] = sha256_file(kwargs["baseline_aliases"])
    with pytest.raises(ValueError, match="crawl_outside_frontier"):
        run_intake(plan, **kwargs)
    assert not (kwargs["root"]/"processed").exists()


def test_resume_refuses_policy_change_and_active_output_root(tmp_path, external):
    plan, kwargs, protected, _ = setup_intake(tmp_path, external, two=False)
    bad = dict(kwargs, root=protected/"staging")
    with pytest.raises(ValueError, match="outside the active submission"):
        run_intake(plan, **bad)
    run_intake(plan, **kwargs)
    kwargs["shard_size"] = 2
    with pytest.raises(ValueError, match="policy changed"):
        run_intake(plan, **kwargs)


def test_duplicate_version_plan_and_output_escape_rejected(tmp_path):
    source = {"snapshot_id": "same", "frontier_sha256": "same"}
    with pytest.raises(ValueError, match="multiple versions"):
        pin_plan(tmp_path, {"sources": [source, dict(source)]})
    with pytest.raises(ValueError, match="campaign"):
        intake_root(tmp_path, "../data")
    bounded = Window(io.BytesIO(b"abcDEFGhij"), 3, 4)
    assert bounded.read() == b"DEFG"
    with pytest.raises(ValueError, match="outside"):
        bounded.seek(5)


def test_drive_range_rejects_unbounded_http_response_before_reading_payload():
    class Response:
        status_code, headers = 200, {}
        raw = None  # Touching the body would fail this test.
        def raise_for_status(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Session:
        def get(self, *args, **kwargs):
            assert kwargs["stream"]
            assert kwargs["headers"]["Range"] == "bytes=0-9"
            return Response()
    with pytest.raises(ValueError, match="bounded Range"):
        DriveRange(Session(), "source-id", {"size": 10}).read(1)


def test_drive_range_cached_headers_seeks_and_exact_payload():
    payload = bytes(range(200))
    class Response:
        status_code = 206
        def __init__(self, start, end):
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
            self.raw = io.BytesIO(payload[start:end+1])
        def raise_for_status(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Session:
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            start, end = map(int, kwargs["headers"]["Range"].split("=")[1].split("-"))
            return Response(start, end)
    session = Session()
    stream = DriveRange(session, "source-id", {"size": len(payload)}, block_size=32)
    assert stream.read(10) == payload[:10]
    stream.seek(4)
    assert stream.read(12) == payload[4:16]
    assert session.calls == 1
    stream.seek(145)
    assert stream.read(50) == payload[145:195]
    assert stream.read(100) == payload[195:]
    assert stream.read(1) == b""


def test_notebook_default_is_cpu_separate_checkout_and_no_publication():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/"notebooks/02_colab_import_new_data.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "VietMedBridge-intake" in code
    assert "data_staging" in code and "intake_code_lock.json" in code
    assert "load_bge_tokenizer" in code and "run_golden_suite" in code
    assert "publish_data_handoff" not in code and "snapshot_dataset" not in code
    plan = read_json(root/"configs/intake_ai_guru_20261019.json")
    assert len(plan["sources"]) == 6
    assert len({(s["snapshot_id"], s["frontier_sha256"]) for s in plan["sources"]}) == 6
    assert sum(s["mode"] == "NEW_SNAPSHOT_SOURCE" for s in plan["sources"]) == 6
    assert set(plan["snapshots"]) == {plan["required_snapshot_id"]}
    assert {s["job_id"] for s in plan["sources"]} == {f"new-{i:05d}" for i in range(6)}
    assert 'CAMPAIGN = "btc-new-71be0e12-20261019-v1"' in code
    # Execute the notebook's scope guard without mounting Drive/importing Colab.
    guard = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)
                 and node.name == "validate_new_btc_plan")
    scope = {}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "scope_guard", "exec"), scope)
    scope["validate_new_btc_plan"](plan)
    mixed = copy.deepcopy(plan)
    mixed["sources"].append(dict(plan["sources"][0], snapshot_id="old", mode="LEGACY_SOURCE_NOT_YET_IMPORTED"))
    with pytest.raises(ValueError, match="snapshot BTC mới"):
        scope["validate_new_btc_plan"](mixed)
    mixed = copy.deepcopy(plan)
    mixed["snapshots"]["old"] = {}
    with pytest.raises(ValueError, match="snapshot BTC mới"):
        scope["validate_new_btc_plan"](mixed)
