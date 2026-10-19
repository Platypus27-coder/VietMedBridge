import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from test_external_import import external, CharacterTokenizer, SPEC, CONFIG, write_parquet
from test_isolated_intake import LocalReader, make_source, setup_intake
from test_intake_resume import loaders, forbidden
from scripts.intake_resume import IntakeResume, bounded_intake_parquet, bounded_intake_sql
from scripts.retained_intake import RetainedAdapter
from scripts.full_intake import FullIntake
from vietmedbridge import isolated_intake as core
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file


def portable_fixture(tmp_path, external, *, document_snapshot="old-capture"):
    source, links, text = external
    rows = pq.read_table(links).to_pylist()
    rows += [{"id": 77777, "url": "https://example.org/new-retained"},
             {"id": 88888, "url": "https://example.org/low-retained"}]
    write_parquet(links, rows)
    descriptor = read_json(source/core.MEMBERS["dataset"])
    descriptor["sources"][0]["sha256"] = sha256_file(links)
    atomic_json(source/core.MEMBERS["dataset"], descriptor)
    mapping, raw, docs = [], [], []
    groups = {}
    for row in rows:
        url = row["url"].split("#")[0]
        groups.setdefault(url, []).append(row)
    for index, (url, aliases) in enumerate(groups.items()):
        key = str(index)
        raw_sha = hashlib.sha256(("HTTP "+url).encode()).hexdigest()
        status = "JAVASCRIPT_CHALLENGE" if url.endswith("failed") else "EXTRACT_EMPTY" if url.endswith("parse") else "EXTRACT_SUCCESS"
        tier = "LOW" if url.endswith("low-retained") else "HIGH"
        for alias in aliases:
            mapping.append({"crawl_url_id": key, "doc_id": alias["id"], "original_url": alias["url"], "source_row_no": alias["id"]})
        original = {"status": "SUCCESS_RAW", "snapshot_id": "old-capture", "raw_sha256": raw_sha,
                    "fetch_url": url, "http_status": 200, "elapsed_ms": 1.}
        raw.append({"crawl_url_id": key, "fetch_url": url, "final_url": url,
            "raw_sha256": raw_sha, "raw_path": "raw/"+raw_sha+".zst", "raw_size_bytes": 1000,
            "content_type": "text/html", "crawl_timestamp": "2026-10-05T00:00:00+00:00",
            "original_snapshot_id": "old-capture", "new_snapshot_id": "team-test",
            "extract_status": status, "quality_tier": tier, "original_crawl_record_json": json.dumps(original)})
        if status == "EXTRACT_SUCCESS":
            docs.append({"crawl_url_id": key, "raw_sha256": raw_sha, "snapshot_id": document_snapshot,
                "extract_status": status, "text": text, "char_count": len(text), "title": "Y sinh",
                "text_markdown": "# Kết quả\n"+text, "quality_tier": "HIGH",
                "language": "vi", "language_method": "hint", "language_confidence": 1.,
                "extractor": "test", "extractor_version": "v1", "source_issue": None})
    folder = tmp_path/"portable-source"
    files = {"mapping": "data/doc_url_map.parquet", "raw_catalog": "data/raw_catalog.parquet",
             "documents": "data/documents.parquet", "dataset": "new_dataset_manifest.json"}
    for kind, content in (("mapping", mapping), ("raw_catalog", raw), ("documents", docs)):
        write_parquet(folder/files[kind], content)
    atomic_json(folder/files["dataset"], descriptor)
    archive = tmp_path/"retained.tar"
    members = {}
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        for kind, name in files.items():
            info = tar.gettarinfo(folder/name, arcname=name)
            with (folder/name).open("rb") as stream:
                tar.addfile(info, stream)
    with tarfile.open(archive) as tar:
        for kind, name in files.items():
            info = tar.getmember(name)
            members[kind] = {"path": name, "offset": info.offset_data, "bytes": info.size,
                             "sha256": sha256_file(folder/name)}
    spec = {"job_id": "retained-test", "mode": "NEW_SNAPSHOT_SOURCE", "format": "RETAINED_PORTABLE_V1",
        "archive": archive.name, "drive_file_id": "retained", "archive_bytes": archive.stat().st_size,
        "snapshot_id": "team-test", "original_snapshot_id": "old-capture",
        "official_links_sha256": sha256_file(links), "members": members}
    return spec, archive


def setup_full(tmp_path, external):
    retained, archive = portable_fixture(tmp_path, external)
    plan, kwargs, protected, before = setup_intake(tmp_path, external, two=False)
    plan["required_snapshot_id"] = "team-test"
    plan["sources"][0]["mode"] = "NEW_SNAPSHOT_SOURCE"
    extra, item = make_source(tmp_path, external, "new-00001")
    item["mode"] = "NEW_SNAPSHOT_SOURCE"
    kwargs["reader"].files.update({"new-00001": extra, "retained": archive})
    scope = {"base_plan": plan, "base_plan_sha256": digest_json(plan), "additional_sources": [item],
             "retained": retained, "audited_coverage": {"test_only": True}}
    return scope, kwargs, protected, before


def full(scope, kwargs, embedded=False):
    classes = dict(FullIntake=FullIntake, IntakeResume=IntakeResume, RetainedAdapter=RetainedAdapter,
                   bounded_intake_sql=bounded_intake_sql, bounded_intake_parquet=bounded_intake_parquet)
    if embedded:
        root = Path(__file__).resolve().parents[1]
        notebook = read_json(root/"notebooks/02_colab_import_new_data.ipynb")
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "FULL_SCOPE = json.loads(" in "".join(c["source"]))
        classes = {}
        exec(compile(cell.split("FULL_SCOPE = json.loads(")[0], "embedded-intake", "exec"), classes)
    return classes["FullIntake"](scope, root=kwargs["root"], work_dir=kwargs["work_dir"], tokenizer_spec=SPEC,
        chunk_config=CONFIG, shard_size=1, controller_type=classes["IntakeResume"], retained_type=classes["RetainedAdapter"],
        sql_context=classes["bounded_intake_sql"], parquet_writer=classes["bounded_intake_parquet"])


@pytest.mark.parametrize("embedded", [False, True])
def test_full_scope_reuses_original_checkpoint_adds_all_phases_and_preserves_capture_snapshot(tmp_path, external, monkeypatch, embedded):
    scope, kwargs, protected, before = setup_full(tmp_path, external)
    original = IntakeResume(scope["base_plan"], root=kwargs["root"], work_dir=kwargs["work_dir"],
        tokenizer_spec=SPEC, chunk_config=CONFIG, shard_size=1)
    original.run(**loaders(kwargs))
    old = {str(p.relative_to(kwargs["root"])): p.read_bytes() for p in kwargs["root"].rglob("*")
           if p.is_file() and p.name != "intake_status.json"}
    result = full(scope, kwargs, embedded).run(**loaders(kwargs))
    assert result["state"] == "FULL_INTAKE_COMPLETE"
    assert result["completed_sources"] == result["total_sources"] == 3
    assert result["counts"] == {"ADDED_VALID_DOCUMENT": 3, "PIPELINE_FAILURE": 7, "REUSED_EXISTING_DOCUMENT": 4}
    assert all((kwargs["root"]/name).read_bytes() == data for name, data in old.items())
    root = kwargs["root"]/"retained-recovery"
    config = read_json(root/"processed/retained-test/config.json")
    assert config["external_snapshot_id"] == "old-capture"
    assert config["official_links_sha256"] == sha256_file(external[1])
    provenance = read_json(root/"processed/retained-test/intake_provenance.json")
    assert provenance["membership_snapshot_id"] == "team-test"
    assert provenance["original_snapshot_id"] == "old-capture"
    outcomes = pq.read_table(root/"sources/retained-test/source_outcomes.parquet").to_pylist()
    assert {r["doc_id"] for r in outcomes} == {583, 921, 4000, 9999, 77777, 88888}
    assert next(r for r in outcomes if r["doc_id"] == 88888)["reason_code"] == "RETAINED_QUALITY_REVIEW_LOW"
    assert before == {str(p.relative_to(protected)): p.read_bytes() for p in protected.rglob("*") if p.is_file()}
    # Reopening the full notebook must not replay the original or retained inputs.
    original_open = Path.open
    def no_parquet(path, *args, **kw):
        if path.suffix == ".parquet":
            pytest.fail("Completed full intake read Parquet payload on resume")
        return original_open(path, *args, **kw)
    monkeypatch.setattr(Path, "open", no_parquet)
    resumed = full(scope, kwargs, embedded).run(**{key: forbidden for key in loaders(kwargs)})
    assert resumed["counts"] == result["counts"]


def test_retained_validated_build_resume_survives_scratch_loss_without_reimport(tmp_path, external, monkeypatch):
    scope, kwargs, _, _ = setup_full(tmp_path, external)
    original = core._write_outcomes
    def stop(root, *args):
        if Path(root).name == "retained-test":
            raise RuntimeError("stopped after retained build")
        return original(root, *args)
    monkeypatch.setattr(core, "_write_outcomes", stop)
    with pytest.raises(RuntimeError, match="after retained build"):
        full(scope, kwargs).run(**loaders(kwargs))
    root = kwargs["root"]/"retained-recovery"
    build_path = root/"processed/retained-test/build.json"
    before = build_path.read_bytes()
    shutil.rmtree(kwargs["work_dir"])
    monkeypatch.setattr(core, "_write_outcomes", original)
    monkeypatch.setattr(core, "import_external_corpus", forbidden)
    result = full(scope, kwargs).run(**{key: forbidden for key in loaders(kwargs)})
    assert result["state"] == "FULL_INTAKE_COMPLETE"
    assert build_path.read_bytes() == before


def test_retained_rejects_mismatched_original_extraction_snapshot(tmp_path, external):
    spec, archive = portable_fixture(tmp_path, external, document_snapshot="wrong")
    reader = LocalReader({"retained": archive})
    work = tmp_path/"work"
    adapter = RetainedAdapter(spec, root=tmp_path/"metadata", work_dir=work, parquet_writer=bounded_intake_parquet)
    empty = tmp_path/"existing.parquet"
    write_parquet(empty, [{"doc_id": -1, "url": "https://other.org"}])
    with bounded_intake_sql():
        metadata = adapter.prepare(reader)
        with pytest.raises(ValueError, match="extraction_binding"):
            adapter.select(reader, metadata, external[1], [empty], tmp_path/"selection")
    assert not (tmp_path/"selection/selection.json").exists()


def test_retained_preserves_alias_group_when_only_one_existing_alias_is_present(tmp_path, external):
    spec, archive = portable_fixture(tmp_path, external)
    reader = LocalReader({"retained": archive})
    baseline = tmp_path/"baseline.parquet"
    write_parquet(baseline, [{"doc_id": 583, "url": "https://example.org/article"}])
    adapter = RetainedAdapter(spec, root=tmp_path/"metadata", work_dir=tmp_path/"work", parquet_writer=bounded_intake_parquet)
    with bounded_intake_sql():
        metadata = adapter.prepare(reader)
        selected = adapter.select(reader, metadata, external[1], [baseline], tmp_path/"selection")
    assert selected["counts"] == {"QUEUED_FOR_PIPELINE": 6}
    rows = pq.read_table(tmp_path/"selection/source_selection.parquet").to_pylist()
    assert {r["doc_id"] for r in rows if r["action"] == "QUEUED_FOR_PIPELINE"} == {583, 921, 4000, 9999, 77777, 88888}


def test_full_scope_rejects_foreign_membership_without_touching_submission(tmp_path, external):
    scope, kwargs, protected, before = setup_full(tmp_path, external)
    scope["additional_sources"][0]["snapshot_id"] = "foreign"
    with pytest.raises(ValueError, match="new BTC"):
        full(scope, kwargs)
    assert before == {str(p.relative_to(protected)): p.read_bytes() for p in protected.rglob("*") if p.is_file()}


def test_retained_download_resumes_committed_block_after_interruption(tmp_path):
    archive, payload = tmp_path/"portable.tar", b"abcdefgh"*(2*1024**2+512)
    name = "data/documents.parquet"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive) as tar:
        offset = tar.getmember(name).offset_data
    spec = {"drive_file_id": "retained", "archive_bytes": archive.stat().st_size,
            "members": {"documents": {"path": name, "offset": offset, "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()}}}
    reader = LocalReader({"retained": archive})
    first_open = reader.open
    reads = []
    class CutStream:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def seek(self, *args): return self.stream.seek(*args)
        def read(self, size):
            reads.append((self.stream.tell(), size))
            if self.stream.tell() >= offset+8*1024**2 and size > 512:
                raise OSError("network interrupted")
            return self.stream.read(size)
    reader.open = lambda *args: CutStream(first_open(*args))
    adapter = RetainedAdapter(spec, root=tmp_path/"members", work_dir=tmp_path/"work", parquet_writer=bounded_intake_parquet)
    with pytest.raises(OSError, match="network interrupted"):
        adapter.member(reader, "documents")
    marker = read_json(tmp_path/"members"/(name+".download.json"))
    assert marker["completed_bytes"] == 8*1024**2
    class TrackStream(CutStream):
        def read(self, size):
            reads.append((self.stream.tell(), size))
            return self.stream.read(size)
    reads.clear()
    reader.open = lambda *args: TrackStream(first_open(*args))
    target = adapter.member(reader, "documents")
    assert target.read_bytes() == payload
    assert all(start >= offset+8*1024**2 for start, size in reads if size > 512)
    assert not target.with_suffix(target.suffix+".partial").exists()


def test_notebook_embeds_complete_scope_and_adapters_without_changing_base_plan():
    root = Path(__file__).resolve().parents[1]
    scope = read_json(root/"configs/intake_ai_guru_full_20261019.json")
    assert scope["base_plan"] == read_json(root/"configs/intake_ai_guru_20261019.json")
    assert len(scope["base_plan"]["sources"])+len(scope["additional_sources"])+1 == 9
    assert len(scope["required_physical_archive_names"]) == 7
    assert {s["job_id"] for s in scope["additional_sources"]} == {"new-00006", "new-00007"}
    notebook = read_json(root/"notebooks/02_colab_import_new_data.ipynb")
    text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "CONTROLLER = FullIntake(" in text
    assert (root/"scripts/retained_intake.py").read_text(encoding="utf-8").strip() in text
    assert (root/"scripts/full_intake.py").read_text(encoding="utf-8").strip() in text
    assert "full_intake_status.json" in text
