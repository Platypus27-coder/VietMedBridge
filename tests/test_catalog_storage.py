"""Real SQLite/gzip transfers: lost Drive DB, failed publication and new-runtime restore."""
from pathlib import Path

import pytest

from test_scale_baseline import external, frozen  # noqa: F401
from vietmedbridge import catalog_storage as storage, disk_catalog as writer
from vietmedbridge.artifacts import read_json, sha256_file
from vietmedbridge.medical_lexical import MedicalAnalyzer


@pytest.fixture
def setup_catalog(tmp_path, frozen, monkeypatch):
    monkeypatch.setattr(storage, "CHUNK_BYTES", 16 * 1024)
    build, candidate, inputs = frozen
    analyzer = MedicalAnalyzer(segmentation=False)
    def prepare(work="work"):
        return storage.prepare_disk_catalog(build, candidate, inputs, tmp_path / "catalogs", analyzer,
                                            work_dir=tmp_path / work)
    return prepare, analyzer


def legacy(tmp_path, frozen, analyzer):
    build, candidate, inputs = frozen
    catalog = writer.prepare_disk_catalog(build, candidate, inputs, tmp_path / "catalogs", analyzer,
                                          work_dir=tmp_path / "work")
    catalog.close()
    marker = next((tmp_path / "catalogs").glob("*/catalog.json"))
    return marker, marker.parent / "catalog.sqlite"


def test_missing_drive_db_repairs_from_verified_local_without_source_rebuild(tmp_path, frozen, setup_catalog, monkeypatch):
    prepare, analyzer = setup_catalog
    marker, database = legacy(tmp_path, frozen, analyzer)
    old = read_json(marker)
    database.unlink()
    monkeypatch.setattr(writer, "rows", lambda *a: pytest.fail("Paid-for local catalog must be reused"))
    catalog = prepare()
    assert catalog.manifest["signature"] == old["signature"]
    assert catalog.manifest["sha256"] == old["sha256"]
    assert catalog.manifest["storage"]["format"] == "gzip-parts-v1"
    assert len(catalog.documents[583]["source_text"]) > 0
    catalog.close()
    storage._verify_remote(marker.parent, read_json(marker))


def test_healthy_legacy_catalog_reuses_identity_without_format_conversion(tmp_path, frozen, setup_catalog, monkeypatch):
    prepare, analyzer = setup_catalog
    marker, database = legacy(tmp_path, frozen, analyzer)
    before = marker.read_bytes()
    monkeypatch.setattr(writer, "rows", lambda *a: pytest.fail("Healthy legacy catalog must not rebuild"))
    catalog = prepare("new-runtime")
    assert catalog.manifest == read_json(marker) and "storage" not in catalog.manifest
    catalog.close()
    assert marker.read_bytes() == before and database.is_file()


def test_missing_drive_and_local_db_rebuild_only_derived_catalog(tmp_path, frozen, setup_catalog, monkeypatch):
    prepare, analyzer = setup_catalog
    marker, database = legacy(tmp_path, frozen, analyzer)
    database.unlink()
    next((tmp_path / "work").glob("catalog-*/catalog.sqlite")).unlink()
    calls = []
    original = writer.rows
    monkeypatch.setattr(writer, "rows", lambda path: (calls.append(Path(path).name), original(path))[1])
    catalog = prepare()
    assert calls and catalog.manifest["counts"]["documents"] == 2
    assert catalog.manifest["unit_count"] == frozen[2]["input_count"]
    catalog.close()
    storage._verify_remote(marker.parent, read_json(marker))


def test_chunked_catalog_new_runtime_restore_and_repeat_preserve_identity(tmp_path, frozen, setup_catalog, monkeypatch):
    prepare, _ = setup_catalog
    catalog = prepare()
    expected = catalog.documents[583]
    identity = catalog.identity
    catalog.close()
    marker = next((tmp_path / "catalogs").glob("*/catalog.json"))
    before = sha256_file(marker), marker.stat().st_mtime_ns
    assert not (marker.parent / "catalog.sqlite").exists()
    monkeypatch.setattr(writer, "rows", lambda *a: pytest.fail("Restore must not rebuild source rows"))
    for _ in range(2):
        restored = prepare("new-runtime")
        assert restored.identity == identity and restored.documents[583] == expected
        restored.close()
    assert (sha256_file(marker), marker.stat().st_mtime_ns) == before


def test_publish_failure_retains_local_db_and_resumes_uploaded_parts(tmp_path, frozen, setup_catalog, monkeypatch):
    prepare, _ = setup_catalog
    publish = storage.publish_file
    attempts = []
    def interrupted(source, target):
        attempts.append(Path(target).name)
        if Path(target).name == "part-00001.sqlite.gz":
            raise OSError("simulated Drive disconnect")
        return publish(source, target)
    monkeypatch.setattr(storage, "publish_file", interrupted)
    with pytest.raises(OSError, match="disconnect"):
        prepare()
    marker = next((tmp_path / "work").glob("catalog-*/catalog.json"))
    assert (marker.parent / "catalog.sqlite").is_file()
    assert not list((tmp_path / "catalogs").glob("*/catalog.json"))
    first = next((tmp_path / "catalogs").glob("*/catalog.parts/part-00000.sqlite.gz"))
    before = sha256_file(first), first.stat().st_mtime_ns
    monkeypatch.setattr(writer, "rows", lambda *a: pytest.fail("Publication retry must not rebuild SQLite"))
    monkeypatch.setattr(storage, "publish_file", publish)
    restored = prepare()
    restored.close()
    assert (sha256_file(first), first.stat().st_mtime_ns) == before


def test_missing_part_heals_from_local_and_corrupt_local_restores_from_parts(tmp_path, frozen, setup_catalog):
    prepare, _ = setup_catalog
    catalog = prepare()
    identity = catalog.identity
    catalog.close()
    marker = next((tmp_path / "catalogs").glob("*/catalog.json"))
    part = next((marker.parent / "catalog.parts").glob("*.gz"))
    original = part.read_bytes()
    part.unlink()
    catalog = prepare()
    assert catalog.identity == identity and part.read_bytes() == original
    catalog.close()
    local = next((tmp_path / "work").glob("catalog-*/catalog.sqlite"))
    local.write_bytes(b"corrupt local cache")
    catalog = prepare()
    assert catalog.identity == identity
    assert sha256_file(local) == read_json(marker)["sha256"]
    catalog.close()


def test_corrupt_manifest_does_not_bypass_integrity_gate(tmp_path, frozen, setup_catalog):
    prepare, _ = setup_catalog
    catalog = prepare()
    catalog.close()
    marker = next((tmp_path / "catalogs").glob("*/catalog.json"))
    marker.write_text(marker.read_text().replace('"state": "COMPLETE"', '"state": "FORGED"'))
    before = marker.read_bytes()
    with pytest.raises(ValueError):
        prepare()
    assert marker.read_bytes() == before


def test_local_disk_full_during_restore_does_not_trigger_expensive_rebuild(tmp_path, frozen, setup_catalog, monkeypatch):
    prepare, analyzer = setup_catalog
    legacy(tmp_path, frozen, analyzer)
    monkeypatch.setattr(writer, "rows", lambda *a: pytest.fail("Local disk error must not trigger source rebuild"))
    monkeypatch.setattr(storage, "publish_file", lambda *a: (_ for _ in ()).throw(OSError("No space left on device")))
    with pytest.raises(OSError, match="No space"):
        prepare("full-runtime")
