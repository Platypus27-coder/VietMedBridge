"""Progress uses the committed SQL, not stale status or counts of payload files."""
import gzip
import hashlib
import json
import sqlite3

import pytest

from scripts.check_maximize_progress_cell import checkpoint_progress, _digest


def saved_checkpoint(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    database = tmp_path / "writer.sqlite"
    binding = {"policy": "collected-delta-raw-recovery-v1", "collection": "fixture"}
    with sqlite3.connect(database) as con:
        con.executescript("""CREATE TABLE contract(value TEXT);
            CREATE TABLE completed(id TEXT PRIMARY KEY,kind TEXT,receipt TEXT);
            CREATE TABLE aliases(id INTEGER PRIMARY KEY,url TEXT,fetch TEXT);
            CREATE TABLE todo(fetch TEXT PRIMARY KEY,state TEXT,candidate TEXT,reason TEXT);
            CREATE TABLE raw_results(sha TEXT,source_url TEXT,quality_review INTEGER,result TEXT);""")
        con.execute("INSERT INTO contract VALUES (?)", [_digest(binding)])
        con.execute("INSERT INTO completed VALUES ('seed','seed',?)", [json.dumps({"counts": {
            "REUSED_VALIDATED_CONTENT": 5, "RAW_AVAILABLE_NEEDS_EXTRACTION": 4}})])
        # Two distinct official IDs may reuse one fetch URL: preserve both.
        con.executemany("INSERT INTO aliases VALUES (?,?,?)", [(1, "a", "a"), (2, "a#x", "a"), (3, "b", "b"), (4, "c", "c")])
        con.executemany("INSERT INTO todo VALUES (?,?,?,NULL)", [
            ("a", "TEXT_READY", json.dumps({"recovery_root": "maximize"})),
            ("b", "CHUNKED", json.dumps({"recovery_root": "maximize"})),
            ("c", "NEEDS_RAW", None)])
        con.execute("INSERT INTO raw_results VALUES ('hash','a',0,'EXTRACTED')")
    raw = database.read_bytes()
    payload = gzip.compress(raw)
    sha = hashlib.sha256(payload).hexdigest()
    (root / ("sql-" + sha + ".gz")).write_bytes(payload)
    pointer = {"binding": binding, "path": "sql-" + sha + ".gz", "sha256": sha,
               "sqlite_sha256": hashlib.sha256(raw).hexdigest(), "updated_at": "fixture-time"}
    pointer["manifest_sha256"] = _digest(pointer)
    (root / "checkpoint.json").write_text(json.dumps(pointer))
    (root / "status.json").write_text("stale, intentionally not JSON")
    return root, database


def test_progress_counts_official_aliases_from_snapshot_without_touching_writer(tmp_path):
    root, database = saved_checkpoint(tmp_path)
    before = database.read_bytes()
    result = checkpoint_progress(root, tmp_path / "scratch", report=lambda _: None)
    assert result["official_ids"] == 9
    assert result["counts_by_official_id"] == {"CHUNKED": 1, "NEEDS_RAW": 1, "TEXT_READY": 2}
    assert result["new_raw_text_ids_by_state"] == {"CHUNKED": 1, "TEXT_READY": 2}
    assert result["usable_text_ids_including_pending_chunks"] == 8
    assert not result["raw_pass_complete"] and not result["all_official_content_ready"]
    assert database.read_bytes() == before
    assert list((tmp_path / "scratch").iterdir()) == []
    assert not (root / "coverage_progress.json").exists()


def test_corrupt_snapshot_rejected_before_reporting_counts(tmp_path):
    root, _ = saved_checkpoint(tmp_path)
    next(root.glob("sql-*.gz")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="snapshot checksum"):
        checkpoint_progress(root, tmp_path / "scratch", report=lambda _: None)


def test_binding_mismatch_rejected(tmp_path):
    root, _ = saved_checkpoint(tmp_path)
    pointer = json.loads((root / "checkpoint.json").read_text())
    pointer["binding"]["collection"] = "another-input"
    pointer["manifest_sha256"] = _digest({k: v for k, v in pointer.items() if k != "manifest_sha256"})
    (root / "checkpoint.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="another input"):
        checkpoint_progress(root, tmp_path / "scratch", report=lambda _: None)


def test_official_id_accounting_does_not_hide_missing_membership(tmp_path):
    root, database = saved_checkpoint(tmp_path)
    with sqlite3.connect(database) as con:
        con.execute("DELETE FROM aliases WHERE id=4")
    pointer = json.loads((root / "checkpoint.json").read_text())
    raw = database.read_bytes()
    payload = gzip.compress(raw)
    sha = hashlib.sha256(payload).hexdigest()
    (root / ("sql-" + sha + ".gz")).write_bytes(payload)
    pointer.update(path="sql-" + sha + ".gz", sha256=sha, sqlite_sha256=hashlib.sha256(raw).hexdigest())
    pointer["manifest_sha256"] = _digest({k: v for k, v in pointer.items() if k != "manifest_sha256"})
    (root / "checkpoint.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="accounting mismatch"):
        checkpoint_progress(root, tmp_path / "scratch", report=lambda _: None)


def test_notebook_embeds_exact_cell_and_is_cpu_audit_only():
    import ast
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / "notebooks/02_colab_check_maximize_progress.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert cells == [(repo / "scripts/check_maximize_progress_cell.py").read_text(encoding="utf-8")]
    ast.parse(cells[0])
