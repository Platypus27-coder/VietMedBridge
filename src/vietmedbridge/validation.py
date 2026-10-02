"""Bounded-memory whole-snapshot validation; never glob data artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .artifacts import digest_json, read_json, verify_file
from .dataset import _connection

SCHEMA_VERSION = "data-v2"


def artifact_paths(build_dir: str | Path, build: dict, kind: str) -> list[str]:
    root = Path(build_dir).resolve()
    result = []
    for part in build["parts"]:
        entry = part["files"][kind]
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact path escapes its build directory.")
        verify_file(path, entry["sha256"])
        result.append(str(path))
    return result


def open_views(connection, build_dir, build):
    for kind in ("documents", "children", "parents", "sections", "failures", "ledger"):
        paths = artifact_paths(build_dir, build, kind)
        if not paths:
            raise ValueError("No completed shards: a corpus snapshot cannot be validated/frozen.")
        connection.read_parquet(paths).create_view(kind)


def validate_snapshot(build_dir: str | Path, build: dict, *, official_links: str | Path | None = None,
                      work_dir: str | Path | None = None) -> dict:
    if build.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected data-v2 schema. Rebuild old data in a new run.")
    config = read_json(Path(build_dir) / "config.json")
    if config["signature"] != build["signature"] or digest_json({key: value for key, value in config.items() if key != "signature"}) != build["signature"]:
        raise ValueError("Build configuration identity changed.")
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vmb-validate-", dir=work_dir) as temporary:
        with _connection(temporary) as con:
            open_views(con, build_dir, build)
            con.execute("CREATE VIEW outcomes AS SELECT doc_id, url FROM documents UNION ALL SELECT doc_id, url FROM failures")
            checks = {
                "duplicate_document_ids": "SELECT count(*) - count(DISTINCT doc_id) FROM documents",
                "duplicate_outcome_ids": "SELECT count(*) - count(DISTINCT doc_id) FROM outcomes",
                "duplicate_input_ids": "SELECT count(*) - count(DISTINCT doc_id) FROM ledger",
                "duplicate_child_ids": "SELECT count(*) - count(DISTINCT chunk_id) FROM children",
                "duplicate_parent_ids": "SELECT count(*) - count(DISTINCT chunk_id) FROM parents",
                "duplicate_section_ids": "SELECT count(*) - count(DISTINCT section_id) FROM sections",
                "outcome_without_input": "SELECT count(*) FROM outcomes o LEFT JOIN ledger l USING (doc_id) WHERE l.doc_id IS NULL OR o.url IS DISTINCT FROM l.url",
                "input_without_outcome": "SELECT count(*) FROM ledger l LEFT JOIN outcomes o USING (doc_id) WHERE o.doc_id IS NULL",
                "source_hash_mismatch": "SELECT count(*) FROM documents WHERE source_text IS NULL OR length(source_text)=0 OR source_text_sha256 IS NULL OR source_text_sha256 IS DISTINCT FROM sha256(source_text)",
                "orphan_sections": "SELECT count(*) FROM sections s LEFT JOIN documents d USING (doc_id) WHERE d.doc_id IS NULL OR s.section_id IS NULL OR s.start_char IS NULL OR s.end_char IS NULL OR s.source_text_sha256 IS DISTINCT FROM d.source_text_sha256 OR s.start_char < 0 OR s.end_char <= s.start_char OR s.end_char > length(d.source_text)",
            }
            for kind in ("children", "parents"):
                checks[f"{kind}_source_span_failure"] = f"""SELECT count(*) FROM {kind} c
                    LEFT JOIN documents d USING (doc_id)
                    WHERE d.doc_id IS NULL OR c.source_text_sha256 IS DISTINCT FROM d.source_text_sha256
                    OR c.chunk_id IS NULL OR c.token_count IS NULL OR c.token_count <= 0
                    OR c.start_char IS NULL OR c.end_char IS NULL OR c.text IS NULL
                    OR c.start_char < 0 OR c.end_char <= c.start_char OR c.end_char > length(d.source_text)
                    OR c.text IS DISTINCT FROM substr(d.source_text, c.start_char + 1, c.end_char - c.start_char)"""
            checks["child_parent_failure"] = """SELECT count(*) FROM children c LEFT JOIN parents p
                ON c.parent_id = p.chunk_id WHERE p.chunk_id IS NULL OR p.doc_id <> c.doc_id
                OR p.source_text_sha256 IS DISTINCT FROM c.source_text_sha256
                OR p.start_char > c.start_char OR p.end_char < c.end_char"""
            checks["child_section_failure"] = """SELECT count(*) FROM children c LEFT JOIN sections s
                ON c.section_id = s.section_id WHERE s.section_id IS NULL OR s.doc_id <> c.doc_id
                OR s.source_text_sha256 IS DISTINCT FROM c.source_text_sha256"""
            checks["parent_section_failure"] = """SELECT count(*) FROM parents p LEFT JOIN sections s
                ON p.section_id = s.section_id WHERE p.section_id IS NOT NULL AND (s.section_id IS NULL
                OR s.doc_id <> p.doc_id OR s.source_text_sha256 IS DISTINCT FROM p.source_text_sha256
                OR p.start_char < s.start_char OR p.end_char > s.end_char)"""
            checks["missing_document_children"] = "SELECT count(*) FROM documents d WHERE NOT EXISTS(SELECT 1 FROM children c WHERE c.doc_id=d.doc_id)"
            chunk_config = config["chunks"]
            if chunk_config["prefer_sections"]:
                checks["child_outside_section"] = """SELECT count(*) FROM children c JOIN sections s
                    ON c.section_id=s.section_id WHERE c.start_char<s.start_char OR c.end_char>s.end_char"""
            checks["token_budget_failures"] = f"""SELECT
                (SELECT count(*) FROM children WHERE token_count>{int(chunk_config['child_tokens'])}
                    OR dense_token_count IS NULL OR dense_token_count<=0 OR dense_token_count>{int(chunk_config['dense_max_tokens'])}
                    OR retrieval_representation_hash IS NULL)
                + (SELECT count(*) FROM parents WHERE token_count>{int(chunk_config['parent_tokens'])})"""
            if official_links is not None:
                con.read_parquet(str(official_links)).create_view("official")
                checks["invalid_official_id_or_url"] = """SELECT count(*) FROM ledger l
                    LEFT JOIN official o ON l.doc_id = o.id WHERE o.id IS NULL OR l.url IS DISTINCT FROM o.url"""
            values = {name: int(con.execute(sql).fetchone()[0]) for name, sql in checks.items()}
            expected = int(con.execute("SELECT count(*) FROM ledger").fetchone()[0])
            values["manifest_input_count_mismatch"] = abs(expected - build["counts"].get("input_records", 0))
            for kind in ("documents", "children", "parents", "sections"):
                values[f"manifest_{kind}_count_mismatch"] = abs(int(con.execute(f"SELECT count(*) FROM {kind}").fetchone()[0]) - build["counts"].get(kind, 0))
            result = {
                "schema_version": SCHEMA_VERSION, "checked_input_records": expected,
                "official_membership": "VERIFIED" if official_links is not None else "NOT_SUPPLIED",
                "checks": values, "passed": not any(values.values()),
            }
    return result
