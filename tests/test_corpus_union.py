"""Union memory regression, conflict checks and resume across execution fixes."""
from dataclasses import asdict
import hashlib

import duckdb
import pytest

from test_external_import import CONFIG, SPEC, external
from test_scale_baseline import Tokenizer, frozen
from vietmedbridge import corpus_union as union
from vietmedbridge.artifacts import code_fingerprint, read_json, sha256_file


def span_views(con, rows=16, text_chars=64, change=None):
    con.execute(f"""CREATE VIEW all_documents AS
        SELECT i AS doc_id,origin FROM range({rows // 8}) t(i) CROSS JOIN range(2) o(origin)""")
    con.execute(f"""CREATE VIEW spans AS SELECT i//8 AS doc_id,origin,
        i::VARCHAR AS chunk_id,i::VARCHAR AS section_id,
        i*{text_chars} AS start_char,(i+1)*{text_chars} AS end_char,
        repeat('x',{text_chars})||i::VARCHAR AS text,
        'policy' AS policy_sha256,'parent-'||i::VARCHAR AS parent_id,
        'repr-'||i::VARCHAR AS retrieval_representation_hash,
        'heading-'||i::VARCHAR AS heading_text
        FROM range({rows}) t(i) CROSS JOIN range(2) o(origin)""")
    for kind in ("children", "parents", "sections"):
        expression = "*"
        if change is not None and kind == change[0]:
            column, value = change[1:]
            expression = f"* REPLACE(CASE WHEN origin=1 AND chunk_id='0' THEN {value} ELSE {column} END AS {column})"
        con.execute(f"CREATE VIEW all_{kind} AS SELECT {expression} FROM spans")


def test_duplicate_span_comparison_stays_within_128mb(tmp_path):
    # More than 300 MB of text: the former ordered aggregate fails at 128 MB.
    # The replacement sorts row hashes and streams their per-document digests.
    with union._union_connection(tmp_path, memory_limit="128MB") as con:
        span_views(con, rows=20000, text_chars=8192)
        with pytest.raises(duckdb.OutOfMemoryException):
            con.execute("""SELECT origin,doc_id,
                md5(string_agg(to_json(row(chunk_id,start_char,end_char,text,policy_sha256,
                    parent_id,retrieval_representation_hash)),'' ORDER BY chunk_id))
                FROM all_children GROUP BY origin,doc_id""").fetchall()
        union._verify_shared_spans(con)


def test_seven_disjoint_sources_skip_chunk_fingerprints(tmp_path, monkeypatch):
    with union._union_connection(tmp_path) as con:
        con.execute("""CREATE VIEW all_documents AS
            SELECT i AS doc_id,i%7 AS origin FROM range(700244) t(i)""")
        # There are deliberately no chunk views: disjoint IDs cannot conflict
        # across sources, so scanning their text would be unnecessary.
        monkeypatch.setattr(union, "_span_fingerprints",
            lambda *a: pytest.fail("Disjoint archives must not scan chunk text"))
        union._verify_shared_spans(con)


def test_staging_large_text_tables_uses_local_disk(tmp_path):
    with union._union_connection(tmp_path, memory_limit="128MB") as con:
        con.execute("""CREATE TABLE selected_documents AS SELECT i AS doc_id,
            repeat(md5(i::VARCHAR),256) AS source_text FROM range(40000) t(i)""")
        assert con.execute("SELECT count(*),sum(length(source_text)) FROM selected_documents").fetchone() == (40000,327680000)
        assert con.execute("SELECT source_text FROM selected_documents WHERE doc_id=39999").fetchone()[0] == hashlib.md5(b"39999").hexdigest()*256
    assert (tmp_path / "union.duckdb").stat().st_size > 1024 * 1024


@pytest.mark.parametrize("change", [
    ("children", "text", "'changed'"),
    ("children", "start_char", "1"),
    ("children", "end_char", "999"),
    ("children", "policy_sha256", "'different'"),
    ("children", "parent_id", "'different'"),
    ("children", "retrieval_representation_hash", "'different'"),
    ("parents", "text", "'changed'"),
    ("sections", "heading_text", "'different'"),
])
def test_shared_span_conflicts_are_not_hidden_by_memory_fix(tmp_path, change):
    with union._union_connection(tmp_path) as con:
        span_views(con, change=change)
        with pytest.raises(ValueError, match="Conflicting successful chunk/section"):
            union._verify_shared_spans(con)


@pytest.mark.parametrize("condition", ["origin=0", "chunk_id<>'0' OR origin=0", "false"])
def test_missing_shared_spans_are_rejected(tmp_path, condition):
    with union._union_connection(tmp_path) as con:
        span_views(con)
        con.execute(f"CREATE OR REPLACE VIEW all_children AS SELECT * FROM spans WHERE {condition}")
        with pytest.raises(ValueError, match="Conflicting successful chunk/section"):
            union._verify_shared_spans(con)


def test_union_resumes_old_producer_parts_after_execution_fix(tmp_path, frozen, external, monkeypatch):
    build, candidate, _ = frozen
    sources = [(build.name, f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json")]
    golden = {"passed":True,"code_sha256":code_fingerprint(),"tokenizer":SPEC,"chunking":asdict(CONFIG)}
    original_publish = union.publish_file
    def interrupt_second_part(local, target):
        if target.name.startswith("part-000000000002"):
            raise RuntimeError("simulated interruption")
        return original_publish(local, target)
    with monkeypatch.context() as patch:
        patch.setattr(union, "code_fingerprint", lambda: "previous-producer-code")
        patch.setattr(union, "publish_file", interrupt_second_part)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            union.compose_candidates(tmp_path,sources,Tokenizer(),run_name="resume-union",
                golden_report=golden | {"code_sha256":"previous-producer-code"},
                official_links=external[1],work_dir=tmp_path / "work",part_size=2)
    output = tmp_path / "processed/resume-union"
    checkpoint = output / "parts/part-000000000000.done.json"
    before = (sha256_file(checkpoint), checkpoint.stat().st_mtime_ns)
    active_before = read_json(tmp_path / "active_data_candidate.json")
    def preserve_completed(local, target):
        assert not target.name.startswith("part-000000000000")
        return original_publish(local, target)
    monkeypatch.setattr(union, "publish_file", preserve_completed)
    result = union.compose_candidates(tmp_path,sources,Tokenizer(),run_name="resume-union",
        golden_report=golden,official_links=external[1],work_dir=tmp_path / "work",part_size=2)
    assert (sha256_file(checkpoint), checkpoint.stat().st_mtime_ns) == before
    assert result["documents"] == 2 and result["children"] == candidate["counts"]["children"]
    assert read_json(output / "build.json")["counts"]["input_records"] == 4
    assert read_json(output / "config.json")["code_sha256"] == "previous-producer-code"
    execution = read_json(output / "union_execution.json")
    assert execution["execution_code_sha256"] == golden["code_sha256"]
    assert active_before["build_run"] == build.name
    assert read_json(tmp_path / "active_data_candidate.json")["build_run"] == "resume-union"
