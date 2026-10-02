import asyncio
import hashlib
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vietmedbridge.artifacts import atomic_json, code_fingerprint, sha256_file
from vietmedbridge.build import build_corpus, verify_spans
from vietmedbridge.chunks import ChunkConfig, chunk_source
from vietmedbridge.crawl import CrawlConfig, crawl_links
from vietmedbridge.gates import authorize_scale, record_milestone
from vietmedbridge.golden import run_golden_suite, run_reviewed_golden
from vietmedbridge.health import freeze_candidate, health_report
from vietmedbridge.inventory import create_stage_a_sample
from vietmedbridge.quality import document_quality
from vietmedbridge.validation import validate_snapshot


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


SPEC = {"model_id": "fixture-char", "revision": "v1"}
FIXTURES = Path(__file__).parent / "golden/cases.json"


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def make_build(tmp_path, *, size=3):
    links = write_parquet(tmp_path / "links.parquet", [
        {"id": 500 + i * 37, "url": f"https://example.org/{i}"} for i in range(size)
    ])
    body = "A clinical study evaluated H. pylori, HbA1c and a dose of 0.5 mg. " * 4
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == f"/{size - 1}":
            return httpx.Response(404)
        return httpx.Response(200, content=f"<html lang='en'><head><title>Title {request.url.path}</title></head><body><article><p>{body}</p></article></body></html>".encode(), headers={"content-type": "text/html"})
    asyncio.run(crawl_links(links, tmp_path / "crawl", config=CrawlConfig(
        shard_size=1, concurrency=1, attempts=1, per_host_delay=0,
    ), transport=httpx.MockTransport(handler), work_dir=tmp_path))
    raw = tmp_path / "crawl/smoke-v1"
    build = build_corpus(raw, tmp_path / "processed", Tokenizer(), SPEC,
                         official_links=links, work_dir=tmp_path)
    return links, raw, tmp_path / "processed/canonical-v1", build


def test_stage_a_is_deterministic_and_includes_rare_strata(tmp_path):
    rows = [{"id": i * 19 + 583, "url": f"https://d{i}.org/article"} for i in range(350)]
    rows += [{"id": 100_000, "url": "https://health.vn/paper.pdf"},
             {"id": 100_037, "url": "https://health.cn/paper.xml"}]
    corpus = write_parquet(tmp_path / "links.parquet", rows)
    queries = write_parquet(tmp_path / "query.parquet", [{"id": 1, "query": "test"}])
    atomic_json(tmp_path / "raw/snapshot.json", {"revision": "fixture", "files": {
        "links_corpus.parquet": {"path": corpus.name, "sha256": sha256_file(corpus)},
        "query.parquet": {"path": queries.name, "sha256": sha256_file(queries)},
    }})
    report = create_stage_a_sample(tmp_path, size=30, work_dir=tmp_path)
    sample = pq.read_table(tmp_path / report["files"]["stage_a_links"]["path"]).to_pylist()
    repeated = create_stage_a_sample(tmp_path, size=30, work_dir=tmp_path)
    assert repeated["files"]["stage_a_links"]["sha256"] == report["files"]["stage_a_links"]["sha256"]
    assert len(sample) == len({row["id"] for row in sample}) == 30
    assert {"pdf", "xml", "html_like"} <= {row["format_hint"] for row in sample}
    assert {"top", "medium", "long_tail"} <= {row["domain_tier"] for row in sample}


def test_golden_and_parent_changes_do_not_change_child_identity():
    result = run_golden_suite(FIXTURES, Tokenizer(), SPEC)
    assert result["passed"], result["cases"]
    source = "Methods\n\nH. pylori is measured with HbA1c and 0.5 mg.\n\nResults\n\n" + "Treatment improved outcomes. " * 30
    doc = {"doc_id": 583, "source_text": source, "source_text_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "heading_hints": ["Methods", "Results"], "title": "Clinical study"}
    children, parents = chunk_source(doc, Tokenizer(), SPEC, ChunkConfig(child_tokens=80, overlap_tokens=15, parent_tokens=200))
    alternatives, _ = chunk_source(doc, Tokenizer(), SPEC, ChunkConfig(child_tokens=80, overlap_tokens=15, parent_tokens=300))
    assert [row["chunk_id"] for row in children] == [row["chunk_id"] for row in alternatives]
    verify_spans(doc, children, parents)
    assert all(row["token_count"] <= 80 and row["dense_token_count"] <= 512 for row in children)
    children[0]["start_char"] = -1
    with pytest.raises(ValueError, match="bounds"):
        verify_spans(doc, children, parents)
    assert document_quality("HbA1c 0.5 mg.")["quality_tier"] == "LOW"


def test_global_dedup_keeps_ids_and_separates_representations(tmp_path):
    links, raw, root, build = make_build(tmp_path)
    report = health_report(root, official_links=links, crawl_dir=raw, work_dir=tmp_path)
    assert report["dedup"]["official_document_ids"] == 2
    assert report["dedup"]["canonical_contents"] == 1
    assert report["dedup"]["aliases_preserved"]
    aliases = pq.read_table(root / report["files"]["canonical_aliases"]["path"])
    assert set(aliases["doc_id"].to_pylist()) == {500, 537}
    mappings = pq.read_table(root / report["files"]["representation_aliases"]["path"]).to_pylist()
    assert {row["retrieval_representation_hash"] for row in mappings if row["doc_id"] == 500}.isdisjoint(
        {row["retrieval_representation_hash"] for row in mappings if row["doc_id"] == 537})
    assert report["coverage"]["requested"] == 3 and report["documents"]["parsed"] == 2
    assert report["retrieval_evaluation"].startswith("NOT_RUN")
    from copy import deepcopy
    baseline = deepcopy(report)
    baseline["distributions"]["source_chars"]["mean"] /= 3
    atomic_json(tmp_path / "baseline.json", baseline)
    compared = health_report(root, official_links=links, work_dir=tmp_path,
                             baseline_report=tmp_path / "baseline.json", drift_relative_threshold=0.5)
    assert compared["distribution_comparison"]["status"] == "WARNING"
    golden = run_golden_suite(FIXTURES, Tokenizer(), SPEC)
    frozen = freeze_candidate(root, report, golden_report=golden)
    assert frozen["state"] == "FROZEN_CANDIDATE"
    altered_golden = deepcopy(golden)
    altered_golden["chunking"]["child_tokens"] = 120
    with pytest.raises(ValueError, match="chunking policies"):
        freeze_candidate(root, report, golden_report=altered_golden)
    # Corruption after health generation is caught again at freeze time.
    docs_path = root / build["parts"][0]["files"]["documents"]["path"]
    with docs_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="changed artifact"):
        freeze_candidate(root, report, golden_report=golden)


def test_official_membership_and_hard_failure_prevent_commit(tmp_path, monkeypatch):
    links, raw, root, build = make_build(tmp_path)
    other = write_parquet(tmp_path / "other.parquet", [{"id": 500, "url": "https://example.org/different"}])
    validation = validate_snapshot(root, build, official_links=other, work_dir=tmp_path)
    assert not validation["passed"] and validation["checks"]["invalid_official_id_or_url"] == 3
    import vietmedbridge.build as builder
    original = builder.chunk_source
    def wrong_source(*args, **kwargs):
        children, parents = original(*args, **kwargs)
        children[0]["text"] = "invented text"
        return children, parents
    monkeypatch.setattr(builder, "chunk_source", wrong_source)
    with pytest.raises(ValueError, match="source offsets"):
        build_corpus(raw, tmp_path / "processed", Tokenizer(), SPEC,
                     run_name="bad-build", official_links=links, work_dir=tmp_path)
    bad = tmp_path / "processed/bad-build"
    assert not list(bad.rglob("*.done.json"))
    assert list(bad.rglob("*.failed.json"))
    monkeypatch.setattr(builder, "chunk_source", original)
    repaired = build_corpus(raw, tmp_path / "processed", Tokenizer(), SPEC,
                            run_name="bad-build", official_links=links, work_dir=tmp_path)
    assert repaired["integrity"]["passed"]
    def programmer_error(*args, **kwargs):
        raise RuntimeError("Unexpected parser implementation bug")
    monkeypatch.setattr(builder, "extract_source", programmer_error)
    with pytest.raises(RuntimeError, match="implementation bug"):
        build_corpus(raw, tmp_path / "processed", Tokenizer(), SPEC,
                     run_name="parser-bug", official_links=links, work_dir=tmp_path)
    assert not list((tmp_path / "processed/parser-bug").rglob("*.done.json"))


def test_scale_gate_does_not_invent_approval_or_qrels(tmp_path):
    golden = run_golden_suite(FIXTURES, Tokenizer(), SPEC)
    assert authorize_scale(tmp_path, rows=1000, corpus_sha256="fixture", golden_report=golden) == "stage_a"
    with pytest.raises(ValueError, match="milestone review"):
        authorize_scale(tmp_path, rows=10000, corpus_sha256="fixture", golden_report=golden)
    atomic_json(tmp_path / "gates/stage_a.json", {"approved": True, "corpus_sha256": "fixture", "code_sha256": code_fingerprint(), "chunking": golden["chunking"], "reviewed_golden_passed": True})
    assert authorize_scale(tmp_path, rows=10000, corpus_sha256="fixture", golden_report=golden) == "stage_b1"
    atomic_json(tmp_path / "gates/stage_b1.json", {"approved": True, "corpus_sha256": "fixture", "retrieval_passed": False, "code_sha256": code_fingerprint(), "chunking": golden["chunking"], "reviewed_golden_passed": True})
    with pytest.raises(ValueError, match="retrieval regression"):
        authorize_scale(tmp_path, rows=100000, corpus_sha256="fixture", golden_report=golden)


def test_partial_range_cannot_approve_milestone_and_resume_completes(tmp_path):
    links = write_parquet(tmp_path / "links.parquet", [
        {"id": i * 37 + 583, "url": f"https://example.org/{i}"} for i in range(3)
    ])
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text="Clinical treatment HbA1c 0.5 mg. " * 10,
                              headers={"content-type": "text/plain"})
    options = {"config": CrawlConfig(shard_size=1, concurrency=1, attempts=1, per_host_delay=0),
               "transport": httpx.MockTransport(handler), "work_dir": tmp_path,
               "origin_corpus_sha256": sha256_file(links)}
    first = asyncio.run(crawl_links(links, tmp_path / "crawl", max_shards=1, **options))
    assert not first["range_complete"] and first["requested_input_records"] == 3
    raw, root = tmp_path / "crawl/smoke-v1", tmp_path / "processed/canonical-v1"
    build_corpus(raw, tmp_path / "processed", Tokenizer(), SPEC, official_links=links, work_dir=tmp_path)
    golden = run_golden_suite(FIXTURES, Tokenizer(), SPEC)
    health = health_report(root, official_links=links, work_dir=tmp_path)
    candidate = freeze_candidate(root, health, golden_report=golden)
    with pytest.raises(ValueError, match="range is incomplete"):
        record_milestone(tmp_path, candidate, stage="stage_a", corpus_sha256=sha256_file(links), reviewer="Fixture reviewer", approved=True)
    resumed = asyncio.run(crawl_links(links, tmp_path / "crawl", **options))
    assert resumed["range_complete"] and resumed["call"]["skipped_shards"] == 1
    build_corpus(raw, tmp_path / "processed", Tokenizer(), SPEC, official_links=links, work_dir=tmp_path)
    health = health_report(root, official_links=links, work_dir=tmp_path)
    candidate = freeze_candidate(root, health, golden_report=golden)
    with pytest.raises(ValueError, match="reviewed real-source"):
        record_milestone(tmp_path, candidate, stage="stage_a", corpus_sha256=sha256_file(links), reviewer="Fixture reviewer", approved=True)
    record = record_milestone(tmp_path, candidate, stage="stage_a", corpus_sha256=sha256_file(links), reviewer="Fixture reviewer", approved=False)
    assert not record["approved"] and not record["retrieval_passed"]


def test_real_golden_requires_human_assertions_and_replays_raw(tmp_path):
    links, raw, root, build = make_build(tmp_path)
    report = health_report(root, official_links=links, work_dir=tmp_path)
    from vietmedbridge.artifacts import read_json
    annotations = root / report["files"]["golden_candidates"]["path"]
    with pytest.raises(ValueError, match="human-reviewed"):
        run_reviewed_golden(annotations, raw, Tokenizer(), SPEC, min_cases=2)
    cases = read_json(annotations)
    for case in cases:
        case.update(annotation_status="REVIEWED", expected_key_snippets=[""])
    atomic_json(tmp_path / "empty-snippets.json", cases)
    with pytest.raises(ValueError, match="nonempty expected"):
        run_reviewed_golden(tmp_path / "empty-snippets.json", raw, Tokenizer(), SPEC, min_cases=2)
    for case in cases:
        case.update(annotation_status="REVIEWED", expected_key_snippets=["H. pylori", "0.5 mg"])
    reviewed = tmp_path / "reviewed.json"
    atomic_json(reviewed, cases)
    assert run_reviewed_golden(reviewed, raw, Tokenizer(), SPEC, min_cases=2)["passed"]
    cases[0]["expected_key_snippets"].append("Snippet that is absent from the source")
    atomic_json(reviewed, cases)
    assert not run_reviewed_golden(reviewed, raw, Tokenizer(), SPEC, min_cases=2)["passed"]


def test_duplicate_across_shards_and_raw_byte_budget_fail_closed(tmp_path):
    from copy import deepcopy
    links, raw, root, build = make_build(tmp_path)
    duplicated = deepcopy(build)
    duplicated["parts"].append(duplicated["parts"][0])
    result = validate_snapshot(root, duplicated, official_links=links, work_dir=tmp_path)
    assert not result["passed"] and result["checks"]["duplicate_input_ids"] == 1
    large = write_parquet(tmp_path / "large.parquet", [
        {"id": 123, "url": "https://example.org/large"},
    ])
    handler = lambda request: httpx.Response(200, content=b"a" * 800, headers={"content-type": "text/plain"})
    with pytest.raises(ValueError, match="Shard byte budget"):
        asyncio.run(crawl_links(large, tmp_path / "crawl", run_name="byte-limit",
                    config=CrawlConfig(max_bytes=900, max_shard_bytes=1000, attempts=1, respect_robots=False),
                    transport=httpx.MockTransport(handler), work_dir=tmp_path))
    assert list((tmp_path / "crawl/byte-limit").rglob("*.failed.json"))
    assert not list((tmp_path / "crawl/byte-limit").rglob("*.done.json"))
