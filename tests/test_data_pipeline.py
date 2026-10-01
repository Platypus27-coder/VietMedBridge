import asyncio
import hashlib
from collections import Counter

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vietmedbridge.artifacts import atomic_json, read_json, sha256_file
from vietmedbridge.build import build_corpus, verify_spans
from vietmedbridge.chunks import ChunkConfig, chunk_source
from vietmedbridge.crawl import CrawlConfig, completed_parts, crawl_links
from vietmedbridge.dataset import audit_dataset, iter_link_shards, validate_link_subset
from vietmedbridge.text import extract_source


class CharacterTokenizer:
    """Deterministic offset test double; production always uses the BGE fast tokenizer."""
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


TOKENIZER_SPEC = {"model_id": "test-char-offsets", "revision": "test-v1"}
HTML = """<!doctype html><html lang="vi"><head><title>Bằng chứng y sinh</title></head>
<body><article><h1>Bằng chứng y sinh</h1>
<p>Người bệnh cần được theo dõi triệu chứng và đánh giá đáp ứng điều trị.
Văn bản này chứa tiếng Việt e\u0301 và 中文医学，用于验证来源位置。
Đây là nội dung nguồn để kiểm tra ID tài liệu và vị trí của từng đoạn.</p>
<p>Chỉ số kiểm tra cần phản ánh nội dung trong tài liệu. Các đoạn văn phải giữ
nguyên vị trí trong văn bản đã trích và có thể được kiểm tra lại.</p>
</article></body></html>""".encode("utf-8")


def parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_audit_noncontiguous_ids_and_duplicate_urls(tmp_path):
    query = parquet(tmp_path / "query.parquet", [{"id": 1, "query": "Câu hỏi một?"}, {"id": 2, "query": "Câu hỏi hai?"}])
    corpus = parquet(tmp_path / "links.parquet", [
        {"id": 583, "url": "https://example.org/a"},
        {"id": 1009, "url": "https://example.org/a"},
        {"id": 1000000, "url": "https://other.org/b"},
    ])
    atomic_json(tmp_path / "raw/snapshot.json", {
        "revision": "test", "files": {
            "query.parquet": {"path": query.name, "sha256": sha256_file(query)},
            "links_corpus.parquet": {"path": corpus.name, "sha256": sha256_file(corpus)},
        },
    })
    report = audit_dataset(tmp_path, work_dir=tmp_path, sample_domains=2, per_domain=2)
    assert report["corpus"]["unique_ids"] == 3
    assert report["corpus"]["min_id"] == 583
    assert report["corpus"]["duplicate_url_rows"] == 1
    assert report["query"]["has_reference_labels"] is False
    sample = tmp_path / report["sample"]["path"]
    validate_link_subset(sample, corpus, work_dir=tmp_path)
    invalid = parquet(tmp_path / "invalid.parquet", [{"id": 1, "url": "https://example.org/a"}])
    with pytest.raises(ValueError, match="official"):
        validate_link_subset(invalid, corpus, work_dir=tmp_path)
    shards = list(iter_link_shards(corpus, start=1, stop=3, shard_size=1))
    assert [part[1][0]["id"] for part in shards] == [1009, 1000000]


def test_unicode_offsets_and_parent_containment():
    source = "Tiếng Việt e\u0301 không được sửa trong output. 中文医学。 " * 20
    doc = {"doc_id": 583, "source_text": source,
           "source_text_sha256": hashlib.sha256(source.encode()).hexdigest(), "language": "vi"}
    config = ChunkConfig(child_tokens=35, overlap_tokens=8, parent_tokens=90)
    children, parents = chunk_source(doc, CharacterTokenizer(), TOKENIZER_SPEC, config)
    verify_spans(doc, children, parents)
    assert children and parents
    assert all(child["token_end"] - child["token_start"] <= 35 for child in children)
    assert chunk_source(doc, CharacterTokenizer(), TOKENIZER_SPEC, config) == (children, parents)
    children[0]["text"] = "invented replacement"
    with pytest.raises(ValueError, match="source offsets"):
        verify_spans(doc, children, parents)


def test_resume_retry_preserves_success_and_official_ids(tmp_path):
    links = parquet(tmp_path / "links.parquet", [
        {"id": 583, "url": "https://example.org/a"},
        {"id": 584, "url": "https://example.org/a"},
        {"id": 1000, "url": "https://example.org/fail"},
    ])
    calls = Counter()
    state = {"fail": True}
    def handler(request):
        path = request.url.path
        calls[path] += 1
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if path == "/fail" and state["fail"]:
            return httpx.Response(503)
        return httpx.Response(200, content=HTML, headers={"content-type": "text/html; charset=utf-8"})
    transport = httpx.MockTransport(handler)
    config = CrawlConfig(concurrency=2, shard_size=3, per_host_delay=0, attempts=1)
    first = asyncio.run(crawl_links(links, tmp_path / "crawl", config=config, transport=transport, work_dir=tmp_path))
    assert first["statuses"]["ok"] == 2
    assert first["call"]["ok_this_call"] == 2
    second = asyncio.run(crawl_links(links, tmp_path / "crawl", config=config, transport=transport, work_dir=tmp_path))
    assert second["call"]["skipped_shards"] == 1
    assert calls["/a"] == 2
    state["fail"] = False
    retried = asyncio.run(crawl_links(
        links, tmp_path / "crawl", config=config, retry_failed=True, transport=transport, work_dir=tmp_path,
    ))
    assert retried["statuses"] == {"ok": 3}
    assert calls["/a"] == 2  # successful sources are not fetched again
    run = tmp_path / "crawl/smoke-v1"
    assert len(list(run.glob("*.raw.jsonl.gz"))) == 2  # old attempt remains intact
    built = build_corpus(
        run, tmp_path / "processed", CharacterTokenizer(), TOKENIZER_SPEC,
        config=ChunkConfig(child_tokens=40, overlap_tokens=10, parent_tokens=90), work_dir=tmp_path,
    )
    assert built["counts"]["documents"] == 3
    document_file = tmp_path / "processed/canonical-v1" / built["parts"][0]["files"]["documents"]["path"]
    docs = pq.read_table(document_file).to_pylist()
    assert {doc["doc_id"] for doc in docs} == {583, 584, 1000}
    assert len({doc["source_text_sha256"] for doc in docs}) == 1
    with pytest.raises(ValueError, match="Input/config changed"):
        asyncio.run(crawl_links(
            links, tmp_path / "crawl", config=CrawlConfig(shard_size=1), transport=transport, work_dir=tmp_path,
        ))
    part = completed_parts(run)[0]
    with (run / part["raw_file"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="changed artifact"):
        completed_parts(run)


def test_robots_redirect_and_blocked_sources(tmp_path):
    links = parquet(tmp_path / "links.parquet", [{"id": 900, "url": "http://example.org/private"}])
    def handler(request):
        if request.url.scheme == "http" and request.url.path == "/robots.txt":
            return httpx.Response(301, headers={"location": "https://example.org/robots.txt"})
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        pytest.fail("Disallowed source should not be downloaded.")
    summary = asyncio.run(crawl_links(
        links, tmp_path / "crawl", config=CrawlConfig(per_host_delay=0, attempts=1),
        transport=httpx.MockTransport(handler), work_dir=tmp_path,
    ))
    assert summary["statuses"] == {"robots_blocked": 1}


def test_xml_source_extraction_and_challenge_quarantine():
    xml = '<article xml:lang="zh"><front><article-title>医学研究</article-title></front><body><sec><title>结果</title><p>研究观察结果。</p></sec></body></article>'.encode()
    extracted = extract_source(xml, "application/xml")
    assert "研究观察结果。" in extracted["source_text"]
    assert extracted["language"] == "zh"
    with pytest.raises(ValueError, match="challenge"):
        extract_source(b"<html><head><title>Just a moment...</title></head></html>", "text/html")
