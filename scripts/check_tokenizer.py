"""Validate real BGE tokenizer without downloading model weights."""

import argparse
import asyncio
import hashlib
import html
import json
import tempfile
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from vietmedbridge.artifacts import atomic_json, read_json, sha256_file
from vietmedbridge.build import build_corpus, verify_spans
from vietmedbridge.chunks import DEFAULT_BGE_REVISION, ChunkConfig, chunk_source, load_bge_tokenizer, parent_for_child
from vietmedbridge.crawl import CrawlConfig, crawl_links
from vietmedbridge.golden import run_golden_suite
from vietmedbridge.health import freeze_candidate, health_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-tokenizer", type=Path, help="Cached pinned tokenizer snapshot; offline only")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.local_tokenizer:
        if args.local_tokenizer.name != DEFAULT_BGE_REVISION:
            raise ValueError("Local snapshot directory must match the pinned BGE revision.")
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer, use_fast=True, local_files_only=True)
        spec = {"model_id": "BAAI/bge-m3", "revision": DEFAULT_BGE_REVISION, "special_tokens": False}
    else:
        tokenizer, spec = load_bge_tokenizer()
    root = Path(__file__).resolve().parents[1]
    report = run_golden_suite(root / "tests/golden/cases.json", tokenizer, spec)
    assert report["passed"], report["cases"]
    texts = [
        "Phương pháp\n\n" + ("H. pylori và HbA1c được đo ở bệnh nhân; liều 0.5 mg, IL-6, β-blocker. " * 80)
        + "\n\nKết quả\n\n" + ("Điều trị cải thiện kết quả lâm sàng, hiệu quả cần đánh giá. " * 100),
        "Methods\n\n" + ("H. pylori, HbA1c 6.5%, eGFR and IL-6 were measured at 0.5 mg. " * 70)
        + "\n\nResults\n\n" + ("The clinical study reported outcomes in treated patients. " * 100),
        "方法\n\n" + ("研究测量患者的血糖和血压；剂量为0.5 mg，保留β、IL-6和HbA1c。" * 100)
        + "\n\n结果\n\n" + ("临床结果需要进一步研究。" * 100),
        ("a\u0301 e\u0302 o\u031b\u0301 👩‍⚕️ µg α β IL-6 <0.5 mg 中文 " * 160),
    ]
    totals = {"children": 0, "parents": 0, "long_documents": len(texts)}
    for i, source in enumerate(texts):
        doc = {"doc_id": 583 + i * 37, "source_text": source,
               "source_text_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "heading_hints": ["Phương pháp", "Kết quả", "Methods", "Results", "方法", "结果"],
               "title": "Study — biomedical symbols β, µg, HbA1c"}
        baseline = None
        for parent_size in (256, 512, 640):
            children, parents = chunk_source(doc, tokenizer, spec, ChunkConfig(parent_tokens=parent_size))
            verify_spans(doc, children, parents)
            ids = [row["chunk_id"] for row in children]
            if baseline is not None:
                assert ids == baseline, "Parent-only change altered children"
            baseline = ids
            assert all(row["token_count"] <= 180 and row["dense_token_count"] <= 512 for row in children)
            assert all(row["token_count"] <= parent_size for row in parents)
        for child in children:
            parent = parent_for_child(doc, child, tokenizer, spec, parent_tokens=256)
            assert parent["start_char"] <= child["start_char"] < child["end_char"] <= parent["end_char"]
        totals["children"] += len(children)
        totals["parents"] += len(parents)
    report["long_source_checks"] = totals
    artifact_root = root / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bge-mock-", dir=artifact_root) as directory:
        work = Path(directory)
        links = work / "official.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"id": doc_id, "url": f"https://example.org/{name}"}
            for doc_id, name in ((583, "a"), (1374, "b"), (9177, "xml"), (9704, "missing"))
        ]), links)
        article = ("<html lang='vi'><head><meta charset='utf-8'><title>Clinical study</title></head>"
                   "<body><article><p>" + html.escape(texts[0]).replace("\n\n", "</p><p>")
                   + "</p></article></body></html>").encode()
        xml = read_json(root / "tests/golden/cases.json")[-1]["body"].encode()
        def handler(request):
            if request.url.path == "/missing":
                return httpx.Response(404)
            return httpx.Response(200, content=xml if request.url.path == "/xml" else article,
                                  headers={"content-type": "application/xml" if request.url.path == "/xml" else "text/html"})
        crawled = asyncio.run(crawl_links(links, work / "crawl", config=CrawlConfig(
            shard_size=2, concurrency=2, per_host_delay=0, attempts=1, respect_robots=False,
        ), work_dir=work, origin_corpus_sha256=sha256_file(links), transport=httpx.MockTransport(handler)))
        assert crawled["range_complete"]
        raw, built = work / "crawl/smoke-v1", work / "processed/canonical-v1"
        build = build_corpus(raw, work / "processed", tokenizer, spec, work_dir=work, official_links=links)
        health = health_report(built, official_links=links, crawl_dir=raw, work_dir=work)
        candidate = freeze_candidate(built, health, golden_report=report)
        assert build["integrity"]["passed"] and health["dedup"]["canonical_contents"] == 2
        assert health["dedup"]["official_document_ids"] == 3
        assert candidate["state"] == "FROZEN_CANDIDATE"
        report["mock_pipeline"] = {"passed": True, "input_records": build["counts"]["input_records"],
                                   "parsed_documents": build["counts"]["documents"],
                                   "state": candidate["state"], "counts": build["counts"],
                                   "scope": "offline HTTP mock, not a real Colab crawl"}
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps({"passed": report["passed"], "golden_cases": len(report["cases"]),
                      "tokenizer": spec, **totals, "mock_pipeline": report["mock_pipeline"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
