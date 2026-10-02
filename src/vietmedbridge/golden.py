"""Offline synthetic source fixtures. Real golden annotations remain a human task."""

from __future__ import annotations

import io
import base64
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from .artifacts import code_fingerprint, read_json, sha256_file, utc_now, verify_file
from .build import verify_spans
from .chunks import ChunkConfig, chunk_source, parent_for_child
from .text import extract_source, normalize_for_retrieval
from .crawl import completed_parts, iter_raw_records


def run_reviewed_golden(annotations: str | Path, crawl_dir: str | Path, tokenizer,
                        tokenizer_spec: dict, *, min_cases: int = 100,
                        config: ChunkConfig | None = None) -> dict:
    """Replay pinned raw sources against assertions entered by a human reviewer."""
    path, root = Path(annotations), Path(crawl_dir)
    config = config or ChunkConfig()
    config.validate()
    if not 1 <= min_cases <= 500:
        raise ValueError("Invalid minimum golden size.")
    cases = read_json(path)
    if not min_cases <= len(cases) <= 500 or len({row["doc_id"] for row in cases}) != len(cases):
        raise ValueError(f"Reviewed golden needs {min_cases}–500 unique source IDs.")
    for case in cases:
        snippets = case.get("expected_key_snippets")
        if (case.get("annotation_status") != "REVIEWED" or not isinstance(snippets, list)
                or not snippets or not all(isinstance(value, str) and value.strip() for value in snippets)):
            raise ValueError("Each golden source needs human-reviewed, nonempty expected_key_snippets.")
    wanted = {row["doc_id"]: row for row in cases}
    raw_files = {row["raw_file"] for row in cases}
    parts = {part["raw_file"]: part for part in completed_parts(root, verify=False)}
    if not raw_files <= parts.keys():
        raise ValueError("Golden raw files are not selected completed attempts in this crawl run.")
    outcomes, seen = [], set()
    for name in sorted(raw_files):
        verify_file(root / name, parts[name]["raw_sha256"])
        for raw in iter_raw_records(root / name):
            case = wanted.get(raw["doc_id"])
            if case is None or case["raw_file"] != name:
                continue
            seen.add(raw["doc_id"])
            try:
                if raw["status"] != "ok" or raw["url"] != case["url"]:
                    raise AssertionError("Reviewed source ID/URL is no longer a successful raw record.")
                body = base64.b64decode(raw["body_base64"], validate=True)
                if hashlib.sha256(body).hexdigest() != raw["body_sha256"]:
                    raise AssertionError("Raw body checksum mismatch.")
                doc = extract_source(body, raw.get("content_type", ""))
                doc["doc_id"] = raw["doc_id"]
                if doc["source_text_sha256"] != case["source_text_sha256"]:
                    raise AssertionError("Reviewed extracted source changed; inspect before updating assertions.")
                for snippet in case["expected_key_snippets"]:
                    if snippet not in doc["source_text"]:
                        raise AssertionError(f"Expected key snippet was lost: {snippet}")
                if case.get("expected_title_contains") and case["expected_title_contains"] not in doc["title"]:
                    raise AssertionError("Expected title text was lost.")
                if case.get("expected_language") and case["expected_language"] != doc["language"]:
                    raise AssertionError("Expected language changed.")
                if len(doc["source_text"]) < case.get("expected_min_text_length", 1):
                    raise AssertionError("Source is shorter than the reviewed minimum.")
                children, parents = chunk_source(doc, tokenizer, tokenizer_spec, config)
                verify_spans(doc, children, parents)
                if case.get("expected_chunk_count_range"):
                    lower, upper = case["expected_chunk_count_range"]
                    if not lower <= len(children) <= upper:
                        raise AssertionError("Chunk count changed outside reviewed bounds.")
                outcomes.append({"doc_id": raw["doc_id"], "passed": True, "children": len(children)})
            except Exception as exc:
                outcomes.append({"doc_id": raw["doc_id"], "passed": False, "error": f"{type(exc).__name__}:{exc}"})
    if seen != wanted.keys():
        raise ValueError("Reviewed golden contains IDs absent from the selected raw attempts.")
    return {"passed": all(row["passed"] for row in outcomes), "cases": outcomes,
            "reviewed_cases": len(cases), "annotation_sha256": sha256_file(path),
            "crawl_signature": read_json(root / "run.json")["signature"],
            "tokenizer": tokenizer_spec, "code_sha256": code_fingerprint(),
            "chunking": asdict(config),
            "created_at": utc_now(), "scope": "reviewed_real_source_regression_not_relevance_labels"}


def _pdf(text: str | None) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    if text is not None:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 40 740 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def run_golden_suite(fixtures: str | Path, tokenizer, tokenizer_spec: dict,
                     config: ChunkConfig | None = None) -> dict:
    path = Path(fixtures)
    config = config or ChunkConfig()
    config.validate()
    cases = read_json(path)
    outcomes = []
    for case in cases:
        try:
            body = _pdf(case.get("body")) if case.get("format") == "pdf" else case["body"].encode("utf-8")
            try:
                doc = extract_source(body, case["content_type"])
            except Exception as exc:
                if case.get("expected_error") and case["expected_error"] in str(exc):
                    outcomes.append({"case": case["name"], "passed": True, "expected_failure": str(exc)})
                    continue
                raise
            if case.get("expected_error"):
                raise AssertionError("Expected extraction failure was not detected.")
            doc["doc_id"] = case["fixture_id"]
            for snippet in case.get("snippets", []):
                if snippet not in doc["source_text"]:
                    raise AssertionError(f"Source lost expected snippet: {snippet}")
                if normalize_for_retrieval(snippet) not in normalize_for_retrieval(doc["source_text"]):
                    raise AssertionError("Retrieval normalization lost a source snippet.")
            if case.get("language") and doc["language"] != case["language"]:
                raise AssertionError("Language declaration was lost.")
            children, parents = chunk_source(doc, tokenizer, tokenizer_spec, config)
            verify_spans(doc, children, parents)
            alternate_budget = max(640, config.parent_tokens)
            alternatives, _ = chunk_source(doc, tokenizer, tokenizer_spec, replace(config, parent_tokens=alternate_budget))
            if [child["chunk_id"] for child in alternatives] != [child["chunk_id"] for child in children]:
                raise AssertionError("Parent-only configuration change altered child identity.")
            for child in children:
                parent = parent_for_child(doc, child, tokenizer, tokenizer_spec, parent_tokens=alternate_budget)
                if not parent["start_char"] <= child["start_char"] < child["end_char"] <= parent["end_char"]:
                    raise AssertionError("Dynamic parent does not contain child.")
            outcomes.append({"case": case["name"], "passed": True,
                             "children": len(children), "source_chars": len(doc["source_text"])})
        except Exception as exc:
            outcomes.append({"case": case["name"], "passed": False,
                             "error": f"{type(exc).__name__}:{str(exc)}"})
    return {"passed": bool(outcomes) and all(row["passed"] for row in outcomes),
            "cases": outcomes, "fixture_sha256": sha256_file(path), "tokenizer": tokenizer_spec,
            "code_sha256": code_fingerprint(), "created_at": utc_now(),
            "chunking": asdict(config),
            "scope": "synthetic offline regression; not real-document gold or retrieval relevance labels"}
