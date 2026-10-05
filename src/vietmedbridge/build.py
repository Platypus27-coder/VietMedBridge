"""Streaming raw-shard extraction and source-span verification."""

from __future__ import annotations

import base64
import hashlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .artifacts import (
    atomic_json, code_fingerprint, digest_json, local_workspace, publish_file, read_json,
    runtime_versions, utc_now, verify_file,
)
from .chunks import ChunkConfig, chunk_source
from .crawl import completed_parts, iter_raw_records, range_complete
from .text import EXPECTED_PARSE_ERRORS, extract_source
from .text import normalize_for_retrieval
from .quality import NORMALIZER_VERSION, document_quality, reason_code
from .structure import parse_structure, validate_structure
from .validation import SCHEMA_VERSION, validate_snapshot
from .dataset import url_host

DOCUMENT_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("url", pa.string()), ("final_url", pa.string()),
    ("title", pa.string()), ("source_text", pa.large_string()),
    ("source_text_sha256", pa.string()), ("body_sha256", pa.string()),
    ("language", pa.string()), ("language_method", pa.string()), ("parser", pa.string()),
    ("language_confidence", pa.float64()),
    ("quality_flags", pa.list_(pa.string())), ("raw_file", pa.string()),
    ("fetched_at", pa.string()),
    ("domain", pa.string()), ("content_type", pa.string()),
    ("quality_tier", pa.string()), ("quality_version", pa.string()),
    ("char_count", pa.int64()), ("paragraph_count", pa.int64()), ("section_count", pa.int64()),
    ("alphabetic_ratio", pa.float64()), ("duplicate_line_ratio", pa.float64()),
    ("biomedical_signal_count", pa.int64()), ("normalizer_version", pa.string()),
    ("content_hash", pa.string()), ("canonical_content_id", pa.string()),
    ("schema_version", pa.string()), ("heading_hints", pa.list_(pa.string())),
    ("raw_has_table", pa.bool_()),
])
SPAN_FIELDS = [
    ("doc_id", pa.int64()), ("chunk_id", pa.string()), ("chunk_order", pa.int64()),
    ("source_text_sha256", pa.string()), ("language", pa.string()), ("policy_sha256", pa.string()),
    ("start_char", pa.int64()), ("end_char", pa.int64()),
    ("token_start", pa.int64()), ("token_end", pa.int64()),
    ("text", pa.large_string()), ("retrieval_text", pa.large_string()),
    ("section_id", pa.string()), ("heading", pa.string()), ("token_count", pa.int64()),
]
PARENT_SCHEMA = pa.schema(SPAN_FIELDS)
CHILD_SCHEMA = pa.schema(SPAN_FIELDS + [
    ("parent_id", pa.string()), ("dense_token_count", pa.int64()),
    ("retrieval_representation_hash", pa.string()), ("representation_builder_version", pa.string()),
    ("quality_flags", pa.list_(pa.string())),
])
FAILURE_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("url", pa.string()), ("phase", pa.string()),
    ("reason", pa.string()), ("body_sha256", pa.string()), ("raw_file", pa.string()),
    ("reason_code", pa.string()), ("domain", pa.string()), ("content_type", pa.string()),
])
SECTION_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("source_text_sha256", pa.string()), ("section_id", pa.string()),
    ("heading_text", pa.string()), ("start_char", pa.int64()), ("end_char", pa.int64()),
    ("section_index", pa.int64()), ("parser_version", pa.string()),
])
LEDGER_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("url", pa.string()), ("domain", pa.string()),
    ("status", pa.string()), ("content_type", pa.string()), ("raw_bytes", pa.int64()),
    ("elapsed_ms", pa.int64()), ("http_status", pa.int64()),
])


def verify_spans(document: dict, children: list[dict], parents: list[dict]) -> None:
    text = document["source_text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != document["source_text_sha256"]:
        raise ValueError("Document source hash mismatch.")
    mapping = {parent["chunk_id"]: parent for parent in parents}
    for rows in (children, parents):
        if len({row["chunk_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate chunk ID.")
    for span in children + parents:
        if span["doc_id"] != document["doc_id"]:
            raise ValueError("Wrong official document ID on source span.")
        if span["source_text_sha256"] != document["source_text_sha256"]:
            raise ValueError("Span source hash mismatch.")
        if not 0 <= span["start_char"] < span["end_char"] <= len(text):
            raise ValueError("Invalid source span bounds.")
        if text[span["start_char"]:span["end_char"]] != span["text"] or not span["text"]:
            raise ValueError("Source span text does not match source offsets.")
    for child in children:
        if child["parent_id"] not in mapping:
            raise ValueError("Orphan child parent ID.")
        parent = mapping[child["parent_id"]]
        if not parent["start_char"] <= child["start_char"] < child["end_char"] <= parent["end_char"]:
            raise ValueError("Parent must contain the child source span.")


def _write(writer, rows: list[dict], schema):
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def build_corpus(
    crawl_dir: str | Path, output_root: str | Path, tokenizer, tokenizer_spec: dict,
    *, run_name: str = "canonical-v1", config: ChunkConfig | None = None,
    work_dir: str | Path | None = None, max_shards: int | None = None,
    official_links: str | Path | None = None,
) -> dict:
    config = config or ChunkConfig()
    config.validate()
    crawl_dir, output_root = Path(crawl_dir), Path(output_root)
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("Use a simple run name.")
    target = output_root / run_name
    target.mkdir(parents=True, exist_ok=True)
    crawl_run = read_json(crawl_dir / "run.json")
    if "requested_input_records" not in crawl_run:
        raise ValueError("Use a data-v2 crawl run; legacy checkpoints do not prove requested-range coverage.")
    identity = {"crawl_signature": crawl_run["signature"],
                "tokenizer": tokenizer_spec, "chunks": asdict(config),
                "extractor_version": "source-text-v2", "schema_version": SCHEMA_VERSION,
                "official_links_sha256": None, "runtime": runtime_versions(),
                "code_sha256": code_fingerprint()}
    signature = digest_json(identity)
    if official_links is not None:
        from .artifacts import sha256_file
        identity["official_links_sha256"] = sha256_file(official_links)
        signature = digest_json(identity)
    config_path = target / "config.json"
    if config_path.exists() and read_json(config_path)["signature"] != signature:
        raise ValueError("Parser/chunker config changed. Use a new canonical run name.")
    if not config_path.exists():
        atomic_json(config_path, {**identity, "signature": signature})
    selected, processed = [], 0
    for part in tqdm(completed_parts(crawl_dir), desc="Extract and chunk", unit="shard"):
        if part["run_signature"] != crawl_run["signature"]:
            raise ValueError("Raw shard belongs to a different crawl run.")
        stem = Path(part["raw_file"]).name.removesuffix(".raw.jsonl.gz")
        bucket = Path(part["raw_file"]).parent
        checkpoint = target / bucket / f"{stem}.done.json"
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if saved["input_raw_sha256"] != part["raw_sha256"] or saved["signature"] != signature:
                raise ValueError(f"Stale canonical checkpoint: {checkpoint}")
            for entry in saved["files"].values():
                verify_file(target / entry["path"], entry["sha256"])
            selected.append(saved)
            continue
        if max_shards is not None and processed >= max_shards:
            break
        counts = Counter()
        with local_workspace(work_dir) as temporary:
            names = {kind: Path(temporary) / f"{stem}.{kind}.parquet"
                     for kind in ("documents", "children", "parents", "sections", "failures", "ledger")}
            schemas = {"documents": DOCUMENT_SCHEMA, "children": CHILD_SCHEMA,
                       "parents": PARENT_SCHEMA, "sections": SECTION_SCHEMA,
                       "failures": FAILURE_SCHEMA, "ledger": LEDGER_SCHEMA}
            writers = {kind: pq.ParquetWriter(path, schemas[kind], compression="zstd")
                       for kind, path in names.items()}
            try:
                seen, input_pairs = set(), []
                for raw in iter_raw_records(crawl_dir / part["raw_file"]):
                    counts["input_records"] += 1
                    if raw["doc_id"] in seen:
                        raise ValueError("Raw shard contains a repeated official ID.")
                    seen.add(raw["doc_id"])
                    input_pairs.append({"id": raw["doc_id"], "url": raw["url"]})
                    try:
                        domain = url_host(raw["url"])
                    except ValueError:
                        domain = "unknown"
                    _write(writers["ledger"], [{
                        "doc_id": raw["doc_id"], "url": raw["url"], "domain": domain,
                        "status": raw["status"], "content_type": raw.get("content_type", "unknown"),
                        "raw_bytes": raw.get("body_bytes", 0), "elapsed_ms": raw.get("elapsed_ms"),
                        "http_status": raw.get("http_status"),
                    }], LEDGER_SCHEMA)
                    failure = {
                        "doc_id": raw["doc_id"], "url": raw["url"],
                        "body_sha256": raw.get("body_sha256"), "raw_file": part["raw_file"],
                        "domain": domain, "content_type": raw.get("content_type", "unknown"),
                    }
                    if raw["status"] != "ok":
                        _write(writers["failures"], [{**failure, "phase": "crawl",
                                                     "reason": raw.get("error", "fetch_failed"),
                                                     "reason_code": reason_code(raw.get("error", "fetch_failed"))}], FAILURE_SCHEMA)
                        counts["crawl_failed"] += 1
                        continue
                    body = base64.b64decode(raw["body_base64"], validate=True)
                    if len(body) != raw["body_bytes"] or hashlib.sha256(body).hexdigest() != raw["body_sha256"]:
                        raise ValueError("Raw source body checksum mismatch.")
                    try:
                        extracted = extract_source(body, raw.get("content_type", ""),
                                                   source_url=raw.get("final_url", raw["url"]),
                                                   requested_url=raw["url"])
                    except EXPECTED_PARSE_ERRORS as exc:
                        _write(writers["failures"], [{**failure, "phase": "parse",
                                                     "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
                                                     "reason_code": reason_code(f"{type(exc).__name__}:{str(exc)[:300]}")}], FAILURE_SCHEMA)
                        counts["parse_failed"] += 1
                        continue
                    document = {
                        **extracted, "doc_id": int(raw["doc_id"]), "url": raw["url"],
                        "final_url": raw.get("final_url", raw["url"]),
                        "body_sha256": raw["body_sha256"], "raw_file": part["raw_file"],
                        "fetched_at": raw["fetched_at"],
                        "domain": domain, "content_type": raw.get("content_type", "unknown"),
                        "normalizer_version": NORMALIZER_VERSION, "schema_version": SCHEMA_VERSION,
                    }
                    structure = parse_structure(document, extracted.get("heading_hints"))
                    validate_structure(document, structure)
                    quality = document_quality(document["source_text"], title=document["title"],
                                               section_count=structure["known_heading_count"],
                                               raw_bytes=raw.get("body_bytes"))
                    document.update(quality, section_count=structure["known_heading_count"])
                    if document.get("raw_has_table"):
                        document["quality_flags"].append("TABLE_TEXT_LINEARIZED_REQUIRES_AUDIT")
                    content_hash = hashlib.sha256(normalize_for_retrieval(document["source_text"]).encode()).hexdigest()
                    document.update(content_hash=content_hash, canonical_content_id=content_hash)
                    children, parents = chunk_source(document, tokenizer, tokenizer_spec, config, structure=structure)
                    verify_spans(document, children, parents)
                    _write(writers["documents"], [document], DOCUMENT_SCHEMA)
                    _write(writers["children"], children, CHILD_SCHEMA)
                    _write(writers["parents"], parents, PARENT_SCHEMA)
                    _write(writers["sections"], structure["sections"], SECTION_SCHEMA)
                    counts["documents"] += 1
                    counts["children"] += len(children)
                    counts["parents"] += len(parents)
                    counts["sections"] += len(structure["sections"])
                    counts[f"language_{document['language']}"] += 1
                if counts["input_records"] != part["records"]:
                    raise ValueError("Raw shard count does not match its manifest.")
                if part.get("input_pairs_sha256") and digest_json(sorted(input_pairs, key=lambda row: row["id"])) != part["input_pairs_sha256"]:
                    raise ValueError("Raw ID/URL pairs differ from the requested shard input.")
            except Exception as exc:
                atomic_json(target / bucket / f"{stem}.failed.json", {
                    "status": "FAILED", "part": part["part"], "signature": signature,
                    "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
                    "last_doc_id": raw.get("doc_id") if "raw" in locals() else None,
                    "input_raw_sha256": part["raw_sha256"], "completed_at": utc_now(),
                })
                raise
            finally:
                for writer in writers.values():
                    writer.close()
            files = {}
            for kind, path in names.items():
                relative = (bucket / path.name).as_posix()
                files[kind] = {"path": relative, "sha256": publish_file(path, target / relative)}
        saved = {
            "part": part["part"], "signature": signature,
            "input_raw_sha256": part["raw_sha256"], "files": files,
            "raw_file": part["raw_file"],
            "first_input_row": part["first_input_row"], "records": part["records"],
            "run_signature": part["run_signature"],
            "counts": dict(counts), "completed_at": utc_now(),
        }
        atomic_json(checkpoint, saved)
        selected.append(saved)
        processed += 1
    total = Counter()
    for part in selected:
        total.update(part["counts"])
    result = {
        "run_name": run_name, "signature": signature,
        "crawl_run": crawl_dir.name, "parts": selected, "counts": dict(total),
        "available_crawl_shards": len(completed_parts(crawl_dir, verify=False)),
        "selected_shards": len(selected), "created_at": utc_now(),
        "requested_input_records": crawl_run["requested_input_records"],
        "requested_range": crawl_run["requested_range"],
        "selected_range_complete": range_complete(crawl_run, selected),
        "origin_corpus_sha256": crawl_run["origin_corpus_sha256"],
        "offset_reference": "Python Unicode character offsets into extracted source_text",
        "scorer_status": "BGE-token chunking; official Stage 3 scorer is not implemented",
        "schema_version": SCHEMA_VERSION, "state": "BUILDING",
        "chunking": asdict(config),
    }
    result["snapshot_sha256"] = digest_json({"signature": signature, "parts": selected})
    if selected:
        validation = validate_snapshot(target, result, official_links=official_links, work_dir=work_dir)
        validation_path = target / "reports" / result["snapshot_sha256"][:16] / "integrity.json"
        atomic_json(validation_path, validation)
        result["integrity"] = validation
        if not validation["passed"]:
            raise ValueError(f"Snapshot integrity failed; see {validation_path}")
        result["state"] = "DATA_VALIDATED"
    atomic_json(target / f"snapshot-{result['snapshot_sha256'][:16]}.json", result)
    atomic_json(target / "build.json", result)
    return result
