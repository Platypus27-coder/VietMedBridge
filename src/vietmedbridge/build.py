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
from .crawl import completed_parts, iter_raw_records
from .text import extract_source

DOCUMENT_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("url", pa.string()), ("final_url", pa.string()),
    ("title", pa.string()), ("source_text", pa.large_string()),
    ("source_text_sha256", pa.string()), ("body_sha256", pa.string()),
    ("language", pa.string()), ("language_method", pa.string()), ("parser", pa.string()),
    ("quality_flags", pa.list_(pa.string())), ("raw_file", pa.string()),
    ("fetched_at", pa.string()),
])
SPAN_FIELDS = [
    ("doc_id", pa.int64()), ("chunk_id", pa.string()), ("chunk_order", pa.int64()),
    ("source_text_sha256", pa.string()), ("language", pa.string()), ("policy_sha256", pa.string()),
    ("start_char", pa.int64()), ("end_char", pa.int64()),
    ("token_start", pa.int64()), ("token_end", pa.int64()),
    ("text", pa.large_string()), ("retrieval_text", pa.large_string()),
]
PARENT_SCHEMA = pa.schema(SPAN_FIELDS)
CHILD_SCHEMA = pa.schema(SPAN_FIELDS + [("parent_id", pa.string())])
FAILURE_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("url", pa.string()), ("phase", pa.string()),
    ("reason", pa.string()), ("body_sha256", pa.string()), ("raw_file", pa.string()),
])


def verify_spans(document: dict, children: list[dict], parents: list[dict]) -> None:
    text = document["source_text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != document["source_text_sha256"]:
        raise ValueError("Document source hash mismatch.")
    mapping = {parent["chunk_id"]: parent for parent in parents}
    for span in children + parents:
        if span["doc_id"] != document["doc_id"]:
            raise ValueError("Wrong official document ID on source span.")
        if span["source_text_sha256"] != document["source_text_sha256"]:
            raise ValueError("Span source hash mismatch.")
        if text[span["start_char"]:span["end_char"]] != span["text"] or not span["text"]:
            raise ValueError("Source span text does not match source offsets.")
    for child in children:
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
) -> dict:
    config = config or ChunkConfig()
    config.validate()
    crawl_dir, output_root = Path(crawl_dir), Path(output_root)
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("Use a simple run name.")
    target = output_root / run_name
    target.mkdir(parents=True, exist_ok=True)
    identity = {"crawl_signature": read_json(crawl_dir / "run.json")["signature"],
                "tokenizer": tokenizer_spec, "chunks": asdict(config),
                "extractor_version": "source-text-v1", "runtime": runtime_versions(),
                "code_sha256": code_fingerprint()}
    signature = digest_json(identity)
    config_path = target / "config.json"
    if config_path.exists() and read_json(config_path)["signature"] != signature:
        raise ValueError("Parser/chunker config changed. Use a new canonical run name.")
    if not config_path.exists():
        atomic_json(config_path, {**identity, "signature": signature})
    selected, processed = [], 0
    for part in tqdm(completed_parts(crawl_dir), desc="Extract and chunk", unit="shard"):
        stem = Path(part["raw_file"]).name.removesuffix(".raw.jsonl.gz")
        checkpoint = target / f"{stem}.done.json"
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
                     for kind in ("documents", "children", "parents", "failures")}
            schemas = {"documents": DOCUMENT_SCHEMA, "children": CHILD_SCHEMA,
                       "parents": PARENT_SCHEMA, "failures": FAILURE_SCHEMA}
            writers = {kind: pq.ParquetWriter(path, schemas[kind], compression="zstd")
                       for kind, path in names.items()}
            try:
                seen = set()
                for raw in iter_raw_records(crawl_dir / part["raw_file"]):
                    counts["input_records"] += 1
                    if raw["doc_id"] in seen:
                        raise ValueError("Raw shard contains a repeated official ID.")
                    seen.add(raw["doc_id"])
                    failure = {
                        "doc_id": raw["doc_id"], "url": raw["url"],
                        "body_sha256": raw.get("body_sha256"), "raw_file": part["raw_file"],
                    }
                    if raw["status"] != "ok":
                        _write(writers["failures"], [{**failure, "phase": "crawl",
                                                     "reason": raw.get("error", "fetch_failed")}], FAILURE_SCHEMA)
                        counts["crawl_failed"] += 1
                        continue
                    body = base64.b64decode(raw["body_base64"], validate=True)
                    if hashlib.sha256(body).hexdigest() != raw["body_sha256"]:
                        raise ValueError("Raw source body checksum mismatch.")
                    try:
                        extracted = extract_source(body, raw.get("content_type", ""))
                    except Exception as exc:
                        _write(writers["failures"], [{**failure, "phase": "parse",
                                                     "reason": f"{type(exc).__name__}:{str(exc)[:300]}"}], FAILURE_SCHEMA)
                        counts["parse_failed"] += 1
                        continue
                    document = {
                        **extracted, "doc_id": int(raw["doc_id"]), "url": raw["url"],
                        "final_url": raw.get("final_url", raw["url"]),
                        "body_sha256": raw["body_sha256"], "raw_file": part["raw_file"],
                        "fetched_at": raw["fetched_at"],
                    }
                    children, parents = chunk_source(document, tokenizer, tokenizer_spec, config)
                    verify_spans(document, children, parents)
                    _write(writers["documents"], [document], DOCUMENT_SCHEMA)
                    _write(writers["children"], children, CHILD_SCHEMA)
                    _write(writers["parents"], parents, PARENT_SCHEMA)
                    counts["documents"] += 1
                    counts["children"] += len(children)
                    counts["parents"] += len(parents)
                    counts[f"language_{document['language']}"] += 1
                if counts["input_records"] != part["records"]:
                    raise ValueError("Raw shard count does not match its manifest.")
            finally:
                for writer in writers.values():
                    writer.close()
            files = {}
            for kind, path in names.items():
                files[kind] = {"path": path.name, "sha256": publish_file(path, target / path.name)}
        saved = {
            "part": part["part"], "signature": signature,
            "input_raw_sha256": part["raw_sha256"], "files": files,
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
        "offset_reference": "Python Unicode character offsets into extracted source_text",
        "scorer_status": "BGE-token chunking; official Stage 3 scorer is not implemented",
    }
    result["snapshot_sha256"] = digest_json({"signature": signature, "parts": selected})
    atomic_json(target / f"snapshot-{result['snapshot_sha256'][:16]}.json", result)
    atomic_json(target / "build.json", result)
    return result
