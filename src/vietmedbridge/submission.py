"""Validate source-backed predictions and export one JSON at the ZIP root."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .artifacts import atomic_json, digest_json, local_workspace, publish_file


def validate_submission(records, queries, catalog, report, *, expected_count=1200):
    if len(queries) != expected_count or len({q["id"] for q in queries}) != expected_count:
        raise ValueError("Official query coverage/count mismatch.")
    if report.get("state") != "COMPLETE" or report["completed_queries"] != expected_count or len(records) != expected_count:
        raise ValueError("Finish every query checkpoint before exporting a submission.")
    by_id = {q["id"]: q for q in queries}
    predictions, seen = [], set()
    for record in records:
        if record["signature"] != report["signature"] or digest_json({k: v for k, v in record.items() if k != "record_sha256"}) != record["record_sha256"]:
            raise ValueError("Query checkpoint integrity mismatch.")
        p = record["prediction"]
        if set(p) != {"id", "relevant_docs", "relevant_chunks"} or type(p["id"]) is not int or p["id"] not in by_id or p["id"] in seen:
            raise ValueError("Submission query ID/schema mismatch.")
        if record["query_sha256"] != digest_json(by_id[p["id"]]):
            raise ValueError("Prediction belongs to another query text.")
        seen.add(p["id"])
        docs, chunks = p["relevant_docs"], p["relevant_chunks"]
        if not isinstance(docs, list) or not docs or any(type(d) is not int or d not in catalog.documents for d in docs) or len(set(docs)) != len(docs):
            raise ValueError("Selected documents need unique official integer IDs.")
        if not isinstance(chunks, list) or not chunks or len(chunks) != len(record["provenance"]):
            raise ValueError("Selected chunks need matching source provenance.")
        selected_parents = set()
        for chunk, origin in zip(chunks, record["provenance"], strict=True):
            if set(chunk) != {"doc_id", "chunk_text"} or type(chunk["doc_id"]) is not int or chunk["doc_id"] not in docs or not isinstance(chunk["chunk_text"], str) or not chunk["chunk_text"].strip():
                raise ValueError("Submission chunk ID/text/schema mismatch.")
            parent = catalog.parents.get(origin["parent_id"])
            anchor = catalog.children.get(origin["anchor_child_id"])
            if not parent or not anchor or anchor["parent_id"] != parent["chunk_id"] or parent["chunk_id"] in selected_parents:
                raise ValueError("Invalid/duplicate source parent or child anchor.")
            selected_parents.add(parent["chunk_id"])
            if any(origin[k] != parent[k] for k in ("doc_id", "start_char", "end_char", "source_text_sha256")) or chunk["doc_id"] != parent["doc_id"]:
                raise ValueError("Chunk provenance does not match frozen parent.")
            doc = catalog.documents[chunk["doc_id"]]
            source = doc["source_text"]
            if hashlib.sha256(source.encode()).hexdigest() != origin["source_text_sha256"] or chunk["chunk_text"] != parent["text"] or chunk["chunk_text"] != source[origin["start_char"]:origin["end_char"]]:
                raise ValueError("Submission text must be the exact frozen source slice.")
        predictions.append(p)
    if seen != set(by_id):
        raise ValueError("Submission must cover every official query exactly once.")
    # Restore official Parquet order; IDs need not be consecutive.
    ordered = {p["id"]: p for p in predictions}
    return [ordered[q["id"]] for q in queries]


def export_submission(records, queries, catalog, report, output_dir, *, evidence,
                      expected_count=1200, work_dir=None):
    predictions = validate_submission(records, queries, catalog, report, expected_count=expected_count)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(predictions, ensure_ascii=False, indent=2) + "\n"
    with local_workspace(work_dir) as temporary:
        local = Path(temporary)
        result_path = local / "results.json"
        result_path.write_text(payload, encoding="utf-8")
        zip_path = local / "submission.zip"
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.write(result_path, arcname="results.json")
        with ZipFile(zip_path) as archive:
            if archive.namelist() != ["results.json"] or json.loads(archive.read("results.json")) != predictions or archive.testzip() is not None:
                raise ValueError("Submission ZIP must contain exactly one valid JSON at its root.")
        result_sha = publish_file(result_path, root / result_path.name)
        zip_sha = publish_file(zip_path, root / zip_path.name)
    manifest = {"catalog": catalog.identity, "queries_sha256": digest_json(queries),
                "prediction_signature": report["signature"], "query_count": len(predictions),
                "evidence": evidence, "json_sha256": result_sha, "zip_sha256": zip_sha,
                "state": "SCHEMA_AND_SOURCE_VALIDATED", "scope": "PILOT_LIMITED_CORPUS",
                "evaluation": "NOT_EVALUATED_NO_REFERENCE_LABELS", "fine_tuned": False}
    manifest["manifest_sha256"] = digest_json(manifest)
    atomic_json(root / "submission_manifest.json", manifest)
    return manifest
