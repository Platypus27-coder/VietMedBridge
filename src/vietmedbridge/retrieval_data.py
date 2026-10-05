"""Read an immutable data-v2 candidate without rebuilding its extractor."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .artifacts import digest_json, read_json
from .representation import build_dense_text, build_sparse_text, representation_hash
from .validation import artifact_paths


@dataclass
class Catalog:
    candidate: dict
    documents: dict
    children: dict
    parents: dict
    units: list
    aliases: dict
    build_config: dict | None = None

    @property
    def identity(self):
        return {"snapshot_sha256": self.candidate["snapshot_sha256"],
                "candidate_manifest_sha256": self.candidate["candidate_manifest_sha256"],
                "representation_ids_sha256": digest_json([u["id"] for u in self.units])}


def _unique(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} in frozen data.")
    return result


def load_catalog(build_dir, candidate_name, tokenizer, *, max_documents=2000, max_children=50000):
    root = Path(build_dir).resolve()
    path = (root / candidate_name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Candidate path escapes the build.")
    candidate = read_json(path)
    if candidate.get("state") != "FROZEN_CANDIDATE":
        raise ValueError("Choose a frozen candidate from notebook 03.")
    if digest_json({k: v for k, v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]:
        raise ValueError("Candidate manifest changed after freeze.")
    if digest_json({"signature": candidate["signature"], "parts": candidate["parts"]}) != candidate["snapshot_sha256"]:
        raise ValueError("Snapshot identity changed.")
    build_config = read_json(root / "config.json")
    if digest_json({k: v for k, v in build_config.items() if k != "signature"}) != candidate["signature"] or build_config["signature"] != candidate["signature"]:
        raise ValueError("Frozen build configuration changed.")
    integrity = candidate.get("integrity", {})
    if not integrity.get("passed") or integrity.get("official_membership") != "VERIFIED":
        raise ValueError("Frozen input must pass integrity and official ID/URL membership.")
    if candidate["counts"]["documents"] > max_documents or candidate["counts"]["children"] > max_children:
        raise ValueError("Pilot memory limit exceeded; benchmark a sharded index before using the 100k corpus.")
    tables = {}
    for kind in ("documents", "children", "parents", "sections", "failures", "ledger"):
        tables[kind] = [r for file in artifact_paths(root, candidate, kind)
                        for r in pq.read_table(file).to_pylist()]
        count_key = "input_records" if kind == "ledger" else kind
        if kind in ("documents", "children", "parents", "sections", "ledger") and len(tables[kind]) != candidate["counts"][count_key]:
            raise ValueError(f"Manifest count mismatch: {kind}")
    documents = _unique(tables["documents"], "doc_id")
    children = _unique(tables["children"], "chunk_id")
    parents = _unique(tables["parents"], "chunk_id")
    sections = _unique(tables["sections"], "section_id")
    failures = _unique(tables["failures"], "doc_id")
    ledger = _unique(tables["ledger"], "doc_id")
    if set(documents) & set(failures) or set(documents) | set(failures) != set(ledger):
        raise ValueError("Frozen outcomes do not conserve input IDs.")
    for d in documents.values():
        if d["url"] != ledger[d["doc_id"]]["url"] or hashlib.sha256(d["source_text"].encode()).hexdigest() != d["source_text_sha256"]:
            raise ValueError("Frozen document URL/source hash mismatch.")
    for span in list(children.values()) + list(parents.values()):
        d = documents[span["doc_id"]]
        a, b = span["start_char"], span["end_char"]
        if not 0 <= a < b <= len(d["source_text"]) or span["text"] != d["source_text"][a:b] or span["source_text_sha256"] != d["source_text_sha256"]:
            raise ValueError("Frozen chunk source span mismatch.")
    unit_text, aliases, sparse_text = {}, defaultdict(list), defaultdict(set)
    for child in children.values():
        parent = parents.get(child["parent_id"])
        section = sections.get(child["section_id"])
        if not parent or parent["doc_id"] != child["doc_id"] or not parent["start_char"] <= child["start_char"] < child["end_char"] <= parent["end_char"]:
            raise ValueError("Child/parent relationship changed.")
        if not section or section["doc_id"] != child["doc_id"] or section["heading_text"] != child["heading"]:
            raise ValueError("Child/section relationship changed.")
        dense = build_dense_text(child, title=documents[child["doc_id"]]["title"], heading=child["heading"],
                                 tokenizer=tokenizer, max_tokens=candidate["chunking"]["dense_max_tokens"])
        key = representation_hash(dense)
        if key != child["retrieval_representation_hash"] or (key in unit_text and unit_text[key] != dense):
            raise ValueError("Retrieval representation changed; use the candidate's pinned tokenizer.")
        unit_text[key] = dense
        sparse_text[key].add(build_sparse_text(child, title=documents[child["doc_id"]]["title"], heading=child["heading"]))
        aliases[key].append(child["chunk_id"])
    if not unit_text:
        raise ValueError("No indexable representations in candidate.")
    for key in aliases:
        aliases[key].sort()
    return Catalog(candidate, documents, children, parents,
                   [{"id": key, "text": unit_text[key], "sparse_text": "\n\n".join(sorted(sparse_text[key]))}
                    for key in sorted(unit_text)], dict(aliases), build_config)


def load_queries(path, *, expected_count=1200):
    rows = pq.read_table(path, columns=["id", "query"]).to_pylist()
    if len(rows) != expected_count or len({r["id"] for r in rows}) != expected_count:
        raise ValueError(f"Expected {expected_count} unique official queries.")
    if any(type(r["id"]) is not int or not isinstance(r["query"], str) or not r["query"].strip() for r in rows):
        raise ValueError("Invalid official query ID/text.")
    return rows
