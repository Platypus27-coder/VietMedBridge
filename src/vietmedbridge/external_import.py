"""Import the team's existing extraction archive into immutable data-v2 shards.

Only the four metadata files are read from the tar. The original archive/raw
locators remain provenance; metadata validation does not certify raw replay or
medical relevance. No network fetching or GPU work occurs here.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .artifacts import (atomic_json, code_fingerprint, digest_json, local_workspace,
                        publish_file, read_json, runtime_versions, sha256_file, utc_now, verify_file)
from .archive_parts import parts_manifest, restore_archive
from .build import (CHILD_SCHEMA, DOCUMENT_SCHEMA, FAILURE_SCHEMA, LEDGER_SCHEMA,
                    PARENT_SCHEMA, SECTION_SCHEMA, verify_spans)
from .chunks import ChunkConfig, chunk_source
from .dataset import _connection, url_host
from .quality import NORMALIZER_VERSION, document_quality, error_page_reason, has_encoded_payload
from .structure import parse_structure
from .text import _redirected_to_homepage, normalize_for_retrieval
from .validation import SCHEMA_VERSION, validate_snapshot

MEMBERS = {
    "dataset": "data/source/dataset_manifest.json",
    "frontier": "data/crawl_shards/shard_00000.parquet",
    "crawl": "data/manifests/crawl_manifest.parquet",
    "extracted": "data/manifests/extraction_manifest.parquet",
}
FRONTIER_MEMBER = re.compile(r"data/crawl_shards/shard_[0-9]{5,}\.parquet")
SCHEMAS = {"documents": DOCUMENT_SCHEMA, "children": CHILD_SCHEMA, "parents": PARENT_SCHEMA,
           "sections": SECTION_SCHEMA, "failures": FAILURE_SCHEMA, "ledger": LEDGER_SCHEMA}


def _name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Use a simple run name.")
    return value


def find_external_source(data_root, explicit=None):
    """Search only known handoff locations; ambiguous inputs require a path."""
    if explicit:
        source = Path(explicit)
        if not source.exists():
            raise FileNotFoundError(f"External source not found: {source}")
        return source
    root = Path(data_root)
    found = []
    for directory in (root / "incoming", root / "data_temp", root.parent / "data_temp"):
        if not directory.is_dir():
            continue
        if (directory / MEMBERS["extracted"]).is_file():
            found.append(directory)
        found.extend(directory.glob("vibiomir_shard_*.tar"))
        for child in directory.iterdir():
            if child.is_dir() and ((child / MEMBERS["extracted"]).is_file()
                                   or (child / "archive_manifest.json").is_file()):
                found.append(child)
    found = list(dict.fromkeys(p.resolve() for p in found))
    if len(found) > 1:
        raise ValueError("Multiple external crawl inputs found. Set EXTERNAL_SOURCE to one exact path.")
    return found[0] if found else None


def _directory_members(source):
    """Resolve the single numbered crawl frontier shard in an extracted archive."""
    source = Path(source)
    frontier_root = source / "data" / "crawl_shards"
    matches = sorted(path for path in frontier_root.glob("shard_*.parquet")
                     if FRONTIER_MEMBER.fullmatch(path.relative_to(source).as_posix())
                     and path.is_file())
    if len(matches) != 1:
        raise ValueError(
            "An extracted external source must contain exactly one numbered "
            f"data/crawl_shards/shard_*.parquet file; found {len(matches)}."
        )
    members = dict(MEMBERS)
    members["frontier"] = matches[0].relative_to(source).as_posix()
    return members


def stage_external_metadata(source, output_dir, *, work_dir):
    """Cache selected metadata on local disk and persist its checksums on Drive."""
    source, target = Path(source).resolve(), Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    local = Path(work_dir) / ("external-source-" + digest_json(str(source))[:16])
    local.mkdir(parents=True, exist_ok=True)
    marker = target / "source.json"
    source_stat = {"path": str(source), "is_directory": source.is_dir()}
    if source.is_file():
        source_stat.update(bytes=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns)
    elif (source / "archive_manifest.json").is_file():
        manifest = parts_manifest(source)
        source_stat.update(archive_sha256=manifest["sha256"],
                           manifest_sha256=sha256_file(source / "archive_manifest.json"))
    else:
        members = _directory_members(source)
        if members == MEMBERS:
            # Preserve the source signature used by existing shard-00000 directory runs.
            source_stat["members"] = {kind: {"bytes": (source/member).stat().st_size,
                "mtime_ns": (source/member).stat().st_mtime_ns} for kind, member in members.items()}
        else:
            source_stat["members"] = {kind: {"path": member, "bytes": (source/member).stat().st_size,
                "mtime_ns": (source/member).stat().st_mtime_ns} for kind, member in members.items()}
    if marker.exists():
        saved = read_json(marker)
        # A renamed/replaced archive must not silently reuse unrelated metadata.
        if saved["source"] != source_stat:
            raise ValueError("External input changed; choose a new import/build run.")
        for entry in saved["files"].values():
            destination = local / entry["name"]
            if not destination.exists() or sha256_file(destination) != entry["sha256"]:
                shutil.copyfile(target / entry["name"], destination)
                verify_file(destination, entry["sha256"])
        return local, saved
    paths = {kind: local / Path(member).name for kind, member in MEMBERS.items()}
    archive_source = source
    if source.is_dir() and (source / "archive_manifest.json").is_file():
        archive_source = restore_archive(source, Path(work_dir) / "restored_archives")
    if archive_source.is_dir():
        for kind, member in _directory_members(archive_source).items():
            shutil.copyfile(source / member, paths[kind])
    else:
        remaining = {kind: member for kind, member in MEMBERS.items() if kind != "frontier"}
        frontier_found = False
        with tarfile.open(archive_source, "r:") as archive:
            for member in archive:
                kind = next((k for k, name in remaining.items() if name == member.name), None)
                if kind is None and FRONTIER_MEMBER.fullmatch(member.name):
                    if frontier_found:
                        raise ValueError("Archive contains multiple numbered crawl frontier shards; import one shard per build.")
                    kind = "frontier"
                if kind is None:
                    continue
                if not member.isfile() or not 0 < member.size <= 512 * 1024**2:
                    raise ValueError(f"Invalid metadata archive member: {member.name}")
                with archive.extractfile(member) as incoming, paths[kind].open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024**2)
                if kind == "frontier":
                    frontier_found = True
                else:
                    del remaining[kind]
        missing = list(remaining.values())
        if not frontier_found:
            missing.append("data/crawl_shards/shard_*.parquet (exactly one required)")
        if missing:
            raise ValueError(f"Archive missing metadata: {missing}")
    files = {kind: {"name": path.name, "sha256": publish_file(path, target / path.name),
                    "bytes": path.stat().st_size} for kind, path in paths.items()}
    saved = {"source": source_stat, "files": files,
             "input_signature": digest_json(files), "created_at": utc_now(),
             "raw_verification": "METADATA_BINDING_ONLY_RAW_PAYLOAD_NOT_REPLAYED"}
    atomic_json(marker, saved)
    return local, saved


def _incoming(con, local, metadata, official_links):
    descriptor = read_json(local / metadata["files"]["dataset"]["name"])
    corpus_sha = sha256_file(official_links)
    sources = [s for s in descriptor.get("sources", []) if s.get("kind") == "links_path"]
    if len(sources) != 1 or sources[0]["sha256"] != corpus_sha:
        raise ValueError("External crawl and official corpus snapshots differ.")
    for kind in ("frontier", "crawl", "extracted"):
        con.read_parquet(str(local / metadata["files"][kind]["name"])).create_view(kind)
    con.read_parquet(str(official_links)).create_view("official")
    checks = {}
    for kind in ("frontier", "crawl", "extracted"):
        checks["duplicate_" + kind] = con.execute(
            f"SELECT count(*)-count(DISTINCT crawl_url_id) FROM {kind}").fetchone()[0]
    checks["frontier_missing_crawl"] = con.execute("""SELECT count(*) FROM frontier f LEFT JOIN crawl c USING(crawl_url_id)
        WHERE c.crawl_url_id IS NULL OR f.fetch_url IS DISTINCT FROM c.fetch_url""").fetchone()[0]
    checks["crawl_outside_frontier"] = con.execute("""SELECT count(*) FROM crawl c LEFT JOIN frontier f USING(crawl_url_id)
        WHERE f.crawl_url_id IS NULL""").fetchone()[0]
    checks["invalid_extraction_binding"] = con.execute("""SELECT count(*) FROM extracted e LEFT JOIN crawl c USING(crawl_url_id)
        WHERE c.status IS DISTINCT FROM 'SUCCESS_RAW' OR e.raw_sha256 IS DISTINCT FROM c.raw_sha256
        OR e.snapshot_id IS DISTINCT FROM c.snapshot_id""").fetchone()[0]
    checks["missing_extraction"] = con.execute("""SELECT count(*) FROM crawl c LEFT JOIN extracted e USING(crawl_url_id)
        WHERE c.status='SUCCESS_RAW' AND e.crawl_url_id IS NULL""").fetchone()[0]
    checks["invalid_source_length"] = con.execute("""SELECT count(*) FROM extracted WHERE extract_status='EXTRACT_SUCCESS'
        AND (text IS NULL OR length(trim(text))=0 OR char_count IS DISTINCT FROM length(text))""").fetchone()[0]
    con.execute("""CREATE TABLE mapped AS SELECT f.crawl_url_id,o.id AS doc_id,o.url AS url
        FROM official o JOIN frontier f ON split_part(trim(o.url),'#',1)=f.fetch_url""")
    checks["unmapped_frontier_or_alias_mismatch"] = con.execute("""SELECT count(*) FROM frontier f LEFT JOIN
        (SELECT crawl_url_id,count(*) AS n FROM mapped GROUP BY 1) m USING(crawl_url_id)
        WHERE m.n IS DISTINCT FROM f.doc_count""").fetchone()[0]
    checks["duplicate_official_id"] = con.execute("SELECT count(*)-count(DISTINCT doc_id) FROM mapped").fetchone()[0]
    checks["snapshot_binding_mismatch"] = con.execute("SELECT count(*) FROM crawl WHERE snapshot_id IS DISTINCT FROM ?",
        [descriptor["snapshot_id"]]).fetchone()[0]
    if any(checks.values()):
        raise ValueError(f"External input integrity failed: {checks}")
    # Sort only IDs, never the complete corpus's text/markdown in memory.
    con.execute("""CREATE TABLE incoming_ids AS SELECT row_number() OVER(ORDER BY doc_id)-1 AS input_row,
        doc_id,url,crawl_url_id FROM mapped""")
    return {"passed": True, "checks": checks, "official_links_sha256": corpus_sha,
            "official_ids": con.execute("SELECT count(*) FROM incoming_ids").fetchone()[0],
            "unique_urls": con.execute("SELECT count(*) FROM frontier").fetchone()[0],
            "external_snapshot_id": descriptor["snapshot_id"]}


def _hold(row, max_source_chars):
    if row["status"] != "SUCCESS_RAW":
        return "crawl", row["status"] or "UNKNOWN_CRAWL_STATUS"
    if row["extract_status"] != "EXTRACT_SUCCESS":
        return "parse", row["extract_status"] or "MISSING_EXTRACTION"
    text = row["text"] or ""
    if not text.strip():
        return "parse", "EMPTY_MAIN_TEXT"
    if len(text) > max_source_chars:
        return "resource", "SOURCE_TOO_LARGE_REQUIRES_SEPARATE_PROCESSING"
    if _redirected_to_homepage(row["url"], row["final_url"] or row["url"]):
        return "quality", "SOURCE_REDIRECT_HOME"
    reason = error_page_reason(row["title"] or "", text)
    if reason:
        return "quality", reason
    if has_encoded_payload(text):
        return "quality", "ENCODED_PAYLOAD_SUSPECTED"
    if row.get("source_issue") in {"SOURCE_REDIRECT_HOME", "JAVASCRIPT_CHALLENGE", "UNRESOLVED_TEMPLATE"}:
        return "quality", row["source_issue"]
    if re.search(r"\{\{\s*(?:item\.|data\.|title|content)|<%[=\s]", text):
        return "quality", "UNRESOLVED_TEMPLATE"
    return None


def _document(row, source):
    text = row["text"]
    # Use headings only when an exact source line exists. No markdown is inserted.
    lines = set(text.splitlines())
    hints = list(dict.fromkeys(m.group(1).strip() for m in re.finditer(
        r"^#{1,6}\s+(.+)$", row.get("text_markdown") or "", re.M)
        if m.group(1).strip() in lines))
    return {"doc_id": row["doc_id"], "url": row["url"], "final_url": row["final_url"] or row["url"],
        "title": row["title"] or "", "source_text": text, "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "body_sha256": row["raw_sha256"], "language": row["language"] or "unknown",
        "language_method": "external:" + (row["language_method"] or "unknown"),
        "language_confidence": row["language_confidence"],
        "parser": "external:" + (row["extractor"] or "unknown") + ":" + (row["extractor_version"] or "unknown"),
        "raw_file": str(source) + "#" + (row["raw_path"] or ""), "fetched_at": row["crawl_timestamp"],
        "domain": url_host(row["url"]), "content_type": row["content_type"] or "unknown",
        "normalizer_version": NORMALIZER_VERSION, "schema_version": SCHEMA_VERSION,
        "heading_hints": hints, "raw_has_table": False}


def import_external_corpus(source, official_links, output_root, tokenizer, tokenizer_spec, *,
                           run_name="team-100k-data-v1", config=None, work_dir=None,
                           shard_size=2048, max_shards=None, max_source_chars=2_000_000):
    """Conserve all official aliases/outcomes; write buffered, atomic Parquet shards."""
    if type(shard_size) is not int or shard_size < 1 or max_source_chars < 1:
        raise ValueError("Invalid shard/source size.")
    if max_shards is not None and (type(max_shards) is not int or max_shards < 0):
        raise ValueError("Invalid new-shard limit.")
    config = config or ChunkConfig()
    config.validate()
    target = Path(output_root) / _name(run_name)
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with local_workspace(work_dir) as temporary:
        local, metadata = stage_external_metadata(source, target / "external_source", work_dir=work_dir or temporary)
        with _connection(temporary) as con:
            con.execute("SET threads=1")
            con.execute("SET preserve_insertion_order=false")
            source_audit = _incoming(con, local, metadata, official_links)
            atomic_json(target / "external_input_audit.json", source_audit)
            identity = {"source_input_signature": metadata["input_signature"], "source_kind": "external-extraction-v1",
                "external_snapshot_id": source_audit["external_snapshot_id"], "tokenizer": tokenizer_spec,
                "chunks": asdict(config), "schema_version": SCHEMA_VERSION, "shard_size": shard_size,
                "max_source_chars": max_source_chars, "official_links_sha256": source_audit["official_links_sha256"],
                "code_sha256": code_fingerprint(), "runtime": runtime_versions()}
            signature = digest_json(identity)
            config_path = target / "config.json"
            if config_path.exists() and read_json(config_path) != identity | {"signature": signature}:
                raise ValueError("Import source/code/chunk policy changed; use a new build run.")
            atomic_json(config_path, identity | {"signature": signature})
            selected, written, cached, processed_seconds = [], 0, 0, 0.
            total_rows = source_audit["official_ids"]
            for first in tqdm(range(0, total_rows, shard_size), desc="Import/chunk existing crawl", unit="shard"):
                records = min(shard_size, total_rows-first)
                part_number = first // shard_size
                stem = f"part-{part_number:06d}"
                marker = target / "parts" / (stem + ".done.json")
                if marker.exists():
                    saved = read_json(marker)
                    if saved["signature"] != signature or saved["first_input_row"] != first or saved["records"] != records:
                        raise ValueError("Imported shard contract changed.")
                    for entry in saved["files"].values():
                        verify_file(target / entry["path"], entry["sha256"])
                    selected.append(saved); cached += 1
                    continue
                if max_shards is not None and written >= max_shards:
                    break
                chunk_start = time.perf_counter()
                counts = Counter()
                names = {kind: Path(temporary) / f"{stem}.{kind}.parquet" for kind in SCHEMAS}
                writers = {kind: pq.ParquetWriter(names[kind], schema, compression="zstd") for kind, schema in SCHEMAS.items()}
                buffers = {kind: [] for kind in SCHEMAS}
                buffered_chars = 0

                def flush():
                    for kind, rows in buffers.items():
                        if rows:
                            writers[kind].write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[kind]))
                            rows.clear()

                try:
                    batches = con.execute("""SELECT m.input_row,m.doc_id,m.url,c.status,c.final_url,c.content_type,
                        c.raw_path,c.raw_sha256,c.raw_size_bytes,c.elapsed_ms,c.http_status,c.crawl_timestamp,
                        e.extract_status,e.text,e.title,e.language,e.language_method,e.language_confidence,
                        e.extractor,e.extractor_version,e.source_issue,e.text_markdown
                        FROM (SELECT * FROM incoming_ids WHERE input_row>=? AND input_row<?) m
                        JOIN crawl c USING(crawl_url_id) LEFT JOIN extracted e USING(crawl_url_id)
                        ORDER BY m.input_row""",
                        [first,first+records]).fetch_record_batch(128)
                    for batch in batches:
                        for row in batch.to_pylist():
                            counts["input_records"] += 1
                            domain = url_host(row["url"])
                            buffers["ledger"].append({"doc_id": row["doc_id"], "url": row["url"], "domain": domain,
                                "status": "ok" if row["status"] == "SUCCESS_RAW" else row["status"],
                                "content_type": row["content_type"] or "unknown", "raw_bytes": row["raw_size_bytes"] or 0,
                                "elapsed_ms": int(row["elapsed_ms"]) if row["elapsed_ms"] is not None else None,
                                "http_status": row["http_status"]})
                            hold = _hold(row, max_source_chars)
                            if hold is None:
                                document = _document(row, source)
                                structure = parse_structure(document, document["heading_hints"])
                                document.update(document_quality(document["source_text"], title=document["title"],
                                    section_count=structure["known_heading_count"], raw_bytes=row["raw_size_bytes"]),
                                    section_count=structure["known_heading_count"])
                                content_hash = hashlib.sha256(normalize_for_retrieval(document["source_text"]).encode()).hexdigest()
                                document.update(content_hash=content_hash, canonical_content_id=content_hash)
                                try:
                                    children, parents = chunk_source(document, tokenizer, tokenizer_spec, config, structure=structure)
                                    verify_spans(document, children, parents)
                                    if not children:
                                        raise ValueError("NO_INDEXABLE_CHILDREN")
                                except ValueError as exc:
                                    hold = ("chunk", "CHUNK_SOURCE_REVIEW:" + str(exc)[:250])
                                else:
                                    buffers["documents"].append(document)
                                    buffers["children"].extend(children); buffers["parents"].extend(parents)
                                    buffers["sections"].extend(structure["sections"])
                                    for kind, n in (("documents",1),("children",len(children)),("parents",len(parents)),("sections",len(structure["sections"]))):
                                        counts[kind] += n
                                    buffered_chars += len(document["source_text"]) + sum(len(c["text"]) for c in children+parents)
                            if hold is not None:
                                phase, reason = hold
                                buffers["failures"].append({"doc_id": row["doc_id"], "url": row["url"], "phase": phase,
                                    "reason": reason, "reason_code": reason.split(":",1)[0], "domain": domain,
                                    "body_sha256": row["raw_sha256"], "content_type": row["content_type"] or "unknown",
                                    "raw_file": str(source) + "#" + (row["raw_path"] or "")})
                                counts["failures"] += 1; counts[phase+"_failed"] += 1
                            if len(buffers["ledger"]) >= 128 or buffered_chars >= 8_000_000:
                                flush(); buffered_chars = 0
                    flush()
                finally:
                    for writer in writers.values():
                        writer.close()
                if counts["input_records"] != records or counts["documents"]+counts["failures"] != records:
                    raise ValueError("Imported shard did not conserve every official input ID.")
                files = {kind: {"path": f"parts/{path.name}", "sha256": publish_file(path, target / "parts" / path.name)}
                         for kind, path in names.items()}
                elapsed = time.perf_counter()-chunk_start
                saved = {"part": part_number, "signature": signature, "first_input_row": first, "records": records,
                         "counts": dict(counts), "files": files, "seconds": round(elapsed,3), "completed_at": utc_now()}
                atomic_json(marker, saved)
                selected.append(saved); written += 1; processed_seconds += elapsed
                atomic_json(target / "import_progress.json", {"signature": signature,"complete_shards":len(selected),
                    "recorded_input_ids":sum(p["records"] for p in selected),"requested_input_ids":total_rows,
                    "updated_at":utc_now()})
            counts = Counter()
            for part in selected:
                counts.update(part["counts"])
            complete = counts["input_records"] == total_rows
            result = {"run_name": run_name,"signature":signature,"parts":selected,"counts":dict(counts),
                "crawl_run":None,"input_kind":"EXTERNAL_EXTRACTION","requested_input_records":total_rows,
                "requested_range":{"start":0,"stop":total_rows},"selected_range_complete":complete,
                "selected_shards":len(selected),"available_crawl_shards":(total_rows+shard_size-1)//shard_size,
                "origin_corpus_sha256":source_audit["official_links_sha256"],"chunking":asdict(config),
                "schema_version":SCHEMA_VERSION,"state":"BUILDING","created_at":utc_now(),
                "offset_reference":"Python Unicode character offsets into the unmodified external extracted text",
                "source_provenance":metadata,"source_audit":source_audit}
            result["snapshot_sha256"] = digest_json({"signature":signature,"parts":selected})
            if selected:
                validation_start = time.perf_counter()
                result["integrity"] = validate_snapshot(target,result,official_links=official_links,work_dir=work_dir)
                validation_seconds = time.perf_counter()-validation_start
                if not result["integrity"]["passed"]:
                    raise ValueError(f"Imported snapshot validation failed: {result['integrity']}")
                result["state"] = "DATA_VALIDATED" if complete else "PARTIAL_DATA_VALIDATED"
            else:
                validation_seconds = 0.
            atomic_json(target / "build.json",result)
            atomic_json(target / f"snapshot-{result['snapshot_sha256'][:16]}.json",result)
            atomic_json(target / "import_runtime.json", {"new_shards":written,"reused_shards":cached,
                "new_shard_seconds":round(processed_seconds,3),"validation_seconds":round(validation_seconds,3),
                "seconds_this_call":round(time.perf_counter()-started,3),"gpu_used":False,
                "complete_input_range":complete})
            return result
