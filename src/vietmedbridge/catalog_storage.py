"""Recover and transfer derived SQLite catalogs without changing their data policy.

The original writer stays byte-for-byte unchanged so existing catalog identities
remain usable. Build on the runtime's local disk first, retain that completed DB,
and publish bounded compressed parts before the Drive completion manifest.
"""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from . import disk_catalog as writer
from .artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, utc_now, verify_file
from .scale_vectors import bound, check_manifest

CHUNK_BYTES = 64 * 1024 * 1024


def _contract(build, candidate, inputs, analyzer):
    check_manifest(inputs)
    if (candidate.get("state") != "FROZEN_CANDIDATE" or not candidate.get("selected_range_complete")
        or not candidate.get("integrity", {}).get("passed")
        or candidate["integrity"].get("official_membership") != "VERIFIED"
        or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]
        or digest_json({"signature":candidate["signature"],"parts":candidate["parts"]}) != candidate["snapshot_sha256"]
        or inputs["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]):
        raise ValueError("Disk catalog requires matching, validated frozen inputs.")
    config = read_json(Path(build) / "config.json")
    if (config["signature"] != candidate["signature"] or
        digest_json({k:v for k,v in config.items() if k != "signature"}) != candidate["signature"]):
        raise ValueError("Frozen build config changed.")
    return {"candidate":candidate["candidate_manifest_sha256"],"inputs":inputs["manifest_sha256"],
        "analyzer":analyzer.identity,"sqlite":sqlite3.sqlite_version,"schema":"disk-source-fts5-v1",
        "code_sha256":sha256_file(Path(writer.__file__)),"field_weights":[2.,1.5,1.,1.5]}


def _manifest(path, policy, candidate, inputs):
    saved = read_json(path)
    check_manifest(saved)
    if (any(saved.get(k) != v for k,v in policy.items()) or saved.get("signature") != digest_json(policy)
        or saved.get("state") != "COMPLETE" or saved.get("path") != "catalog.sqlite"
        or saved.get("counts") != {k:candidate["counts"][k] for k in ("documents","children","parents")}
        or saved.get("unit_count") != inputs["input_count"]):
        raise ValueError("Disk catalog contract changed.")
    return saved


def _matches(path, checksum):
    return Path(path).is_file() and sha256_file(path) == checksum


def _storage_parts(saved):
    storage = saved["storage"]
    if (storage.get("format") != "gzip-parts-v1" or type(storage.get("bytes")) is not int
        or storage["bytes"] <= 0 or not storage.get("parts")):
        raise ValueError("Invalid catalog storage manifest.")
    offset = 0
    for index, part in enumerate(storage["parts"]):
        if (part.get("path") != f"catalog.parts/part-{index:05d}.sqlite.gz" or part.get("offset") != offset
            or type(part.get("bytes")) is not int or not 0 < part["bytes"] <= CHUNK_BYTES
            or type(part.get("compressed_bytes")) is not int or part["compressed_bytes"] <= 0):
            raise ValueError("Invalid catalog part ordering/size.")
        offset += part["bytes"]
        yield part
    if offset != storage["bytes"]:
        raise ValueError("Incomplete catalog storage range.")


def _verify_remote(target, saved):
    if "storage" not in saved:
        verify_file(target / saved["path"], saved["sha256"])
        return
    for part in _storage_parts(saved):
        path = bound(target, part["path"])
        verify_file(path, part["sha256"])
        if path.stat().st_size != part["compressed_bytes"]:
            raise ValueError("Catalog part size changed.")


def _restore(target, saved, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    if "storage" not in saved:
        verify_file(target / saved["path"], saved["sha256"])
        if publish_file(target / saved["path"], local) != saved["sha256"]:
            raise ValueError("Catalog cache copy changed.")
        return
    temporary = local.with_name(f".{uuid4().hex}.restore.tmp")
    try:
        with temporary.open("wb") as output:
            for index, part in enumerate(_storage_parts(saved), 1):
                path = bound(target, part["path"])
                verify_file(path, part["sha256"])
                if path.stat().st_size != part["compressed_bytes"]:
                    raise ValueError("Catalog part size changed.")
                with gzip.open(path, "rb") as source:
                    data = source.read(part["bytes"] + 1)
                if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["raw_sha256"]:
                    raise ValueError("Catalog decoded part changed.")
                output.write(data)
                print(f"Restore CPU catalog: {index}/{len(saved['storage']['parts'])} parts", flush=True)
        verify_file(temporary, saved["sha256"])
        os.replace(temporary, local)
    finally:
        temporary.unlink(missing_ok=True)


def _publish(local, target, saved):
    verify_file(local, saved["sha256"])
    parts = []
    size = local.stat().st_size
    total = (size + CHUNK_BYTES - 1) // CHUNK_BYTES
    temporary = local.with_name(f".{uuid4().hex}.publish.tmp")
    progress = target / "catalog_progress.json"
    try:
        with local.open("rb") as stream:
            for index in range(total):
                data = stream.read(CHUNK_BYTES)
                temporary.write_bytes(gzip.compress(data, compresslevel=1, mtime=0))
                relative = f"catalog.parts/part-{index:05d}.sqlite.gz"
                destination = bound(target, relative)
                checksum = sha256_file(temporary)
                reused = _matches(destination, checksum)
                if not reused:
                    if publish_file(temporary, destination) != checksum:
                        raise ValueError("Catalog part copy changed.")
                    verify_file(destination, checksum)
                parts.append({"path":relative,"offset":index * CHUNK_BYTES,"bytes":len(data),
                    "raw_sha256":hashlib.sha256(data).hexdigest(),"sha256":checksum,
                    "compressed_bytes":temporary.stat().st_size})
                atomic_json(progress, {"state":"PUBLISHING","completed_parts":len(parts),"total_parts":total,
                    "source_sha256":saved["sha256"],"bytes":size,"updated_at":utc_now()})
                print(f"Save CPU catalog: {index+1}/{total} parts | {'reused' if reused else 'new'}", flush=True)
        report = {k:v for k,v in saved.items() if k not in ("manifest_sha256","storage")}
        report["storage"] = {"format":"gzip-parts-v1","bytes":size,"parts":parts}
        report["manifest_sha256"] = digest_json(report)
        # All final payload filenames are read back before COMPLETE is published.
        _verify_remote(target, report)
        atomic_json(target / "catalog.json", report)
        atomic_json(progress, {"state":"COMPLETE","completed_parts":len(parts),"total_parts":total,
            "source_sha256":saved["sha256"],"bytes":size,"updated_at":utc_now()})
        return report
    finally:
        temporary.unlink(missing_ok=True)


def _preserve_bad_local(local_dir):
    tag = uuid4().hex
    for name in ("catalog.json", "catalog.sqlite"):
        path = local_dir / name
        if path.exists():
            path.rename(local_dir / f"{name}.invalid-{tag}")


def prepare_disk_catalog(build, candidate, inputs, output_dir, analyzer, *, work_dir):
    """Reuse healthy catalogs, repair from local bytes, otherwise rebuild only this derived index."""
    policy = _contract(build, candidate, inputs, analyzer)
    signature = digest_json(policy)
    target = Path(output_dir) / ("catalog-" + signature[:16])
    local_dir = Path(work_dir) / target.name
    if target.resolve() == local_dir.resolve():
        raise ValueError("Persistent catalog output must differ from the runtime work directory.")
    local_dir.mkdir(parents=True, exist_ok=True)
    local, marker = local_dir / "catalog.sqlite", target / "catalog.json"
    saved = _manifest(marker, policy, candidate, inputs) if marker.exists() else None
    if saved is not None:
        # A corrupt contract is never ignored. Only derived payload availability
        # can be repaired; inputs and the current data policy were checked above.
        if _matches(local, saved["sha256"]):
            try:
                _verify_remote(target, saved)
            except (ValueError, FileNotFoundError) as error:
                print(f"Repair CPU catalog from verified local copy: {error}", flush=True)
                saved = _publish(local, target, saved)
            return writer.DiskCatalog(local, candidate, inputs, analyzer, saved)
        try:
            print("Restore completed CPU catalog to local disk", flush=True)
            _restore(target, saved, local)
            return writer.DiskCatalog(local, candidate, inputs, analyzer, saved)
        except (ValueError, FileNotFoundError, gzip.BadGzipFile, EOFError) as error:
            print(f"CPU catalog payload unavailable; rebuild derived catalog only: {error}", flush=True)
    local_marker = local_dir / "catalog.json"
    if local_marker.exists():
        cached = _manifest(local_marker, policy, candidate, inputs)
        if not _matches(local, cached["sha256"]):
            _preserve_bad_local(local_dir)
    elif local.exists():
        # Old runtimes retained a DB without its local marker. Reuse only if it
        # matches the independently sealed Drive manifest byte for byte.
        if saved is not None and _matches(local, saved["sha256"]):
            atomic_json(local_marker, saved)
        else:
            _preserve_bad_local(local_dir)
    # The unchanged writer targets local disk. Its completed DB/marker survive
    # publication failure in this runtime instead of being deleted with a tempdir.
    base = writer.prepare_disk_catalog(build, candidate, inputs, Path(work_dir), analyzer, work_dir=work_dir)
    try:
        saved = _publish(local, target, base.manifest)
    finally:
        base.close()
    return writer.DiskCatalog(local, candidate, inputs, analyzer, saved)
