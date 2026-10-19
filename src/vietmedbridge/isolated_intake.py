"""CPU-only, snapshot-bound intake that never publishes a submission pointer.

Archives are read through bounded seekable streams; even nested TARs do not
need to be copied/unpacked. Only metadata and normal data-v2 outputs are saved.
The frozen baseline and source plan are immutable inputs, not active pointers.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import tarfile
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .artifacts import (atomic_json, code_fingerprint, digest_json, publish_file, read_json,
                        runtime_versions, sha256_file, utc_now, verify_file)
from .dataset import _connection
from .external_import import MEMBERS, import_external_corpus


class Window(io.RawIOBase):
    """A bounded view of a seekable archive; offsets cannot escape its member."""
    def __init__(self, stream, start, size):
        self.stream, self.start, self.size, self.position = stream, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0 or position > self.size:
            raise ValueError("Archive seek outside bounded member.")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        self.stream.seek(self.start+self.position)
        payload = self.stream.read(size)
        if len(payload) != size:
            raise EOFError("Truncated archive member.")
        self.position += size
        return payload


class DriveRange(io.RawIOBase):
    """Read-only Drive API media ranges using an already authenticated session.

    Fail closed if a server ignores Range; never download a whole multi-GB TAR.
    One small block cache makes TAR header reads inexpensive.
    """
    def __init__(self, session, file_id, metadata, *, block_size=1024**2):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValueError("Invalid Drive file ID.")
        self.session, self.file_id, self.metadata = session, file_id, metadata
        self.size, self.position = int(metadata["size"]), 0
        self.block_size, self.block_start, self.block = block_size, -1, b""
        self.url = f"https://www.googleapis.com/drive/v3/files/{file_id}"

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or not 0 <= position <= self.size:
            raise ValueError("Drive seek outside file.")
        self.position = position
        return position

    def _get_block(self, start):
        end = min(self.size, start+self.block_size)-1
        for attempt in range(4):
            with self.session.get(self.url, params={"alt": "media", "supportsAllDrives": "true"},
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    stream=True, timeout=90) as response:
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    print(f"Drive range retry {attempt+1}/3", flush=True)
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                expected = f"bytes {start}-{end}/{self.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Drive did not honor the bounded Range request.")
                payload = response.raw.read(end-start+2)
                if len(payload) != end-start+1:
                    raise EOFError("Incomplete Drive range.")
                self.block_start, self.block = start, payload
                return

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        result = bytearray()
        while len(result) < size:
            if not self.block_start <= self.position < self.block_start+len(self.block):
                self._get_block(self.position)
            start = self.position-self.block_start
            count = min(size-len(result), len(self.block)-start)
            result.extend(self.block[start:start+count])
            self.position += count
        return bytes(result)


class DriveSource:
    """GET-only adapter. Credentials stay in the caller's runtime."""
    def __init__(self, session):
        self.session = session

    def metadata(self, file_id):
        response = self.session.get(f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"fields": "id,name,size,modifiedTime,md5Checksum,trashed", "supportsAllDrives": "true"},
            timeout=60)
        response.raise_for_status()
        value = response.json()
        if value.get("trashed") or not value.get("size"):
            raise ValueError("Source is trashed or is not a downloadable file.")
        return value

    def open(self, file_id, metadata):
        return DriveRange(self.session, file_id, metadata)


def _inside(root, relative):
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Intake artifact escapes its root.")
    return path


def _parquet(con, query, destination, work_dir):
    # DuckDB writes/seeks on local disk. The Drive filename is published only
    # after the whole file exists; interruption cannot expose a partial table.
    with tempfile.TemporaryDirectory(prefix="intake-table-", dir=work_dir) as temporary:
        path = Path(temporary)/"table.parquet"
        con.sql(query).write_parquet(str(path), compression="zstd")
        return publish_file(path, destination)


def intake_root(data_root, campaign):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", campaign):
        raise ValueError("Invalid intake campaign name.")
    # A sibling cannot be picked up by submission discovery under data/processed.
    return Path(data_root).resolve().parent/"data_staging"/campaign


def pin_plan(root, plan):
    if not plan.get("sources"):
        raise ValueError("An intake plan must contain at least one source.")
    identities = [(s["snapshot_id"], s["frontier_sha256"]) for s in plan["sources"]]
    if len(set(identities)) != len(identities):
        raise ValueError("The plan contains multiple versions of the same frontier.")
    jobs = [s["job_id"] for s in plan["sources"]]
    if len(set(jobs)) != len(jobs):
        raise ValueError("The plan contains duplicate job IDs.")
    path = Path(root)/"intake_plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Intake plan changed; use a new campaign, retaining previous checkpoints.")
    if not path.exists():
        atomic_json(path, plan)


def stage_metadata(reader, source, target):
    """Persist four pinned metadata files, never raw HTML or an inner TAR.

    Completion marker is last. An interrupted file copy is retried. Source
    metadata identity prevents silently mixing different cumulative versions.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    marker = target/"metadata.json"
    current = reader.metadata(source["drive_file_id"])
    if int(current["size"]) != source["archive_bytes"]:
        raise ValueError("Audited archive size changed; re-audit before importing.")
    if marker.exists():
        saved = read_json(marker)
        if saved["source"] != source or saved["remote"] != current:
            raise ValueError("Archive version changed; use a newly audited intake plan.")
        for item in saved["files"].values():
            verify_file(_inside(target, item["path"]), item["sha256"])
        print(f"Reused metadata: {source['job_id']}", flush=True)
        return saved
    names = dict(MEMBERS, frontier=source["frontier"])
    remaining = {name: kind for kind, name in names.items()}
    files = {}
    with reader.open(source["drive_file_id"], current) as remote:
        stream = remote
        if source.get("inner_tar"):
            with tarfile.open(fileobj=remote, mode="r:") as outer:
                for index, member in enumerate(outer):
                    if index >= 128:
                        raise ValueError("Inner TAR not found within bounded header scan.")
                    if member.name == source["inner_tar"]:
                        if not member.isfile():
                            raise ValueError("Inner TAR must be a regular file.")
                        stream = Window(remote, member.offset_data, member.size)
                        break
                else:
                    raise ValueError("Selected inner TAR is missing.")
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for index, member in enumerate(archive):
                if index >= 128:
                    raise ValueError("Required metadata missing from bounded header scan.")
                if member.name not in remaining:
                    continue
                kind = remaining[member.name]
                if not member.isfile() or not 0 < member.size <= 512*1024**2:
                    raise ValueError("Invalid metadata member size/type.")
                destination = _inside(target, member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix+".partial")
                print(f"Read metadata: {source['job_id']} | {kind} | {member.size/1024**2:.1f} MiB", flush=True)
                with archive.extractfile(member) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024**2)
                if temporary.stat().st_size != member.size:
                    raise EOFError("Truncated metadata payload.")
                checksum = sha256_file(temporary)
                expected = member.pax_headers.get("vibiomir.sha256")
                if (expected and checksum != expected) or (kind == "frontier" and checksum != source["frontier_sha256"]):
                    raise ValueError("Archive metadata SHA-256 mismatch.")
                os.replace(temporary, destination)
                files[kind] = {"path": member.name, "sha256": checksum, "bytes": member.size}
                del remaining[member.name]
                if not remaining:
                    break
    if remaining:
        raise ValueError(f"Archive missing required metadata: {sorted(remaining)}")
    if reader.metadata(source["drive_file_id"]) != current:
        raise ValueError("Archive changed while metadata was being read; no checkpoint published.")
    descriptor = read_json(target/names["dataset"])
    if descriptor["snapshot_id"] != source["snapshot_id"]:
        raise ValueError("Source snapshot differs from the pinned intake plan.")
    result = {"source": source, "remote": current, "files": files,
              "raw_verification": "METADATA_ONLY_RAW_PAYLOAD_NOT_REPLAYED"}
    atomic_json(marker, result)
    return result


def fetch_reference(reader, reference, target):
    """Download a small pinned ID mapping once, then reuse its checksum."""
    target = Path(target)
    if target.exists():
        verify_file(target, reference["sha256"])
        print(f"Reused pinned reference: {target.name}", flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    metadata = reader.metadata(reference["drive_file_id"])
    size, received = int(metadata["size"]), 0
    print(f"Read pinned reference: {target.name} | {size/1024**2:.1f} MiB", flush=True)
    with reader.open(reference["drive_file_id"], metadata) as incoming, temporary.open("wb") as outgoing:
        while block := incoming.read(1024**2):
            outgoing.write(block)
            received += len(block)
            if received % (16*1024**2) == 0 or received == size:
                print(f"{target.name}: {received/1024**2:.1f}/{size/1024**2:.1f} MiB", flush=True)
    verify_file(temporary, reference["sha256"])
    os.replace(temporary, target)
    return target


def _local_mapping(source, checksum, work):
    """Read a pinned mapping from Drive once per runtime, then join locally."""
    target = Path(work)/"reference_cache"/(checksum+".parquet")
    if target.exists():
        verify_file(target, checksum)
    elif publish_file(source, target) != checksum:
        target.unlink(missing_ok=True)
        raise ValueError("Pinned ID mapping changed while being staged locally.")
    return target


def select_delta(source_dir, metadata, links, existing_paths, output_dir, work_dir):
    """Conserve native aliases in an audit ledger; skip only fully reused URLs.

    Reuse requires the same native ID AND exact URL in a validated build. A
    missing alias causes the whole URL's aliases to be processed, preserving
    frontier doc_count. Failed/pending inputs remain queued for normal import.
    """
    source_dir, output, work = Path(source_dir), Path(output_dir), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"metadata": metadata, "links_sha256": sha256_file(links),
                "existing": [{"sha256": sha256_file(p)} for p in existing_paths]}
    marker = output/"selection.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["identity"] != identity:
            raise ValueError("Delta selection inputs changed; use a new campaign.")
        for item in saved["files"].values():
            verify_file(_inside(output, item["path"]), item["sha256"])
        return saved
    with _connection(work) as con:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        for kind in ("frontier", "crawl", "extracted"):
            con.read_parquet(str(source_dir/metadata["files"][kind]["path"])).create_view(kind)
        con.read_parquet(str(links)).create_view("official")
        con.read_parquet([str(p) for p in existing_paths], union_by_name=True).create_view("existing")
        # The existing pipeline performs its full audit again on the filtered
        # metadata. Check the original source first so filtering cannot hide a
        # foreign record or an invalid raw/extraction binding.
        from .external_import import _incoming
        local_descriptor = work/"dataset_manifest.json"
        shutil.copyfile(source_dir/metadata["files"]["dataset"]["path"], local_descriptor)
        original = {"files": {k: {"name": str((source_dir/v["path"]).resolve())}
                              for k, v in metadata["files"].items()}}
        _incoming(con, work, original, links, report_dir=output)
        conflict = con.execute("""SELECT count(*) FROM mapped m JOIN existing e USING(doc_id)
            WHERE m.url IS DISTINCT FROM e.url""").fetchone()[0]
        if conflict:
            raise ValueError(f"Native ID/URL conflict with existing validated documents: {conflict}")
        con.execute("""CREATE TABLE selection AS SELECT m.*, CASE WHEN NOT EXISTS (
            SELECT 1 FROM mapped a WHERE a.crawl_url_id=m.crawl_url_id AND NOT EXISTS (
                SELECT 1 FROM existing e WHERE e.doc_id=a.doc_id AND e.url=a.url))
            THEN 'REUSED_EXISTING_DOCUMENT' ELSE 'QUEUED_FOR_PIPELINE' END AS action FROM mapped m""")
        con.execute("CREATE VIEW needed AS SELECT DISTINCT crawl_url_id FROM selection WHERE action='QUEUED_FOR_PIPELINE'")
        paths = dict(MEMBERS, frontier=metadata["source"]["frontier"])
        for kind in ("frontier", "crawl", "extracted"):
            destination = _inside(output, paths[kind])
            destination.parent.mkdir(parents=True, exist_ok=True)
            _parquet(con, f"SELECT s.* FROM {kind} s JOIN needed n USING(crawl_url_id)", destination, work)
        destination = _inside(output, paths["dataset"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_file(local_descriptor, destination)
        _parquet(con, "SELECT * FROM selection ORDER BY doc_id", output/"source_selection.parquet", work)
        counts = dict(con.execute("SELECT action,count(*) FROM selection GROUP BY 1").fetchall())
        files = {kind: {"path": path, "sha256": sha256_file(output/path)} for kind, path in paths.items()}
        files["selection"] = {"path": "source_selection.parquet", "sha256": sha256_file(output/"source_selection.parquet")}
        result = {"identity": identity, "counts": counts, "official_ids": sum(counts.values()), "files": files}
        atomic_json(marker, result)
        return result


def restore_import_source(selection_dir, selection, local):
    """Stable local names/mtime make import resume independent of Drive aliases."""
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    for kind in MEMBERS:
        item = selection["files"][kind]
        source, destination = Path(selection_dir)/item["path"], _inside(local, item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != item["sha256"]:
            shutil.copyfile(source, destination)
        os.utime(destination, ns=(1700000000000000000, 1700000000000000000))
    return local


def _write_outcomes(root, selection_dir, selection, build_dir, build, work_dir):
    """Reconcile EVERY original ID, including reuse, quality/crawl failures."""
    with _connection(work_dir) as con:
        con.read_parquet(str(Path(selection_dir)/"source_selection.parquet")).create_view("selection")
        if build and build["parts"]:
            for kind in ("documents", "failures"):
                paths = [str(Path(build_dir)/p["files"][kind]["path"]) for p in build["parts"]]
                con.read_parquet(paths).create_view(kind)
            con.execute("""CREATE VIEW outcomes AS SELECT s.doc_id,s.url,s.crawl_url_id,
                CASE WHEN s.action='REUSED_EXISTING_DOCUMENT' THEN s.action
                     WHEN d.doc_id IS NOT NULL THEN 'ADDED_VALID_DOCUMENT'
                     WHEN f.doc_id IS NOT NULL THEN 'PIPELINE_FAILURE' ELSE 'PENDING' END AS state,
                f.reason_code FROM selection s LEFT JOIN documents d USING(doc_id)
                LEFT JOIN failures f USING(doc_id)""")
        else:
            con.execute("CREATE VIEW outcomes AS SELECT doc_id,url,crawl_url_id,action AS state,NULL::VARCHAR AS reason_code FROM selection")
        counts = dict(con.execute("SELECT state,count(*) FROM outcomes GROUP BY 1").fetchall())
        if sum(counts.values()) != selection["official_ids"] or counts.get("PENDING", 0) or counts.get("QUEUED_FOR_PIPELINE", 0):
            raise ValueError("Intake outcomes do not account for all native input IDs.")
        _parquet(con, "SELECT * FROM outcomes ORDER BY doc_id", Path(root)/"source_outcomes.parquet", work_dir)
        return counts


def run_intake(plan, *, root, reader, links_by_snapshot, baseline_aliases,
               tokenizer, tokenizer_spec, chunk_config, work_dir, shard_size=2048):
    """Serial CPU intake. Safe alongside a submission in another runtime.

    A source completes before the next starts. Per-shard importer checkpoints
    survive interruption. COMPLETE source receipts skip importer/validation on
    resume after checking their small newly-added ID mapping.
    """
    root, work = Path(root), Path(work_dir)
    if any((ancestor/"active_data_candidate.json").exists() for ancestor in (root, *root.parents)):
        raise ValueError("Intake output must be outside the active submission data root.")
    work.mkdir(parents=True, exist_ok=True)
    pin_plan(root, plan)
    contract = {"tokenizer": tokenizer_spec, "chunking": asdict(chunk_config),
        "shard_size": shard_size, "code_sha256": code_fingerprint(), "runtime": runtime_versions()}
    contract_path = root/"intake_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("Intake code/runtime/chunk policy changed; restore the pinned intake environment.")
    if not contract_path.exists():
        atomic_json(contract_path, contract)
    baseline_local = _local_mapping(baseline_aliases, plan["baseline"]["sha256"], work)
    available, completed, total = [baseline_local], [], Counter()
    for index, source in enumerate(plan["sources"], 1):
        job = source["job_id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job):
            raise ValueError("Invalid source job ID.")
        target = root/"sources"/job
        receipt = target/"complete.json"
        links = Path(links_by_snapshot[source["snapshot_id"]])
        verify_file(links, plan["snapshots"][source["snapshot_id"]]["sha256"])
        if receipt.exists():
            saved = read_json(receipt)
            if saved["source"] != source or saved["plan_sha256"] != digest_json(plan):
                raise ValueError("Completed intake source binding changed.")
            verify_file(target/"added_ids.parquet", saved["added_ids_sha256"])
            verify_file(target/"source_outcomes.parquet", saved["outcomes_sha256"])
            print(f"Intake {index}/{len(plan['sources'])}: {job} | reused COMPLETE checkpoint", flush=True)
        else:
            print(f"Intake {index}/{len(plan['sources'])}: {job} | CPU only", flush=True)
            metadata = stage_metadata(reader, source, target/"metadata")
            selection = select_delta(target/"metadata", metadata, links, available, target/"selection", work)
            print(f"{job}: {selection['counts']} native IDs", flush=True)
            build, build_dir = None, root/"processed"/job
            if selection["counts"].get("QUEUED_FOR_PIPELINE", 0):
                staged = restore_import_source(target/"selection", selection, work/"sources"/job)
                build = import_external_corpus(staged, links, root/"processed", tokenizer, tokenizer_spec,
                    run_name=job, config=chunk_config, work_dir=work, shard_size=shard_size)
                if not build["selected_range_complete"] or not build["integrity"]["passed"]:
                    raise ValueError("Intake build is not complete and valid.")
                atomic_json(build_dir/"intake_provenance.json", {
                    "source": source, "original_metadata": metadata["files"],
                    "plan_sha256": digest_json(plan),
                    "raw_locator": "Drive file ID -> optional inner_tar -> suffix after # in document.raw_file",
                    "raw_verification": "METADATA_ONLY_RAW_PAYLOAD_NOT_REPLAYED"})
            counts = _write_outcomes(target, target/"selection", selection, build_dir, build, work)
            with _connection(work) as con:
                if build:
                    paths = [str(build_dir/p["files"]["documents"]["path"]) for p in build["parts"]]
                    con.read_parquet(paths).create_view("added")
                    _parquet(con, "SELECT doc_id,url FROM added", target/"added_ids.parquet", work)
                else:
                    _parquet(con, "SELECT NULL::BIGINT AS doc_id,NULL::VARCHAR AS url WHERE false", target/"added_ids.parquet", work)
            saved = {"source": source, "plan_sha256": digest_json(plan), "state": "COMPLETE",
                "counts": counts, "build_run": job if build else None,
                "build_snapshot_sha256": build["snapshot_sha256"] if build else None,
                "data_counts": build["counts"] if build else {},
                "added_ids_sha256": sha256_file(target/"added_ids.parquet"),
                "outcomes_sha256": sha256_file(target/"source_outcomes.parquet"), "updated_at": utc_now()}
            atomic_json(receipt, saved)
        available.append(_local_mapping(target/"added_ids.parquet", saved["added_ids_sha256"], work))
        completed.append(saved)
        total.update(saved["counts"])
        status = {"state": "INTAKE_COMPLETE" if index == len(plan["sources"]) else "INTAKE_IN_PROGRESS",
            "plan_sha256": digest_json(plan), "completed_sources": index, "total_sources": len(plan["sources"]),
            "counts": dict(total), "builds": [{"build_run": s["build_run"], "snapshot_sha256": s["build_snapshot_sha256"],
                "snapshot_id": s["source"]["snapshot_id"]} for s in completed if s["build_run"]],
            "scope": "CPU import, quality and exact-span chunks only; not embedded/frozen/promoted or submitted",
            "submission_candidate_changed": False, "updated_at": utc_now()}
        atomic_json(root/"intake_status.json", status)
        print(f"{job} COMPLETE | cumulative outcomes: {dict(total)}", flush=True)
    return status
