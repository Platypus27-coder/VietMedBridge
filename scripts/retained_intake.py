"""Portable retained adapter, embedded outside the pinned chunk producer.

Only audited TAR member ranges are downloaded. New BTC membership and original
crawl/extraction snapshots are distinct, checked identities throughout.
"""
import hashlib
import os
import tarfile
from pathlib import Path

from vietmedbridge import isolated_intake as intake_core
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file, verify_file


class RetainedAdapter:
    def __init__(self, spec, *, root, work_dir, parquet_writer):
        self.spec, self.root, self.work = spec, Path(root), Path(work_dir)
        self.write = parquet_writer
        self.root.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)

    def member(self, reader, kind):
        item = self.spec["members"][kind]
        path = intake_core._inside(self.root, item["path"])
        marker = path.with_suffix(path.suffix+".download.json")
        partial = path.with_suffix(path.suffix+".partial")
        identity = {"file_id": self.spec["drive_file_id"], "archive_bytes": self.spec["archive_bytes"], **item}
        saved = read_json(marker) if marker.exists() else None
        if saved and saved["identity"] != identity:
            raise ValueError("Retained member download identity changed.")
        if path.exists():
            if not saved or path.stat().st_size != item["bytes"]:
                raise ValueError("Unbound or truncated retained member; preserve it for inspection.")
            return path  # SHA checked by _local_mapping when used, never for status only.
        remote_meta = reader.metadata(self.spec["drive_file_id"])
        if int(remote_meta["size"]) != self.spec["archive_bytes"]:
            raise ValueError("Retained archive size changed.")
        if saved and saved["remote"] != remote_meta:
            raise ValueError("Retained archive changed during member download.")
        if not (512 <= item["offset"] and 0 < item["bytes"] <= 2*1024**3
                and item["offset"]+item["bytes"] <= self.spec["archive_bytes"]):
            raise ValueError("Retained member range is outside the audited archive.")
        offset = saved["completed_bytes"] if saved else 0
        if offset and (not partial.exists() or partial.stat().st_size < offset):
            raise ValueError("Retained download checkpoint is missing committed bytes.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with reader.open(self.spec["drive_file_id"], remote_meta) as remote:
            remote.seek(item["offset"]-512)
            header = tarfile.TarInfo.frombuf(remote.read(512), "utf-8", "surrogateescape")
            if header.name != item["path"] or header.size != item["bytes"] or not header.isfile():
                raise ValueError("Retained member TAR header differs from the audited range.")
            # Bound each request, including the large documents member.
            if isinstance(remote, intake_core.DriveRange):
                remote.block_size = 8*1024**2
            with partial.open("r+b" if partial.exists() else "w+b") as outgoing:
                if offset:
                    outgoing.seek(saved["last_start"])
                    if hashlib.sha256(outgoing.read(offset-saved["last_start"])).hexdigest() != saved["last_sha256"]:
                        raise ValueError("Committed retained download block changed.")
                outgoing.truncate(offset)  # discard only this unpublished member's uncommitted tail
                outgoing.seek(offset)
                remote.seek(item["offset"]+offset)
                print(f"RETAINED_MEMBER {kind} | resume {offset:,}/{item['bytes']:,} bytes", flush=True)
                while offset < item["bytes"]:
                    start = offset
                    payload = remote.read(min(8*1024**2, item["bytes"]-offset))
                    if not payload:
                        raise EOFError("Truncated retained member range.")
                    outgoing.write(payload)
                    outgoing.flush()
                    offset += len(payload)
                    saved = {"identity": identity, "remote": remote_meta, "completed_bytes": offset,
                        "last_start": start, "last_sha256": hashlib.sha256(payload).hexdigest()}
                    atomic_json(marker, saved)
                    print(f"RETAINED_DOWNLOAD_CHECKPOINT {kind} | {offset:,}/{item['bytes']:,} bytes", flush=True)
        if reader.metadata(self.spec["drive_file_id"]) != remote_meta:
            raise ValueError("Retained archive changed during download; no member published.")
        verify_file(partial, item["sha256"])
        os.replace(partial, path)
        return path

    def local(self, reader, kind):
        return intake_core._local_mapping(self.member(reader, kind), self.spec["members"][kind]["sha256"], self.work)

    def prepare(self, reader):
        marker = self.root/"metadata.json"
        if marker.exists():
            saved = read_json(marker)
            if saved["portable_origin"] != self.spec:
                raise ValueError("Retained portable source changed; preserve existing checkpoints.")
            for kind in ("frontier", "dataset"):
                verify_file(self.root/saved["files"][kind]["path"], saved["files"][kind]["sha256"])
            return saved
        mapping, raw = self.local(reader, "mapping"), self.local(reader, "raw_catalog")
        descriptor = read_json(self.local(reader, "dataset"))
        if descriptor["snapshot_id"] != self.spec["snapshot_id"]:
            raise ValueError("Retained membership snapshot differs from BTC snapshot.")
        descriptor.update(snapshot_id=self.spec["original_snapshot_id"],
            membership_snapshot_id=self.spec["snapshot_id"],
            adapter="retained-portable-v1; original crawl/extraction snapshots preserved")
        with intake_core._connection(self.work) as con:
            con.read_parquet(str(mapping)).create_view("mapping")
            con.read_parquet(str(raw)).create_view("raw")
            name = intake_core.MEMBERS["frontier"]
            self.write(con, """SELECT m.crawl_url_id,r.fetch_url,count(*) AS doc_count
                FROM mapping m JOIN raw r USING(crawl_url_id)
                GROUP BY m.crawl_url_id,r.fetch_url ORDER BY m.crawl_url_id""", self.root/name, self.work)
        atomic_json(self.root/intake_core.MEMBERS["dataset"], descriptor)
        files = {kind: {"path": name, "sha256": sha256_file(self.root/name), "bytes": (self.root/name).stat().st_size}
                 for kind, name in (("frontier", intake_core.MEMBERS["frontier"]),
                                    ("dataset", intake_core.MEMBERS["dataset"]))}
        source = {k: v for k, v in self.spec.items() if k != "members"}
        source.update(frontier=files["frontier"]["path"], frontier_sha256=files["frontier"]["sha256"],
            portable_source_sha256=digest_json(self.spec))
        metadata = {"source": source, "files": files, "portable_origin": self.spec,
            "raw_verification": "PINNED_METADATA_ONLY_RAW_PAYLOAD_NOT_REPLAYED"}
        atomic_json(marker, metadata)
        return metadata

    def select(self, reader, metadata, links, existing_paths, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        identity = {"metadata": metadata, "links_sha256": sha256_file(links),
                    "existing": [{"sha256": sha256_file(p)} for p in existing_paths]}
        expected = self.spec["official_links_sha256"]
        if identity["links_sha256"] != expected:
            raise ValueError("Retained BTC links checksum differs from the audited package.")
        mapping, raw = self.local(reader, "mapping"), self.local(reader, "raw_catalog")
        with intake_core._connection(self.work) as con:
            for view, path in (("mapping", mapping), ("raw", raw), ("official", links)):
                con.read_parquet(str(path)).create_view(view)
            con.read_parquet([str(p) for p in existing_paths], union_by_name=True).create_view("existing")
            checks = {}
            checks["duplicate_mapping_id"] = con.sql("SELECT count(*)-count(DISTINCT doc_id) FROM mapping").fetchone()[0]
            checks["duplicate_raw_id"] = con.sql("SELECT count(*)-count(DISTINCT crawl_url_id) FROM raw").fetchone()[0]
            queries = {
                "unmapped_native_id": "SELECT count(*) FROM mapping m ANTI JOIN official o ON m.doc_id=o.id AND m.original_url=o.url",
                "mapping_missing_raw": "SELECT count(*) FROM mapping m ANTI JOIN raw r USING(crawl_url_id)",
                "raw_outside_mapping": "SELECT count(*) FROM raw r ANTI JOIN mapping m USING(crawl_url_id)",
                "native_url_conflict": "SELECT count(*) FROM mapping m JOIN existing e USING(doc_id) WHERE m.original_url IS DISTINCT FROM e.url",
                "fetch_url_mismatch": "SELECT count(*) FROM mapping m JOIN raw r USING(crawl_url_id) WHERE split_part(trim(m.original_url),'#',1) IS DISTINCT FROM r.fetch_url",
                "invalid_raw_locator": "SELECT count(*) FROM raw WHERE raw_path IS NULL OR NOT starts_with(raw_path,'raw/') OR contains(raw_path,'..') OR contains(raw_path,chr(92))",
                "invalid_raw_metadata": "SELECT count(*) FROM raw WHERE raw_sha256 IS NULL OR NOT regexp_full_match(raw_sha256,'[0-9a-f]{64}') OR raw_size_bytes IS NULL OR raw_size_bytes<=0",
                "invalid_original_crawl": """SELECT count(*) FROM raw WHERE NOT json_valid(original_crawl_record_json)
                    OR json_extract_string(original_crawl_record_json,'$.status') IS DISTINCT FROM 'SUCCESS_RAW'
                    OR json_extract_string(original_crawl_record_json,'$.snapshot_id') IS DISTINCT FROM original_snapshot_id
                    OR json_extract_string(original_crawl_record_json,'$.raw_sha256') IS DISTINCT FROM raw_sha256
                    OR json_extract_string(original_crawl_record_json,'$.fetch_url') IS DISTINCT FROM fetch_url""",
            }
            for name, query in queries.items():
                checks[name] = con.sql(query).fetchone()[0]
            checks["snapshot_mismatch"] = con.execute("""SELECT count(*) FROM raw WHERE
                new_snapshot_id IS DISTINCT FROM ? OR original_snapshot_id IS DISTINCT FROM ?""",
                [self.spec["snapshot_id"], self.spec["original_snapshot_id"]]).fetchone()[0]
            con.execute("CREATE TABLE mapped AS SELECT crawl_url_id,doc_id,original_url AS url FROM mapping")
            checks["missing_official_alias"] = con.sql("""SELECT count(*) FROM official o JOIN raw r
                ON split_part(trim(o.url),'#',1)=r.fetch_url ANTI JOIN mapped m ON m.doc_id=o.id AND m.url=o.url""").fetchone()[0]
            self._audit(output, checks)
            con.execute("""CREATE TABLE selection AS SELECT m.*, CASE WHEN NOT EXISTS (
                SELECT 1 FROM mapped a WHERE a.crawl_url_id=m.crawl_url_id AND NOT EXISTS (
                    SELECT 1 FROM existing e WHERE e.doc_id=a.doc_id AND e.url=a.url))
                THEN 'REUSED_EXISTING_DOCUMENT' ELSE 'QUEUED_FOR_PIPELINE' END AS action FROM mapped m""")
            con.execute("CREATE VIEW needed AS SELECT DISTINCT crawl_url_id FROM selection WHERE action='QUEUED_FOR_PIPELINE'")
            counts = dict(con.sql("SELECT action,count(*) FROM selection GROUP BY 1").fetchall())
            print(f"RETAINED_SELECTION | {counts} native IDs | original snapshot retained", flush=True)
            if counts.get("QUEUED_FOR_PIPELINE", 0):
                documents = self.local(reader, "documents")
                con.read_parquet(str(documents)).create_view("portable_documents")
                checks["duplicate_extraction"] = con.sql("SELECT count(*)-count(DISTINCT crawl_url_id) FROM portable_documents").fetchone()[0]
                checks["extraction_binding"] = con.sql("""SELECT count(*) FROM portable_documents e LEFT JOIN raw r USING(crawl_url_id)
                    WHERE r.crawl_url_id IS NULL OR e.snapshot_id IS DISTINCT FROM r.original_snapshot_id
                    OR e.raw_sha256 IS DISTINCT FROM r.raw_sha256 OR e.extract_status IS DISTINCT FROM r.extract_status
                    OR e.extract_status IS DISTINCT FROM 'EXTRACT_SUCCESS'""").fetchone()[0]
                checks["missing_successful_extraction"] = con.sql("""SELECT count(*) FROM raw r ANTI JOIN
                    portable_documents e USING(crawl_url_id) WHERE r.extract_status='EXTRACT_SUCCESS'""").fetchone()[0]
                checks["invalid_source_length"] = con.sql("""SELECT count(*) FROM portable_documents WHERE text IS NULL
                    OR length(trim(text))=0 OR char_count IS DISTINCT FROM length(text)""").fetchone()[0]
                self._audit(output, checks)
                con.execute("""CREATE VIEW normalized_extracted AS SELECT r.crawl_url_id,r.raw_sha256,
                    r.original_snapshot_id AS snapshot_id,
                    CASE WHEN e.extract_status='EXTRACT_SUCCESS' AND (
                        upper(coalesce(e.quality_tier,'')) IN ('LOW','QUARANTINE') OR
                        upper(coalesce(r.quality_tier,'')) IN ('LOW','QUARANTINE'))
                        THEN 'RETAINED_QUALITY_REVIEW_'||CASE
                            WHEN upper(coalesce(e.quality_tier,'')) IN ('LOW','QUARANTINE')
                            THEN upper(e.quality_tier) ELSE upper(r.quality_tier) END
                        ELSE coalesce(e.extract_status,r.extract_status,'MISSING_EXTRACTION') END AS extract_status,
                    e.text,e.title,e.char_count,e.text_markdown,e.language,e.language_method,e.language_confidence,
                    e.extractor,e.extractor_version,e.source_issue FROM raw r JOIN needed n USING(crawl_url_id)
                    LEFT JOIN portable_documents e USING(crawl_url_id)""")
            else:
                con.execute("""CREATE VIEW normalized_extracted AS SELECT NULL::VARCHAR crawl_url_id,
                    NULL::VARCHAR raw_sha256,NULL::VARCHAR snapshot_id,NULL::VARCHAR extract_status,
                    NULL::VARCHAR text,NULL::VARCHAR title,NULL::BIGINT char_count,NULL::VARCHAR text_markdown,
                    NULL::VARCHAR language,NULL::VARCHAR language_method,NULL::DOUBLE language_confidence,
                    NULL::VARCHAR extractor,NULL::VARCHAR extractor_version,NULL::VARCHAR source_issue WHERE false""")
            con.read_parquet(str(self.root/metadata["files"]["frontier"]["path"])).create_view("frontier")
            queries = {
                "frontier": "SELECT f.* FROM frontier f JOIN needed n USING(crawl_url_id) ORDER BY f.crawl_url_id",
                "crawl": """SELECT r.crawl_url_id,r.fetch_url,'SUCCESS_RAW' AS status,r.final_url,r.content_type,
                    r.raw_path,r.raw_sha256,r.raw_size_bytes,
                    try_cast(json_extract_string(r.original_crawl_record_json,'$.elapsed_ms') AS DOUBLE) AS elapsed_ms,
                    try_cast(json_extract_string(r.original_crawl_record_json,'$.http_status') AS INTEGER) AS http_status,
                    r.crawl_timestamp,r.original_snapshot_id AS snapshot_id,r.new_snapshot_id
                    FROM raw r JOIN needed n USING(crawl_url_id) ORDER BY r.crawl_url_id""",
                "extracted": "SELECT * FROM normalized_extracted ORDER BY crawl_url_id",
            }
            files = {}
            for kind, query in queries.items():
                name = intake_core.MEMBERS[kind]
                checksum = self.write(con, query, output/name, self.work)
                files[kind] = {"path": name, "sha256": checksum}
            name = intake_core.MEMBERS["dataset"]
            files["dataset"] = {"path": name, "sha256": intake_core.publish_file(self.root/metadata["files"]["dataset"]["path"], output/name)}
            checksum = self.write(con, "SELECT * FROM selection ORDER BY doc_id", output/"source_selection.parquet", self.work)
            files["selection"] = {"path": "source_selection.parquet", "sha256": checksum}
            result = {"identity": identity, "counts": counts, "official_ids": sum(counts.values()), "files": files}
            atomic_json(output/"selection.json", result)
            return result

    def _audit(self, output, checks):
        report = {"passed": not any(checks.values()), "checks": checks,
            "extraction_payload_checked": "extraction_binding" in checks,
            "membership_snapshot_id": self.spec["snapshot_id"], "original_snapshot_id": self.spec["original_snapshot_id"],
            "raw_verification": "PINNED_METADATA_BINDINGS_ONLY_RAW_PAYLOAD_NOT_REPLAYED"}
        atomic_json(output/"retained_input_audit.json", report)
        if not report["passed"]:
            raise ValueError(f"Retained integrity failed: {checks}")
