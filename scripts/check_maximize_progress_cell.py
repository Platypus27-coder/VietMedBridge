"""Paste into a second Colab CPU session; inspect one durable SQL snapshot only."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime, timezone

CAMPAIGN = "btc-maximize-71be0e12-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def checkpoint_progress(root, work, report=print):
    """No source TAR, recovered Parquet, corpus, vector, or writer DB is opened."""
    root, work = Path(root), Path(work)
    pointer = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    if _digest({k: v for k, v in pointer.items() if k != "manifest_sha256"}) != pointer.get("manifest_sha256"):
        raise ValueError("Checkpoint manifest checksum mismatch.")
    if pointer["binding"].get("policy") != "collected-delta-raw-recovery-v1":
        raise ValueError("Unexpected recovery campaign policy.")
    name = pointer["path"]
    checksum = pointer["sha256"]
    if (len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)
            or name != "sql-" + checksum + ".gz"):
        raise ValueError("Unexpected snapshot path.")
    work.mkdir(parents=True, exist_ok=True)
    report("Đọc một SQL checkpoint trên Drive; không mở lại TAR/corpus/vector.")
    with tempfile.TemporaryDirectory(prefix="progress-", dir=work) as temporary:
        compressed, database = Path(temporary) / "snapshot.gz", Path(temporary) / "snapshot.sqlite"
        sha = hashlib.sha256()
        with (root / name).open("rb") as incoming, compressed.open("wb") as out:
            for block in iter(lambda: incoming.read(1024 ** 2), b""):
                sha.update(block)
                out.write(block)
        if sha.hexdigest() != pointer["sha256"]:
            raise ValueError("Saved SQL snapshot checksum mismatch.")
        sha, size = hashlib.sha256(), 0
        with gzip.open(compressed, "rb") as incoming, database.open("wb") as out:
            for block in iter(lambda: incoming.read(1024 ** 2), b""):
                size += len(block)
                if size > 2 * 1024 ** 3:
                    raise ValueError("Unexpected SQL metadata size; audit stopped.")
                sha.update(block)
                out.write(block)
        if sha.hexdigest() != pointer["sqlite_sha256"]:
            raise ValueError("Restored SQL checksum mismatch.")
        con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            con.execute("PRAGMA query_only=ON")
            con.execute("PRAGMA cache_size=-16384")
            contract = con.execute("SELECT value FROM contract").fetchone()
            if contract != (_digest(pointer["binding"]),):
                raise ValueError("SQL belongs to another input/policy.")
            seed = con.execute("SELECT receipt FROM completed WHERE id='seed'").fetchone()
            if not seed:
                raise ValueError("Seed membership has not been committed yet.")
            original = json.loads(seed[0])["counts"]
            expected = sum(original.values())
            reused = original.get("REUSED_VALIDATED_CONTENT", 0)
            counts = dict(con.execute("""SELECT t.state,count(*) FROM aliases a
                JOIN todo t USING(fetch) GROUP BY t.state"""))
            alias_count = con.execute("SELECT count(*) FROM aliases").fetchone()[0]
            if sum(counts.values()) != alias_count or alias_count + reused != expected:
                raise ValueError("Official ID accounting mismatch; do not advance to GPU.")
            raw_ids = dict(con.execute("""SELECT t.state,count(*) FROM aliases a JOIN todo t USING(fetch)
                WHERE json_extract(t.candidate,'$.recovery_root')='maximize' GROUP BY t.state"""))
            frozen = {key: json.loads(value) for key, value in con.execute(
                "SELECT id,receipt FROM completed WHERE kind='freeze'")}
            cursor = con.execute("SELECT id,receipt FROM completed WHERE kind='cursor' ORDER BY rowid DESC LIMIT 1").fetchone()
            raw_complete = bool(con.execute("SELECT 1 FROM completed WHERE id='raw-pass'").fetchone())
            result = {
                "state": "DURABLE_CHECKPOINT_PROGRESS",
                "checkpoint_updated_at": pointer["updated_at"],
                "checkpoint_manifest_sha256": pointer["manifest_sha256"],
                "official_ids": expected,
                "reused_validated_ids": reused,
                "counts_by_official_id": counts,
                "new_raw_text_ids_by_state": raw_ids,
                "usable_text_ids_including_pending_chunks": reused + counts.get("CHUNKED", 0) + counts.get("TEXT_READY", 0),
                "raw_pass_complete": raw_complete,
                "completed_raw_sources": con.execute("SELECT count(*) FROM completed WHERE kind='raw_source'").fetchone()[0],
                "raw_attempts_by_result": dict(con.execute("SELECT result,count(*) FROM raw_results GROUP BY result")),
                "frozen_delta_receipts": frozen,
                "latest_raw_cursor": {"id": cursor[0], **json.loads(cursor[1])} if cursor else None,
                "official_id_accounting_complete": True,
                "all_official_content_ready": reused + counts.get("CHUNKED", 0) == expected,
                "scope": "Saved SQL metadata only; text/vector payloads were not revalidated; embedding coverage is separate.",
                "audited_at": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            con.close()
    return result


def run_colab_check():
    if not Path("/var/colab/hostname").is_file():
        raise RuntimeError("Chỉ chạy cell này trên Colab CPU, không tải dữ liệu về máy cá nhân.")
    from google.colab import drive
    drive.mount("/content/drive")
    candidates = [Path("/content/drive/MyDrive/VietMedBridge"),
                  Path("/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge")]
    root = next((p / "data_staging" / CAMPAIGN for p in candidates
                 if (p / "data_staging" / CAMPAIGN / "checkpoint.json").is_file()), None)
    if root is None:
        raise FileNotFoundError("Chưa thấy checkpoint maximize trong Drive đang kết nối.")
    result = checkpoint_progress(root, Path("/content/vmb_progress_work"),
                                 report=lambda message: print(message, flush=True))
    # This separate report never overwrites the writer's status, pointer, or SQL.
    destination = root / "coverage_progress.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("Đã lưu:", destination, flush=True)


if __name__ == "__main__":
    run_colab_check()
