"""Immutable Hub snapshots and bounded-memory Parquet audits."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RevisionNotFoundError

from .artifacts import atomic_json, read_json, runtime_versions, sha256_file, utc_now, verify_file

DATASET_ID = "AIGuruTinix/ViBioMIR"
DATASET_REVISION = "0148f6f80ffafed5c005af6d506ccfd9d3fb47a7"


def snapshot_dataset(data_root: str | Path, revision: str = DATASET_REVISION) -> dict:
    root = Path(data_root)
    # Reuse a verified immutable snapshot without requiring the Hub on every rerun.
    pinned_target = root / "raw" / "hub" / revision
    pinned_manifest = pinned_target / "snapshot.json"
    if pinned_manifest.exists():
        manifest = read_json(pinned_manifest)
        for entry in manifest["files"].values():
            verify_file(root / entry["path"], entry["sha256"])
        atomic_json(root / "raw" / "snapshot.json", manifest)
        return manifest

    api = HfApi()
    resolved_from = revision
    try:
        info = api.dataset_info(DATASET_ID, revision=revision, files_metadata=True)
    except RevisionNotFoundError:
        # Hub repositories can be force-pushed. Fall forward only when an immutable
        # requested revision disappeared, then record the actual SHA for replay.
        info = api.dataset_info(DATASET_ID, revision="main", files_metadata=True)
        resolved_from = "main"
        warnings.warn(
            f"Dataset revision {revision!r} no longer exists; using current main "
            f"({info.sha}). The resolved SHA is recorded in the snapshot manifest.",
            RuntimeWarning,
            stacklevel=2,
        )
    target = root / "raw" / "hub" / info.sha
    target.mkdir(parents=True, exist_ok=True)
    existing = target / "snapshot.json"
    if existing.exists():
        manifest = read_json(existing)
        for entry in manifest["files"].values():
            verify_file(root / entry["path"], entry["sha256"])
        atomic_json(root / "raw" / "snapshot.json", manifest)
        return manifest
    remote_files = {item.rfilename: item for item in info.siblings}
    files = {}
    for name in ("query.parquet", "links_corpus.parquet", "README.md"):
        path = Path(hf_hub_download(
            DATASET_ID, name, repo_type="dataset", revision=info.sha, local_dir=target,
        ))
        checksum = sha256_file(path)
        remote = remote_files[name]
        expected = getattr(remote.lfs, "sha256", None) if remote.lfs else None
        if expected and checksum != expected:
            raise ValueError(f"Hub LFS checksum mismatch: {name}")
        entry = {"path": path.relative_to(root).as_posix(), "sha256": checksum,
                 "bytes": path.stat().st_size}
        if name.endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            entry.update(rows=parquet.metadata.num_rows, schema=str(parquet.schema_arrow))
        files[name] = entry
    manifest = {
        "dataset_id": DATASET_ID, "revision": info.sha,
        "requested_revision": revision, "resolved_from": resolved_from,
        "created_at": utc_now(),
        "configs": {"query": {"split": "train", "file": "query.parquet"},
                    "corpus": {"split": "train", "file": "links_corpus.parquet"}},
        "files": files, "runtime": runtime_versions(),
    }
    atomic_json(existing, manifest)
    atomic_json(root / "raw" / "snapshot.json", manifest)
    return manifest


def load_snapshot(data_root: str | Path, verify: bool = True) -> dict:
    root = Path(data_root)
    snapshot = read_json(root / "raw" / "snapshot.json")
    if verify:
        for entry in snapshot["files"].values():
            verify_file(root / entry["path"], entry["sha256"])
    return snapshot


def parquet_path(data_root: str | Path, name: str) -> Path:
    snapshot = load_snapshot(data_root, verify=False)
    entry = snapshot["files"][name]
    path = Path(data_root) / entry["path"]
    verify_file(path, entry["sha256"])
    return path


def require_schema(path: str | Path, text_column: str) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    if "id" not in schema.names or text_column not in schema.names:
        raise ValueError(f"Expected columns id/{text_column}, got {schema.names}")
    if not pa.types.is_integer(schema.field("id").type):
        raise ValueError("Official IDs must be integers; no reindexing/coercion is allowed.")
    if not (pa.types.is_string(schema.field(text_column).type)
            or pa.types.is_large_string(schema.field(text_column).type)):
        raise ValueError(f"Expected string column: {text_column}")


def _connection(temp_dir: str | Path, memory_limit: str = "512MB"):
    connection = duckdb.connect()
    connection.execute("SET memory_limit = ?", [memory_limit])
    connection.execute("SET temp_directory = ?", [str(temp_dir)])
    connection.execute("SET threads = 2")
    return connection


def audit_dataset(data_root: str | Path, *, work_dir: str | Path | None = None,
                  sample_domains: int = 30, per_domain: int = 3) -> dict:
    """Count IDs in DuckDB with disk spill, without a Python set of millions of IDs."""
    if sample_domains < 1 or per_domain < 1:
        raise ValueError("Sample sizes must be positive.")
    root = Path(data_root)
    snapshot = load_snapshot(root)
    queries = root / snapshot["files"]["query.parquet"]["path"]
    corpus = root / snapshot["files"]["links_corpus.parquet"]["path"]
    require_schema(queries, "query")
    require_schema(corpus, "url")
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = {"created_at": utc_now(), "dataset_revision": snapshot["revision"],
              "corpus_sha256": snapshot["files"]["links_corpus.parquet"]["sha256"]}
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vmb-audit-", dir=work_dir) as temporary:
        with _connection(temporary) as connection:
            for name, path, column in (("query", queries, "query"), ("corpus", corpus, "url")):
                # column is an internal constant; paths are bound SQL parameters.
                result = connection.execute(
                    f"""SELECT count(*), count(DISTINCT id), count(*) FILTER (WHERE id IS NULL),
                        min(id), max(id), count(*) FILTER (
                        WHERE {column} IS NULL OR length(trim({column})) = 0)
                        FROM read_parquet(?)""", [str(path)],
                ).fetchone()
                report[name] = dict(zip(
                    ("rows", "unique_ids", "null_ids", "min_id", "max_id", "empty_text"), result,
                ))
                if result[0] != result[1] or result[2] or result[5]:
                    raise ValueError(f"Invalid {name} dataset: {report[name]}")
            invalid_urls = connection.execute(
                """SELECT count(*) FROM read_parquet(?)
                   WHERE NOT regexp_matches(url, '(?i)^https?://[^/?#\\s]+')""", [str(corpus)],
            ).fetchone()[0]
            report["corpus"]["invalid_http_urls"] = invalid_urls
            report["corpus"]["unique_urls"] = connection.execute(
                "SELECT count(DISTINCT url) FROM read_parquet(?)", [str(corpus)],
            ).fetchone()[0]
            report["corpus"]["duplicate_url_rows"] = (
                report["corpus"]["rows"] - report["corpus"]["unique_urls"]
            )
            domains = connection.execute(
                """SELECT lower(regexp_extract(url, '^https?://([^/?#]+)', 1)) AS domain,
                          count(*) AS rows
                   FROM read_parquet(?) GROUP BY domain ORDER BY rows DESC, domain""",
                [str(corpus)],
            ).fetch_arrow_table()
            pq.write_table(domains, reports / "domains.parquet", compression="zstd")
            report["top_domains"] = domains.slice(0, sample_domains).to_pylist()
            lengths = connection.execute(
                """SELECT min(length(query)), quantile_cont(length(query), 0.5),
                          quantile_cont(length(query), 0.95), max(length(query))
                   FROM read_parquet(?)""", [str(queries)],
            ).fetchone()
            report["query"]["length_characters"] = dict(zip(("min", "median", "p95", "max"), lengths))
            report["query"]["has_reference_labels"] = any(
                name in pq.ParquetFile(queries).schema_arrow.names
                for name in ("relevant_docs", "relevant_chunks")
            )
            chosen_domains = [item["domain"] for item in report["top_domains"] if item["domain"]]
            sample = connection.execute(
                """WITH candidates AS (
                      SELECT id, url, lower(regexp_extract(url, '^https?://([^/?#]+)', 1)) AS domain
                      FROM read_parquet(?)
                   ), ranked AS (
                      SELECT *, row_number() OVER (PARTITION BY domain ORDER BY id) AS position
                      FROM candidates WHERE domain IN (SELECT unnest(?))
                   )
                   SELECT id, url FROM ranked WHERE position <= ? ORDER BY domain, id""",
                [str(corpus), chosen_domains, per_domain],
            ).fetch_arrow_table()
            pq.write_table(sample, reports / "sample_links.parquet", compression="zstd")
    report["sample"] = {
        "path": "reports/sample_links.parquet", "rows": sample.num_rows,
        "sha256": sha256_file(reports / "sample_links.parquet"),
        "method": "first IDs per top domain; crawl smoke test, not an evaluation set",
    }
    report["domain_report_sha256"] = sha256_file(reports / "domains.parquet")
    atomic_json(reports / "dataset_audit.json", report)
    return report


def validate_link_subset(subset: str | Path, official: str | Path,
                         *, work_dir: str | Path | None = None) -> None:
    """A crawl subset must retain the original ID/URL pair, including duplicate URLs."""
    require_schema(subset, "url")
    with tempfile.TemporaryDirectory(prefix="vmb-subset-", dir=work_dir) as temporary:
        with _connection(temporary) as connection:
            counts = connection.execute(
                "SELECT count(*), count(DISTINCT id) FROM read_parquet(?)", [str(subset)],
            ).fetchone()
            if counts[0] != counts[1]:
                raise ValueError("Crawl input contains duplicated/null IDs.")
            invalid = connection.execute(
                """SELECT count(*) FROM read_parquet(?) s
                   LEFT JOIN read_parquet(?) o ON s.id = o.id
                   WHERE o.id IS NULL OR s.url IS DISTINCT FROM o.url""",
                [str(subset), str(official)],
            ).fetchone()[0]
            if invalid:
                raise ValueError(f"{invalid} crawl rows do not match official ID/URL pairs.")


def iter_link_shards(path: str | Path, *, start: int = 0, stop: int | None = None,
                     shard_size: int = 128):
    """Offsets are physical Parquet row positions, never replacement document IDs."""
    if start < 0 or shard_size < 1 or (stop is not None and stop < start):
        raise ValueError("Invalid row range or shard size.")
    parquet = pq.ParquetFile(path)
    stop = parquet.metadata.num_rows if stop is None else min(stop, parquet.metadata.num_rows)
    cursor, rows, first = 0, [], start
    for batch in parquet.iter_batches(batch_size=8192, columns=["id", "url"]):
        end = cursor + batch.num_rows
        if cursor >= stop:
            break
        if end > start:
            lower, upper = max(0, start - cursor), min(batch.num_rows, stop - cursor)
            for row in batch.slice(lower, upper - lower).to_pylist():
                rows.append(row)
                if len(rows) == shard_size:
                    yield first, rows
                    first += len(rows)
                    rows = []
        cursor = end
    if rows:
        yield first, rows


def url_host(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username:
        raise ValueError("Expected a public HTTP(S) source URL without credentials.")
    return parts.hostname.lower()
