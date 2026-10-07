"""Global dedup reduction and human-readable audit of one validated snapshot."""

from __future__ import annotations

import html
import json
import shutil
import tempfile
from pathlib import Path

from .artifacts import atomic_json, code_fingerprint, digest_json, publish_file, read_json, sha256_file, utc_now, verify_file
from .dataset import _connection
from .inventory import copy_parquet
from .validation import artifact_paths, open_views, validate_snapshot


def _rows(con, sql):
    cursor = con.execute(sql)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _highlight(text, start, end):
    return html.escape(text[:start]) + "<mark>" + html.escape(text[start:end]) + "</mark>" + html.escape(text[end:])


def health_report(build_dir: str | Path, *, official_links: str | Path,
                  crawl_dir: str | Path | None = None, work_dir: str | Path | None = None,
                  audit_size: int = 50, baseline_report: str | Path | None = None,
                  drift_relative_threshold: float | None = None) -> dict:
    if not 1 <= audit_size <= 500:
        raise ValueError("Use an audit size between 1 and 500.")
    if drift_relative_threshold is not None and drift_relative_threshold <= 0:
        raise ValueError("Calibrated drift threshold must be positive.")
    root = Path(build_dir)
    build = read_json(root / "build.json")
    if build["snapshot_sha256"] != digest_json({"signature": build["signature"], "parts": build["parts"]}):
        raise ValueError("Build snapshot identity changed.")
    config = read_json(root / "config.json")
    if config["official_links_sha256"] != sha256_file(official_links):
        raise ValueError("Official corpus file differs from the validated build input.")
    integrity = validate_snapshot(root, build, official_links=official_links, work_dir=work_dir)
    if not integrity["passed"]:
        raise ValueError("Hard integrity gate failed; do not generate/promote a valid health report.")
    target = root / "reports" / build["snapshot_sha256"][:16]
    target.mkdir(parents=True, exist_ok=True)
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vmb-health-", dir=work_dir) as temporary:
        with _connection(temporary) as con:
            open_views(con, root, build)
            con.execute("""CREATE VIEW aliases AS SELECT canonical_content_id, doc_id,
                content_hash, source_text_sha256, url FROM documents""")
            # These mappings span ALL selected shards, not just the current shard.
            canonical = Path(temporary) / "canonical_aliases.parquet"
            representations = Path(temporary) / "representation_aliases.parquet"
            copy_parquet(con, "SELECT * FROM aliases ORDER BY canonical_content_id, doc_id", canonical)
            copy_parquet(con, """SELECT retrieval_representation_hash, chunk_id, doc_id, source_text_sha256,
                representation_builder_version FROM children
                ORDER BY retrieval_representation_hash, chunk_id""", representations)
            alias_count, unique_alias_ids, canonical_count = con.execute(
                "SELECT count(*), count(DISTINCT doc_id), count(DISTINCT canonical_content_id) FROM aliases"
            ).fetchone()
            if alias_count != unique_alias_ids or alias_count != build["counts"].get("documents", 0):
                raise ValueError("Canonical aliases must represent each parsed official ID exactly once.")
            report = {
                "snapshot_sha256": build["snapshot_sha256"], "created_at": utc_now(),
                "integrity": integrity, "manual_review": "NOT_REVIEWED",
                "retrieval_evaluation": "NOT_RUN_NO_INDEX_OR_QRELS",
                "coverage": _rows(con, """SELECT count(*) AS requested, count(*) FILTER (WHERE status='ok') AS crawl_success,
                    sum(raw_bytes) AS raw_body_bytes, approx_quantile(elapsed_ms, 0.5) AS fetch_p50_ms,
                    approx_quantile(elapsed_ms, 0.95) AS fetch_p95_ms,
                    approx_quantile(raw_bytes, 0.95) FILTER (WHERE status='ok') AS body_p95_bytes FROM ledger""")[0],
                "documents": _rows(con, """SELECT count(*) AS parsed, avg(char_count) AS mean_chars,
                    approx_quantile(char_count, 0.5) AS median_chars, approx_quantile(char_count, 0.95) AS p95_chars
                    FROM documents""")[0],
                "chunks": _rows(con, """SELECT count(*) AS children, avg(token_count) AS mean_source_tokens,
                    approx_quantile(token_count, 0.95) AS p95_source_tokens,
                    avg(dense_token_count) AS mean_dense_tokens_including_special,
                    count(DISTINCT retrieval_representation_hash) AS distinct_representations FROM children""")[0],
                "distributions": {},
                "quality_tiers": _rows(con, "SELECT quality_tier, count(*) AS documents FROM documents GROUP BY quality_tier ORDER BY quality_tier"),
                "languages": _rows(con, "SELECT language, language_method, count(*) AS documents FROM documents GROUP BY language, language_method ORDER BY documents DESC"),
                "failure_reasons": _rows(con, "SELECT phase, reason_code, count(*) AS documents FROM failures GROUP BY phase, reason_code ORDER BY documents DESC"),
                "quality_reasons": _rows(con, "SELECT reason, count(*) AS documents FROM (SELECT unnest(quality_flags) AS reason FROM documents) GROUP BY reason ORDER BY documents DESC"),
                "by_domain": _rows(con, """WITH p AS (SELECT domain, count(*) AS parsed,
                    avg(char_count) AS mean_chars, approx_quantile(char_count,0.95) AS p95_chars FROM documents GROUP BY domain),
                    f AS (SELECT domain, count(*) AS failed FROM failures GROUP BY domain)
                    SELECT l.domain, count(*) AS requested, count(*) FILTER (WHERE status='ok') AS crawl_success,
                    coalesce(p.parsed,0) AS parsed, coalesce(f.failed,0) AS failed,
                    p.mean_chars, p.p95_chars FROM ledger l
                    LEFT JOIN p USING (domain) LEFT JOIN f USING (domain)
                    GROUP BY l.domain, p.parsed, f.failed, p.mean_chars, p.p95_chars ORDER BY requested DESC, l.domain"""),
                "by_content_type": _rows(con, """WITH p AS (SELECT content_type, count(*) AS parsed FROM documents GROUP BY content_type)
                    SELECT l.content_type, count(*) AS requested,
                    count(*) FILTER (WHERE status='ok') AS crawl_success, coalesce(p.parsed,0) AS parsed FROM ledger l
                    LEFT JOIN p USING (content_type) GROUP BY l.content_type, p.parsed ORDER BY requested DESC"""),
                "domain_languages": _rows(con, "SELECT domain, language, count(*) AS parsed FROM documents GROUP BY domain, language ORDER BY domain, parsed DESC"),
                "dedup": {"canonical_contents": canonical_count, "official_document_ids": alias_count,
                          "exact_normalized_duplicate_documents": alias_count - canonical_count,
                          "aliases_preserved": alias_count == unique_alias_ids},
                "quantile_method": "DuckDB approx_quantile (T-Digest); bounded aggregate state",
                "limitations": ["Language stats cover extracted content; failed-source language is unknown",
                                 "LOW quality documents remain eligible; quality is not relevance",
                                 "Pilot rates and bytes are not population-weighted full-corpus forecasts",
                                 "Representation grouping requires the same model/tokenizer/encoding policy before embedding reuse"],
            }
            metrics = {
                "source_chars": ("documents", "char_count"),
                "paragraphs_per_doc": ("documents", "paragraph_count"),
                "sections_per_doc": ("documents", "section_count"),
                "child_tokens": ("children", "token_count"),
                "parent_tokens": ("parents", "token_count"),
                "children_per_doc": ("(SELECT d.doc_id, count(c.chunk_id) AS value FROM documents d LEFT JOIN children c USING(doc_id) GROUP BY d.doc_id)", "value"),
            }
            for metric, (table, column) in metrics.items():
                report["distributions"][metric] = _rows(con, f"""SELECT avg({column}) AS mean,
                    approx_quantile({column},0.05) AS p05, approx_quantile({column},0.5) AS p50,
                    approx_quantile({column},0.95) AS p95, approx_quantile({column},0.99) AS p99 FROM {table}""")[0]
            # Stable, diverse selection plus very short/long/noisy cases. Never pull all text into RAM.
            audit = _rows(con, f"""WITH ranked AS (
                SELECT doc_id, row_number() OVER (PARTITION BY domain, language, quality_tier ORDER BY doc_id) AS r,
                    row_number() OVER (ORDER BY char_count DESC, doc_id) AS longest,
                    row_number() OVER (ORDER BY char_count, doc_id) AS shortest
                FROM documents), selected AS (SELECT doc_id FROM ranked
                ORDER BY CASE WHEN r=1 OR longest<=5 OR shortest<=5 THEN 0 ELSE 1 END,
                    least(longest, shortest), hash(doc_id) LIMIT {int(audit_size)})
                SELECT d.doc_id, url, title, language, quality_tier, quality_flags, source_text_sha256,
                    substr(source_text, 1, 6000) AS preview, char_count, raw_file
                FROM selected JOIN documents d USING(doc_id) ORDER BY d.doc_id""")
            entries = []
            for doc in audit:
                child = _rows(con, f"SELECT start_char, end_char FROM children WHERE doc_id={int(doc['doc_id'])} ORDER BY chunk_order LIMIT 1")
                preview = doc["preview"]
                highlighted = _highlight(preview, child[0]["start_char"], min(child[0]["end_char"], len(preview))) if child and child[0]["start_char"] < len(preview) else html.escape(preview)
                url = doc["url"]
                link = f'<a href="{html.escape(url, quote=True)}" rel="noopener noreferrer">source</a>' if url.startswith(("https://", "http://")) else html.escape(url)
                entries.append(f"<article><h2>{doc['doc_id']} — {html.escape(doc['title'])}</h2><p>{link} | {html.escape(doc['language'])} | {doc['quality_tier']}</p><p>{html.escape(', '.join(doc['quality_flags']))}</p><pre>{highlighted}</pre><small>source SHA-256: {doc['source_text_sha256']} | preview only; full source {doc['char_count']} chars</small></article>")
            audit_html = Path(temporary) / "audit.html"
            audit_html.write_text("<!doctype html><meta charset='utf-8'><title>VietMedBridge source audit</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#f5f7fa}article{background:white;padding:24px;margin:20px 0;border-radius:12px}pre{white-space:pre-wrap;line-height:1.6}mark{background:#ffe8a0}small{overflow-wrap:anywhere}</style><h1>Source audit — human review pending</h1><p>First child is highlighted. This report is a preview, not a relevance label.</p>" + "".join(entries), encoding="utf-8")
            review_candidates = Path(temporary) / "golden_candidates.json"
            review_candidates.write_text(json.dumps([
                {key: doc[key] for key in ("doc_id", "url", "source_text_sha256", "raw_file", "language")}
                | {"expected_key_snippets": [], "annotation_status": "PENDING_HUMAN_REVIEW"} for doc in audit
            ], ensure_ascii=False, indent=2), encoding="utf-8")
        files = {}
        for kind, local in (("canonical_aliases", canonical), ("representation_aliases", representations),
                            ("audit", audit_html), ("golden_candidates", review_candidates)):
            relative = (Path("reports") / build["snapshot_sha256"][:16] / local.name).as_posix()
            files[kind] = {"path": relative, "sha256": publish_file(local, root / relative)}
    raw_compressed = None
    if crawl_dir is not None:
        raw_compressed = 0
        for part in build["parts"]:
            raw_path = Path(crawl_dir) / part["raw_file"]
            verify_file(raw_path, part["input_raw_sha256"])
            raw_compressed += raw_path.stat().st_size
    report["storage"] = {"selected_compressed_raw_bytes": raw_compressed,
                         "selected_processed_bytes": sum((root / entry["path"]).stat().st_size for part in build["parts"] for entry in part["files"].values()),
                         "local_free_bytes": shutil.disk_usage(work_dir or root).free}
    if baseline_report:
        baseline = read_json(baseline_report)
        changes = []
        for metric, stats in report["distributions"].items():
            old = baseline.get("distributions", {}).get(metric, {}).get("mean")
            new = stats["mean"]
            delta = (new - old) / old if old and new is not None else None
            changes.append({"metric": metric, "baseline_mean": old, "candidate_mean": new,
                            "relative_change": delta,
                            "warning": bool(delta is not None and drift_relative_threshold is not None and abs(delta) > drift_relative_threshold)})
        report["distribution_comparison"] = {
            "baseline_snapshot_sha256": baseline["snapshot_sha256"],
            "relative_threshold": drift_relative_threshold, "changes": changes,
            "status": "WARNING" if any(row["warning"] for row in changes) else "REVIEW_REQUIRED",
            "note": "Different sample/domain composition may change distributions; calibrate threshold in pilot.",
        }
    report["files"] = files
    atomic_json(target / "health.json", report)
    return report


def freeze_candidate(build_dir: str | Path, health: dict, *, golden_report: dict) -> dict:
    root = Path(build_dir)
    build = read_json(root / "build.json")
    if build.get("input_kind") == "EXTERNAL_EXTRACTION" and not build.get("selected_range_complete"):
        raise ValueError("Complete every external import shard before freezing the 100k candidate.")
    if build["snapshot_sha256"] != digest_json({"signature": build["signature"], "parts": build["parts"]}):
        raise ValueError("Build snapshot identity changed.")
    if health["snapshot_sha256"] != build["snapshot_sha256"] or not health["integrity"]["passed"]:
        raise ValueError("Health report does not validate this snapshot.")
    if health["integrity"]["official_membership"] != "VERIFIED":
        raise ValueError("Official ID/URL membership must be verified before freeze.")
    if not golden_report.get("passed") or golden_report.get("code_sha256") != code_fingerprint():
        raise ValueError("Golden regression must pass with the current code before freeze.")
    config = read_json(root / "config.json")
    if config["code_sha256"] != code_fingerprint():
        raise ValueError("Build was generated by different code. Rebuild in a new run before freeze.")
    if golden_report.get("tokenizer") != config["tokenizer"]:
        raise ValueError("Golden and build tokenizer policies differ.")
    if golden_report.get("chunking") != config["chunks"]:
        raise ValueError("Golden and build chunking policies differ.")
    if build["counts"].get("documents", 0) == 0:
        raise ValueError("No extracted documents: inspect failures before freezing a retrieval candidate.")
    for kind in ("documents", "children", "parents", "sections", "failures", "ledger"):
        artifact_paths(root, build, kind)
    for entry in health["files"].values():
        verify_file(root / entry["path"], entry["sha256"])
    # The immutable snapshot is an input to index experiments, not an approved retrieval release.
    candidate = {**build, "state": "FROZEN_CANDIDATE", "health": health,
                 "golden": golden_report, "promotion": "PENDING_RETRIEVAL_RESOURCE_AND_HUMAN_REVIEW"}
    candidate["candidate_manifest_sha256"] = digest_json(candidate)
    atomic_json(root / f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json", candidate)
    return candidate
