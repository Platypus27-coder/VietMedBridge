"""Replay captured browser articles into a versioned extraction review bundle.

This stage performs no requests and does not change a crawl/canonical corpus.
Every selected capture stays linked to its experiment, official ID and hashes.
"""

from __future__ import annotations

import ast
import csv
import html
from collections import Counter
from pathlib import Path

from .artifacts import (
    atomic_json, code_fingerprint, digest_json, local_workspace, publish_file,
    read_json, sha256_file, utc_now, verify_file,
)
from .quality import document_quality
from .structure import parse_structure, validate_structure
from .text import EXPECTED_PARSE_ERRORS, extract_source


def _asset_path(root: Path, key: str, asset: dict) -> Path:
    prefix = f"reports/crawl_recovery/stage-a-v2/experiments/{key}/"
    value = asset["path"]
    if not value.startswith(prefix):
        raise ValueError("Capture path does not belong to this experiment.")
    path = (root / value[len(prefix):]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Capture path escapes the experiment directory.")
    verify_file(path, asset["sha256"])
    return path


def review_browser_captures(experiment_dir: str | Path, output_root: str | Path, *,
                            expected_links: dict[int, str], expected_attempts_sha256: str,
                            work_dir: str | Path | None = None) -> dict:
    """Verify the experiment and extract one source document per official ID.

    Prefer actual browser response bytes. If only rendered DOM has the article,
    label it rendered_dom and preserve its separate raw hash/provenance.
    Canonical ingestion/promotion remains a separate step after review.
    """
    root, output_root = Path(experiment_dir), Path(output_root)
    manifest, summary = read_json(root / "experiment.json"), read_json(root / "summary.json")
    key = summary["probe_experiment"]
    if digest_json(manifest)[:16] != key or summary["attempts_sha256"] != expected_attempts_sha256:
        raise ValueError("Probe manifest or attempts digest differs from the reviewed input.")
    verify_file(root / "attempts.csv", expected_attempts_sha256)
    records = [read_json(p) for p in sorted((root / "markers").glob("*.json"))]
    pairs = {(r["doc_id"], r["method"]) for r in records}
    if len(pairs) != len(records) or len(records) != summary["completed_methods"]:
        raise ValueError("Missing or duplicate probe methods.")
    if {r["doc_id"] for r in records} != set(expected_links) or len(expected_links) != summary["completed_ids"]:
        raise ValueError("Probe IDs do not match the expected official ID set.")
    if dict(Counter(r["outcome"] for r in records)) != summary["outcomes"]:
        raise ValueError("Probe outcome counts differ from summary.")
    with (root / "attempts.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_pairs = {(int(r["doc_id"]), r["method"]): r for r in csv_rows}
    if len(csv_pairs) != len(csv_rows) or set(csv_pairs) != pairs:
        raise ValueError("Probe markers differ from pinned attempts.csv.")
    for row in records:
        if expected_links[row["doc_id"]] != row["url"]:
            raise ValueError("Probe URL differs from its official ID mapping.")
        csv_row = csv_pairs[(row["doc_id"], row["method"])]
        if any(csv_row.get(name, "") != str(row.get(name) or "") for name in (
            "url", "outcome", "final_url", "source_text_sha256",
        )):
            raise ValueError("Probe marker metadata differs from pinned attempts.csv.")
        # Pandas writes nested objects using Python repr. literal_eval reads
        # values only; it never executes document text or Python expressions.
        serialized_assets = csv_row.get("assets", "{}")
        if len(serialized_assets) > 16000 or ast.literal_eval(serialized_assets) != row.get("assets", {}):
            raise ValueError("Probe marker assets differ from pinned attempts.csv.")
        for asset in row.get("assets", {}).values():
            _asset_path(root, key, asset)
    identity = {"source_experiment": key, "source_attempts_sha256": expected_attempts_sha256,
                "official_pairs_sha256": digest_json(sorted(expected_links.items())),
                "source_markers_sha256": digest_json(records),
                "code_sha256": code_fingerprint(), "policy": "browser-capture-extract-review-v1"}
    signature = digest_json(identity)
    target = output_root / signature[:16]
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "experiment.json"
    if config_path.exists() and read_json(config_path) != identity:
        raise ValueError("Extraction review configuration changed.")
    if not config_path.exists():
        atomic_json(config_path, identity)
    browser_rows = {r["doc_id"]: r for r in records if r["method"] == "dynamic_browser"}
    results = []
    for doc_id, url in sorted(expected_links.items()):
        marker = target / "markers" / f"{doc_id}.json"
        if marker.exists():
            saved = read_json(marker)
            if saved["doc_id"] != doc_id or saved["url"] != url or saved["source_experiment"] != key:
                raise ValueError("Extraction checkpoint identity changed.")
            for asset in saved.get("assets", {}).values():
                path = (target / asset["path"]).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise ValueError("Extraction checkpoint path escapes its directory.")
                verify_file(path, asset["sha256"])
            results.append(saved)
            continue
        row = browser_rows.get(doc_id)
        result = {"doc_id": doc_id, "url": url, "state": "EXTRACTION_REJECTED",
                  "source_experiment": key, "reviewed_at": utc_now(), "assets": {}}
        if row is None or row.get("http_status") != 200 or row.get("rendered_http_status") != 200 or row.get("guard_errors"):
            result["error"] = "browser_capture_not_eligible"
        else:
            source_errors = []
            for name, kind in (("response.html", "response_bytes"), ("rendered.html", "rendered_dom")):
                asset = row.get("assets", {}).get(name)
                if not asset:
                    continue
                body = _asset_path(root, key, asset).read_bytes()
                try:
                    extracted = extract_source(body, "text/html", source_url=url)
                except EXPECTED_PARSE_ERRORS as exc:
                    source_errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:250]}")
                    continue
                document = {**extracted, "doc_id": doc_id, "url": url,
                            "final_url": row["final_url"], "source_method": "dynamic_browser",
                            "capture_kind": kind, "body_sha256": asset["sha256"],
                            "source_experiment": key, "source_asset": asset["path"],
                            "fetched_at": row["finished_at"], "state": "READY_FOR_EXTRACTION_REVIEW"}
                structure = parse_structure(document, extracted["heading_hints"])
                validate_structure(document, structure)
                quality = document_quality(extracted["source_text"], title=extracted["title"],
                                           section_count=structure["known_heading_count"], raw_bytes=len(body))
                if extracted.get("raw_has_table"):
                    quality["quality_flags"].append("TABLE_TEXT_LINEARIZED_REQUIRES_AUDIT")
                document.update(quality, section_count=structure["known_heading_count"])
                with local_workspace(work_dir) as temporary:
                    text_path = Path(temporary) / f"{doc_id}.txt"
                    text_path.write_bytes(extracted["source_text"].encode("utf-8"))
                    doc_path = Path(temporary) / f"{doc_id}.json"
                    atomic_json(doc_path, {"document": document, "structure": structure})
                    for local in (text_path, doc_path):
                        relative = f"documents/{local.name}"
                        checksum = publish_file(local, target / relative)
                        result["assets"][local.suffix] = {"path": relative, "sha256": checksum}
                result.update(state=document["state"], capture_kind=kind,
                              title=document["title"], source_chars=len(extracted["source_text"]),
                              source_text_sha256=extracted["source_text_sha256"],
                              body_sha256=asset["sha256"], parser=document["parser"],
                              section_count=document["section_count"], quality_flags=quality["quality_flags"],
                              paragraph_count=quality["paragraph_count"], source_errors=source_errors)
                break
            else:
                result["error"] = "no_parseable_browser_article"
                result["source_errors"] = source_errors
        atomic_json(marker, result)
        results.append(result)
    # Static review contains exact escaped source text; no scripts or remote content.
    fragments = []
    for result in results:
        if ".txt" not in result["assets"]:
            fragments.append(f'<section><h2>{result["doc_id"]}: extraction rejected</h2></section>')
            continue
        text = (target / result["assets"][".txt"]["path"]).read_text(encoding="utf-8")
        fragments.append(f'<section><h2>{result["doc_id"]}: {html.escape(result["title"])}</h2>'
                         f'<p>{result["section_count"]} sections · {result["source_chars"]} characters</p>'
                         f'<pre>{html.escape(text)}</pre></section>')
    report = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Long Châu extraction review</title>'
              '<style>body{max-width:960px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui}'
              'section{border-top:1px solid #ccc;margin:28px 0}pre{white-space:pre-wrap;font:inherit}</style>'
              f'<h1>Long Châu — {len(expected_links)} article extraction review</h1>'
              '<p>Source-derived captures. This bundle does not modify the canonical corpus.</p>'
              + ''.join(fragments) + '</html>')
    with local_workspace(work_dir) as temporary:
        local = Path(temporary) / "review.html"
        local.write_bytes(report.encode("utf-8"))
        review_hash = publish_file(local, target / "review.html")
    result = {"signature": signature, "output_dir": str(target), "input_ids": len(expected_links),
              "ready_for_review": sum(r["state"] == "READY_FOR_EXTRACTION_REVIEW" for r in results),
              "states": dict(Counter(r["state"] for r in results)), "records": results,
              "review_html_sha256": review_hash, "created_at": utc_now()}
    atomic_json(target / "summary.json", result)
    return result
