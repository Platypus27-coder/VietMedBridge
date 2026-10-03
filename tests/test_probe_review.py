"""Recovery replay verifies ID/URL and captured evidence before extracting text."""

import csv
import hashlib

import pytest

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.probe_review import review_browser_captures


URL = "https://nhathuoclongchau.com.vn/bai-viet/source.html"
BODY = f'''<html><head><title>Bài nguồn</title><link rel="canonical" href="{URL}"></head><body>
<div data-lcpr="prr-id-articles-content"><div><h1>Bài nguồn</h1><div><p>Đoạn dẫn.</p></div>
<div data-theme-element="article"><h2>Phần nguồn</h2><p>Văn bản nguồn tiếng Việt.</p></div>
<aside>Cookie Consent</aside></div></div></body></html>'''.encode()


def fixture_probe(tmp_path, *, raw_body=BODY, rendered_body=BODY):
    root = tmp_path / "probe"
    root.mkdir()
    manifest = {"policy": "fixture-probe"}
    key = digest_json(manifest)[:16]
    assets = {}
    for name, body in (("response.html", raw_body), ("rendered.html", rendered_body)):
        relative = f"assets/583/dynamic_browser/{name}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        assets[name] = {"path": f"reports/crawl_recovery/stage-a-v2/experiments/{key}/{relative}",
                        "sha256": sha256_file(path)}
    row = {"doc_id": 583, "url": URL, "final_url": URL, "method": "dynamic_browser",
           "outcome": "article_candidate_needs_review", "http_status": 200,
           "rendered_http_status": 200, "guard_errors": [], "assets": assets,
           "finished_at": "2026-10-03T02:20:00+00:00"}
    atomic_json(root / "experiment.json", manifest)
    atomic_json(root / "markers/583-dynamic_browser.json", row)
    with (root / "attempts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    checksum = sha256_file(root / "attempts.csv")
    atomic_json(root / "summary.json", {"probe_experiment": key, "completed_ids": 1,
                 "completed_methods": 1, "outcomes": {row["outcome"]: 1}, "attempts_sha256": checksum})
    return root, checksum


def run_review(root, checksum, tmp_path, *, links=None):
    return review_browser_captures(root, tmp_path / "review", expected_links=links or {583: URL},
                                   expected_attempts_sha256=checksum, work_dir=tmp_path)


def test_replay_preserves_capture_identity_exact_text_and_checkpoint(tmp_path):
    root, checksum = fixture_probe(tmp_path)
    first = run_review(root, checksum, tmp_path)
    record = first["records"][0]
    assert first["ready_for_review"] == 1
    assert record["capture_kind"] == "response_bytes"
    assert record["body_sha256"] == hashlib.sha256(BODY).hexdigest()
    from pathlib import Path
    target = Path(first["output_dir"])
    document = read_json(target / "documents/583.json")["document"]
    text = (target / "documents/583.txt").read_bytes()
    assert document["source_asset"].endswith("response.html")
    assert document["source_text_sha256"] == hashlib.sha256(text).hexdigest()
    assert "Cookie Consent" not in document["source_text"]
    assert "Phần nguồn" in document["source_text"]
    second = run_review(root, checksum, tmp_path)
    assert second["records"] == first["records"]


def test_rendered_dom_fallback_is_explicitly_labelled(tmp_path):
    root, checksum = fixture_probe(tmp_path, raw_body=b"<html><title>Bai</title><body>Empty shell</body></html>")
    report = run_review(root, checksum, tmp_path)
    assert report["records"][0]["capture_kind"] == "rendered_dom"
    assert report["records"][0]["source_errors"]


def test_modified_raw_asset_rejected_before_any_extraction(tmp_path):
    root, checksum = fixture_probe(tmp_path)
    (root / "assets/583/dynamic_browser/response.html").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Missing or changed"):
        run_review(root, checksum, tmp_path)


def test_wrong_official_url_is_rejected(tmp_path):
    root, checksum = fixture_probe(tmp_path)
    with pytest.raises(ValueError, match="official ID mapping"):
        run_review(root, checksum, tmp_path, links={583: URL.replace("source", "other")})


def test_marker_tampering_cannot_override_pinned_csv_asset_mapping(tmp_path):
    root, checksum = fixture_probe(tmp_path)
    p = root / "markers/583-dynamic_browser.json"
    row = read_json(p)
    row["assets"]["response.html"] = row["assets"]["rendered.html"]
    atomic_json(p, row)
    with pytest.raises(ValueError, match="assets differ"):
        run_review(root, checksum, tmp_path)
