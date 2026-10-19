"""Restore a sealed submission handoff using JSON only; never start CPU preparation.

Embedded in the notebook so old producer/code locks and corpus fingerprints stay
unchanged. This checks identities, not catalog/vector payload integrity. Native
readers still validate payloads when inference actually consumes them.
"""
from pathlib import Path

from vietmedbridge.artifacts import atomic_json, digest_json, read_json
from vietmedbridge.official_scope import discover_sources
from vietmedbridge.scale_benchmark import load_handoff


def restore_submission_preparation_only(data_root, checkout):
    root, checkout = Path(data_root), Path(checkout)
    print("RESUME_ONLY: đọc manifest/báo cáo JSON; không mở corpus, vector hoặc catalog/BM25.", flush=True)
    build, candidate, inputs = load_handoff(root)
    config_sha = digest_json(read_json(checkout / "configs/retrieval_full.json"))
    source_names = discover_sources(root)
    corpus_sha = read_json(root / "raw/snapshot.json")["files"]["links_corpus.parquet"]["sha256"]
    identity = {"snapshot_sha256": candidate["snapshot_sha256"],
        "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
        "index_inputs_manifest_sha256": inputs["manifest_sha256"]}
    pointer = root / "retrieval/current_submission.json"
    reports = [pointer, build / "submission_preparation.json",
        root / "retrieval/cpu_preparation" / (candidate["candidate_manifest_sha256"][:16] + ".json")]
    for path in reports:
        if not path.is_file():
            continue
        saved = read_json(path)
        if (saved.get("state") != "CPU_PREPARATION_COMPLETE"
            or saved.get("build_run") != build.name
            or saved.get("candidate_manifest_sha256") != candidate["candidate_manifest_sha256"]
            or saved.get("retrieval_config_sha256") != config_sha
            or saved.get("official_sha256") != candidate["origin_corpus_sha256"]
            or candidate.get("input_kind") != "AVAILABLE_CORPUS"
            or saved.get("legacy_urls_retained") is not True
            or saved.get("source_builds") != source_names
            or any(saved.get("catalog", {}).get(k) != v for k, v in identity.items())
            or corpus_sha != candidate["origin_corpus_sha256"]):
            continue
        result = saved | {"results_root": str(root.parent / "submission_runs" / build.name)}
        if not pointer.is_file() or read_json(pointer) != result:
            atomic_json(pointer, result)  # only a small handoff pointer; no stage output is rebuilt
        print(f"RESUME_READY: {build.name} | {saved['documents']:,} documents | CPU preparation reused.", flush=True)
        return result
    raise RuntimeError(
        "RESUME_ONLY_STOP: thiếu hoặc không khớp báo cáo CPU_PREPARATION_COMPLETE. "
        "Chưa đọc lại corpus/vector/catalog và chưa bắt đầu preparation. "
        "Kiểm tra DATA_ROOT/candidate/code lock; chỉ bật ALLOW_CPU_PREPARATION khi chủ động chuẩn bị dữ liệu mới.")
