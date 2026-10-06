"""Immutable vector parts; a final checksum marker makes each part resumable."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, utc_now, verify_file


def unit_signature(units):
    ids = [u["id"] for u in units]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(u["text"], str) or not u["text"].strip() for u in units):
        raise ValueError("Embedding inputs need unique IDs and nonempty text.")
    return digest_json([{ "id": u["id"], "text_sha256": hashlib.sha256(u["text"].encode()).hexdigest()}
                        for u in units])


def validate_vectors(vectors, rows, dimension):
    if vectors.shape != (rows, dimension) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Vector shape/dtype/finite-value check failed.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("Cosine retrieval needs nonzero L2-normalized vectors.")


def embed_units(units, encoder, output_dir, *, part_size=256, batch_size=16,
                max_new_parts=None, work_dir=None):
    if part_size < 1 or batch_size < 1 or (max_new_parts is not None and max_new_parts < 0):
        raise ValueError("Invalid embedding batch/part limits.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"units_sha256": unit_signature(units), "input_count": len(units),
                "encoder": encoder.identity, "dimension": encoder.dimension, "part_size": part_size,
                "format": "float32-unit-vectors-v1"}
    signature = digest_json(identity)
    config = root / "config.json"
    if config.exists() and read_json(config) != {**identity, "signature": signature}:
        raise ValueError("Embedding input/model/policy changed. Use a new retrieval run name.")
    if not config.exists():
        atomic_json(config, {**identity, "signature": signature})
    parts, written = [], 0
    started, encoding_seconds, cached_parts = time.perf_counter(), 0., 0
    label = f"Embedding {encoder.identity.get('model_id', 'encoder')} | {len(units)} texts | batch {batch_size}"
    for start in tqdm(range(0, len(units), part_size), desc=label):
        selected = units[start:start+part_size]
        stem = f"part-{start:010d}-{len(selected):05d}"
        marker = root / (stem + ".done.json")
        input_sha = unit_signature(selected)
        if marker.exists():
            saved = read_json(marker)
            if saved["signature"] != signature or saved["start"] != start or saved["rows"] != len(selected) or saved["input_sha256"] != input_sha or saved["path"] != stem + ".npy":
                raise ValueError("Embedding checkpoint input mismatch.")
            verify_file(root / saved["path"], saved["sha256"])
            validate_vectors(np.load(root / saved["path"], allow_pickle=False), len(selected), encoder.dimension)
            cached_parts += 1
        else:
            if max_new_parts is not None and written >= max_new_parts:
                break
            encoding_started = time.perf_counter()
            vectors = np.asarray(encoder.encode([u["text"] for u in selected], batch_size=batch_size), dtype=np.float32)
            encoding_seconds += time.perf_counter() - encoding_started
            validate_vectors(vectors, len(selected), encoder.dimension)
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / (stem + ".npy")
                np.save(local, vectors, allow_pickle=False)
                checksum = publish_file(local, root / local.name)
            saved = {"signature": signature, "start": start, "rows": len(selected),
                     "input_sha256": input_sha, "path": stem + ".npy", "sha256": checksum,
                     "completed_at": utc_now()}
            atomic_json(marker, saved)
            written += 1
        parts.append(saved)
    complete = sum(p["rows"] for p in parts) == len(units)
    manifest = {**identity, "signature": signature, "parts": parts,
                "state": "COMPLETE" if complete else "IN_PROGRESS"}
    manifest["manifest_sha256"] = digest_json(manifest)
    atomic_json(root / "embeddings.json", manifest)
    atomic_json(root / "runtime_profile.json", {"model": encoder.identity.get("model_id"),
        "input_count": len(units), "batch_size_requested": batch_size, "part_size": part_size,
        "new_parts": written, "cached_parts": cached_parts,
        "encoding_seconds": round(encoding_seconds, 3), "seconds_this_call": round(time.perf_counter() - started, 3)})
    return manifest


def embedding_matrix(root, manifest):
    root = Path(root).resolve()
    if manifest.get("state") != "COMPLETE" or digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Resume all embedding parts before building/searching an index.")
    matrices, cursor = [], 0
    for part in manifest["parts"]:
        path = (root / part["path"]).resolve()
        if not path.is_relative_to(root) or part["start"] != cursor or part["signature"] != manifest["signature"]:
            raise ValueError("Vector parts are out of order or belong to another run.")
        verify_file(path, part["sha256"])
        vectors = np.load(path, allow_pickle=False)
        validate_vectors(vectors, part["rows"], manifest["dimension"])
        matrices.append(vectors)
        cursor += part["rows"]
    if cursor != manifest["input_count"]:
        raise ValueError("Incomplete embedding coverage.")
    return np.ascontiguousarray(np.concatenate(matrices), dtype=np.float32)
