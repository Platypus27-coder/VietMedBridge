"""Bounded-memory FAISS shards from verified vector parts, with atomic resume."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, verify_file
from .embeddings import validate_vectors


def build_dense_shards(embedding_root, manifest, output_dir, *, rows_per_shard=50000, work_dir=None, max_new_shards=None):
    import faiss

    if manifest.get("state") != "COMPLETE" or digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Sharded index requires verified complete embeddings.")
    if type(rows_per_shard) is not int or rows_per_shard < 1:
        raise ValueError("Invalid shard row budget.")
    if max_new_shards is not None and (type(max_new_shards) is not int or max_new_shards < 0):
        raise ValueError("Invalid new-shard limit.")
    root, source = Path(output_dir), Path(embedding_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"embeddings": manifest["manifest_sha256"], "dimension": manifest["dimension"],
        "rows": manifest["input_count"], "rows_per_shard": rows_per_shard, "backend": "faiss-exact-shards-v1"}
    signature = digest_json(identity)
    if (root / "config.json").exists() and read_json(root / "config.json") != identity:
        raise ValueError("Sharded index policy changed.")
    atomic_json(root / "config.json", identity)
    parts, written, cursor, index = [], 0, 0, faiss.IndexFlatIP(manifest["dimension"])

    def commit_shard(start, shard):
        nonlocal written
        stem = f"shard-{start:012d}-{shard.ntotal:06d}"
        marker = root / (stem + ".done.json")
        expected = {"signature": signature, "start": start, "rows": shard.ntotal, "file": stem + ".faiss"}
        if marker.exists():
            saved = read_json(marker)
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError("Sharded checkpoint identity mismatch.")
            verify_file(root / saved["file"], saved["sha256"])
        else:
            if max_new_shards is not None and written >= max_new_shards:
                return False
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / expected["file"]
                faiss.write_index(shard, str(local))
                checksum = publish_file(local, root / local.name)
            saved = expected | {"sha256": checksum}
            atomic_json(marker, saved)
            written += 1
        parts.append(saved)
        return True

    for part in manifest["parts"]:
        path = (source / part["path"]).resolve()
        if not path.is_relative_to(source) or part["start"] != cursor or part["signature"] != manifest["signature"]:
            raise ValueError("Sharded vector part order/path mismatch.")
        verify_file(path, part["sha256"])
        vectors = np.load(path, allow_pickle=False, mmap_mode="r")
        validate_vectors(vectors, part["rows"], manifest["dimension"])
        offset = 0
        while offset < len(vectors):
            count = min(rows_per_shard - index.ntotal, len(vectors) - offset)
            index.add(np.ascontiguousarray(vectors[offset:offset+count]))
            offset += count
            cursor += count
            if index.ntotal == rows_per_shard:
                if not commit_shard(cursor - index.ntotal, index):
                    break
                index = faiss.IndexFlatIP(manifest["dimension"])
        else:
            continue
        break
    else:
        if index.ntotal:
            commit_shard(cursor - index.ntotal, index)
    report = identity | {"signature": signature, "parts": parts,
        "state": "COMPLETE" if sum(p["rows"] for p in parts) == manifest["input_count"] else "IN_PROGRESS"}
    report["manifest_sha256"] = digest_json(report)
    atomic_json(root / "index.json", report)
    return report


class ShardedDenseIndex:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "index.json")
        if self.manifest["state"] != "COMPLETE" or digest_json({k: v for k, v in self.manifest.items() if k != "manifest_sha256"}) != self.manifest["manifest_sha256"]:
            raise ValueError("Complete every dense shard before searching.")
        self.d = self.manifest["dimension"]
        self.ntotal = self.manifest["rows"]

    def search(self, vectors, k):
        import faiss
        if type(k) is not int or k < 1:
            raise ValueError("Invalid dense search limit.")
        validate_vectors(vectors, len(vectors), self.d)
        values = np.empty((len(vectors), 0), np.float32)
        ids = np.empty((len(vectors), 0), np.int64)
        cursor = 0
        for part in self.manifest["parts"]:
            path = (self.root / part["file"]).resolve()
            if not path.is_relative_to(self.root) or part["start"] != cursor or part["signature"] != self.manifest["signature"]:
                raise ValueError("Unsafe/out-of-order dense shard.")
            verify_file(path, part["sha256"])
            index = faiss.read_index(str(path))
            if index.ntotal != part["rows"] or index.d != self.d:
                raise ValueError("Dense shard shape mismatch.")
            scores, positions = index.search(vectors, min(k, part["rows"]))
            positions += part["start"]
            values, ids = np.concatenate([values, scores], axis=1), np.concatenate([ids, positions], axis=1)
            order = np.argsort(-values, axis=1, kind="stable")[:, :k]
            values, ids = np.take_along_axis(values, order, axis=1), np.take_along_axis(ids, order, axis=1)
            cursor += part["rows"]
            del index
        if cursor != self.ntotal:
            raise ValueError("Incomplete dense shard coverage.")
        return values, ids
