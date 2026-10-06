"""Discover original, verified vector artifacts across retrieval and training runs."""
from pathlib import Path

from .artifacts import atomic_json, digest_json, read_json
from .embeddings import unit_signature
from .retrieval_cache import reuse_bge_cache


def find_embeddings(data_root, units, spec, *, family, role="corpus", preferred=()):
    """Reuse vectors only when input order, weights and inference policy match.

    Partition size and the query set of another experiment do not change corpus
    vectors. We retain the original producer manifest and files, without copying
    weights/vectors onto Drive or relabelling their producer.
    """
    if family not in ("bge", "qwen") or role not in ("corpus", "query"):
        raise ValueError("Invalid shared embedding family/role.")
    root = Path(data_root).resolve()
    inputs_sha = unit_signature(units)
    key = digest_json({"inputs": inputs_sha, "spec": spec, "family": family, "role": role})
    pointer = root / "cache" / "embedding_producers" / (key + ".json")
    candidates = [Path(p) for p in preferred]
    if pointer.is_file():
        saved = read_json(pointer)
        candidate = (root / saved["directory"]).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Shared embedding pointer escapes DATA_ROOT.")
        candidates.append(candidate)
    # Only artifact directories: no recursive scan of crawls or query reports.
    names = ("corpus_embeddings", "query_embeddings", "document_embeddings") if family == "bge" else ("qwen_corpus", "qwen_queries")
    patterns = ("retrieval/*/{name}/embeddings.json", "training/*/experiments/*/{name}/embeddings.json")
    seen = set()
    for candidate in candidates:
        seen.add(candidate.resolve())
    for pattern in patterns:
        for name in names:
            for manifest_path in root.glob(pattern.format(name=name)):
                candidate = manifest_path.parent.resolve()
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Embedding producer must remain inside DATA_ROOT.")
        if candidate in checked:
            continue
        checked.add(candidate)
        path = candidate / "embeddings.json"
        if not path.is_file():
            continue
        manifest = read_json(path)
        encoder = manifest.get("encoder", {})
        if (manifest.get("state") != "COMPLETE" or manifest.get("units_sha256") != inputs_sha
            or manifest.get("input_count") != len(units)
            or any(encoder.get(k) != v for k, v in spec.items())
            or (family == "qwen" and encoder.get("input_role") != role)):
            continue
        # A matching but corrupt artifact must fail, never silently hide tampering.
        if family == "bge":
            cached = reuse_bge_cache(candidate, units, spec, part_size=manifest["part_size"])
        else:
            from .qwen_models import cached_qwen_embeddings
            cached = cached_qwen_embeddings(candidate, units, spec, role)
            identity = {k: manifest[k] for k in ("units_sha256", "input_count", "encoder", "dimension", "part_size", "format")}
            if (digest_json(identity) != manifest["signature"]
                or read_json(candidate / "config.json") != {**identity, "signature": manifest["signature"]}):
                raise ValueError("Shared Qwen embedding configuration integrity mismatch.")
            for part in manifest["parts"]:
                if read_json(candidate / (Path(part["path"]).stem + ".done.json")) != part:
                    raise ValueError("Shared Qwen embedding completion marker mismatch.")
        if cached is not None:
            vectors, verified = cached
            atomic_json(pointer, {"directory": str(candidate.relative_to(root)),
                "manifest_sha256": verified["manifest_sha256"], "key": key})
            return vectors, verified, candidate
    return None
