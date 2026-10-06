"""Read verified old BGE vectors without relabelling their producer/runtime."""
from pathlib import Path

from .artifacts import digest_json, read_json, sha256_file
from .embeddings import embedding_matrix, unit_signature


def reuse_bge_cache(root, units, spec, *, part_size):
    root = Path(root)
    path = root / "embeddings.json"
    if not path.exists():
        return None
    manifest = read_json(path)
    if manifest.get("state") != "COMPLETE":
        return None
    encoder = manifest["encoder"]
    if manifest["units_sha256"] != unit_signature(units) or manifest["input_count"] != len(units) or manifest["part_size"] != part_size:
        raise ValueError("Cached embedding inputs/order/part size differ from the requested candidate.")
    if any(encoder.get(k) != v for k,v in spec.items()) or encoder.get("pooling") != "cls-l2-v1" or encoder.get("truncation") is not False or encoder.get("query_instruction") is not None or encoder.get("precision") != "torch.float16" or encoder.get("device_class") != "cuda" or encoder.get("inference_code_sha256") != sha256_file(Path(__file__).with_name("retrieval_models.py")) or encoder.get("dimension") != manifest["dimension"] or manifest.get("format") != "float32-unit-vectors-v1":
        raise ValueError("Cached BGE inference policy differs from the requested model.")
    identity = {k:manifest[k] for k in ("units_sha256", "input_count", "encoder", "dimension", "part_size", "format")}
    if digest_json(identity) != manifest["signature"] or read_json(root / "config.json") != {**identity, "signature": manifest["signature"]}:
        raise ValueError("Cached embedding configuration integrity mismatch.")
    for part in manifest["parts"]:
        if read_json(root / (Path(part["path"]).stem + ".done.json")) != part:
            raise ValueError("Cached vector part lacks its original completion marker.")
    return embedding_matrix(root, manifest), manifest
