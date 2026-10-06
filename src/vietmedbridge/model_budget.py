"""Conservative aggregate model budget, independent of VRAM and quantization."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct

from .artifacts import digest_json, read_json, sha256_file, verify_file

MAX_MODEL_PARAMETERS = 15_000_000_000
MODEL_ROLES = ("dense", "second_dense", "translation", "reranker")


def adapter_parameter_count(directory):
    """Count verified saved adapter tensor shapes without loading weights."""
    root = Path(directory)
    manifest = read_json(root / "adapter_manifest.json")
    if digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Adapter budget requires a verified manifest.")
    path = root / "adapter_model.safetensors"
    verify_file(path, manifest["files"][path.name])
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Invalid adapter safetensors header.")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 0 < length <= min(16 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError("Invalid adapter safetensors header size.")
        header = json.loads(stream.read(length))
    count = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape = tensor.get("shape")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid adapter tensor shape.")
        count += math.prod(shape)
    if count <= 0:
        raise ValueError("Empty adapter parameter budget.")
    return {"manifest_sha256": manifest["manifest_sha256"], "parameters": count,
            "weights_sha256": sha256_file(path)}


def add_adapter_parameters(report, parameters):
    """Also used immediately after LoRA creation, before starting training."""
    if type(parameters) is not int or parameters < 0:
        raise ValueError("Invalid additional adapter parameter count.")
    if (not isinstance(report, dict) or report.get("scope") != "ALL_ACTIVE_MODELS_BEFORE_QUANTIZATION"
        or report.get("limit_parameters") != MAX_MODEL_PARAMETERS
        or type(report.get("base_parameters")) is not int
        or report["base_parameters"] <= 0):
        raise ValueError("Training requires a verified aggregate model budget.")
    existing = report.get("adapter_parameters", 0)
    if type(existing) is not int or existing < 0:
        raise ValueError("Invalid existing adapter parameter count.")
    parameters += existing
    total = report["base_parameters"] + parameters
    if total > MAX_MODEL_PARAMETERS:
        raise ValueError(f"Aggregate model budget exceeds 15B: {total:,} parameters including adapters.")
    return {**report, "adapter_parameters": parameters, "total_parameters": total,
            "remaining_parameters": MAX_MODEL_PARAMETERS - total}


def model_budget_report(config, registry, *, adapter_paths=()):
    """Count all distinct active checkpoints, even when run sequentially."""
    budget = config.get("model_budget", {})
    if (budget.get("max_parameters") != MAX_MODEL_PARAMETERS
        or budget.get("scope") != "ALL_ACTIVE_MODELS_BEFORE_QUANTIZATION"):
        raise ValueError("Configure the aggregate model budget as at most 15B before quantization.")
    registered = {(m["model_id"], m["revision"]): m for m in registry["models"]}
    if len(registered) != len(registry["models"]):
        raise ValueError("Duplicate model checkpoints in parameter registry.")
    models = {}
    for role in MODEL_ROLES:
        spec = config[role]
        key = spec["model_id"], spec["revision"]
        entry = registered.get(key)
        if not entry or type(entry.get("parameters")) is not int or entry["parameters"] <= 0:
            raise ValueError("Missing parameter evidence in model registry.")
        if key not in models:
            models[key] = {"model_id": key[0], "revision": key[1], "parameters": entry["parameters"], "roles": []}
        models[key]["roles"].append(role)
    adapters = [adapter_parameter_count(path) for path in adapter_paths]
    report = {"scope": budget["scope"], "limit_parameters": MAX_MODEL_PARAMETERS,
        "base_parameters": sum(m["parameters"] for m in models.values()),
        "models": list(models.values()), "adapters": adapters,
        "registry_sha256": digest_json(registry),
        "policy_source": "USER_REQUEST_AGGREGATE_15B_NOT_BTC_INTERPRETATION",
        "btc_approval": False}
    return add_adapter_parameters(report, sum(a["parameters"] for a in adapters))
