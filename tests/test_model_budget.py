"""Each model, including its own adapter, must fit the confirmed 15B rule."""
from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.model_budget import add_adapter_parameters, adapter_parameter_count, model_budget_report
from vietmedbridge.qwen_models import review_model_registry


@pytest.fixture
def models():
    root = Path(__file__).resolve().parents[1]
    return read_json(root / "configs/retrieval_full.json"), read_json(root / "configs/strong_model_manifest.json")


def test_per_model_budget_records_total_without_capping_it(models):
    config, registry = models
    report = model_budget_report(config, registry)
    assert report["total_parameters"] == 20_346_066_432
    assert report["max_model_parameters"] == 8_188_548_096
    assert report["remaining_parameters"] == 6_811_451_904
    assert len(report["models"]) == 4 and not report["btc_approval"]
    changed = deepcopy(config)
    changed["second_dense"]["quantization"] = "nf4"
    changed["reranker"]["quantization"] = "fp16"
    assert model_budget_report(changed, registry) == report


def test_stack_above_15b_total_is_valid_but_one_oversized_model_is_rejected(models):
    config, registry = models
    embedding = next(m for m in registry["models"] if m["role"] == "second_dense")
    assert all(m["parameters"] <= 15_000_000_000 for m in registry["models"])
    review_model_registry(config, registry)
    embedding["parameters"] = 15_000_000_001
    with pytest.raises(ValueError,match="exceeds 15B"):
        model_budget_report(config,registry)


def test_total_budget_at_limit_and_adapter_overflow(models):
    config, registry = models
    report = model_budget_report(config, registry)
    at_limit = add_adapter_parameters(report, report["remaining_parameters"])
    assert at_limit["max_model_parameters"] == 15_000_000_000
    assert at_limit["total_parameters"] > 15_000_000_000
    with pytest.raises(ValueError, match="including adapters"):
        add_adapter_parameters(at_limit, 1)
    first = add_adapter_parameters(report, 10)
    assert add_adapter_parameters(first, 20)["total_parameters"] == report["base_parameters"] + 30


def test_unknown_parameters_missing_scope_or_expanded_limit_fail(models):
    config, registry = models
    unknown = deepcopy(registry)
    unknown["models"][0]["parameters"] = None
    with pytest.raises(ValueError, match="evidence"):
        model_budget_report(config, unknown)
    for replacement in ({}, {"max_parameters": 30_000_000_000, "scope": config["model_budget"]["scope"]}):
        config["model_budget"] = replacement
        with pytest.raises(ValueError, match="per-model"):
            model_budget_report(config, registry)


def test_shared_checkpoint_is_counted_once_across_roles(models):
    config, registry = models
    config["second_dense"] = dict(config["dense"])
    report = model_budget_report(config, registry)
    assert len(report["models"]) == 3
    assert report["total_parameters"] == 12_778_770_944
    assert report["models"][0]["roles"] == ["dense", "second_dense"]


def test_adapter_header_count_checksum_and_manifest_integrity(tmp_path, models):
    header = json.dumps({"lora_A.weight": {"shape": [2, 3], "dtype": "F32", "data_offsets": [0, 24]},
        "lora_B.weight": {"shape": [4, 2], "dtype": "F32", "data_offsets": [24, 56]},
        "__metadata__": {"format": "pt"}}).encode()
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(56))
    manifest = {"files": {path.name: sha256_file(path)},
        "base_model":models[0]["reranker"]["model_id"],"base_revision":models[0]["reranker"]["revision"]}
    manifest["manifest_sha256"] = digest_json(manifest)
    atomic_json(tmp_path / "adapter_manifest.json", manifest)
    assert adapter_parameter_count(tmp_path)["parameters"] == 14
    report = model_budget_report(*models, adapter_paths=[tmp_path])
    assert report["total_parameters"] == 20_346_066_446
    assert report["adapter_parameters"] == 14
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        adapter_parameter_count(tmp_path)


def test_adapter_budget_is_attached_to_its_own_model(models):
    report=model_budget_report(*models)
    encoder=next(m for m in report['models'] if 'second_dense' in m['roles'])
    limit=add_adapter_parameters(report,encoder['remaining_parameters'],role='second_dense')
    assert next(m for m in limit['models'] if 'second_dense' in m['roles'])['total_parameters']==15_000_000_000
    assert next(m for m in limit['models'] if 'reranker' in m['roles'])['adapter_parameters']==0
    with pytest.raises(ValueError,match='exceeds 15B'):
        add_adapter_parameters(limit,1,role='second_dense')
    # A different model may still acquire an adapter even when the encoder is at its limit.
    assert add_adapter_parameters(limit,100)['total_parameters']==limit['total_parameters']+100
