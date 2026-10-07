"""Bounded GPU timing on disk-partitioned inputs; never a retrieval score."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, read_json, runtime_versions, sha256_file, verify_file
from .dataset import parquet_path
from .model_budget import model_budget_report
from .qwen_models import TorchQwenEncoder, TorchQwenReranker, pair_windows, review_model_registry
from .retrieval_data import load_queries
from .retrieval_models import TorchDenseEncoder
from .runtime_profile import inference_batches


def _bound(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Resource benchmark path escapes its root.")
    return path


def load_handoff(data_root):
    root = Path(data_root)
    active = read_json(root / "active_data_candidate.json")
    build = _bound(root / "processed", active["build_run"])
    candidate = read_json(_bound(build, active["candidate_name"]))
    units = read_json(build / "index_inputs/units.json")
    for value, key in ((candidate, "candidate_manifest_sha256"), (units, "manifest_sha256")):
        if digest_json({k:v for k,v in value.items() if k != key}) != value[key]:
            raise ValueError("Resource benchmark input manifest changed.")
    if (candidate["state"] != "FROZEN_CANDIDATE" or not candidate["selected_range_complete"]
        or units["state"] != "COMPLETE"
        or active["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]
        or active["index_inputs_manifest_sha256"] != units["manifest_sha256"]
        or units["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]
        or units["input_count"] != sum(p["rows"] for p in units["parts"])):
        raise ValueError("Resource benchmark handoff/candidate/inputs mismatch.")
    return build, candidate, units


def sample_inputs(build, units, sample_size=64):
    """Read at most eight parts; keep bounded samples from across the corpus."""
    if type(sample_size) is not int or not 8 <= sample_size <= 256:
        raise ValueError("Use a bounded resource sample of 8 to 256 texts.")
    parts = units["parts"]
    if not parts or any(p["rows"] <= 0 for p in parts):
        raise ValueError("No nonempty resource input parts available.")
    indexes = sorted(set(np.linspace(0, len(parts)-1, min(8,len(parts)), dtype=int).tolist()))
    selected = []
    each = (sample_size + len(indexes)-1) // len(indexes)
    for index in indexes:
        part = parts[index]
        path = _bound(Path(build) / "index_inputs", part["path"])
        verify_file(path, part["sha256"])
        if pq.ParquetFile(path).metadata.num_rows != part["rows"]:
            raise ValueError("Resource sample part row count changed.")
        remaining = min(each,part["rows"])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=remaining, columns=["id","text"]):
            selected.extend(batch.to_pylist()[:remaining])
            break
    # A reproducible timing sample, not a population-weighted evaluation set.
    return sorted(selected[:sample_size], key=lambda r:(len(r["text"]),r["id"]))


def run_scale_benchmark(data_root, checkout, *, sample_size=64):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU for the bounded resource benchmark.")
    root, checkout = Path(data_root), Path(checkout)
    build, candidate, inputs = load_handoff(root)
    snapshot = read_json(root / "raw/snapshot.json")
    if snapshot["files"]["links_corpus.parquet"]["sha256"] != candidate["origin_corpus_sha256"]:
        raise ValueError("Benchmark candidate/official corpus snapshot mismatch.")
    config = read_json(checkout / "configs/retrieval_full.json")
    registry = read_json(checkout / "configs/strong_model_manifest.json")
    review_model_registry(config,registry)
    if any(candidate["golden"]["tokenizer"][k] != config["dense"][k] for k in ("model_id","revision")):
        raise ValueError("Benchmark candidate/tokenizer mismatch.")
    sample = sample_inputs(build,inputs,sample_size)
    if not sample:
        raise ValueError("No model inputs available.")
    queries = load_queries(parquet_path(root,"query.parquet"),expected_count=1200)[:8]
    device = {"name":torch.cuda.get_device_name(0),
        "memory_bytes":torch.cuda.get_device_properties(0).total_memory,"torch":torch.__version__}
    contract = {"candidate":candidate["candidate_manifest_sha256"],"inputs":inputs["manifest_sha256"],
        "sample_sha256":digest_json(sample),"queries_sha256":digest_json(queries),
        "config":config,"device":device,"runtime":runtime_versions(),
        "code":{name:sha256_file(Path(__file__).with_name(name)) for name in
            ("scale_benchmark.py","qwen_models.py","retrieval_models.py","runtime_profile.py")}}
    signature = digest_json(contract)
    target = root / "retrieval" / ("resource-benchmark-"+signature[:16])
    atomic_json(target / "contract.json",contract)
    stages = []
    batches = inference_batches(device["memory_bytes"])
    texts = [row["text"] for row in sample]
    for name, factory, spec in (("bge_dense",TorchDenseEncoder,config["dense"]),
        ("qwen_dense",TorchQwenEncoder,config["second_dense"]),
        ("qwen_reranker",TorchQwenReranker,config["reranker"])):
        marker = target / (name+".json")
        if marker.exists():
            saved = read_json(marker)
            if saved["signature"] != signature:
                raise ValueError("Resource benchmark checkpoint contract changed.")
            if saved["state"] == "COMPLETE":
                stages.append(saved)
                print(f"Reuse benchmark: {name}",flush=True)
                continue
        model = None
        row = {"stage":name,"signature":signature,"state":"RUNNING"}
        atomic_json(marker,row)
        try:
            print(f"Benchmark {name}: load/download weights, then bounded inference",flush=True)
            started = time.perf_counter()
            model = factory(spec)
            torch.cuda.synchronize()
            row["load_and_download_seconds"] = round(time.perf_counter()-started,3)
            if name == "qwen_reranker":
                pairs = []
                for i,text in enumerate(texts):
                    query = queries[i % len(queries)]["query"]
                    pairs.extend((query,w["text"]) for w in pair_windows(
                        model.tokenizer,query,text,spec["max_length"]))
                items = pairs
                batch = batches["reranker"]
                infer = lambda values: model.score(values,batch_size=batch)
            else:
                items = texts
                batch = batches["embedding"]
                infer = lambda values: model.encode(values,batch_size=batch)
            infer(items[:min(batch,len(items))])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(2):
                started = time.perf_counter()
                values = infer(items)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-started)
                if len(values) != len(items) or not np.isfinite(values).all():
                    raise ValueError("Invalid benchmark model outputs.")
            rate = 2*len(items)/sum(durations)
            row.update(state="COMPLETE",items=len(items),repeats=2,batch_size=batch,
                safe_batch_size=getattr(model,"safe_batch_size",batch),
                inference_seconds=durations,items_per_second=rate,
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                oom_backoffs=model.oom_backoffs,encoder_identity=model.identity)
            if name != "qwen_reranker":
                row["sample_rate_projection_child_embeddings_hours"] = inputs["input_count"]/rate/3600
                row["projection_scope"] = "Child inputs only; excludes model loading, document dense, I/O, index and query stages"
        except Exception as exc:
            row.update(state="FAILED",error=f"{type(exc).__name__}: {exc}")
        finally:
            if model is not None:
                model.close()
            atomic_json(marker,row)
        stages.append(row)
        print(f"Benchmark {name}: {row['state']}",flush=True)
    report = {"state":"RESOURCE_BENCHMARK_COMPLETE" if all(s["state"]=="COMPLETE" for s in stages)
        else "RESOURCE_BENCHMARK_PARTIAL","signature":signature,"device":device,
        "documents":candidate["counts"]["documents"],"unique_child_inputs":inputs["input_count"],
        "sample_size":len(sample),"stages":stages,"model_parameter_budget":model_budget_report(config,registry),
        "submission_created":False,"scope":"Resource timing only; no predictions, labels or relevance score",
        "limitations":["Convenience sample, not a full-corpus time guarantee",
            "Query LLM, document dense, sparse index, ANN, fusion and full cascade not timed",
            "Full-scale retrieval still requires disk-backed catalog/index integration"],
        "report_path":str(target / "benchmark.json")}
    atomic_json(target / "benchmark.json",report)
    return report
