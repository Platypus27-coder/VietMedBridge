"""Resumable submissions over all available legacy data, as requested by the user."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import json

from .artifacts import atomic_json, digest_json, read_json, sha256_file, utc_now, verify_file
from .official_scope import activate_scope, derive_official_scope, discover_sources, rebind_cached_family, stage_scope_catalog


def ensure_submission_preparation(data_root, checkout, work_dir, *, gpu_session=False):
    """Restore the durable handoff, or resume CPU work even on a GPU runtime.

    Colab runtime variables and local files are disposable. Only matching Drive
    reports can authorize inference; absence is a resumable preparation stage,
    not a requirement to manually switch runtimes or fabricate a success marker.
    """
    from .scale_benchmark import load_handoff
    from .colab_storage import run_with_drive_recovery
    root, checkout, work = Path(data_root), Path(checkout), Path(work_dir)

    def restore():
        build, candidate, inputs = load_handoff(root)
        config_sha = digest_json(read_json(checkout / "configs/retrieval_full.json"))
        source_names = discover_sources(root)
        reports = [root / "retrieval/current_submission.json", build / "submission_preparation.json",
            root / "retrieval/cpu_preparation" / (candidate["candidate_manifest_sha256"][:16]+".json")]
        for path in reports:
            if not path.exists():
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
                or any(saved.get("catalog",{}).get(k) != v for k,v in {
                    "snapshot_sha256":candidate["snapshot_sha256"],
                    "candidate_manifest_sha256":candidate["candidate_manifest_sha256"],
                    "index_inputs_manifest_sha256":inputs["manifest_sha256"]}.items())
                or read_json(root / "raw/snapshot.json")["files"]["links_corpus.parquet"]["sha256"] != candidate["origin_corpus_sha256"]):
                continue
            # Recreate only small pointers. Catalog/vector payloads are validated
            # by their native readers before inference; nothing is re-encoded.
            saved = saved | {"results_root":str(root.parent / "submission_runs" / build.name)}
            atomic_json(root / "retrieval/current_submission.json",saved)
            print(f"Restored preparation from Drive: {path} | {saved['documents']:,} documents",flush=True)
            return saved
        return None

    saved = run_with_drive_recovery(restore,root,work,max_recoveries=3)
    if saved is not None:
        return saved
    print("Preparation chưa hoàn tất trên Drive. Tiếp tục checkpoint CPU hiện có; chưa nạp model weights.",flush=True)
    if gpu_session:
        print("Đang dùng runtime GPU: bước chuẩn bị này vẫn chạy bằng CPU; sau khi đủ checkpoint sẽ tự sang inference.",flush=True)
    return prepare_current_submission(root,checkout,work)


def prepare_current_submission(data_root, checkout, work_dir):
    """CPU only: keep legacy URLs, combine available sources, reuse vectors/BM25."""
    from .catalog_storage import prepare_disk_catalog
    from .chunks import ChunkConfig, load_bge_tokenizer
    from .golden import run_golden_suite
    from .medical_lexical import MedicalAnalyzer
    from .portable_embeddings import resources_relative
    from .scale_benchmark import load_handoff
    from .colab_storage import run_with_drive_recovery

    root,checkout,work = Path(data_root),Path(checkout),Path(work_dir)
    work.mkdir(parents=True,exist_ok=True)
    config = read_json(checkout / "configs/retrieval_full.json")
    # User explicitly requested retaining URLs removed in the revised BTC corpus.
    # The original snapshot remains the ID/URL provenance contract. Do not
    # silently relabel this run as passing membership in the revised corpus.
    snapshot = read_json(root / "raw/snapshot.json")
    sources = discover_sources(root)
    first = read_json(root / "processed" / sources[0] / "config.json")
    tokenizer,spec = load_bge_tokenizer(config["dense"]["revision"])
    if spec != first["tokenizer"]:
        raise ValueError("Source tokenizer differs from the pinned submission tokenizer.")
    golden = run_golden_suite(checkout / "tests/golden/cases.json",tokenizer,spec,ChunkConfig(**first["chunks"]))
    if not golden["passed"]:
        raise ValueError("Current source/chunk golden regression did not pass.")
    _,previous,_ = load_handoff(root)
    previous_resources = root / resources_relative(previous,config)
    producers = {family:read_json(previous_resources / family / "config.json")
                 for family in ("bge","qwen") if (previous_resources / family / "config.json").exists()}
    build,candidate,inputs = run_with_drive_recovery(lambda:derive_official_scope(
        root,sources,snapshot,tokenizer,golden,work_dir=work,keep_all_legacy=True),root,work,max_recoveries=3)
    report_path = build / "submission_preparation.json"
    if report_path.exists():
        old = read_json(report_path)
        if (old.get("state") == "CPU_PREPARATION_COMPLETE"
            and old.get("candidate_manifest_sha256") == candidate["candidate_manifest_sha256"]
            and old.get("retrieval_config_sha256") == digest_json(config)):
            activate_scope(root,build,candidate,inputs,snapshot)
            atomic_json(root / "retrieval/current_submission.json",old)
            print("Reused completed all-available-data CPU preparation; legacy URLs retained.",flush=True)
            return old
    resources = root / resources_relative(candidate,config)
    origin_path = build / "submission_cache_producers.json"
    if origin_path.exists():
        producers = read_json(origin_path)
    else:
        atomic_json(origin_path,producers)
    cache = {}
    for family,producer in producers.items():
        spec_name = "dense" if family == "bge" else "second_dense"
        if any(producer["encoder"].get(k) != v for k,v in config[spec_name].items()):
            raise ValueError("Existing vector model differs from the submission configuration.")
        model_file = "retrieval_models.py" if family == "bge" else "qwen_models.py"
        if producer["encoder"].get("inference_code_sha256") != sha256_file(checkout / "src/vietmedbridge" / model_file):
            raise ValueError("Restore the matching model producer code; paid-for vectors cannot be relabeled.")
        print("CPU vector cache inventory:",family,flush=True)
        cache[family] = run_with_drive_recovery(lambda:rebind_cached_family(
            root,build,candidate,inputs,producer,resources / family,work_dir=work),root,work,max_recoveries=3)
    activate_scope(root,build,candidate,inputs,snapshot)
    glossary = root / "labels/medical_aliases.json"
    analyzer = MedicalAnalyzer(segmentation=config["lexical_segmentation"],
                              glossary_path=glossary if glossary.is_file() else None)
    print("CPU catalog: all available validated documents, including legacy URLs; source spans unchanged.",flush=True)
    base = run_with_drive_recovery(lambda:prepare_disk_catalog(build,candidate,inputs,
        root / "retrieval/catalogs",analyzer,work_dir=work,
        stage_build=lambda:stage_scope_catalog(build,candidate,inputs,work_dir=work),
        stage_delta=lambda indexes:stage_scope_catalog(build,candidate,inputs,work_dir=work,skip_parts=indexes)),
        root,work,max_recoveries=3)
    try:
        identity = base.identity
        reused_catalog = base.manifest.get("reused_catalog")
    finally:
        base.close()
    report = {"state":"CPU_PREPARATION_COMPLETE","build_run":build.name,
        "candidate_manifest_sha256":candidate["candidate_manifest_sha256"],"catalog":identity,
        "documents":candidate["counts"]["documents"],"children":candidate["counts"]["children"],
        "unique_dense_inputs":inputs["input_count"],"official_revision":snapshot["revision"],
        "official_sha256":candidate["origin_corpus_sha256"],"cache":cache,
        "retrieval_config_sha256":digest_json(config),
        "results_root":str(root.parent / "submission_runs" / build.name),"updated_at":utc_now(),
        "scope":"ALL_AVAILABLE_VALIDATED_DATA_LEGACY_URLS_RETAINED",
        "legacy_urls_retained":True,"current_btc_membership_validated":False,
        "source_builds":sources,"reused_catalog":reused_catalog}
    atomic_json(report_path,report)
    atomic_json(root / "retrieval/cpu_preparation" / (candidate["candidate_manifest_sha256"][:16]+".json"),report)
    atomic_json(root / "retrieval/current_submission.json",report)
    return report


def verify_current_submission(zip_path, queries, official_links):
    """Independent final ID and ZIP readback; source validation precedes this."""
    import pyarrow.parquet as pq
    ids = set(pq.read_table(official_links,columns=["id"]).column("id").to_pylist())
    with ZipFile(zip_path) as archive:
        if archive.namelist() != ["results.json"] or archive.testzip() is not None:
            raise ValueError("Submission ZIP must contain exactly results.json.")
        records = json.loads(archive.read("results.json"))
    expected = [q["id"] for q in queries]
    if len(records) != len(expected) or [r["id"] for r in records] != expected or len(set(expected)) != len(expected):
        raise ValueError("Submission must preserve every official query exactly once and in official order.")
    for record in records:
        if set(record) != {"id","relevant_docs","relevant_chunks"}:
            raise ValueError("Unexpected submission schema.")
        docs = record["relevant_docs"]
        if any(type(i) is not int or i not in ids for i in docs) or len(set(docs)) != len(docs):
            raise ValueError("Submission contains a removed/nonofficial/duplicate document ID.")
        for chunk in record["relevant_chunks"]:
            if (set(chunk) != {"doc_id","chunk_text"} or chunk["doc_id"] not in docs
                or type(chunk["doc_id"]) is not int or not isinstance(chunk["chunk_text"],str)
                or not chunk["chunk_text"].strip()):
                raise ValueError("Submission chunk binding/schema mismatch.")
    return {"state":"DECLARED_SNAPSHOT_SUBMISSION_VERIFIED","queries":len(records),
            "zip_sha256":sha256_file(zip_path),"official_links_sha256":sha256_file(official_links)}


def _submission_context(data_root, checkout):
    from .scale_benchmark import load_handoff
    root,checkout = Path(data_root),Path(checkout)
    report = read_json(root / "retrieval/current_submission.json")
    build,candidate,inputs = load_handoff(root)
    config = read_json(checkout / "configs/retrieval_full.json")
    if (report.get("state") != "CPU_PREPARATION_COMPLETE"
        or report["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]
        or report["retrieval_config_sha256"] != digest_json(config)
        or candidate.get("input_kind") != "AVAILABLE_CORPUS" or not report.get("legacy_urls_retained")
        or read_json(root / "raw/snapshot.json")["files"]["links_corpus.parquet"]["sha256"] != candidate["origin_corpus_sha256"]):
        raise ValueError("Prepare all available legacy data on CPU before allocating models.")
    return root,checkout,report,build,candidate,inputs,config


def _ensure_primary(context, work):
    from .colab_storage import run_with_drive_recovery
    from .content_embeddings import content_embeddings
    from .portable_embeddings import resources_relative
    from .qwen_models import review_model_registry
    from .retrieval_models import TorchDenseEncoder
    from .runtime_profile import inference_batches
    from .scale_baseline import imported_bge_manifest
    root,checkout,report,build,candidate,inputs,config = context
    review_model_registry(config,read_json(checkout / "configs/strong_model_manifest.json"))
    resources = root / resources_relative(candidate,config)
    def primary():
        _, existing = imported_bge_manifest(root,candidate,inputs,config)
        if existing is not None and existing["state"] == "COMPLETE":
            print("BGE corpus cache complete; no child corpus encoding needed.",flush=True)
            return existing
        # The generic execution profile also serves the much larger Qwen
        # encoder. Honor BGE's own throughput batch instead of capping a T4 at
        # Qwen's batch of 16; the unchanged encoder halves batches on CUDA OOM.
        configured_batch = read_json(checkout / "configs/retrieval_scale.json")["bge_batch_size"]
        if type(configured_batch) is not int or configured_batch < 1:
            raise ValueError("BGE batch size must be a positive integer.")
        batch = max(configured_batch,inference_batches()["embedding"])
        print(f"BGE corpus: batch {batch}; existing content vectors reused; CUDA OOM halves batch",flush=True)
        encoder = TorchDenseEncoder(config["dense"])
        try:
            return content_embeddings(root,build / "index_inputs",inputs,encoder,resources / "bge",
                work_dir=work,batch_size=batch)
        finally:
            encoder.close()
    result = run_with_drive_recovery(primary,root,work,max_recoveries=3)
    if result["state"] != "COMPLETE":
        raise ValueError("Complete the primary BGE corpus before submission inference.")


def _publish_submission(result, root, outputs, report, name):
    from .dataset import parquet_path
    from .retrieval_data import load_queries
    outputs = Path(outputs)
    atomic_json(outputs / (name+"_status.json"),result["ready"] | {"diagnostics":result["diagnostics_path"]})
    if result["ready"]["state"] != "READY_FOR_MANUAL_UPLOAD":
        return
    ready = result["ready"]
    verify_file(ready["zip_path"],ready["zip_sha256"])
    validated = verify_current_submission(ready["zip_path"],
        load_queries(parquet_path(root,"query.parquet"),expected_count=1200),parquet_path(root,"links_corpus.parquet"))
    entry = ready | {"declared_snapshot_validation":validated,"official_revision":report["official_revision"],
        "candidate_manifest_sha256":report["candidate_manifest_sha256"],"legacy_urls_retained":True,
        "retrieval_config_sha256":report["retrieval_config_sha256"],
        "current_btc_membership_validated":False,"diagnostics":result["diagnostics_path"],"variant":name}
    atomic_json(outputs / (name+"_validation.json"),validated)
    atomic_json(root / "retrieval" / (name+"_submission.json"),entry)
    latest = root / "retrieval/latest_submission.json"
    previous = read_json(latest) if latest.exists() else {}
    if name == "strong" or previous.get("variant") != "strong" or previous.get("candidate_manifest_sha256") != report["candidate_manifest_sha256"]:
        atomic_json(latest,entry)
    print(f"Verified {name} ZIP saved on Drive: {ready['zip_path']}",flush=True)


def _saved_submission(context, name):
    from .dataset import parquet_path
    from .retrieval_data import load_queries
    root,checkout,report,*_ = context
    path = root / "retrieval" / (name+"_submission.json")
    if not path.exists():
        return None
    saved = read_json(path)
    if (saved.get("state") != "READY_FOR_MANUAL_UPLOAD" or saved.get("variant") != name
        or saved.get("candidate_manifest_sha256") != report["candidate_manifest_sha256"]
        or saved.get("retrieval_config_sha256") != report["retrieval_config_sha256"]):
        return None
    verify_file(saved["zip_path"],saved["zip_sha256"])
    verify_current_submission(saved["zip_path"],load_queries(parquet_path(root,"query.parquet"),expected_count=1200),
                              parquet_path(root,"links_corpus.parquet"))
    print(f"Reused verified {name} submission ZIP; no model loading or inference.",flush=True)
    return {"ready":saved,"status":read_json(saved["diagnostics"]),"diagnostics_path":saved["diagnostics"],"samples":[]}


def run_current_baseline(data_root, checkout, work_dir, *, code_commit, max_new_queries=None):
    """Save a first submittable ZIP before the slower LLM enhancement stages."""
    from .colab_storage import run_with_drive_recovery
    from .scale_baseline import run_large_baseline
    context = _submission_context(data_root,checkout)
    saved = _saved_submission(context,"baseline")
    if saved is not None:
        return saved
    root,checkout,report,*_ = context
    work = Path(work_dir)
    _ensure_primary(context,work)
    outputs = Path(report["results_root"])
    result = run_with_drive_recovery(lambda:run_large_baseline(root,checkout,code_commit=code_commit,
        work_dir=work,reuse_content_only=True,read_only_source=True,output_root=outputs,
        max_new_queries=max_new_queries),root,work,readonly=False,max_recoveries=3)
    _publish_submission(result,root,outputs,report,"baseline")
    return result


def run_current_submission(data_root, checkout, work_dir, *, code_commit, max_new_queries=None):
    """Translation/document dense/Qwen rerank/ZIP; no incomplete Qwen corpus encoding."""
    from .full_scale_runtime import run_scale_full_pipeline
    context = _submission_context(data_root,checkout)
    saved = _saved_submission(context,"strong")
    if saved is not None:
        return saved
    root,checkout,report,*_ = context
    work = Path(work_dir)
    _ensure_primary(context,work)
    outputs = Path(report["results_root"])
    print("Strong submission: full selected BGE/BM25 corpus + document dense + translation/expansion + Qwen reranker.",flush=True)
    print("Qwen corpus dense is enabled only when complete; no missing Qwen corpus is encoded by this submission run.",flush=True)
    result = run_scale_full_pipeline(root,checkout,code_commit=code_commit,work_dir=work,
        execution_mode="cached_strong",output_root=outputs,second_dense_mode="if_complete",
        document_dense_enabled=True,max_new_queries=max_new_queries)
    _publish_submission(result,root,outputs,report,"strong")
    return result
