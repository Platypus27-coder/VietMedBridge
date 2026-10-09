"""Full architecture on frozen disk inputs, shared content vectors and QLoRA adapters."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, read_json, sha256_file, verify_file, utc_now
from .competition_pilot import MASTER_PLAN, bind_pilot_run, finish_pilot
from .content_embeddings import ContentVectorParts, checked, content_embeddings, seal
from .dataset import parquet_path
from .catalog_storage import prepare_disk_catalog
from .disk_full import AdaptiveDiskCatalog, DiskStrongIndex
from .embeddings import embed_units, embedding_matrix, unit_signature
from .full_plan_runtime import calibration_context
from .medical_lexical import MedicalAnalyzer
from .model_budget import model_budget_report
from .query_expansion import cached_expansions, expand_queries
from .query_translation import cached_translations, translate_queries
from .qwen_models import TorchQwenEncoder, TorchQwenReranker, review_model_registry
from .retrieval_data import load_queries
from .retrieval_models import TorchDenseEncoder
from .retrieval_diagnostics import diagnose
from .retrieval_cache import reuse_bge_cache
from .runtime_profile import RuntimeProfile, inference_batches
from .scale_benchmark import load_handoff
from .scale_vectors import bound, search_parts
from .strong_retrieval import StrongConfig, auxiliary_units, predict_strong
from .translation_model import TorchQueryTranslator


def is_large_handoff(root,config):
    if not (Path(root) / "active_data_candidate.json").exists():
        return False
    _,candidate,_ = load_handoff(root)
    limits = config["pilot_limits"]
    return candidate["counts"]["documents"] > limits["max_documents"] or candidate["counts"]["children"] > limits["max_children"]


def training_source_pool(catalog,limit):
    """Bound the reviewed source pool; negatives still come from the full index."""
    from types import SimpleNamespace
    ids = catalog.base.con.execute("""SELECT doc_id FROM
        (SELECT doc_id,row_number() OVER(PARTITION BY json_extract(payload,'$.language') ORDER BY doc_id) AS ordinal FROM documents)
        ORDER BY ordinal,doc_id LIMIT ?""",(limit,)).fetchall()
    documents,children = {},{}
    for (doc,) in ids:
        documents[doc] = catalog.documents[doc]
        children.update({row["chunk_id"]:row for row,_ in catalog.base.document_children(doc)})
    return SimpleNamespace(documents=documents,children=children,identity=catalog.identity)


def load_scale_catalog(root,checkout,work,tokenizer):
    build,candidate,inputs = load_handoff(root)
    config = read_json(Path(checkout) / "configs/retrieval_full.json")
    if any(candidate["golden"]["tokenizer"].get(k) != config["dense"].get(k) for k in ("model_id","revision")):
        raise ValueError("Frozen source tokenizer differs from the full architecture.")
    snapshot = read_json(Path(root) / "raw/snapshot.json")
    if snapshot["files"]["links_corpus.parquet"]["sha256"] != candidate["origin_corpus_sha256"]:
        raise ValueError("Full system/official snapshot mismatch.")
    glossary = Path(root) / "labels/medical_aliases.json"
    analyzer = MedicalAnalyzer(segmentation=config["lexical_segmentation"],glossary_path=glossary if glossary.is_file() else None)
    base = prepare_disk_catalog(build,candidate,inputs,Path(root) / "retrieval/catalogs",analyzer,work_dir=work)
    policy = StrongConfig(**config["retrieval"])
    return AdaptiveDiskCatalog(base,tokenizer,(policy.parent_short_tokens,policy.parent_long_tokens)),analyzer


def prepare_scale_cpu_resources(root,checkout,work):
    """Publish the disk catalog before allocating a GPU or any model weights."""
    from transformers import AutoTokenizer
    import traceback
    root,checkout = Path(root),Path(checkout)
    build,candidate,inputs = load_handoff(root)
    path = root / "retrieval/cpu_preparation" / (candidate["candidate_manifest_sha256"][:16]+".json")
    status = {"state":"RUNNING","stage":"CPU_CATALOG","build_run":build.name,
        "candidate_manifest_sha256":candidate["candidate_manifest_sha256"],"updated_at":utc_now()}
    atomic_json(path,status)
    print(f"CPU catalog preparation — progress: {path}",flush=True)
    try:
        spec = read_json(checkout / "configs/retrieval_full.json")["dense"]
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"],revision=spec["revision"],trust_remote_code=False)
        catalog,_ = load_scale_catalog(root,checkout,work,tokenizer)
        try:
            report = status | {"state":"CPU_PREPARATION_COMPLETE","catalog":catalog.base.identity,
                "documents":candidate["counts"]["documents"],"children":candidate["counts"]["children"],
                "unique_dense_inputs":inputs["input_count"],"updated_at":utc_now(),
                "scope":"Frozen data, model inputs and CPU catalog ready; corpus vectors still require GPU workers"}
        finally:
            catalog.close()
        atomic_json(path,report)
        return report
    except Exception as error:
        try:
            atomic_json(path,status | {"state":"FAILED","updated_at":utc_now(),
                "error":{"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}})
        except OSError:
            pass
        raise


def _document_inputs(catalog,tokenizer,target,max_length=512,part_size=1024):
    """Document title/opening representations streamed to bounded Parquet parts."""
    target = Path(target)
    identity = {"catalog":catalog.base.identity,"tokenizer":catalog.candidate["golden"]["tokenizer"],
        "max_length":max_length,"part_size":part_size,"builder":"document-title-bounded-opening-v1"}
    signature = digest_json(identity)
    if (target / "units.json").exists():
        saved = checked(read_json(target / "units.json"))
        if saved.get("signature") != signature:
            raise ValueError("Document representation policy changed.")
        return saved
    target.mkdir(parents=True,exist_ok=True)
    rows,parts,cursor = [],[],0
    def publish():
        nonlocal rows,cursor
        path = target / f"units-{cursor:012d}.parquet"
        marker = path.with_suffix(".done.json")
        rows_sha = digest_json(rows)
        if marker.exists():
            part = checked(read_json(marker))
            if part["signature"] != signature or part["rows_sha256"] != rows_sha:
                raise ValueError("Document input checkpoint changed.")
            verify_file(path,part["sha256"])
        else:
            from .artifacts import local_workspace,publish_file
            with local_workspace() as temporary:
                local = Path(temporary) / path.name
                pq.write_table(pa.Table.from_pylist(rows),local,compression="zstd")
                part = seal({"signature":signature,"start":cursor,"rows":len(rows),"path":path.name,
                    "rows_sha256":rows_sha,"sha256":publish_file(local,path)})
            atomic_json(marker,part)
        parts.append(part)
        cursor += len(rows)
        rows = []
    for identifier in catalog.documents:
        doc = catalog.documents[identifier]
        title,source = doc.get("title","").strip(),doc["source_text"]
        offsets = tokenizer(title,add_special_tokens=False,return_offsets_mapping=True,truncation=False,verbose=False)["offset_mapping"]
        if len(offsets) > 64:
            title = title[:offsets[63][1]]
        limit = max_length-tokenizer.num_special_tokens_to_add(pair=False)
        title_cost = len(tokenizer(title,add_special_tokens=False,truncation=False,verbose=False)["input_ids"])
        length = min(len(source),max_length*16)
        while True:
            offsets = tokenizer(source[:length],add_special_tokens=False,return_offsets_mapping=True,truncation=False,verbose=False)["offset_mapping"]
            if len(offsets) >= limit or length == len(source):
                break
            length = min(len(source),length*2)
        count = min(len(offsets),limit-title_cost-4)
        while count > 0:
            excerpt = source[:offsets[count-1][1]]
            text = title+"\n\n"+excerpt if title else excerpt
            if len(tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)["input_ids"]) <= limit:
                break
            count -= 1
        if count < 1:
            raise ValueError("Document cannot fit its embedding budget.")
        rows.append({"id":str(identifier),"text":text})
        if len(rows) == part_size:
            publish()
    if rows:
        publish()
    saved = seal(identity | {"signature":signature,"state":"COMPLETE","input_count":cursor,"parts":parts})
    atomic_json(target / "units.json",saved)
    return saved


def _store(root,target,manifest,work,cache_bytes):
    return ContentVectorParts(root,target,manifest,Path(work) / "content_vector_cache",local_cache_bytes=cache_bytes)


def baseline_seed_sources(root,candidate):
    """Discover verified ancestor baseline vectors when first upgrading a union."""
    pending,seen,result = list(candidate.get("sources",[])),set(),[]
    while pending:
        origin = pending.pop()
        sha = origin["candidate_manifest_sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        build = bound(Path(root) / "processed",origin["build_run"])
        source_path = build / f"candidate-{sha[:16]}.json"
        if not source_path.exists():
            continue  # union owns its source copies; ancestor cache is optional
        source = read_json(source_path)
        if digest_json({k:v for k,v in source.items() if k != "candidate_manifest_sha256"}) != sha:
            raise ValueError("Union ancestor candidate changed.")
        vectors = Path(root) / "retrieval" / (build.name+"-bge-bm25-v1-"+sha[:8]) / "corpus_embeddings"
        if (vectors / "config.json").exists() and (build / "index_inputs/units.json").exists():
            inputs = read_json(build / "index_inputs/units.json")
            if inputs["candidate_manifest_sha256"] != sha:
                raise ValueError("Union ancestor input/candidate mismatch.")
            result.append((build / "index_inputs",inputs,vectors))
        pending.extend(source.get("sources",[]))
    return result


def prepare_scale_resources(root,checkout,work,queries,*,catalog=None,worker_id=None,workers=1,
                            max_new_embedding_parts=None,max_new_translations=None,search_device="cuda"):
    """All four model roles, bounded corpus reads, query cache independent of corpus."""
    from transformers import AutoTokenizer
    root,checkout,work = Path(root),Path(checkout),Path(work)
    build,candidate,inputs = load_handoff(root)
    config = read_json(checkout / "configs/retrieval_full.json")
    scale = read_json(checkout / "configs/retrieval_scale.json")
    registry = read_json(checkout / "configs/strong_model_manifest.json")
    review_model_registry(config,registry)
    config["scale_execution"] = scale
    tokenizer = AutoTokenizer.from_pretrained(config["dense"]["model_id"],revision=config["dense"]["revision"],trust_remote_code=False)
    if catalog is None:
        catalog,analyzer = load_scale_catalog(root,checkout,work,tokenizer)
    else:
        analyzer = catalog.base.analyzer
    config["analyzer"] = analyzer.identity
    resources = root / "retrieval/full_resources" / candidate["candidate_manifest_sha256"][:16] / digest_json({
        "dense":config["dense"],"second_dense":config["second_dense"],"document_builder":"document-title-bounded-opening-v1"})[:12]
    query_root = root / "retrieval/full_query_cache" / digest_json({"queries":queries,
        "models":{k:config[k] for k in ("dense","second_dense","translation")},"expansion":config["expansion_enabled"],
        "query_code":[sha256_file(Path(__file__).with_name(n)) for n in ("query_translation.py","query_expansion.py","translation_model.py")]})[:20]
    profile = RuntimeProfile(resources / ("coordinator-profile.json" if worker_id is None else f"worker-{worker_id}-profile.json"))
    old_baseline = root / "retrieval" / (build.name+"-bge-bm25-v1-"+candidate["candidate_manifest_sha256"][:8])
    coordinator = worker_id is None
    encode_missing = not coordinator or workers == 1
    def pending(stage,report,input_parts=inputs["parts"]):
        profile.finish(stage)
        catalog.close()
        return {"state":stage,"completed_parts":len(report["parts"]),"requested_parts":len(input_parts),
            "worker_id":worker_id,"workers":workers,"resources":str(resources)}
    print(f"Full system: worker={worker_id}, team={workers}, resources={resources}",flush=True)
    dense = TorchDenseEncoder(config["dense"])
    try:
        cm = content_embeddings(root,build / "index_inputs",inputs,dense,resources / "bge",work_dir=work,
            batch_size=scale["bge_batch_size"],max_new_parts=max_new_embedding_parts,worker_id=worker_id,workers=workers,
            encode_missing=encode_missing,seed_root=old_baseline / "corpus_embeddings",seed_sources=baseline_seed_sources(root,candidate))
        if coordinator and cm["state"] != "COMPLETE":
            return pending("WAITING_FOR_BGE_CORPUS_PARTS",cm)
        if coordinator:
            query_units = [{"id":q["id"],"text":q["query"]} for q in queries]
            previous_query_manifest = old_baseline / "query_embeddings/embeddings.json"
            available = None
            if previous_query_manifest.exists() and read_json(previous_query_manifest).get("units_sha256") == unit_signature(query_units):
                available = reuse_bge_cache(old_baseline / "query_embeddings",query_units,config["dense"],part_size=256)
            if available is not None:
                qv,qm = available
                if qm["encoder"] != cm["encoder"]:
                    raise ValueError("BGE corpus/query encoder identity mismatch.")
            else:
                qm = embed_units(query_units,dense,query_root / "bge",work_dir=work,**config["embedding"])
                qv = embedding_matrix(query_root / "bge",qm)
            di = _document_inputs(catalog,tokenizer,resources / "document_inputs",config["dense"]["max_length"])
            dm = content_embeddings(root,resources / "document_inputs",di,dense,resources / "document_dense",work_dir=work,
                batch_size=scale["bge_batch_size"],max_new_parts=max_new_embedding_parts)
            if dm["state"] != "COMPLETE":
                return pending("DOCUMENT_EMBEDDINGS_IN_PROGRESS",dm,di["parts"])
    finally:
        dense.close()
    if coordinator:
        with profile.stage("query_translation_expansion"):
            translations_cached = cached_translations(queries,config["translation"],query_root / "translations")
            translations = translations_cached[0] if translations_cached else None
            expansions = cached_expansions(queries,translations,config["translation"],query_root / "expansions",
                enabled=config["expansion_enabled"]) if translations is not None else None
            if expansions is None:
                translator = TorchQueryTranslator(config["translation"])
                try:
                    if translations is None:
                        translations,report = translate_queries(queries,translator,query_root / "translations",max_new_queries=max_new_translations)
                        if report["state"] != "COMPLETE":
                            profile.finish("TRANSLATIONS_IN_PROGRESS")
                            catalog.close()
                            return {"state":"TRANSLATIONS_IN_PROGRESS","report":report}
                    expansions = expand_queries(queries,translator,query_root / "expansions",translations=translations,enabled=config["expansion_enabled"])
                finally:
                    translator.close()
        aux = auxiliary_units(queries,expansions)
    # Keep corpus and query encoding in the same model load. All other large
    # models have been released before Qwen is allocated on the GPU.
    encoder = TorchQwenEncoder(config["second_dense"])
    try:
        with profile.stage("qwen_corpus_embeddings"):
            sm = content_embeddings(root,build / "index_inputs",inputs,encoder.for_role("corpus"),resources / "qwen",work_dir=work,
                batch_size=inference_batches()["embedding"],max_new_parts=max_new_embedding_parts,worker_id=worker_id,workers=workers,
                encode_missing=encode_missing)
        if not coordinator:
            return pending("CORPUS_WORKER_CHECKPOINTED",sm)
        if sm["state"] != "COMPLETE":
            return pending("WAITING_FOR_QWEN_CORPUS_PARTS",sm)
        with profile.stage("qwen_query_embeddings"):
            am = embed_units(aux,encoder.for_role("query"),query_root / "qwen",work_dir=work,
                **(config["qwen_embedding"] | {"batch_size":inference_batches()["embedding"]}))
            av = embedding_matrix(query_root / "qwen",am)
            if {k:v for k,v in am["encoder"].items() if k != "input_role"} != {k:v for k,v in sm["encoder"].items() if k != "input_role"}:
                raise ValueError("Qwen corpus/query encoder identity mismatch.")
    finally:
        encoder.close()
    primary,secondary,documents = (_store(root,resources / name,m,work,scale["local_vector_cache_bytes_per_branch"])
        for name,m in (("bge",cm),("qwen",sm),("document_dense",dm)))
    searches = []
    for name,store,vectors in (("bge",primary,qv),("qwen",secondary,av),("documents",documents,qv)):
        with profile.stage("dense_search_"+name):
            searches.append(search_parts(store,vectors,resources / "search" / digest_json({"queries":queries,"aux":aux})[:16] / name,
                k=scale["dense_top_k"] if name != "documents" else config["retrieval"]["doc_candidate_k"],work_dir=work,device=search_device))
    ps,ss,ds = searches
    primary_search = {q["query"]:(ps[0][i],ps[1][i]) for i,q in enumerate(queries)}
    secondary_search = {u["id"]:(ss[0][i],ss[1][i]) for i,u in enumerate(aux)}
    document_search = {q["query"]:(ds[0][i],ds[1][i]) for i,q in enumerate(queries)}
    document_ids = [int(u) for p in di["parts"] for u in pq.read_table(resources / "document_inputs" / p["path"],columns=["id"]).column("id").to_pylist()]
    identity = {"catalog":catalog.identity,"primary":cm["manifest_sha256"],"secondary":sm["manifest_sha256"],
        "document_dense":dm["manifest_sha256"],"auxiliary":am["manifest_sha256"],"expansions":digest_json(expansions),
        "searches":[s[2]["manifest_sha256"] for s in searches],"scale":scale,"analyzer":analyzer.identity,
        "code_sha256":sha256_file(Path(__file__).with_name("disk_full.py"))}
    manifest = seal(identity | {"signature":digest_json(identity)})
    index = DiskStrongIndex(catalog,primary,secondary,qv,{u["id"]:v for u,v in zip(aux,av,strict=True)},queries,expansions,
        primary_search,secondary_search,document_search,document_ids,tokenizer,manifest)
    profile.finish("FULL_RESOURCES_READY")
    return {"state":"FULL_RESOURCES_READY","index":index,"vectors":qv,"query_manifest":qm,"translations":translations,
        "config":config,"budget":model_budget_report(config,registry),"registry":registry,"resources":str(resources)}


def scale_adapter(root,catalog,queries,config):
    path = Path(root) / "training/stage-a-qlora-v3-per-model-15b"
    pointer = checked(read_json(path / "active_adapter.json")) if (path / "active_adapter.json").exists() else None
    if pointer is not None:
        path = bound(path,pointer["directory"])
    if not (path / "calibrated_policy.json").exists():
        return None,None
    document = checked(read_json(path / "calibrated_policy.json"))
    adapter = checked(read_json(path / "selected_adapter/adapter_manifest.json"))
    if pointer is not None and (pointer["calibration_sha256"] != document["manifest_sha256"]
        or pointer["adapter_manifest_sha256"] != adapter["manifest_sha256"]):
        raise ValueError("Active trained adapter pointer changed.")
    if (document.get("reviewed") is not True or document.get("split") != "dev" or document["reranker_spec"] != config["reranker"]
        or adapter["manifest_sha256"] != document["adapter_manifest_sha256"]
        or adapter["base_model"] != config["reranker"]["model_id"] or adapter["base_revision"] != config["reranker"]["revision"]
        or set(document["dev_query_ids"]) & {q["id"] for q in queries}
        or set(document["dev_query_texts_sha256"]) & {digest_json(q["query"]) for q in queries}):
        raise ValueError("Adapter/dev selection integrity or contest leakage check failed.")
    if set(document["policy"]) != {"doc_top_k","doc_score_margin","chunk_top_k","chunk_score_margin"}:
        raise ValueError("Dev calibration may only change the reviewed output cutoffs.")
    for name,sha in adapter["files"].items():
        verify_file(bound(path / "selected_adapter",name),sha)
    exact = (document["catalog"] == catalog.identity and document["calibration_context"] == calibration_context(config,catalog.base.analyzer)
        and document["contest_queries_sha256"] == digest_json(queries))
    return document | {"applies_to_current_corpus":exact},path / "selected_adapter"


def run_scale_full_pipeline(data_root,checkout,*,code_commit,work_dir,worker_id=None,workers=1,
                            max_new_embedding_parts=None,max_new_translations=None,max_new_queries=None):
    root,checkout = Path(data_root),Path(checkout)
    queries = load_queries(parquet_path(root,"query.parquet"),expected_count=1200)
    prepared = prepare_scale_resources(root,checkout,work_dir,queries,worker_id=worker_id,workers=workers,
        max_new_embedding_parts=max_new_embedding_parts,max_new_translations=max_new_translations)
    if prepared["state"] != "FULL_RESOURCES_READY":
        return {"ready":{"state":prepared["state"],"query_count":0},"status":prepared,"samples":[],
            "diagnostics_path":prepared.get("resources",str(root / "retrieval/full_resources"))}
    index,config = prepared["index"],prepared["config"]
    try:
        calibration,adapter = scale_adapter(root,index.catalog,queries,config)
        policy = StrongConfig(**config["retrieval"])
        if calibration is not None:
            config["adapter_manifest_sha256"] = calibration["adapter_manifest_sha256"]
            if calibration["applies_to_current_corpus"]:
                policy = replace(policy,**calibration["policy"])
                config["retrieval"] = asdict(policy)
        budget = model_budget_report(config,prepared["registry"],adapter_paths=[adapter] if adapter else [])
        run = root / "retrieval/full_system" / digest_json({"catalog":index.catalog.identity,"config":config})[:20]
        contract = bind_pilot_run(run,plan_path=checkout / MASTER_PLAN,catalog=index.catalog,queries=queries,config=config,code_commit=code_commit)
        model = TorchQwenReranker(config["reranker"],adapter_path=adapter)
        try:
            records,report = predict_strong(index,queries,prepared["vectors"],model,run / "queries",
                query_embedding_manifest=prepared["query_manifest"],translations=prepared["translations"],config=policy,
                batch_size=inference_batches()["reranker"],max_new_queries=max_new_queries)
            model_identity = dict(model.identity)
        finally:
            model.close()
        status = {"state":report["state"],"architecture":config["architecture"],"query_count":len(records),
            "documents":len(index.catalog.documents),"full_plan_inference":True,"fine_tuned":adapter is not None,
            "model_parameter_budget":budget,"official_score":None,"full_corpus":False,
            "calibration":"DEV_SELECTED" if calibration and calibration["applies_to_current_corpus"] else "UNTUNED_ON_CURRENT_CORPUS",
            "candidate_search":"bounded global child top-k plus document dense plus translated lexical",
            "supervised_stage":"SELECTED_ADAPTER_LOADED" if adapter else "WAITING_FOR_REVIEWED_LABELS_AND_TRAINING",
            "index":index.manifest,"resources":prepared["resources"]}
        if records:
            diagnostic = diagnose(records,queries,index,run / "diagnostics.json")
            diagnostic["eligibility_scope"] = "Source counts cover the frozen corpus; quality holds are checked lazily on retrieved candidates."
            atomic_json(run / "diagnostics.json",diagnostic)
            status["diagnostics_path"] = str(run / "diagnostics.json")
        if report["state"] != "COMPLETE":
            ready = {"state":"QUERIES_IN_PROGRESS","query_count":len(records)}
        else:
            evidence = {"git_commit":code_commit,"master_plan":MASTER_PLAN,"index":index.manifest,"reranker":model_identity,
                "model_parameter_budget":budget,"retrieval_config":asdict(policy),"calibration":calibration,
                "inference_scope":"FULL_ARCHITECTURE_DISK_PARTIAL_CORPUS"}
            _,ready = finish_pilot(records,queries,index.catalog,report,run,contract=contract,tokenizer=index.tokenizer,evidence=evidence,work_dir=work_dir)
            status["state"] = ready["state"]
        diagnostics = run / "full_plan_status.json"
        atomic_json(diagnostics,status)
        return {"ready":ready,"status":status,"diagnostics_path":str(diagnostics),
            "samples":[{"query":q,"prediction":r["prediction"]} for q,r in zip(queries[:3],records[:3])]}
    finally:
        index.catalog.close()
