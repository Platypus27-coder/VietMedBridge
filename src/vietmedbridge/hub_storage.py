"""Immutable, resumable Hub snapshots of the active pipeline's real artifacts.

Drive is a read-only migration source. Large immutable objects are committed in
bounded batches; the snapshot and latest pointer are published only after every
object is verified. GPU runtimes restore selected roles to their local disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil

import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, publish_file, read_json, sha256_file, verify_file
from .content_embeddings import checked, seal
from .portable_embeddings import relative_path, resources_relative
from .scale_benchmark import load_handoff
from .scale_vectors import bound

DEFAULT_REPO = "platypus123/biobridgeir-checkpoints"
SCHEMA = "vmb-hub-storage-v1"
PROFILES = {"bge_submit":{"metadata","catalog","bge"},
            "all":{"metadata","catalog","bge","qwen","sources","inputs"}}


def validate_plan(plan):
    checked(plan)
    if (plan.get("schema") != SCHEMA or not isinstance(plan.get("files"),list)
        or not re.fullmatch(r"[0-9a-f]{64}",plan.get("candidate",""))
        or not re.fullmatch(r"[0-9a-f]{64}",plan.get("inputs",""))):
        raise ValueError("Invalid pipeline storage snapshot.")
    seen, sizes = set(), {}
    for item in plan["files"]:
        relative_path(item["path"])
        if (item["path"] in seen or not re.fullmatch(r"[0-9a-f]{64}",item["sha256"])
            or type(item["bytes"]) is not int or item["bytes"] < 0
            or not item["roles"] or not set(item["roles"]) <= PROFILES["all"]):
            raise ValueError("Invalid or duplicate snapshot file.")
        seen.add(item["path"])
        if item["sha256"] in sizes and sizes[item["sha256"]] != item["bytes"]:
            raise ValueError("One content address cannot have different sizes.")
        sizes[item["sha256"]] = item["bytes"]
    return plan


def plan_active_storage(data_root, checkout, *, include_sources=True):
    """Inventory only the active corpus, its catalog and referenced vector blocks.

    Known checksums come from sealed pipeline manifests, avoiding a preliminary
    rehash of all vectors on Drive. Upload verifies each new payload before commit.
    Incoming TARs, old builds, model weights and live temporary files are excluded.
    """
    root = Path(data_root)
    build,candidate,inputs = load_handoff(root)
    full = read_json(Path(checkout) / "configs/retrieval_full.json")
    files = {}
    def add(relative, role, checksum=None):
        relative_path(relative)
        path = bound(root,relative)
        if not path.is_file():
            raise ValueError(f"Storage source artifact missing: {relative}")
        value = {"path":relative,"sha256":checksum or sha256_file(path),
                 "bytes":path.stat().st_size,"roles":[role]}
        if relative in files:
            old = files[relative]
            if old["sha256"] != value["sha256"] or old["bytes"] != value["bytes"]:
                raise ValueError("Snapshot source changed during inventory.")
            value["roles"] = sorted(set(old["roles"]) | {role})
        files[relative] = value
    active = read_json(root / "active_data_candidate.json")
    for path in ("active_data_candidate.json","raw/snapshot.json",
                 f"processed/{build.name}/{active['candidate_name']}",
                 f"processed/{build.name}/config.json",f"processed/{build.name}/index_inputs/units.json"):
        add(path,"metadata")
    for path in ("retrieval_embedding_runtime_lock.json","labels/medical_aliases.json"):
        if (root / path).is_file():
            add(path,"metadata")
    for entry in read_json(root / "raw/snapshot.json")["files"].values():
        add(entry["path"],"metadata",entry["sha256"])
    for part in inputs["parts"]:
        add(f"processed/{build.name}/index_inputs/{part['path']}","inputs",part["sha256"])
    if include_sources:
        for part in candidate["parts"]:
            for entry in part["files"].values():
                add(f"processed/{build.name}/{entry['path']}","sources",entry["sha256"])
    coverage = {}
    for family,spec_name in (("bge","dense"),("qwen","second_dense")):
        view = root / resources_relative(candidate,full) / family
        refs, complete = {},0
        if (view / "config.json").is_file():
            identity = read_json(view / "config.json")
            if (identity["input_manifest_sha256"] != inputs["manifest_sha256"]
                or identity["input_count"] != inputs["input_count"]
                or any(identity["encoder"].get(k) != v for k,v in full[spec_name].items())):
                raise ValueError("Storage vector view belongs to another model/corpus.")
            add((view / "config.json").relative_to(root).as_posix(),family)
            for part in inputs["parts"]:
                marker = view / f"map-{part['start']:012d}.done.json"
                if not marker.is_file():
                    continue
                saved = checked(read_json(marker))
                if any(saved.get(k) != v for k,v in {"signature":digest_json(identity),
                    "start":part["start"],"rows":part["rows"],"input_sha256":part["sha256"],
                    "path":f"map-{part['start']:012d}.parquet"}.items()):
                    raise ValueError("Storage mapping differs from frozen input order.")
                mapping = bound(view,saved["path"])
                verify_file(mapping,saved["sha256"])
                rows = pq.read_table(mapping).to_pylist()
                if len(rows) != part["rows"]:
                    raise ValueError("Storage mapping row count differs.")
                for ref in rows:
                    relative_path(ref["path"])
                    if ref["path"] in refs and refs[ref["path"]] != ref["sha256"]:
                        raise ValueError("Conflicting vector checksums.")
                    refs[ref["path"]] = ref["sha256"]
                add(mapping.relative_to(root).as_posix(),family,saved["sha256"])
                add(marker.relative_to(root).as_posix(),family)
                complete += 1
            cache_identity = {"encoder":identity["encoder"],"dimension":identity["dimension"],"format":"content-float32-l2-v1"}
            directory = root / "model_cache" / digest_json(cache_identity)[:24]
            found = set()
            for marker in sorted(directory.glob("*.done.json")):
                try:
                    block = read_json(marker)
                except ValueError:
                    continue  # referenced blocks still must be found below
                if block.get("path") not in refs:
                    continue
                checked(block)
                if block["identity"] != cache_identity or block["sha256"] != refs[block["path"]]:
                    raise ValueError("Storage vector producer differs.")
                add(marker.relative_to(root).as_posix(),family)
                add(block["path"],family,block["sha256"])
                found.add(block["path"])
            if found != set(refs):
                raise ValueError("A referenced vector block has no producer receipt.")
        coverage[family] = {"completed_parts":complete,"total_parts":len(inputs["parts"]),
                            "complete":complete == len(inputs["parts"])}
    catalogs, uncompressed = 0,0
    for marker in sorted((root / "retrieval/catalogs").glob("*/catalog.json")):
        saved = read_json(marker)
        if saved.get("candidate") != candidate["candidate_manifest_sha256"] or saved.get("inputs") != inputs["manifest_sha256"]:
            continue
        checked(saved)
        if saved.get("state") != "COMPLETE":
            continue
        add(marker.relative_to(root).as_posix(),"catalog")
        entries = saved["storage"]["parts"] if "storage" in saved else [saved]
        for entry in entries:
            add((marker.parent / entry["path"]).relative_to(root).as_posix(),"catalog",entry["sha256"])
        size = saved["storage"]["bytes"] if "storage" in saved else (marker.parent / saved["path"]).stat().st_size
        uncompressed = max(uncompressed,size)
        catalogs += 1
    return seal({"schema":SCHEMA,"candidate":candidate["candidate_manifest_sha256"],
        "inputs":inputs["manifest_sha256"],"counts":candidate["counts"],"coverage":coverage,
        "catalogs":catalogs,"catalog_uncompressed_bytes":uncompressed,
        "includes_source_shards":include_sources,"retrieval_config_sha256":digest_json(full),
        "files":sorted(files.values(),key=lambda item:item["path"])})


def storage_summary(plan):
    validate_plan(plan)
    objects = {item["sha256"]:item["bytes"] for item in plan["files"]}
    return {"state":"STORAGE_PLAN_READY","snapshot":plan["manifest_sha256"],
        "files":len(plan["files"]),"unique_objects":len(objects),"bytes":sum(objects.values()),
        "decimal_GB":round(sum(objects.values()) / 1e9,3),"counts":plan["counts"],
        "coverage":plan["coverage"],"catalogs":plan["catalogs"],
        "scope":"Active candidate only; incoming archives and old builds remain at their source."}


def object_path(item):
    return f"objects/{item['sha256'][:2]}/{item['sha256']}"


def _download(repo_id,filename,revision,cache,api):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id,filename,repo_type="dataset",revision=revision,
        cache_dir=cache,token=api.token))


def _remote_matches(repo_id,item,remote,revision,cache,api):
    if remote is None:
        return False
    if remote.size != item["bytes"]:
        raise ValueError("Hub object size changed at an immutable content address.")
    if remote.lfs is not None:
        if remote.lfs.sha256 != item["sha256"]:
            raise ValueError("Hub object checksum differs from its content address.")
    else:
        verify_file(_download(repo_id,object_path(item),revision,cache,api),item["sha256"])
    return True


def upload_storage(data_root,plan,repo_id=DEFAULT_REPO,*,work_dir,max_snapshot_bytes=90_000_000_000,api=None):
    """Idempotent after loss of local state; never delete or write to the source."""
    from huggingface_hub import CommitOperationAdd, HfApi
    validate_plan(plan)
    if type(max_snapshot_bytes) is not int or max_snapshot_bytes < 1:
        raise ValueError("Set an explicit positive snapshot byte budget.")
    summary = storage_summary(plan)
    if summary["bytes"] > max_snapshot_bytes:
        return summary | {"state":"UPLOAD_NOT_STARTED_BUDGET_EXCEEDED","budget_bytes":max_snapshot_bytes}
    root,work = Path(data_root),Path(work_dir)
    if work.resolve().is_relative_to(root.resolve()):
        raise ValueError("Migration work_dir must be outside Drive DATA_ROOT.")
    work.mkdir(parents=True,exist_ok=True)
    api = api or HfApi()
    api.create_repo(repo_id,repo_type="dataset",private=True,exist_ok=True)
    info = api.dataset_info(repo_id)
    if info.private is not True:
        raise ValueError("Checkpoint storage requires a PRIVATE dataset repo. Set its visibility to private, then retry.")
    objects = {object_path(item):item for item in plan["files"]}
    items = list(objects.values())
    uploaded,reused = 0,0
    # One bounded transaction at a time: verified prior objects survive every
    # interruption, even when the runtime's progress files have disappeared.
    for start in range(0,len(items),20):
        batch = items[start:start+20]
        remote = {f.path:f for f in api.get_paths_info(repo_id,[object_path(i) for i in batch],
            repo_type="dataset",revision="main")}
        missing = []
        for item in batch:
            if _remote_matches(repo_id,item,remote.get(object_path(item)),"main",work / "hub-cache",api):
                reused += 1
            else:
                source = bound(root,item["path"])
                verify_file(source,item["sha256"])
                missing.append(CommitOperationAdd(path_in_repo=object_path(item),path_or_fileobj=str(source)))
        if missing:
            api.create_commit(repo_id,repo_type="dataset",operations=missing,
                commit_message=f"Store immutable pipeline objects {start+1}-{start+len(batch)}",num_threads=4)
            uploaded += len(missing)
        # Verify the committed server hashes before counting the batch complete.
        remote = {f.path:f for f in api.get_paths_info(repo_id,[object_path(i) for i in batch],
            repo_type="dataset",revision="main")}
        for item in batch:
            if not _remote_matches(repo_id,item,remote.get(object_path(item)),"main",work / "hub-cache",api):
                raise ValueError("Hub commit did not publish every required object.")
        progress = summary | {"state":"UPLOADING","new_objects":uploaded,"reused_objects":reused,
            "verified_objects":start+len(batch),"repo_id":repo_id}
        atomic_json(work / "storage_upload_progress.json",progress)
        print(f"HF storage: {start+len(batch)}/{len(items)} objects verified | new {uploaded}, reused {reused}",flush=True)
    manifest_path = f"snapshots/{plan['manifest_sha256']}.json"
    manifest_commit = api.upload_file(repo_id=repo_id,repo_type="dataset",path_in_repo=manifest_path,
        path_or_fileobj=(json.dumps(plan,ensure_ascii=False)+"\n").encode(),commit_message="Seal verified active pipeline snapshot")
    saved = checked(read_json(_download(repo_id,manifest_path,manifest_commit.oid,work / "hub-cache",api)))
    if saved != plan:
        raise ValueError("Published Hub snapshot differs from verified inventory.")
    pointer = seal({"schema":SCHEMA,"manifest":manifest_path,"snapshot_sha256":plan["manifest_sha256"]})
    commit = api.upload_file(repo_id=repo_id,repo_type="dataset",path_in_repo="vmb-latest.json",
        path_or_fileobj=json.dumps(pointer).encode(),commit_message="Publish completed pipeline storage snapshot")
    revision = commit.oid
    if checked(read_json(_download(repo_id,"vmb-latest.json",revision,work / "hub-cache",api))) != pointer:
        raise ValueError("Published Hub locator differs from the verified snapshot.")
    result = summary | {"state":"HF_STORAGE_VERIFIED","repo_id":repo_id,"revision":revision,
        "manifest":manifest_path,"new_objects":uploaded,"reused_objects":reused,
        "source_deleted":False,"source_writes":False}
    atomic_json(work / "hf-storage-locator.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def restore_storage(repo_id=DEFAULT_REPO,*,data_root,work_dir,profile="bge_submit",revision=None,api=None,checkout=None):
    from huggingface_hub import HfApi
    if profile not in PROFILES:
        raise ValueError("Storage profile must be bge_submit or all.")
    root,work = Path(data_root),Path(work_dir)
    root.mkdir(parents=True,exist_ok=True)
    work.mkdir(parents=True,exist_ok=True)
    api = api or HfApi()
    revision = api.dataset_info(repo_id,revision=revision or "main").sha
    if not re.fullmatch(r"[0-9a-f]{40}",revision):
        raise ValueError("Hub snapshot needs an immutable commit SHA.")
    cache = work / "hub-cache"
    pointer = checked(read_json(_download(repo_id,"vmb-latest.json",revision,cache,api)))
    if (pointer.get("schema") != SCHEMA or not re.fullmatch(r"[0-9a-f]{64}",pointer.get("snapshot_sha256",""))
        or pointer.get("manifest") != f"snapshots/{pointer.get('snapshot_sha256')}.json"):
        raise ValueError("Invalid Hub storage locator.")
    plan = validate_plan(read_json(_download(repo_id,pointer["manifest"],revision,cache,api)))
    if plan["manifest_sha256"] != pointer["snapshot_sha256"]:
        raise ValueError("Hub storage locator/manifest differ.")
    if profile == "bge_submit" and (not plan["coverage"]["bge"]["complete"] or not plan["catalogs"]):
        raise ValueError("BGE submit requires complete BGE and a completed CPU catalog in this snapshot.")
    if checkout is not None and digest_json(read_json(Path(checkout) / "configs/retrieval_full.json")) != plan["retrieval_config_sha256"]:
        raise ValueError("Hub snapshot belongs to another retrieval model/configuration.")
    selected = [item for item in plan["files"] if set(item["roles"]) & PROFILES[profile]]
    selected.sort(key=lambda item:(0 if "metadata" in item["roles"] else
        1 if "catalog" in item["roles"] and item["path"].endswith("/catalog.json") else 2,item["path"]))
    binding_path = root / "hf_storage_binding.json"
    if binding_path.exists() and read_json(binding_path)["snapshot"] != plan["manifest_sha256"]:
        raise ValueError("Local DATA_ROOT contains another Hub snapshot; use a fresh local data folder.")
    missing = [item for item in selected if not bound(root,item["path"]).is_file()]
    # Budget the logical data, uncompressed catalog and a vector/search reserve.
    needed = sum({i["sha256"]:i["bytes"] for i in missing}.values()) + plan["catalog_uncompressed_bytes"] + 2 * 2**30
    if needed > shutil.disk_usage(root).free:
        raise ValueError(f"Insufficient runtime disk: need approximately {needed/1e9:.1f} GB for selected data/catalog; use a runtime with more disk.")
    binding = {"schema":SCHEMA,"repo_id":repo_id,"revision":revision,"snapshot":plan["manifest_sha256"],
        "profile":profile,"candidate":plan["candidate"],"retrieval_config_sha256":plan["retrieval_config_sha256"]}
    atomic_json(binding_path,binding | {"state":"STAGING"})
    checked_catalog = profile != "bge_submit" or checkout is None
    for index,item in enumerate(selected,1):
        if not checked_catalog and "metadata" not in item["roles"] and not item["path"].endswith("/catalog.json"):
            from .catalog_storage import _contract
            from .medical_lexical import MedicalAnalyzer
            build,candidate,inputs = load_handoff(root)
            config = read_json(Path(checkout) / "configs/retrieval_full.json")
            glossary = root / "labels/medical_aliases.json"
            analyzer = MedicalAnalyzer(segmentation=config["lexical_segmentation"],
                glossary_path=glossary if glossary.is_file() else None)
            expected = digest_json(_contract(build,candidate,inputs,analyzer))
            if not any(checked(read_json(path))["signature"] == expected
                for path in (root / "retrieval/catalogs").glob("*/catalog.json")):
                raise ValueError("Saved CPU catalog differs from this runtime. Use the original SQLite/analyzer runtime or restore all sources to rebuild locally; no large vectors downloaded.")
            checked_catalog = True
        target = bound(root,item["path"])
        if target.exists():
            verify_file(target,item["sha256"])
        else:
            source = _download(repo_id,object_path(item),revision,cache,api)
            verify_file(source,item["sha256"])
            target.parent.mkdir(parents=True,exist_ok=True)
            # Hub cache and logical data share immutable bytes on local disk.
            try:
                os.link(source,target)
            except OSError:
                if publish_file(source,target) != item["sha256"]:
                    raise ValueError("Restored Hub object checksum changed.")
        if index % 100 == 0 or index == len(selected):
            print(f"HF restore ({profile}): {index}/{len(selected)} verified files",flush=True)
    build,candidate,inputs = load_handoff(root)
    if candidate["candidate_manifest_sha256"] != plan["candidate"] or inputs["manifest_sha256"] != plan["inputs"]:
        raise ValueError("Restored active candidate differs from the Hub snapshot.")
    if profile == "bge_submit" and checkout is not None:
        from .scale_baseline import imported_bge_manifest
        _, bge = imported_bge_manifest(root,candidate,inputs,read_json(Path(checkout) / "configs/retrieval_full.json"))
        if bge is None or bge["state"] != "COMPLETE":
            raise ValueError("Hub snapshot omitted required BGE maps; local data is not ready for submit.")
    binding = binding | {"restored_files":len(selected),"state":"HF_DATA_READY_LOCAL"}
    atomic_json(binding_path,binding)
    return binding
