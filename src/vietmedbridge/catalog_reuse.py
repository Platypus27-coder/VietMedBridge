"""Inherit a verified catalog when an available scope includes its entire source.

Scope receipts prove the selected successful rows are unchanged even when
Parquet encoding differs. Never seed an official whitelist, partial selection,
different analyzer, different chunk policy or a different source snapshot.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sqlite3

from .artifacts import atomic_json, digest_json, publish_file, read_json, verify_file
from .scale_vectors import bound, check_manifest

KINDS = ("documents","children","parents")


def _sealed(value):
    if digest_json({k:v for k,v in value.items() if k != "manifest_sha256"}) != value.get("manifest_sha256"):
        raise ValueError("Catalog inheritance receipt changed.")
    return value


def _candidate(value):
    if (value.get("state") != "FROZEN_CANDIDATE" or not value.get("selected_range_complete")
        or not value.get("integrity",{}).get("passed") or value["integrity"].get("official_membership") != "VERIFIED"
        or digest_json({k:v for k,v in value.items() if k != "candidate_manifest_sha256"}) != value.get("candidate_manifest_sha256")
        or digest_json({"signature":value["signature"],"parts":value["parts"]}) != value.get("snapshot_sha256")):
        raise ValueError("Catalog inheritance requires a validated frozen source.")


def _exact_prefix(old,current):
    def key(part):
        return tuple(part["files"][kind]["sha256"] for kind in KINDS)
    now = {key(part):index for index,part in enumerate(current["parts"])}
    parts = [part for part in old["parts"] if part["counts"].get("documents",0)]
    if any(key(part) not in now for part in parts):
        return None
    indexes = sorted(now[key(part)] for part in parts)
    if indexes != list(range(len(parts))):
        return None
    if any(sum(current["parts"][i]["counts"].get(k,0) for i in indexes) != old["counts"][k] for k in KINDS):
        return None
    return indexes


def find_seed(build,candidate,inputs,catalog_root,policy):
    """Return the largest complete catalog for the first, unfiltered source."""
    if candidate.get("input_kind") != "AVAILABLE_CORPUS" or not candidate.get("source_builds"):
        return None
    build,catalog_root = Path(build),Path(catalog_root)
    config = read_json(build / "config.json")
    if config.get("selection") != "all-validated-successes-keep-legacy-urls-v1":
        return None
    source = candidate["source_builds"][0]
    old_build = bound(build.parent,source["build_run"])
    options = []
    for marker in sorted(catalog_root.glob("catalog-*/catalog.json")):
        saved = read_json(marker)
        if saved.get("candidate") == candidate["candidate_manifest_sha256"]:
            continue
        filename = f"candidate-{str(saved.get('candidate',''))[:16]}.json"
        source_files = [old_build / filename,*build.parent.glob("available-*/"+filename)]
        source_file = next((p for p in source_files if p.is_file()),None)
        if source_file is None:
            continue
        old = read_json(source_file)
        exact = _exact_prefix(old,candidate)
        if exact is None and old.get("snapshot_sha256") != source["snapshot_sha256"]:
            continue
        _candidate(old)
        check_manifest(saved)
        if (old["candidate_manifest_sha256"] != saved["candidate"]
            or old["origin_corpus_sha256"] != candidate["origin_corpus_sha256"]
            or old["chunking"] != candidate["chunking"]
            or old["golden"]["tokenizer"] != candidate["golden"]["tokenizer"]
            or saved.get("signature") != digest_json({k:saved[k] for k in (
                "candidate","inputs","analyzer","sqlite","schema","code_sha256","field_weights")})
            or saved.get("path") != "catalog.sqlite"
            or saved.get("counts") != {k:old["counts"][k] for k in KINDS}):
            raise ValueError("Existing source catalog provenance changed; refusing to reindex it silently.")
        if any(saved.get(k) != policy[k] for k in ("analyzer","schema","code_sha256","field_weights")):
            raise ValueError("Existing source catalog analyzer/encoding differs; cannot inherit its BM25 postings.")
        old_inputs = read_json(source_file.parent / "index_inputs/units.json")
        check_manifest(old_inputs)
        if (old_inputs.get("candidate_manifest_sha256") != old["candidate_manifest_sha256"]
            or old_inputs["manifest_sha256"] != saved["inputs"] or old_inputs["input_count"] != saved["unit_count"]):
            raise ValueError("Existing source catalog input binding changed.")
        options.append((saved["counts"]["documents"],marker,old,saved,exact))
    if not options:
        return None
    _,marker,old,saved,exact = max(options,key=lambda item:item[0])
    binding = {"format":"catalog-inheritance-v1","candidate":candidate["candidate_manifest_sha256"],
        "source":source,"source_candidate":old["candidate_manifest_sha256"],
        "source_catalog_manifest":saved["manifest_sha256"],"analyzer":policy["analyzer"],
        "proof":"exact-source-parts" if exact is not None else "first-source-selection-receipts"}
    plan_file = catalog_root / ("reuse-"+digest_json(binding)[:24]+".json")
    if plan_file.is_file():
        plan = _sealed(read_json(plan_file))
        check_manifest(plan)
        if any(plan.get(k) != v for k,v in binding.items()):
            raise ValueError("Catalog inheritance plan binding changed.")
    else:
        by_part = {digest_json(part):index for index,part in enumerate(candidate["parts"])}
        def verify_receipt(index):
            part = old["parts"][index]
            path = build / "scope_checkpoints" / f"source-000-{index:06d}.json"
            receipt = _sealed(read_json(path))
            witness = {"scope_signature":candidate["signature"],"source_build":source["build_run"],
                "source_snapshot":source["snapshot_sha256"],"source_part":part}
            selected = receipt.get("selected_ids",[])
            target = receipt.get("part")
            if receipt.get("witness") != witness or len(selected) != len(set(selected)):
                raise ValueError("Catalog source selection receipt changed.")
            if not part["counts"].get("documents",0):
                if target is not None or selected or any(part["counts"].get(k,0) for k in KINDS):
                    raise ValueError("Empty source shard selection changed.")
                return None
            if (len(selected) != part["counts"]["documents"] or target is None
                or any(target["counts"][k] != part["counts"][k] for k in KINDS)
                or target["records"] != len(selected) or digest_json(target) not in by_part):
                raise ValueError("Scope does not retain the whole source catalog; inheritance is unsafe.")
            return by_part[digest_json(target)]
        indexes = exact if exact is not None else []
        if exact is None:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for completed,index in enumerate(pool.map(verify_receipt,range(len(old["parts"]))),1):
                    if index is not None:
                        indexes.append(index)
                    if completed % 64 == 0 or completed == len(old["parts"]):
                        print(f"Catalog inheritance proof: {completed}/{len(old['parts'])} small source receipts",flush=True)
        if sorted(indexes) != list(range(len(indexes))):
            raise ValueError("Source catalog does not cover the unchanged target prefix.")
        totals = {k:sum(candidate["parts"][i]["counts"][k] for i in indexes) for k in KINDS}
        if totals != saved["counts"]:
            raise ValueError("Inherited source coverage/counts changed.")
        plan = binding | {"part_indexes":sorted(indexes),"counts":totals,
            "source_sha256":saved["sha256"],"state":"COMPLETE"}
        plan["manifest_sha256"] = digest_json(plan)
        atomic_json(plan_file,plan)
    indexes = plan["part_indexes"]
    if (indexes != list(range(len(indexes))) or any(type(i) is not int or not 0 <= i < len(candidate["parts"]) for i in indexes)
        or {k:sum(candidate["parts"][i]["counts"][k] for i in indexes) for k in KINDS} != saved["counts"]
        or plan["counts"] != saved["counts"] or plan["source_sha256"] != saved["sha256"]):
        raise ValueError("Catalog inheritance plan coverage changed.")
    print(f"Catalog reuse: {plan['counts']['documents']:,} documents, {len(indexes)}/{len(candidate['parts'])} shards already indexed;"
          f" delta {candidate['counts']['documents']-plan['counts']['documents']:,} documents",flush=True)
    return {"plan":plan,"manifest":saved,"directory":marker.parent}


def initialize_seed(local_dir,candidate,policy,seed,work_dir):
    """Clone completed bytes; a failed clone never replaces the local progress DB."""
    if seed is None:
        return
    from . import catalog_storage as storage
    local_dir = Path(local_dir)
    building = local_dir / "building.sqlite"
    inherited = set(seed["plan"]["part_indexes"])
    prefix = 0
    while prefix in inherited:
        prefix += 1
    previous = None
    if building.exists():
        with sqlite3.connect(building) as con:
            previous = json.loads(con.execute("SELECT payload FROM metadata").fetchone()[0])
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.execute("PRAGMA journal_mode=DELETE")
        con.close()
        if previous == policy:
            return
        if previous.get("policy") != policy:
            raise ValueError("Partial catalog policy changed before inheritance.")
        if previous.get("seed"):
            if previous["seed"]["plan_manifest_sha256"] != seed["plan"]["manifest_sha256"]:
                raise ValueError("Partial catalog inheritance source changed.")
            return
        # Work beyond the inherited source is already useful: keep it intact.
        if any(i not in inherited for i in range(previous["next_part"])):
            print("Catalog local progress already includes delta shards; preserving it",flush=True)
            return
    saved = seed["manifest"]
    size = saved["storage"]["bytes"] if "storage" in saved else (seed["directory"] / saved["path"]).stat().st_size
    if shutil.disk_usage(local_dir).free < size+512*1024**2:
        raise OSError("Insufficient local disk to inherit the completed catalog; no source work was deleted.")
    temporary = local_dir / "seed-copy.sqlite"
    if not storage._matches(temporary,saved["sha256"]):
        cached = Path(work_dir) / seed["directory"].name / "catalog.sqlite"
        if storage._matches(cached,saved["sha256"]):
            publish_file(cached,temporary)
        else:
            print("Restore completed source catalog once; retaining existing BM25 postings",flush=True)
            storage._restore(seed["directory"],saved,temporary)
    verify_file(temporary,saved["sha256"])
    con = sqlite3.connect(temporary,uri=True)
    try:
        source_policy = json.loads(con.execute("SELECT payload FROM metadata").fetchone()[0])
        if source_policy != {k:saved[k] for k in (
                "candidate","inputs","analyzer","sqlite","schema","code_sha256","field_weights")}:
            raise ValueError("Inherited database producer policy differs from its manifest.")
        if any(con.execute(f"SELECT count(*) FROM {kind}").fetchone()[0] != saved["counts"][kind] for kind in KINDS):
            raise ValueError("Inherited catalog row counts changed.")
        con.executescript("BEGIN IMMEDIATE; DROP TABLE units; CREATE TABLE units(position INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL);")
        cursor = 0
        if previous is not None:
            con.execute("ATTACH DATABASE ? AS progress",(building.resolve().as_uri()+"?mode=ro",))
            cursor = previous["units_cursor"]
            if con.execute("SELECT count(*) FROM progress.units").fetchone()[0] != cursor:
                raise ValueError("Partial unit cursor changed before inheritance.")
            con.execute("INSERT INTO units SELECT * FROM progress.units")
        reuse = {"plan_manifest_sha256":seed["plan"]["manifest_sha256"],"source_catalog_sha256":saved["sha256"],
            "source_candidate":saved["candidate"],"part_indexes":sorted(inherited),"counts":saved["counts"]}
        state = {"builder":"parallel-catalog-v1","policy":policy,"units_cursor":cursor,"next_part":prefix,
            "counts":saved["counts"],"seed":reuse}
        con.execute("DELETE FROM metadata")
        con.execute("INSERT INTO metadata VALUES (?)",(json.dumps(state,ensure_ascii=False,separators=(",",":")),))
        con.commit()
        if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Inherited SQLite integrity check failed.")
    finally:
        con.close()
    # On a normal interruption keep the old DB as well as the completed seed.
    # Atomic replacement below is the only point at which the new copy is active.
    if building.exists():
        backup = local_dir / "before-inheritance.sqlite"
        shutil.copyfile(building,backup)
    os.replace(temporary,building)
    atomic_json(local_dir / "catalog_seed.json",{"policy":policy,"seed":reuse})
    print(f"Inherited catalog ready: skip {len(inherited)} existing source shards; preserved {cursor:,} unit positions",flush=True)
