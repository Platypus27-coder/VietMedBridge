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
import threading
import time

from .artifacts import atomic_json, digest_json, publish_file, read_json
from .scale_vectors import bound, check_manifest

KINDS = ("documents","children","parents")
UNIT_COPY_ROWS = 65536


class _Progress:
    """Report long local SQL/I/O phases without touching Drive or the DB."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.phase = "start"
        self.started = time.monotonic()
        self.sql_steps = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def report(self, phase):
        self.phase = phase
        self._write()

    def _write(self):
        sizes = {}
        for name in ("seed-copy.sqlite", "seed-copy.sqlite-wal", "seed-copy.sqlite-journal"):
            try:
                sizes[name] = (self.directory / name).stat().st_size
            except FileNotFoundError:
                pass
        print(f"Catalog inheritance: {self.phase} | elapsed {time.monotonic()-self.started:.0f}s"
              f" | SQL steps {self.sql_steps:,} | local bytes {sum(sizes.values()):,}", flush=True)

    def _heartbeat(self):
        while not self.stop.wait(15):
            try:
                self._write()
            except OSError:
                pass  # Observability must not abort a valid SQL transaction.

    def sql_progress(self):
        self.sql_steps += 100000
        return 0

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def _copy_units(con, state, total, progress):
    """Commit the copy cursor with each bounded INSERT; retain it on interruption."""
    cursor = state["units_cursor"]
    while cursor < total:
        end = min(total, cursor+UNIT_COPY_ROWS)
        progress.report(f"copy dense positions {cursor:,}/{total:,}")
        with con:
            added = con.execute("INSERT INTO units SELECT position,id FROM progress.units "
                "WHERE position>=? AND position<? ORDER BY position", (cursor,end)).rowcount
            if added != end-cursor:
                raise ValueError("Partial unit position range changed before inheritance.")
            state["units_cursor"] = end
            con.execute("UPDATE metadata SET payload=?",(json.dumps(state,separators=(",",":")),))
        cursor = end
    progress.report(f"dense positions saved {cursor:,}/{total:,}")


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
    with _Progress(local_dir) as progress:
        _initialize_seed(local_dir,candidate,policy,seed,work_dir,progress)


def _initialize_seed(local_dir,candidate,policy,seed,work_dir,progress):
    from . import catalog_storage as storage
    local_dir = Path(local_dir)
    building = local_dir / "building.sqlite"
    inherited = set(seed["plan"]["part_indexes"])
    prefix = 0
    while prefix in inherited:
        prefix += 1
    previous = None
    if building.exists():
        progress.report("read existing local SQL checkpoint")
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
    temporary = local_dir / "seed-copy.sqlite"
    reuse = {"plan_manifest_sha256":seed["plan"]["manifest_sha256"],"source_catalog_sha256":saved["sha256"],
        "source_candidate":saved["candidate"],"part_indexes":sorted(inherited),"counts":saved["counts"]}
    state = None
    if temporary.exists():
        # A v7 run may already have committed the complete remap before being
        # interrupted in quick_check/copyfile. Do not restore 220 parts again.
        progress.report("inspect restored source / migration checkpoint")
        with sqlite3.connect(temporary) as pending:
            state = json.loads(pending.execute("SELECT payload FROM metadata").fetchone()[0])
        pending.close()
        if state.get("policy") == policy and state.get("seed") == reuse:
            if (state.get("builder") != "parallel-catalog-v1" or state.get("counts") != saved["counts"]
                or state.get("next_part") != prefix):
                raise ValueError("Partial inheritance state changed.")
            progress.report(f"resume inherited dense positions {state['units_cursor']:,}")
        else:
            state = None
    if state is None:
        progress.report("verify restored source bytes")
        if not storage._matches(temporary,saved["sha256"]):
            size = saved["storage"]["bytes"] if "storage" in saved else (seed["directory"] / saved["path"]).stat().st_size
            if shutil.disk_usage(local_dir).free < size+512*1024**2:
                raise OSError("Insufficient local disk to inherit the completed catalog; no source work was deleted.")
            cached = Path(work_dir) / seed["directory"].name / "catalog.sqlite"
            if storage._matches(cached,saved["sha256"]):
                if publish_file(cached,temporary) != saved["sha256"]:
                    raise ValueError("Inherited catalog copy changed.")
            else:
                progress.report("restore completed source catalog; keep existing BM25 postings")
                storage._restore(seed["directory"],saved,temporary)
        # _matches/publish_file/_restore have each verified the full checksum.
        # Do not read the same 14+ GB file for a second verification here.
    con = sqlite3.connect(temporary,uri=True)
    try:
        con.set_progress_handler(progress.sql_progress,100000)
        con.executescript("PRAGMA synchronous=NORMAL; PRAGMA cache_size=-131072; PRAGMA temp_store=FILE;")
        if state is None:
            source_policy = json.loads(con.execute("SELECT payload FROM metadata").fetchone()[0])
            if source_policy != {k:saved[k] for k in (
                    "candidate","inputs","analyzer","sqlite","schema","code_sha256","field_weights")}:
                raise ValueError("Inherited database producer policy differs from its manifest.")
            for kind in KINDS:
                progress.report(f"validate inherited {kind} row count")
                if con.execute(f"SELECT count(*) FROM {kind}").fetchone()[0] != saved["counts"][kind]:
                    raise ValueError("Inherited catalog row counts changed.")
            state = {"builder":"parallel-catalog-v1","policy":policy,"units_cursor":0,"next_part":prefix,
                "counts":saved["counts"],"seed":reuse,"copy_target":previous["units_cursor"] if previous else 0}
            progress.report("replace old dense-position map; retain BM25/source tables")
            con.executescript("BEGIN IMMEDIATE; DROP TABLE units; CREATE TABLE units(position INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL);")
            con.execute("UPDATE metadata SET payload=?",(json.dumps(state,separators=(",",":")),))
            con.commit()
        # Change the journal mode only after resumable metadata is committed.
        # An interruption during read-only source validation keeps its SHA valid.
        con.execute("PRAGMA journal_mode=WAL")
        total = state.get("copy_target",state["units_cursor"])
        cursor = state["units_cursor"]
        if not 0 <= cursor <= total:
            raise ValueError("Partial inheritance unit cursor changed.")
        progress.report(f"validate dense-position cursor {cursor:,}/{total:,}")
        count,first,last = con.execute("SELECT count(*),min(position),max(position) FROM units").fetchone()
        if count != cursor or (cursor and (first != 0 or last != cursor-1)):
            raise ValueError("Partial inheritance unit position range changed.")
        if previous is not None:
            con.execute("ATTACH DATABASE ? AS progress",(building.resolve().as_uri()+"?mode=ro",))
            if (previous["units_cursor"] != total
                or con.execute("SELECT count(*) FROM progress.units").fetchone()[0] != total):
                raise ValueError("Partial unit cursor changed before inheritance.")
            _copy_units(con,state,total,progress)
        elif cursor != total:
            raise ValueError("Partial source unit map is missing; keep the runtime and its local building.sqlite.")
        state.pop("copy_target",None)
        cursor = state["units_cursor"]
        with con:
            con.execute("UPDATE metadata SET payload=?",(json.dumps(state,separators=(",",":")),))
        # Source bytes were SHA-verified and only units/metadata have changed.
        # The final builder performs full quick_check after appending the delta;
        # checking the entire inherited FTS catalog here would duplicate that.
        progress.report("flush committed inheritance checkpoint")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("PRAGMA journal_mode=DELETE")
    finally:
        con.close()
    # On a normal interruption keep the old DB as well as the completed seed.
    # Atomic replacement below is the only point at which the new copy is active.
    if building.exists():
        backup = local_dir / "before-inheritance.sqlite"
        if not backup.exists():
            progress.report("preserve previous SQL checkpoint")
            try:
                os.link(building,backup)
            except OSError:
                shutil.copyfile(building,backup)
    os.replace(temporary,building)
    atomic_json(local_dir / "catalog_seed.json",{"policy":policy,"seed":reuse})
    print(f"Inherited catalog ready: skip {len(inherited)} existing source shards; preserved {cursor:,} unit positions",flush=True)
