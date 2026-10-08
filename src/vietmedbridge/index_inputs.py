"""Prepare deduplicated dense inputs on disk, without an in-memory pilot catalog."""
from __future__ import annotations

from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, utc_now, verify_file
from .dataset import _connection
from .representation import BUILDER_VERSION, build_dense_text, representation_hash
from .validation import open_views

UNIT_SCHEMA = pa.schema([("id",pa.string()),("text",pa.large_string())])
LINEAGE_SCHEMA_VERSION = 1


def _part_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Index input part escapes its directory.")
    return path


def prepare_index_inputs(build_dir, candidate_name, tokenizer, *, work_dir=None, part_size=4096):
    if type(part_size) is not int or part_size < 1:
        raise ValueError("Invalid input part size.")
    root = Path(build_dir).resolve()
    path = (root / candidate_name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Candidate path escapes its build.")
    candidate = read_json(path)
    if (candidate.get("state") != "FROZEN_CANDIDATE" or not candidate.get("selected_range_complete")
        or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]):
        raise ValueError("Complete and freeze every imported input shard first.")
    if candidate["golden"]["tokenizer"] != read_json(root / "config.json")["tokenizer"]:
        raise ValueError("Index input tokenizer contract mismatch.")
    target = root / "index_inputs"
    target.mkdir(parents=True,exist_ok=True)
    policy = {"candidate_manifest_sha256":candidate["candidate_manifest_sha256"],
        "tokenizer":candidate["golden"]["tokenizer"],"builder":BUILDER_VERSION,
        "dense_max_tokens":candidate["chunking"]["dense_max_tokens"],"part_size":part_size}
    signature = digest_json(policy)
    config_path = target / "config.json"
    if config_path.exists() and read_json(config_path) != policy:
        raise ValueError("Index input source/policy changed; use a new build.")
    atomic_json(config_path,policy)
    manifest_path = target / "units.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if (saved["signature"] != signature or saved["state"] != "COMPLETE"
            or digest_json({k:v for k,v in saved.items() if k != "manifest_sha256"}) != saved["manifest_sha256"]):
            raise ValueError("Index input manifest changed.")
        for part in saved["parts"]:
            verify_file(_part_path(target,part["path"]),part["sha256"])
        return saved
    started = time.perf_counter()
    parts, cursor = [],0
    with local_workspace(work_dir) as temporary:
        with _connection(temporary) as con:
            open_views(con,root,candidate)
            # Keep all official aliases in the candidate/health mapping. Only the
            # model's identical text is encoded once, sorted by its content hash.
            con.execute("""CREATE TABLE representatives AS SELECT row_number() OVER(ORDER BY retrieval_representation_hash)-1 AS position,*
                FROM (SELECT retrieval_representation_hash,min(chunk_id) AS chunk_id FROM children GROUP BY 1)""")
            total = con.execute("SELECT count(*) FROM representatives").fetchone()[0]
            while cursor < total:
                stem = f"units-{cursor:012d}"
                marker = target / (stem+".done.json")
                expected = min(part_size,total-cursor)
                if marker.exists():
                    part = read_json(marker)
                    if part["signature"] != signature or part["start"] != cursor or part["rows"] != expected:
                        raise ValueError("Index input checkpoint changed.")
                    verify_file(_part_path(target,part["path"]),part["sha256"])
                else:
                    rows = con.execute("""SELECT r.retrieval_representation_hash,c.text,c.heading,d.title
                        FROM (SELECT * FROM representatives WHERE position>=? AND position<?) r
                        JOIN children c USING(chunk_id) JOIN documents d USING(doc_id) ORDER BY r.position""",
                        [cursor,cursor+expected]).fetchall()
                    units = []
                    for key,text,heading,title in rows:
                        dense = build_dense_text({"text":text},title=title,heading=heading,tokenizer=tokenizer,
                            max_tokens=policy["dense_max_tokens"])
                        if representation_hash(dense) != key:
                            raise ValueError("Stored representation differs from the pinned tokenizer/builder.")
                        units.append({"id":key,"text":dense})
                    local = Path(temporary)/(stem+".parquet")
                    pq.write_table(pa.Table.from_pylist(units,schema=UNIT_SCHEMA),local,compression="zstd")
                    part = {"signature":signature,"start":cursor,"rows":len(units),"path":local.name,
                        "sha256":publish_file(local,target/local.name)}
                    atomic_json(marker,part)
                parts.append(part);cursor += part["rows"]
                print(f"Index inputs: {cursor:,}/{total:,} unique texts",flush=True)
    report = policy | {"signature":signature,"input_count":cursor,"parts":parts,"state":"COMPLETE",
        "scope":"Disk-partitioned model inputs, not encoded vectors or a completed retrieval index",
        "seconds_this_call":round(time.perf_counter()-started,3),"created_at":utc_now()}
    report["manifest_sha256"] = digest_json(report)
    atomic_json(manifest_path,report)
    return report


def _candidate_reference(build_run, candidate_name, manifest_sha256):
    if not all(isinstance(value, str) and value for value in (build_run, candidate_name, manifest_sha256)):
        raise ValueError("Candidate lineage reference is incomplete.")
    if Path(build_run).name != build_run or Path(candidate_name).name != candidate_name:
        raise ValueError("Candidate lineage reference must use local names.")
    return {"build_run": build_run, "candidate_name": candidate_name,
            "candidate_manifest_sha256": manifest_sha256}


def _read_lineage_file(path):
    if not path.is_file():
        return None
    saved = read_json(path)
    payload = {key: value for key, value in saved.items() if key != "lineage_sha256"}
    if (saved.get("schema_version") != LINEAGE_SCHEMA_VERSION
        or digest_json(payload) != saved.get("lineage_sha256")
        or not isinstance(saved.get("sources"), list) or not saved["sources"]):
        raise ValueError("Candidate lineage manifest is invalid; preserve it for inspection before rebuilding.")
    refs, seen = [], set()
    for item in saved["sources"]:
        ref = _candidate_reference(item.get("build_run"), item.get("candidate_name"),
                                   item.get("candidate_manifest_sha256"))
        if ref["candidate_manifest_sha256"] not in seen:
            refs.append(ref)
            seen.add(ref["candidate_manifest_sha256"])
    return refs


def read_candidate_lineage(data_root):
    """Return ordered frozen candidates to include in the next cumulative corpus."""
    root = Path(data_root)
    refs = _read_lineage_file(root / "candidate_lineage.json")
    if refs is not None:
        return refs
    active_path = root / "active_data_candidate.json"
    if not active_path.is_file():
        return []
    active = read_json(active_path)
    return [_candidate_reference(active.get("build_run"), active.get("candidate_name"),
                                 active.get("candidate_manifest_sha256"))]


def publish_data_handoff(data_root, build_run, candidate, inputs, *, lineage_mode="append"):
    if inputs["candidate_manifest_sha256"] != candidate["candidate_manifest_sha256"]:
        raise ValueError("Candidate/index-input handoff mismatch.")
    if lineage_mode not in ("append", "replace"):
        raise ValueError("lineage_mode must be append or replace.")
    root = Path(data_root)
    candidate_name = f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json"
    current_ref = _candidate_reference(build_run, candidate_name, candidate["candidate_manifest_sha256"])
    value = {"build_run":build_run,"candidate_name":f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",
        "candidate_manifest_sha256":candidate["candidate_manifest_sha256"],
        "index_inputs_manifest_sha256":inputs["manifest_sha256"],"documents":candidate["counts"]["documents"],
        "children":candidate["counts"]["children"],"unique_dense_inputs":inputs["input_count"],
        "state":"DATA_READY_RESOURCE_BENCHMARK_REQUIRED","updated_at":utc_now()}
    lineage_path = root / "candidate_lineage.json"
    if lineage_mode == "replace":
        refs = [current_ref]
    else:
        refs = _read_lineage_file(lineage_path) or []
        active_path = root / "active_data_candidate.json"
        if active_path.is_file():
            active = read_json(active_path)
            active_ref = _candidate_reference(active.get("build_run"), active.get("candidate_name"),
                                              active.get("candidate_manifest_sha256"))
            if all(ref["candidate_manifest_sha256"] != active_ref["candidate_manifest_sha256"] for ref in refs):
                refs.append(active_ref)
        if all(ref["candidate_manifest_sha256"] != current_ref["candidate_manifest_sha256"] for ref in refs):
            refs.append(current_ref)
        if not refs:
            refs = [current_ref]
    lineage = {"schema_version":LINEAGE_SCHEMA_VERSION,"sources":refs,
        "state":"PENDING_UNION" if len(refs) > 1 else "SINGLE_SOURCE","updated_at":utc_now()}
    lineage["lineage_sha256"] = digest_json(lineage)
    if lineage_mode == "replace":
        # If interrupted before lineage replacement, the pending source list
        # causes the coordinator to resume this same deterministic union.
        atomic_json(root / "active_data_candidate.json",value)
        atomic_json(lineage_path,lineage)
    else:
        # Record both candidates before moving the active pointer. If interrupted,
        # the old pointer remains usable and the union can still be resumed.
        atomic_json(lineage_path,lineage)
        atomic_json(root / "active_data_candidate.json",value)
    return value
