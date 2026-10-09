"""Compose complete frozen batches with official IDs and source offsets intact."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path
import traceback

import duckdb

from .artifacts import atomic_json, code_fingerprint, digest_json, local_workspace, publish_file, read_json, sha256_file, utc_now, verify_file
from .health import freeze_candidate, health_report
from .index_inputs import prepare_index_inputs, publish_data_handoff
from .validation import artifact_paths, validate_snapshot

KINDS = ("documents","children","parents","sections","failures","ledger")


@contextmanager
def _union_stage(output, signature, stage):
    """Keep the last stage on Drive even when Colab loses its cell output."""
    path = output / "union_progress.json"
    status = {"signature":signature,"stage":stage,"state":"RUNNING","updated_at":utc_now()}
    atomic_json(path,status)
    print(f"Union stage: {stage} — progress: {path}",flush=True)
    try:
        yield
    except BaseException as error:
        try:
            atomic_json(path,status | {"state":"FAILED","updated_at":utc_now(),
                "error":{"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}})
        except OSError:
            pass  # Keep the original failure if Drive itself is unavailable.
        raise
    else:
        atomic_json(path,status | {"state":"STAGE_COMPLETE","updated_at":utc_now()})


def _verified_completed_build(output, config):
    """Resume a validated export; recheck its bytes without selecting/exporting again."""
    build = read_json(output / "build.json")
    integrity = build.get("integrity",{})
    if (build.get("state") != "DATA_VALIDATED" or not build.get("selected_range_complete")
        or build.get("run_name") != output.name or build.get("signature") != config["signature"]
        or build.get("sources") != config["sources"] or build.get("input_kind") != config["input_kind"]
        or build.get("schema_version") != config["schema_version"] or build.get("chunking") != config["chunks"]
        or build.get("origin_corpus_sha256") != config["official_links_sha256"]
        or digest_json({"signature":build["signature"],"parts":build["parts"]}) != build.get("snapshot_sha256")
        or integrity.get("passed") is not True or integrity.get("official_membership") != "VERIFIED"
        or not integrity.get("checks") or any(integrity["checks"].values())
        or integrity.get("checked_input_records") != build["counts"]["input_records"]):
        raise ValueError("Validated union build checkpoint changed.")
    cursor,total = 0,Counter()
    for part in build["parts"]:
        expected = min(config["part_size"],build["counts"]["input_records"]-cursor)
        if (expected <= 0 or part["signature"] != config["signature"]
            or part["first_input_row"] != cursor or part["records"] != expected
            or part["counts"]["input_records"] != expected or set(part["files"]) != set(KINDS)
            or read_json(output / "parts" / f"part-{cursor:012d}.done.json") != part):
            raise ValueError("Validated union part checkpoint changed.")
        cursor += expected
        total.update(part["counts"])
    if (cursor != build["requested_input_records"] or dict(total) != build["counts"]
        or build["selected_shards"] != len(build["parts"])):
        raise ValueError("Validated union coverage checkpoint changed.")
    for kind in KINDS:
        artifact_paths(output,build,kind)
    print(f"Reused validated union: {cursor:,} input IDs; no source selection or export",flush=True)
    return build


def _finish_union(root, output, build, tokenizer, golden_report, official_links, work_dir):
    signature = build["signature"]
    frozen_path = output / "union_frozen_candidate.json"
    if frozen_path.exists():
        with _union_stage(output,signature,"VERIFY_FROZEN_CANDIDATE"):
            candidate = read_json(frozen_path)
            if (candidate["snapshot_sha256"] != build["snapshot_sha256"]
                or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]):
                raise ValueError("Union frozen checkpoint changed.")
            for entry in candidate["health"]["files"].values():
                verify_file(output / entry["path"],entry["sha256"])
    else:
        with _union_stage(output,signature,"HEALTH_REPORT"):
            health = health_report(output,official_links=official_links,work_dir=work_dir)
        with _union_stage(output,signature,"FREEZE_CANDIDATE"):
            candidate = freeze_candidate(output,health,golden_report=golden_report)
            atomic_json(frozen_path,candidate)
    with _union_stage(output,signature,"INDEX_INPUTS"):
        inputs = prepare_index_inputs(output,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",tokenizer,work_dir=work_dir)
    with _union_stage(output,signature,"PUBLISH_HANDOFF"):
        handoff = publish_data_handoff(root,output.name,candidate,inputs,lineage_mode="replace")
        atomic_json(output / "union_handoff.json",handoff)
    atomic_json(output / "union_progress.json",{"signature":signature,"stage":"PUBLISH_HANDOFF",
        "state":"COMPLETE","snapshot_sha256":build["snapshot_sha256"],"updated_at":utc_now()})
    return handoff


def _union_connection(temporary, memory_limit="512MB"):
    # Persist staging tables on Colab's local disk, separate from Drive outputs.
    con = duckdb.connect(str(Path(temporary) / "union.duckdb"))
    con.execute("SET memory_limit = ?", [memory_limit])
    con.execute("SET temp_directory = ?", [str(Path(temporary) / "spill")])
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = false")
    return con


def _span_fingerprints(con, kind):
    fields = (["section_id", "start_char", "end_char", "heading_text"] if kind == "sections" else
        ["chunk_id", "start_char", "end_char", "text", "policy_sha256"]+
        (["parent_id", "retrieval_representation_hash"] if kind == "children" else []))
    # Each field is a fixed-size hex digest (or a distinct NULL sentinel). This
    # also avoids creating large JSON strings inside DuckDB's JSON allocator.
    field_hashes = ",".join(f"coalesce(sha256(a.{field}::VARCHAR),'NULL')" for field in fields)
    sort_key = "section_id" if kind == "sections" else "chunk_id"
    # Sort only fixed-size row hashes and IDs. Ordered string_agg holds all text
    # in aggregate states that DuckDB cannot spill, even with temp_directory set.
    con.execute(f"""CREATE TABLE span_hashes AS SELECT a.doc_id,a.origin,s.origins,
        a.{sort_key} AS sort_key,sha256(concat_ws('|',{field_hashes})) AS span_hash
        FROM all_{kind} a JOIN shared_successes s ON a.doc_id=s.doc_id""")
    cursor = con.execute("""SELECT doc_id,origin,origins,span_hash FROM span_hashes
        ORDER BY doc_id,origin,sort_key""")
    group = None
    digest, rows, origins = None, 0, None
    while batch := cursor.fetchmany(4096):
        for doc_id, origin, expected_origins, span_hash in batch:
            if group != (doc_id, origin):
                if group is not None:
                    yield (*group, origins, (rows, digest.digest()))
                group, digest, rows = (doc_id, origin), hashlib.sha256(), 0
                origins = expected_origins
            digest.update(bytes.fromhex(span_hash))
            rows += 1
    if group is not None:
        yield (*group, origins, (rows, digest.digest()))
    con.execute("DROP TABLE span_hashes")


def _verify_shared_spans(con):
    """Compare duplicate successes without retaining their text in aggregate RAM."""
    con.execute("""CREATE TABLE shared_successes AS SELECT doc_id,count(DISTINCT origin) AS origins
        FROM all_documents GROUP BY doc_id HAVING count(DISTINCT origin)>1""")
    shared_count = con.execute("SELECT count(*) FROM shared_successes").fetchone()[0]
    print(f"Union overlap: {shared_count:,} successful IDs need span comparison", flush=True)
    if not shared_count:
        return
    for kind in ("children", "parents", "sections"):
        previous_doc, reference, expected, observed, seen = None, None, None, 0, 0
        for doc_id, origin, origins, fingerprint in _span_fingerprints(con, kind):
            if doc_id != previous_doc:
                if previous_doc is not None and observed != expected:
                    raise ValueError("Conflicting successful chunk/section policies across batches.")
                previous_doc, reference, expected, observed = doc_id, fingerprint, origins, 0
                seen += 1
            if fingerprint != reference:
                raise ValueError("Conflicting successful chunk/section policies across batches.")
            observed += 1
        if seen != shared_count or observed != expected:
            raise ValueError("Conflicting successful chunk/section policies across batches.")
        print(f"Union overlap: verified {kind} for {seen:,} shared IDs", flush=True)


def compose_candidates(data_root, sources, tokenizer, *, run_name, golden_report, official_links,
                       work_dir, part_size=2048):
    """sources: [(build_run, candidate_filename), ...], ordered for stable ties.

    A recovered success replaces an earlier failure. Different successful text
    or chunk policies for one ID require explicit reconciliation, never a guess.
    Completed output parts resume without re-extracting or re-chunking sources.
    """
    root = Path(data_root).resolve()
    if Path(run_name).name != run_name or run_name in (".","..") or not sources or type(part_size) is not int or part_size < 1:
        raise ValueError("Use explicit frozen sources, a simple output name and a positive part size.")
    loaded = []
    for build_name,filename in sources:
        build = (root / "processed" / build_name).resolve()
        path = (build / filename).resolve()
        if not build.is_relative_to(root / "processed") or not path.is_relative_to(build):
            raise ValueError("Union source escapes processed data.")
        candidate = read_json(path)
        if (candidate.get("state") != "FROZEN_CANDIDATE" or not candidate.get("selected_range_complete")
            or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]
            or not candidate["health"]["integrity"]["passed"] or candidate["health"]["integrity"]["official_membership"] != "VERIFIED"):
            raise ValueError("Only complete, verified frozen candidates can be composed.")
        loaded.append((build,candidate))
    first = loaded[0][1]
    for _,candidate in loaded:
        if (candidate["origin_corpus_sha256"] != sha256_file(official_links)
            or candidate["chunking"] != first["chunking"] or candidate["golden"]["tokenizer"] != first["golden"]["tokenizer"]):
            raise ValueError("Union corpus/tokenizer/chunking policies differ.")
    if (golden_report.get("passed") is not True or golden_report.get("code_sha256") != code_fingerprint()
        or golden_report.get("tokenizer") != first["golden"]["tokenizer"] or golden_report.get("chunking") != first["chunking"]):
        raise ValueError("Run the current golden suite with the frozen source tokenizer/chunking before composing.")
    output = root / "processed" / run_name
    if any(output == b for b,_ in loaded):
        raise ValueError("Union output must be a new build, separate from its sources.")
    identity = {"input_kind":"FROZEN_CANDIDATE_UNION","sources":[{"build_run":b.name,
        "candidate_manifest_sha256":c["candidate_manifest_sha256"]} for b,c in loaded],
        "tokenizer":first["golden"]["tokenizer"],"chunks":first["chunking"],"part_size":part_size,
        "official_links_sha256":first["origin_corpus_sha256"],"schema_version":first["schema_version"],
        "code_sha256":code_fingerprint(),"selection":"success-before-failure-identical-success-required-v1"}
    signature = digest_json(identity)
    config = identity | {"signature":signature}
    if (output / "config.json").exists():
        saved = read_json(output / "config.json")
        saved_identity = {k:v for k,v in saved.items() if k != "signature"}
        if digest_json(saved_identity) != saved.get("signature"):
            raise ValueError("Union configuration checksum changed.")
        # A memory/execution fix does not change selected rows. Retain producer
        # provenance and existing part signatures; all semantic inputs must match.
        if ({k:v for k,v in saved_identity.items() if k != "code_sha256"}
            != {k:v for k,v in identity.items() if k != "code_sha256"}):
            raise ValueError("Union source/policy changed; choose a new output run.")
        identity, signature, config = saved_identity, saved["signature"], saved
    else:
        atomic_json(output / "config.json",config)
    atomic_json(output / "union_execution.json", {
        "signature":signature,"producer_code_sha256":identity["code_sha256"],
        "execution_code_sha256":code_fingerprint(),"span_validation":"streamed-shared-successes-v2",
        "memory_limit":"512MB","threads":1,"staging":"local-disk-duckdb",
        "updated_at":utc_now()})
    if (output / "union_handoff.json").exists():
        saved = read_json(output / "union_handoff.json")
        candidate = read_json(output / saved["candidate_name"])
        inputs = read_json(output / "index_inputs/units.json")
        if (digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != saved["candidate_manifest_sha256"]
            or digest_json({k:v for k,v in inputs.items() if k != "manifest_sha256"}) != saved["index_inputs_manifest_sha256"]):
            raise ValueError("Completed union manifest changed.")
        for kind in KINDS:
            artifact_paths(output,candidate,kind)
        for part in inputs["parts"]:
            verify_file(output / "index_inputs" / part["path"],part["sha256"])
        return publish_data_handoff(root,run_name,candidate,inputs,lineage_mode="replace")
    if (output / "build.json").exists():
        with _union_stage(output,signature,"VERIFY_SAVED_BUILD"):
            build = _verified_completed_build(output,config)
        return _finish_union(root,output,build,tokenizer,golden_report,official_links,work_dir)
    with _union_stage(output,signature,"SELECT_AND_EXPORT"), local_workspace(work_dir) as temporary, _union_connection(temporary) as con:
        for kind in KINDS:
            selects = []
            for i,(build,candidate) in enumerate(loaded):
                con.read_parquet(artifact_paths(build,candidate,kind)).create_view(f"source_{i}_{kind}")
                selects.append(f"SELECT *,{i} AS origin FROM source_{i}_{kind}")
            con.execute(f"CREATE VIEW all_{kind} AS "+" UNION ALL ".join(selects))
        if con.execute("SELECT count(*) FROM (SELECT doc_id FROM all_ledger GROUP BY doc_id HAVING count(DISTINCT url)>1)").fetchone()[0]:
            raise ValueError("One official ID has conflicting URLs across batches.")
        if con.execute("""SELECT count(*) FROM (SELECT doc_id FROM all_documents GROUP BY doc_id
            HAVING count(DISTINCT source_text_sha256)>1 OR count(DISTINCT title)>1 OR count(DISTINCT language)>1)""").fetchone()[0]:
            raise ValueError("Conflicting successful sources for an official ID; reconcile the versions explicitly.")
        # Ignore fetch timestamps/raw paths, but retain exact span/input conflict checks.
        # Disjoint archives need no cross-source chunk fingerprint scan at all.
        _verify_shared_spans(con)
        con.execute("""CREATE TABLE chosen AS SELECT row_number() OVER(ORDER BY doc_id)-1 AS position,doc_id,origin FROM
            (SELECT l.doc_id,l.origin,row_number() OVER(PARTITION BY l.doc_id
                ORDER BY CASE WHEN d.doc_id IS NULL THEN 1 ELSE 0 END,l.origin) AS rank
             FROM all_ledger l LEFT JOIN all_documents d ON l.doc_id=d.doc_id AND l.origin=d.origin) WHERE rank=1""")
        count = con.execute("SELECT count(*) FROM chosen").fetchone()[0]
        if any(not (output / "parts" / f"part-{start:012d}.done.json").exists() for start in range(0,count,part_size)):
            # Scan each source collection once into bounded DuckDB staging.
            # Rejoining remote Parquet for every output part scales quadratically.
            for kind in KINDS:
                print(f"Union staging on local disk: {kind}", flush=True)
                con.execute(f"""CREATE TABLE selected_{kind} AS SELECT c.position,a.* EXCLUDE(origin)
                    FROM all_{kind} a JOIN chosen c ON a.doc_id=c.doc_id AND a.origin=c.origin""")
        parts,total = [],Counter()
        for start in range(0,count,part_size):
            marker = output / "parts" / f"part-{start:012d}.done.json"
            expected = min(part_size,count-start)
            if marker.exists():
                part = read_json(marker)
                if part["signature"] != signature or part["first_input_row"] != start or part["records"] != expected:
                    raise ValueError("Union part checkpoint changed.")
                for entry in part["files"].values():
                    verify_file(output / entry["path"],entry["sha256"])
            else:
                files,counts = {},{}
                for kind in KINDS:
                    local = Path(temporary) / f"{kind}.parquet"
                    con.sql(f"""SELECT * EXCLUDE(position) FROM selected_{kind}
                        WHERE position>=? AND position<? ORDER BY doc_id""",params=[start,start+expected]).write_parquet(str(local),compression="zstd")
                    import pyarrow.parquet as pq
                    counts["input_records" if kind == "ledger" else kind] = pq.ParquetFile(local).metadata.num_rows
                    relative = f"parts/part-{start:012d}.{kind}.parquet"
                    files[kind] = {"path":relative,"sha256":publish_file(local,output / relative)}
                part = {"signature":signature,"first_input_row":start,"records":expected,"files":files,"counts":counts}
                atomic_json(marker,part)
            parts.append(part)
            total.update(part["counts"])
            print(f"Union: {start+expected:,}/{count:,} official input IDs",flush=True)
    build = {"run_name":run_name,"signature":signature,"parts":parts,"counts":dict(total),
        "input_kind":"FROZEN_CANDIDATE_UNION","schema_version":first["schema_version"],"chunking":first["chunking"],
        "selected_shards":len(parts),"selected_range_complete":True,"requested_input_records":count,
        "requested_range":{"kind":"union_of_explicit_official_ids"},"origin_corpus_sha256":first["origin_corpus_sha256"],
        "sources":identity["sources"],"created_at":utc_now(),"offset_reference":first["offset_reference"],
        "snapshot_sha256":digest_json({"signature":signature,"parts":parts}),"state":"DATA_VALIDATED"}
    with _union_stage(output,signature,"VALIDATE_EXPORTED_BUILD"):
        integrity = validate_snapshot(output,build,official_links=official_links,work_dir=work_dir)
        if not integrity["passed"]:
            raise ValueError(f"Union integrity failed: {integrity}")
        build["integrity"] = integrity
        atomic_json(output / "build.json",build)
    return _finish_union(root,output,build,tokenizer,golden_report,official_links,work_dir)
