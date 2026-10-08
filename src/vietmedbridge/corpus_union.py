"""Compose complete frozen batches with official IDs and source offsets intact."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .artifacts import atomic_json, code_fingerprint, digest_json, local_workspace, publish_file, read_json, sha256_file, utc_now, verify_file
from .dataset import _connection
from .health import freeze_candidate, health_report
from .index_inputs import prepare_index_inputs, publish_data_handoff
from .validation import artifact_paths, validate_snapshot

KINDS = ("documents","children","parents","sections","failures","ledger")


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
    if (output / "config.json").exists() and read_json(output / "config.json") != config:
        raise ValueError("Union source/policy changed; choose a new output run.")
    atomic_json(output / "config.json",config)
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
    with local_workspace(work_dir) as temporary, _connection(temporary) as con:
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
        # A semantic fingerprint ignores fetch timestamps/raw paths while proving
        # duplicate successes have the same source spans and retrieval inputs.
        for kind in ("children","parents","sections"):
            fields = ("section_id,start_char,end_char,heading_text" if kind == "sections" else
                "chunk_id,start_char,end_char,text,policy_sha256"+(",parent_id,retrieval_representation_hash" if kind == "children" else ""))
            sort_key = "section_id" if kind == "sections" else "chunk_id"
            con.execute(f"""CREATE TEMP TABLE fingerprints_{kind} AS SELECT origin,doc_id,
                md5(string_agg(to_json(row({fields})),'' ORDER BY {sort_key})) AS fingerprint
                FROM all_{kind} GROUP BY origin,doc_id""")
            if con.execute(f"SELECT count(*) FROM (SELECT doc_id FROM fingerprints_{kind} GROUP BY doc_id HAVING count(DISTINCT fingerprint)>1)").fetchone()[0]:
                raise ValueError("Conflicting successful chunk/section policies across batches.")
        con.execute("""CREATE TABLE chosen AS SELECT row_number() OVER(ORDER BY doc_id)-1 AS position,doc_id,origin FROM
            (SELECT l.doc_id,l.origin,row_number() OVER(PARTITION BY l.doc_id
                ORDER BY CASE WHEN d.doc_id IS NULL THEN 1 ELSE 0 END,l.origin) AS rank
             FROM all_ledger l LEFT JOIN all_documents d ON l.doc_id=d.doc_id AND l.origin=d.origin) WHERE rank=1""")
        count = con.execute("SELECT count(*) FROM chosen").fetchone()[0]
        if any(not (output / "parts" / f"part-{start:012d}.done.json").exists() for start in range(0,count,part_size)):
            # Scan each source collection once into bounded DuckDB staging.
            # Rejoining remote Parquet for every output part scales quadratically.
            for kind in KINDS:
                con.execute(f"""CREATE TABLE selected_{kind} AS SELECT c.position,a.* EXCLUDE(origin)
                    FROM all_{kind} a JOIN chosen c ON a.doc_id=c.doc_id AND a.origin=c.origin ORDER BY c.position""")
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
    integrity = validate_snapshot(output,build,official_links=official_links,work_dir=work_dir)
    if not integrity["passed"]:
        raise ValueError(f"Union integrity failed: {integrity}")
    build["integrity"] = integrity
    if (output / "build.json").exists():
        previous = read_json(output / "build.json")
        if previous["snapshot_sha256"] != build["snapshot_sha256"]:
            raise ValueError("Union build snapshot changed.")
        build = previous
    else:
        atomic_json(output / "build.json",build)
    frozen_path = output / "union_frozen_candidate.json"
    if frozen_path.exists():
        candidate = read_json(frozen_path)
        if (candidate["snapshot_sha256"] != build["snapshot_sha256"]
            or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]):
            raise ValueError("Union frozen checkpoint changed.")
    else:
        health = health_report(output,official_links=official_links,work_dir=work_dir)
        candidate = freeze_candidate(output,health,golden_report=golden_report)
        atomic_json(frozen_path,candidate)
    inputs = prepare_index_inputs(output,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",tokenizer,work_dir=work_dir)
    # Change the active pointer only after all source, span and input checks pass.
    handoff = publish_data_handoff(root,run_name,candidate,inputs,lineage_mode="replace")
    atomic_json(output / "union_handoff.json",handoff)
    return handoff
