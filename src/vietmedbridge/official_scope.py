"""Derive a corpus from validated builds without re-chunking.

Source parts and content-addressed vector blocks remain immutable. Every derived
part is checkpointed independently; model inputs and the official pointer are
published only after full source-span validation of the selected corpus.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .artifacts import atomic_json, code_fingerprint, digest_json, publish_file, read_json, sha256_file, utc_now, verify_file
from .build import FAILURE_SCHEMA
from .content_embeddings import ContentBlocks, checked, content_embeddings, key
from .health import freeze_candidate, health_report
from .index_inputs import prepare_index_inputs, publish_data_handoff
from .validation import SCHEMA_VERSION

BTC_REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
BTC_CORPUS_SHA256 = "87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7"
KINDS = ("documents", "children", "parents", "sections", "failures", "ledger")


def _bound(root, relative):
    root = Path(root).resolve()
    value = (root / relative).resolve()
    if not value.is_relative_to(root):
        raise ValueError("Scope artifact escapes its source directory.")
    return value


def stage_btc_snapshot(data_root, work_dir, *, revision=BTC_REVISION):
    """Download an immutable snapshot without changing the active snapshot."""
    from huggingface_hub import HfApi, hf_hub_download
    from .retrieval_data import load_queries

    root, work = Path(data_root), Path(work_dir)
    target = root / "raw/hub" / revision
    saved = target / "submission_snapshot.json"
    if saved.is_file():
        result = read_json(saved)
        if result["revision"] != revision:
            raise ValueError("Official scope revision changed.")
        for entry in result["files"].values():
            verify_file(_bound(root, entry["path"]), entry["sha256"])
        return result
    info = HfApi().dataset_info("AIGuruTinix/ViBioMIR", revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("The requested official revision must resolve exactly; no main fallback.")
    remote = {x.rfilename: x for x in info.siblings}
    corpus = "links_corpus_vi_en_zh.parquet"
    files = {}
    for logical, actual in (("links_corpus.parquet", corpus), ("query.parquet", "query.parquet"), ("README.md", "README.md")):
        local = Path(hf_hub_download("AIGuruTinix/ViBioMIR", actual, repo_type="dataset",
            revision=revision, local_dir=work / "official_download" / revision))
        checksum = sha256_file(local)
        expected = getattr(remote[actual].lfs, "sha256", None) if remote[actual].lfs else None
        if expected and expected != checksum:
            raise ValueError("Official Hub file checksum mismatch.")
        if actual == corpus and revision == BTC_REVISION and checksum != BTC_CORPUS_SHA256:
            raise ValueError("Official corpus differs from the audited revision.")
        destination = target / actual
        publish_file(local, destination)
        files[logical] = {"path": destination.relative_to(root).as_posix(), "source_name": actual,
                          "sha256": checksum, "bytes": destination.stat().st_size}
        if actual.endswith(".parquet"):
            files[logical]["rows"] = pq.ParquetFile(destination).metadata.num_rows
    load_queries(root / files["query.parquet"]["path"], expected_count=1200)
    result = {"dataset_id": "AIGuruTinix/ViBioMIR", "revision": revision,
              "files": files, "created_at": utc_now()}
    atomic_json(saved, result)
    return result


def discover_sources(data_root):
    """Preserve active-source precedence, then add the latest completed batch."""
    root = Path(data_root)
    active = read_json(root / "active_data_candidate.json")
    current = read_json(_bound(root / "processed" / active["build_run"], active["candidate_name"]))
    refs = [x["build_run"] for x in current["source_builds"]] if current.get("input_kind") in {"OFFICIAL_SCOPE","AVAILABLE_CORPUS"} else [active["build_run"]]
    batch_path = root / "active_data_batch.json"
    if batch_path.exists():
        batch = read_json(batch_path)
        if batch.get("state") != "COMPLETE" or any(x.get("state") != "COMPLETE" for x in batch.get("builds", [])):
            raise ValueError("Complete the current data batch before preparing a submission scope.")
        refs += [x["build_run"] for x in batch["builds"]]
    pilot = "stage-a-data-v5-clean-text"
    if (root / "processed" / pilot / "build.json").is_file():
        refs.append(pilot)
    return list(dict.fromkeys(refs))


def _copy_checked(source, destination, checksum):
    """One source read, verified while copying onto Colab local disk."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == checksum:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    with Path(source).open("rb") as reader, temporary.open("wb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
            writer.write(block)
    if digest.hexdigest() != checksum:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Changed source artifact: {Path(source).name}")
    temporary.replace(destination)
    return destination


def _subset(table, ids):
    return table.filter(pc.is_in(table["doc_id"], value_set=pa.array(ids, type=pa.int64())))


def _restore_scope_freeze(target, local, signature):
    """Restore sealed metadata without replaying every source receipt/payload."""
    frozen = checked(read_json(target / "scope_freeze.json"))
    candidate = frozen["candidate"]
    if (frozen["scope_signature"] != signature or candidate.get("signature") != signature
        or candidate.get("state") != "FROZEN_CANDIDATE" or not candidate.get("selected_range_complete")
        or not candidate.get("integrity",{}).get("passed")
        or candidate["integrity"].get("official_membership") != "VERIFIED"
        or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]
        or digest_json({"signature":signature,"parts":candidate["parts"]}) != candidate["snapshot_sha256"]):
        raise ValueError("Submission freeze checkpoint changed.")
    names = set()
    for entry in frozen["files"]:
        if Path(entry["path"]).parts[0] in {"parts","index_inputs"}:
            raise ValueError("Scope freeze must contain metadata, not corpus payloads.")
        names.add(entry["path"])
        _copy_checked(_bound(target,entry["path"]),_bound(local,entry["path"]),entry["sha256"])
    filename = f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json"
    if not {"build.json",filename} <= names:
        raise ValueError("Scope freeze metadata is incomplete.")
    build, config = read_json(local / "build.json"), read_json(local / "config.json")
    if (read_json(local / filename) != candidate or build.get("run_name") != target.name
        or build.get("state") != "DATA_VALIDATED" or not build.get("selected_range_complete")
        or build.get("signature") != signature or build.get("snapshot_sha256") != candidate["snapshot_sha256"]
        or build.get("parts") != candidate["parts"] or build.get("counts") != candidate["counts"]
        or candidate.get("chunking") != config["chunks"]
        or candidate["golden"]["tokenizer"] != config["tokenizer"]
        or candidate.get("origin_corpus_sha256") != config["official_links_sha256"]
        or build.get("source_builds") != config["source_builds"]):
        raise ValueError("Scope freeze build/config binding changed.")
    return frozen, candidate


def _complete_scope(target, local, signature, candidate, snapshot, tokenizer, work, *, duplicates=0):
    filename = f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json"
    def restore_input_source(entry):
        _copy_checked(_bound(target,entry["path"]),_bound(local,entry["path"]),entry["sha256"])
    inputs = prepare_index_inputs(local,filename,tokenizer,work_dir=work,
        checkpoint_root=target / "index_inputs",source_loader=restore_input_source)
    handoff = {"scope_signature":signature,"candidate_name":filename,
        "candidate_manifest_sha256":candidate["candidate_manifest_sha256"],
        "index_inputs_manifest_sha256":inputs["manifest_sha256"],"official_snapshot":snapshot,
        "documents":candidate["counts"]["documents"],"duplicate_successes_kept_from_first_source":duplicates}
    atomic_json(target / "scope_handoff.json",handoff)
    atomic_json(target / "scope_progress.json", {"state":"SCOPE_READY",
        "documents":candidate["counts"]["documents"],"updated_at":utc_now()})
    return target,candidate,inputs


def stage_scope_catalog(build, candidate, inputs, *, work_dir):
    """Copy only catalog inputs, and only when a catalog actually needs building."""
    target = Path(build)
    local = Path(work_dir) / "official_scope" / target.name
    _copy_checked(target / "config.json",local / "config.json",sha256_file(target / "config.json"))
    for ordinal, part in enumerate(candidate["parts"],1):
        for kind in ("documents","children","parents"):
            entry = part["files"][kind]
            _copy_checked(_bound(target,entry["path"]),_bound(local,entry["path"]),entry["sha256"])
        if ordinal % 16 == 0 or ordinal == len(candidate["parts"]):
            print(f"Catalog input staging: {ordinal:,}/{len(candidate['parts']):,} parts"
                  " | documents/children/parents only",flush=True)
    for part in inputs["parts"]:
        _copy_checked(_bound(target / "index_inputs",part["path"]),
                      _bound(local / "index_inputs",part["path"]),part["sha256"])
    atomic_json(local / "index_inputs/units.json",inputs)
    return local


def derive_official_scope(data_root, sources, snapshot, tokenizer, golden_report, *, work_dir,
                          expected_urls=None, keep_all_legacy=False):
    root, work = Path(data_root).resolve(), Path(work_dir)
    official_path = _bound(root, snapshot["files"]["links_corpus.parquet"]["path"])
    verify_file(official_path, snapshot["files"]["links_corpus.parquet"]["sha256"])
    official = None
    if not keep_all_legacy:
        official_rows = pq.read_table(official_path, columns=["id", "url"]).to_pylist()
        official = {x["id"]: x["url"] for x in official_rows}
        if len(official) != len(official_rows) or len({x["url"] for x in official_rows}) != len(official_rows):
            raise ValueError("Official IDs and URLs must be unique.")
        del official_rows
    elif expected_urls is not None:
        raise ValueError("Legacy-all scope must not apply a current-BTC URL whitelist.")
    expected = None
    if expected_urls is not None:
        rows = pq.read_table(expected_urls, columns=["doc_id", "url"]).to_pylist()
        expected = {x["doc_id"]: x["url"] for x in rows}
        if len(expected) != len(rows) or any(official.get(k) != v for k, v in expected.items()):
            raise ValueError("The processed-URL inventory differs from the official scope.")
    if len(set(sources)) != len(sources):
        raise ValueError("Submission sources must not contain duplicate build names.")
    loaded = []
    for name in sources:
        build_root = _bound(root / "processed", name)
        b, config = read_json(build_root / "build.json"), read_json(build_root / "config.json")
        if (b.get("run_name") != build_root.name or b.get("state") != "DATA_VALIDATED" or b.get("selected_range_complete") is not True
            or not b.get("integrity", {}).get("passed") or b["integrity"].get("official_membership") != "VERIFIED"
            or any(b["integrity"].get("checks", {}).values()) or b.get("schema_version") != SCHEMA_VERSION
            or digest_json({"signature": b["signature"], "parts": b["parts"]}) != b["snapshot_sha256"]
            or digest_json({k:v for k,v in config.items() if k != "signature"}) != b["signature"]
            or config["signature"] != b["signature"]):
            raise ValueError(f"Source build is incomplete or changed: {name}")
        if config["tokenizer"] != golden_report["tokenizer"] or b["chunking"] != golden_report["chunking"]:
            raise ValueError("Source tokenizer/chunk policy differs; reconcile explicitly.")
        if keep_all_legacy and b["origin_corpus_sha256"] != snapshot["files"]["links_corpus.parquet"]["sha256"]:
            raise ValueError("Legacy corpus and source snapshots differ; restore the original source snapshot.")
        loaded.append((build_root, b))
    if not loaded or not golden_report.get("passed") or golden_report["code_sha256"] != code_fingerprint():
        raise ValueError("Current golden regression and validated sources are required.")
    identity = {"input_kind":"AVAILABLE_CORPUS" if keep_all_legacy else "OFFICIAL_SCOPE", "schema_version":SCHEMA_VERSION,
        "source_builds":[{"build_run":p.name,"snapshot_sha256":b["snapshot_sha256"]} for p,b in loaded],
        "official_links_sha256":snapshot["files"]["links_corpus.parquet"]["sha256"],
        "official_revision":snapshot["revision"], "tokenizer":golden_report["tokenizer"],
        "chunks":golden_report["chunking"], "selection":("all-validated-successes-keep-legacy-urls-v1" if keep_all_legacy
            else "all-exact-id-url-successes-with-audited-minimum-v2"),
        "expected_urls_sha256":sha256_file(expected_urls) if expected_urls is not None else None,
        "code_sha256":code_fingerprint()}
    run = ("available-" if keep_all_legacy else "official-") + snapshot["revision"][:8] + "-" + digest_json({k:v for k,v in identity.items() if k != "code_sha256"})[:12]
    target, local = root / "processed" / run, work / "official_scope" / run
    target.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    if (target / "config.json").exists():
        config = read_json(target / "config.json")
        saved = {k:v for k,v in config.items() if k != "signature"}
        if digest_json(saved) != config["signature"] or {k:v for k,v in saved.items() if k != "code_sha256"} != {k:v for k,v in identity.items() if k != "code_sha256"}:
            raise ValueError("Submission scope source/policy changed.")
        identity = saved
    signature = digest_json(identity)
    config = identity | {"signature": signature}
    atomic_json(target / "config.json", config)
    atomic_json(local / "config.json", config)
    if (target / "scope_handoff.json").exists():
        saved = read_json(target / "scope_handoff.json")
        if saved["scope_signature"] != signature:
            raise ValueError("Scope handoff signature changed.")
        candidate = read_json(target / saved["candidate_name"])
        inputs = read_json(target / "index_inputs/units.json")
        if (digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]
            or digest_json({k:v for k,v in inputs.items() if k != "manifest_sha256"}) != inputs["manifest_sha256"]):
            raise ValueError("Completed scope manifest changed.")
        return target, candidate, inputs
    if (target / "scope_freeze.json").exists():
        frozen,candidate = _restore_scope_freeze(target,local,signature)
        print(f"Restored frozen scope: {candidate['counts']['documents']:,} documents"
              " | source selection already complete; resuming Index inputs",flush=True)
        atomic_json(target / "scope_progress.json", {"state":"RESUMING_INDEX_INPUTS",
            "scope_signature":signature,"documents":candidate["counts"]["documents"],"updated_at":utc_now()})
        return _complete_scope(target,local,signature,candidate,snapshot,tokenizer,work,
                               duplicates=frozen.get("duplicate_successes_kept_from_first_source",0))
    parts, seen, total, cursor, duplicates = [], set(), Counter(), 0, 0
    for origin, (source, b) in enumerate(loaded):
        for ordinal, source_part in enumerate(b["parts"]):
            witness = {"scope_signature":signature,"source_build":source.name,
                       "source_snapshot":b["snapshot_sha256"],"source_part":source_part}
            receipt_path = target / "scope_checkpoints" / f"source-{origin:03d}-{ordinal:06d}.json"
            if receipt_path.exists():
                action = "restored"
                receipt = checked(read_json(receipt_path))
                if receipt["witness"] != witness or any(i in seen or (official is not None and official.get(i) is None) for i in receipt["selected_ids"]):
                    raise ValueError("Scope part checkpoint identity/coverage changed.")
                part = receipt["part"]
                if part is not None:
                    if part["first_input_row"] != cursor or part["records"] != len(receipt["selected_ids"]):
                        raise ValueError("Scope checkpoint ordering changed.")
                    for entry in part["files"].values():
                        _copy_checked(_bound(target, entry["path"]), _bound(local, entry["path"]), entry["sha256"])
            else:
                action = "new"
                scratch = work / "official_scope_source"
                doc_spec = source_part["files"]["documents"]
                doc_file = _copy_checked(_bound(source,doc_spec["path"]),scratch / "documents.parquet",doc_spec["sha256"])
                documents = pq.read_table(doc_file)
                ids = []
                for row in documents.select(["doc_id","url"]).to_pylist():
                    doc_id, url = row["doc_id"],row["url"]
                    if official is not None and official.get(doc_id) != url:
                        continue
                    if doc_id in seen:
                        duplicates += 1
                        continue
                    ids.append(doc_id)
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate document ID in a selected source part.")
                part = None
                if ids:
                    files, counts = {}, {}
                    for kind in KINDS:
                        if kind == "documents":
                            table = _subset(documents, ids)
                        elif kind == "failures":
                            table = pa.Table.from_pylist([], schema=FAILURE_SCHEMA)
                        else:
                            spec = source_part["files"][kind]
                            path = _copy_checked(_bound(source,spec["path"]),scratch / f"{kind}.parquet",spec["sha256"])
                            table = _subset(pq.read_table(path), ids)
                        relative = f"parts/part-{cursor:012d}.{kind}.parquet"
                        path = _bound(local,relative)
                        path.parent.mkdir(exist_ok=True)
                        pq.write_table(table,path,compression="zstd")
                        files[kind] = {"path":relative,"sha256":publish_file(path,target / relative)}
                        counts["input_records" if kind == "ledger" else kind] = table.num_rows
                    if counts["input_records"] != len(ids) or counts["documents"] != len(ids):
                        raise ValueError("Selected source input/document conservation failed.")
                    part = {"signature":signature,"first_input_row":cursor,"records":len(ids),"counts":counts,"files":files}
                receipt = witness | {"witness":witness,"selected_ids":ids,"part":part}
                receipt["manifest_sha256"] = digest_json(receipt)
                atomic_json(receipt_path,receipt)
            seen.update(receipt["selected_ids"])
            if part is not None:
                parts.append(part);total.update(part["counts"]);cursor += part["records"]
            atomic_json(target / "scope_progress.json", {"state":"FILTERING", "scope_signature":signature,
                "source":source.name,"source_part":ordinal+1,"source_parts":len(b["parts"]),
                "selected_documents":cursor,"updated_at":utc_now()})
            print(f"Official scope {origin+1}/{len(loaded)} source parts {ordinal+1}/{len(b['parts'])}: {cursor:,} documents | {action}",flush=True)
    if not seen or (expected is not None and not set(expected) <= seen):
        raise ValueError(f"Scope coverage mismatch; expected={len(expected) if expected is not None else None}, selected={len(seen)}")
    build = {"run_name":run,"signature":signature,"parts":parts,"counts":dict(total),
        "schema_version":SCHEMA_VERSION,"input_kind":identity["input_kind"],"source_builds":identity["source_builds"],
        "chunking":identity["chunks"],"origin_corpus_sha256":identity["official_links_sha256"],
        "selected_shards":len(parts),"selected_range_complete":True,"requested_input_records":cursor,
        "requested_range":{"kind":"available-successes-in-current-official-corpus"},
        "snapshot_sha256":digest_json({"signature":signature,"parts":parts}),"state":"BUILDING",
        "offset_reference":"Original source text and Python Unicode character offsets retained unchanged",
        "created_at":utc_now()}
    freeze_path = target / "scope_freeze.json"
    if freeze_path.exists():
        frozen = checked(read_json(freeze_path))
        candidate = frozen["candidate"]
        if (frozen["scope_signature"] != signature or candidate["snapshot_sha256"] != build["snapshot_sha256"]
            or digest_json({k:v for k,v in candidate.items() if k != "candidate_manifest_sha256"}) != candidate["candidate_manifest_sha256"]):
            raise ValueError("Submission freeze checkpoint changed.")
        for entry in frozen["files"]:
            _copy_checked(_bound(target,entry["path"]),_bound(local,entry["path"]),entry["sha256"])
    else:
        atomic_json(local / "build.json",build)
        health = health_report(local,official_links=official_path,work_dir=work)
        if not health["integrity"]["passed"]:
            raise ValueError("Filtered corpus failed source-span/official-ID validation.")
        build.update(state="DATA_VALIDATED",integrity=health["integrity"])
        atomic_json(local / "build.json",build)
        candidate = freeze_candidate(local,health,golden_report=golden_report)
        files = []
        names = ["build.json",f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json"]
        names += [p.relative_to(local).as_posix() for p in sorted((local / "reports").rglob("*")) if p.is_file()]
        for name in names:
            files.append({"path":name,"sha256":publish_file(local / name,target / name)})
        frozen = {"scope_signature":signature,"candidate":candidate,"files":files,
                  "duplicate_successes_kept_from_first_source":duplicates}
        frozen["manifest_sha256"] = digest_json(frozen)
        atomic_json(freeze_path,frozen)
    return _complete_scope(target,local,signature,candidate,snapshot,tokenizer,work,duplicates=duplicates)


def activate_scope(data_root, build, candidate, inputs, snapshot):
    """Journal the paired official/candidate pointer change for safe retries."""
    root = Path(data_root)
    if candidate["origin_corpus_sha256"] != snapshot["files"]["links_corpus.parquet"]["sha256"]:
        raise ValueError("Scope candidate and official snapshot differ.")
    journal = root / "submission_scope_activation.json"
    saved = read_json(journal) if journal.exists() else None
    record = {"state":"ACTIVATING","build_run":Path(build).name,
        "candidate_manifest_sha256":candidate["candidate_manifest_sha256"],
        "previous_candidate":read_json(root / "active_data_candidate.json"),
        "previous_snapshot":read_json(root / "raw/snapshot.json"),"new_snapshot":snapshot}
    if saved and saved["candidate_manifest_sha256"] == candidate["candidate_manifest_sha256"]:
        record = saved | {"state":"ACTIVATING"}
    atomic_json(journal,record)
    atomic_json(root / "raw/snapshot.json",snapshot)
    result = publish_data_handoff(root,Path(build).name,candidate,inputs,lineage_mode="replace")
    atomic_json(journal,record | {"state":"COMPLETE","updated_at":utc_now()})
    return result


def rebind_cached_family(data_root, build, candidate, inputs, producer, output_dir, *, work_dir):
    """CPU-only cache inventory and maps; never initializes a neural model."""
    encoder = SimpleNamespace(identity=producer["encoder"],dimension=producer["dimension"])
    cached = 0
    with_cache = ContentBlocks(data_root,encoder,Path(work_dir) / "scope-cache-inventory")
    try:
        for part in inputs["parts"]:
            path = Path(build) / "index_inputs" / part["path"]
            verify_file(path,part["sha256"])
            keys = [key(x) for x in pq.read_table(path,columns=["text"]).column("text").to_pylist()]
            for start in range(0,len(keys),900):
                batch = keys[start:start+900]
                cached += with_cache.con.execute("SELECT count(*) FROM vectors WHERE key IN ("+
                    ",".join("?" for _ in batch)+")",batch).fetchone()[0]
    finally:
        with_cache.close()
    mapped = content_embeddings(data_root,Path(build) / "index_inputs",inputs,encoder,output_dir,
        work_dir=work_dir,encode_missing=False)
    report = {"model_id":encoder.identity.get("model_id"),"cached_inputs":cached,
        "missing_inputs":inputs["input_count"]-cached,"total_inputs":inputs["input_count"],
        "complete_mapped_parts":len(mapped["parts"]),"total_parts":len(inputs["parts"]),
        "neural_inference_used":False,"candidate_manifest_sha256":candidate["candidate_manifest_sha256"]}
    atomic_json(Path(output_dir) / "scope_cache_coverage.json",report)
    return report
