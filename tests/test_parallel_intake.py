import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from test_external_import import external, CharacterTokenizer, SPEC, CONFIG, write_parquet
from test_isolated_intake import LocalReader, make_source
from test_full_intake import portable_fixture
from scripts.intake_resume import IntakeResume, bounded_intake_sql, bounded_intake_parquet
from scripts.retained_intake import RetainedAdapter
from scripts.parallel_intake import ParallelIntake
from scripts.full_intake import FullIntake
from vietmedbridge import isolated_intake as core
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file


def worker(scope, root, work, worker_id):
    return ParallelIntake(scope, root=root, work_dir=work, worker_id=worker_id,
        tokenizer_spec=SPEC, chunk_config=CONFIG, shard_size=1,
        controller_type=IntakeResume, retained_type=RetainedAdapter,
        sql_context=bounded_intake_sql, parquet_writer=bounded_intake_parquet)


def setup_parallel(tmp_path, external):
    retained, archive = portable_fixture(tmp_path, external)
    links, text = external[1], external[2]
    native = pq.read_table(links).to_pylist()
    for index in range(8):
        for original in pq.read_table(external[0]/core.MEMBERS["frontier"]).to_pylist():
            url = original["fetch_url"].replace("example.org", f"source{index}.example.org")
            n = int(original["crawl_url_id"])
            native.append({"id": 10000+index*100+n*2, "url": url})
            if original["doc_count"] == 2:
                native.append({"id": 10000+index*100+n*2+1, "url": url+"#evidence"})
    write_parquet(links, native)
    source_descriptor = read_json(external[0]/core.MEMBERS["dataset"])
    source_descriptor["sources"][0]["sha256"] = sha256_file(links)
    atomic_json(external[0]/core.MEMBERS["dataset"], source_descriptor)
    atomic_json(tmp_path/"portable-source/new_dataset_manifest.json", source_descriptor)
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        for item in retained["members"].values():
            info = tar.gettarinfo(tmp_path/"portable-source"/item["path"], arcname=item["path"])
            with (tmp_path/"portable-source"/item["path"]).open("rb") as stream:
                tar.addfile(info, stream)
    with tarfile.open(archive) as tar:
        for item in retained["members"].values():
            info = tar.getmember(item["path"])
            item.update(offset=info.offset_data, bytes=info.size,
                        sha256=sha256_file(tmp_path/"portable-source"/item["path"]))
    retained.update(archive_bytes=archive.stat().st_size, official_links_sha256=sha256_file(links))
    files, sources = {"retained": archive}, []
    for index in range(8):
        folder = tmp_path/f"input-{index}"
        for kind, name in core.MEMBERS.items():
            path = folder/name
            if kind == "dataset":
                atomic_json(path, source_descriptor)
            else:
                rows = pq.read_table(external[0]/name).to_pylist()
                for row in rows:
                    for key in ("fetch_url", "final_url"):
                        if key in row:
                            row[key] = row[key].replace("example.org", f"source{index}.example.org")
                write_parquet(path, rows)
        path, source = make_source(tmp_path, (folder, links, text), f"new-{index:05d}")
        source["mode"] = "NEW_SNAPSHOT_SOURCE"
        files[source["job_id"]] = path
        sources.append(source)
    baseline = tmp_path/"baseline.parquet"
    write_parquet(baseline, [{"doc_id": 583, "url": "https://example.org/article"}])
    base = {"sources": sources[:6], "required_snapshot_id": "team-test",
            "snapshots": {"team-test": {"sha256": sha256_file(links)}},
            "baseline": {"sha256": sha256_file(baseline)}}
    scope = {"base_plan": base, "base_plan_sha256": digest_json(base), "additional_sources": sources[6:],
             "retained": retained, "audited_coverage": {"union_native_ids": len(native)}}
    root, work = tmp_path/"project/data_staging/campaign", tmp_path/"work-0"
    protected = tmp_path/"project/data"
    atomic_json(protected/"active_data_candidate.json", {"unchanged": True})
    resource = {"load_reader": lambda: LocalReader(files), "load_links": lambda _: links,
                "load_baseline": lambda: baseline, "load_tokenizer": lambda: (CharacterTokenizer(), SPEC)}
    return scope, root, work, resource, files, links, baseline, protected


def runner(path):
    config = read_json(path)
    w = worker(config["scope"], Path(config["root"]), Path(config["work"]), config["worker_id"])
    result = w.run(load_reader=lambda: LocalReader({k: Path(v) for k, v in config["files"].items()}),
        load_links=lambda _: Path(config["links"]), load_baseline=lambda: Path(config["baseline"]),
        load_tokenizer=lambda: (CharacterTokenizer(), SPEC))
    atomic_json(Path(config["report"]), result)


def test_four_processes_cover_every_id_and_reuse_old_partial_checkpoint(tmp_path, external, monkeypatch):
    import vietmedbridge.external_import as importer
    scope, root, work, resources, files, links, baseline, protected = setup_parallel(tmp_path, external)
    serial = IntakeResume(scope["base_plan"], root=root, work_dir=work, tokenizer_spec=SPEC, chunk_config=CONFIG, shard_size=1)
    serial.run(**resources, max_new_sources=1)
    receipt = root/"sources/new-00000/complete.json"
    receipt_before = receipt.read_bytes()
    original = importer.chunk_source
    def stop(doc, *args, **kwargs):
        if doc["doc_id"] == 10101:
            raise RuntimeError("stop old serial source")
        return original(doc, *args, **kwargs)
    monkeypatch.setattr(importer, "chunk_source", stop)
    with pytest.raises(RuntimeError, match="old serial"):
        serial.run(**resources, max_new_sources=1)
    committed = root/"processed/new-00001/parts/part-000000.done.json"
    checkpoint = committed.read_bytes()
    monkeypatch.setattr(importer, "chunk_source", original)
    processes, outputs, logs = [], [], []
    command = "import sys,faiss; sys.path.insert(0,'tests'); from test_parallel_intake import runner; runner(sys.argv[1])"
    try:
        for owner in range(4):
            report, config = tmp_path/f"worker-{owner}.json", tmp_path/f"config-{owner}.json"
            atomic_json(config, {"scope": scope, "root": str(root), "work": str(tmp_path/f"work-{owner}"),
                "worker_id": owner, "files": {k: str(v) for k, v in files.items()}, "links": str(links),
                "baseline": str(baseline), "report": str(report)})
            log = (tmp_path/f"worker-{owner}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, "-X", "utf8", "-c", command, str(config)],
                cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()))
            outputs.append(report)
        for owner, process in enumerate(processes):
            assert process.wait(timeout=120) == 0, (tmp_path/f"worker-{owner}.log").read_text(encoding="utf-8")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    reports = [read_json(path) for path in outputs]
    assert [r["completed_sources"] for r in reports] == [3, 2, 2, 2]
    assert all(r["state"] == "CPU_WORKER_COMPLETE" for r in reports)
    assert receipt.read_bytes() == receipt_before
    assert committed.read_bytes() == checkpoint
    coordinator = worker(scope, root, work, 0)
    result = coordinator.collect()
    assert result["state"] == "FULL_INTAKE_COMPLETE"
    assert result["coverage"] == {"passed": True, "input_native_ids": 38, "unique_native_ids": 38}
    assert sum(result["counts"].values()) == 38
    assert read_json(protected/"active_data_candidate.json") == {"unchanged": True}
    # Completed workers/collection should only read JSON/stat on another run.
    original_open = Path.open
    def no_payload(path, *args, **kwargs):
        if path.suffix == ".parquet":
            pytest.fail("Re-read completed corpus payload")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_payload)
    def reject(*args, **kwargs): pytest.fail("Loaded an unnecessary runtime resource")
    assert coordinator.run(**{key: reject for key in resources})["state"] == "CPU_WORKER_COMPLETE"
    assert coordinator.collect() == result


def test_production_ownership_and_presets_cover_nine_sources_once():
    root = Path(__file__).resolve().parents[1]
    scope = read_json(root/"configs/intake_ai_guru_full_20261019.json")
    sources = scope["base_plan"]["sources"]+scope["additional_sources"]+[scope["retained"]]
    owners = {s["job_id"]: ParallelIntake.OWNERS[i] for i, s in enumerate(sources)}
    assert [j for j, w in owners.items() if w == 0] == ["new-00000", "new-00001", "new-00005"]
    assert [j for j, w in owners.items() if w == 1] == ["new-00002", "new-00003"]
    assert [j for j, w in owners.items() if w == 2] == ["new-00004", "new-00006"]
    assert [j for j, w in owners.items() if w == 3] == ["new-00007", "retained-71be0e12"]
    import ast
    for owner in range(4):
        book = read_json(root/f"notebooks/parallel_cpu/02_cpu_worker_{owner}.ipynb")
        text = "\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
        ast.parse(text)
        assert f"WORKER_ID = {owner} #@param" in text and 'RUN_MODE = "parallel" #@param' in text
        assert (root/"scripts/parallel_intake.py").read_text(encoding="utf-8").strip() in text


def test_collect_waits_without_reading_old_payloads_and_serial_mode_is_blocked(tmp_path, external, monkeypatch):
    scope, root, work, resources, _, _, _, _ = setup_parallel(tmp_path, external)
    coordinator = worker(scope, root, work, 0)
    assert coordinator.collect()["pending_sources"] == list(coordinator.owners)
    serial = FullIntake(scope, root=root, work_dir=work, tokenizer_spec=SPEC, chunk_config=CONFIG,
        controller_type=IntakeResume, retained_type=RetainedAdapter,
        sql_context=bounded_intake_sql, parquet_writer=bounded_intake_parquet, shard_size=1)
    with pytest.raises(ValueError, match="4 CPU workers"):
        serial.status()
    with pytest.raises(ValueError, match="4 CPU workers"):
        serial.run(**resources)


def test_collector_rejects_duplicate_ids_even_if_source_receipt_was_rehashed(tmp_path, external):
    scope, root, work, resources, _, _, _, _ = setup_parallel(tmp_path, external)
    for owner in range(4):
        worker(scope, root, work.parent/f"work-{owner}", owner).run(**resources)
    coordinator = worker(scope, root, work, 0)
    ref, receipt, marker = coordinator._ref("new-00000")
    path = marker.parent/receipt["files"]["outcomes"]["path"]
    rows = pq.read_table(path).to_pylist()
    rows[0]["doc_id"] = 10100  # duplicate an ID in source 00001
    write_parquet(path, rows)
    checksum = sha256_file(path)
    receipt["outcomes_sha256"] = checksum
    receipt["files"]["outcomes"].update(sha256=checksum, bytes=path.stat().st_size)
    atomic_json(marker, receipt)
    ref["receipt_sha256"] = sha256_file(marker)
    atomic_json(root/"parallel_cpu4/results/new-00000.json", ref)
    with pytest.raises(ValueError, match="Parallel coverage failed"):
        coordinator.collect()
    assert not (root/"parallel_cpu4/collection.json").exists()


def test_parallel_can_finalize_old_retained_build_without_loading_resources(tmp_path, external, monkeypatch):
    scope, root, work, resources, _, _, _, _ = setup_parallel(tmp_path, external)
    original = core._write_outcomes
    def stop(path, *args):
        if Path(path).name == "retained-test":
            raise RuntimeError("retained handoff interrupted")
        return original(path, *args)
    monkeypatch.setattr(core, "_write_outcomes", stop)
    old = FullIntake(scope, root=root, work_dir=work, tokenizer_spec=SPEC, chunk_config=CONFIG,
        controller_type=IntakeResume, retained_type=RetainedAdapter,
        sql_context=bounded_intake_sql, parquet_writer=bounded_intake_parquet, shard_size=1)
    with pytest.raises(RuntimeError, match="retained handoff"):
        old.run(**resources)
    build = root/"retained-recovery/processed/retained-test/build.json"
    before = build.read_bytes()
    monkeypatch.setattr(core, "_write_outcomes", original)
    def reject(*args, **kwargs): pytest.fail("Replayed an unnecessary retained resource")
    monkeypatch.setattr(core, "import_external_corpus", reject)
    result = worker(scope, root, work, 3).run(**{key: reject for key in resources})
    assert result["state"] == "CPU_WORKER_COMPLETE"
    assert build.read_bytes() == before
