"""Independent ranges must merge into one complete, build-compatible capture."""

import asyncio
import gzip
import json
import sys
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from team_crawl import merge_team_crawls, team_plan  # noqa: E402

from vietmedbridge.crawl import (CrawlConfig, completed_parts, crawl_links,
                                 iter_raw_records, read_completed_crawl)
from vietmedbridge.artifacts import atomic_json, sha256_file


def make_team(tmp_path, *, skip_member=None):
    links = tmp_path / "links.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": 100 + i * 3, "url": f"https://example.test/article/{i}"}
        for i in range(11)
    ]), links)
    plan = team_plan(links, shard_size=2, run_prefix="team-test")
    assert [(p["start_row"], p["stop_row"]) for p in plan["parts"]] == [
        (0, 4), (4, 8), (8, 11),
    ]

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, text=f"Body for {request.url.path}",
                              headers={"content-type": "text/plain"})

    source_dirs = []
    for part in plan["parts"]:
        source = tmp_path / f"drive-{part['member']}" / "crawl" / part["run_name"]
        if part["member"] != skip_member:
            asyncio.run(crawl_links(
                links, source.parent, run_name=part["run_name"],
                config=CrawlConfig(shard_size=2, attempts=1, per_host_delay=0),
                transport=httpx.MockTransport(handler), work_dir=tmp_path,
                start=part["start_row"], stop=part["stop_row"],
                origin_corpus_sha256="official-snapshot",
            ))
        source_dirs.append(source)
    return links, plan, source_dirs


def test_merge_three_independent_drives(tmp_path):
    links, plan, sources = make_team(tmp_path)
    result = merge_team_crawls(plan, links_path=links,
                               origin_corpus_sha256="official-snapshot",
                               source_dirs=sources, output_root=tmp_path / "combined")
    merged = Path(result["run_dir"])
    assert result["range_complete"]
    assert result["recorded_documents"] == 11
    assert read_completed_crawl(merged, links_path=links,
                                origin_corpus_sha256="official-snapshot")["range_complete"]
    ids = [record["doc_id"] for part in completed_parts(merged)
           for record in iter_raw_records(merged / part["raw_file"])]
    assert len(ids) == len(set(ids)) == 11
    assert sorted(ids) == [100 + i * 3 for i in range(11)]
    assert all(source.exists() for source in sources)


def test_merge_rejects_missing_member_without_publishing(tmp_path):
    links, plan, sources = make_team(tmp_path, skip_member=3)
    with pytest.raises(FileNotFoundError):
        merge_team_crawls(plan, links_path=links,
                          origin_corpus_sha256="official-snapshot",
                          source_dirs=sources, output_root=tmp_path / "combined")
    assert not (tmp_path / "combined" / plan["merged_run_name"]).exists()


def test_merge_rejects_modified_plan(tmp_path):
    links, plan, sources = make_team(tmp_path)
    plan["parts"][1]["start_row"] += 1
    with pytest.raises(ValueError, match="plan was changed"):
        merge_team_crawls(plan, links_path=links,
                          origin_corpus_sha256="official-snapshot",
                          source_dirs=sources, output_root=tmp_path / "combined")


def test_merge_resumes_staging_and_reuses_complete_target(tmp_path):
    links, plan, sources = make_team(tmp_path)
    options = {"links_path": links, "origin_corpus_sha256": "official-snapshot",
               "source_dirs": sources, "output_root": tmp_path / "combined"}
    first = merge_team_crawls(plan, **options)
    target = Path(first["run_dir"])
    staging = target.with_name(f".{target.name}.staging")
    target.rename(staging)
    resumed = merge_team_crawls(plan, **options)
    assert resumed["range_complete"] and Path(resumed["run_dir"]) == target
    assert merge_team_crawls(plan, **options)["recorded_documents"] == 11


def test_merge_rejects_corrupt_source(tmp_path):
    links, plan, sources = make_team(tmp_path)
    part = completed_parts(sources[1])[0]
    raw = sources[1] / part["raw_file"]
    raw.write_bytes(raw.read_bytes() + b"damage")
    with pytest.raises(ValueError, match="Missing or changed artifact"):
        merge_team_crawls(plan, links_path=links,
                          origin_corpus_sha256="official-snapshot",
                          source_dirs=sources, output_root=tmp_path / "combined")


def test_merge_checks_raw_ids_even_if_checkpoint_hash_was_updated(tmp_path):
    links, plan, sources = make_team(tmp_path)
    part = completed_parts(sources[0])[0]
    raw = sources[0] / part["raw_file"]
    records = list(iter_raw_records(raw))
    records[0]["doc_id"] = -1
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    part["raw_sha256"] = sha256_file(raw)
    pointer = next(sources[0].rglob(f"{part['part']}.done.json"))
    atomic_json(pointer, part)
    with pytest.raises(ValueError, match="Raw ID/URL mismatch"):
        merge_team_crawls(plan, links_path=links,
                          origin_corpus_sha256="official-snapshot",
                          source_dirs=sources, output_root=tmp_path / "combined")
