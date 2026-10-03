import asyncio
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from vietmedbridge.artifacts import atomic_json, read_json, sha256_file
from vietmedbridge.chrome_transport import ChromeRobotsTransport
from vietmedbridge.stage_a_recovery import load_remaining_plan, recover_remaining, select_remaining


BODY = ('<html lang="vi"><head><title>Bài nguồn</title></head><body><article><h1>Bài nguồn</h1>'
        '<p>' + 'Nghiên cứu y sinh và điều trị cần giữ nguyên nội dung nguồn tiếng Việt. ' * 12 +
        '</p></article></body></html>').encode()
URL = "https://source.test/article"


async def no_sleep(_seconds):
    pass


def response(url, status=200, body=BODY, headers=None):
    return SimpleNamespace(url=url, status=status, body=body,
                           headers=headers or {"content-type": "text/html"})


def plan_for_ids():
    expected = {i: f"https://source.test/article-{i}" for i in range(1, 7)}
    failures = [{"doc_id": i, "url": expected[i], "error": "http_403"} for i in range(2, 7)]
    robots = [{"doc_id": i, "url": expected[i], "robots_state": "ROBOTS_OK_ALLOWED"} for i in range(2, 7)]
    return select_remaining(expected, failures, robots, cached_browser_ids={2}, legacy_candidate_ids={3})


def test_selection_keeps_all_ids_and_rejects_overlapping_or_missing_groups():
    plan = plan_for_ids()
    assert plan["baseline_http_captures"] == 1 and plan["remaining_ids"] == 3
    assert {r["doc_id"] for r in plan["records"] if r["state"] == "PENDING_RECOVERY"} == {4, 5, 6}
    expected = {1: URL}
    failed = [{"doc_id": 1, "url": URL}]
    policies = [{"doc_id": 1, "url": URL, "robots_state": "ROBOTS_403"}]
    with pytest.raises(ValueError, match="overlap"):
        select_remaining(expected, failed, policies, cached_browser_ids={1}, legacy_candidate_ids={1})
    with pytest.raises(ValueError, match="cover every"):
        select_remaining(expected, failed, [], cached_browser_ids=set(), legacy_candidate_ids=set())
    with pytest.raises(ValueError, match="official"):
        select_remaining(expected, [{"doc_id": 1, "url": URL + '-wrong'}], policies,
                         cached_browser_ids=set(), legacy_candidate_ids=set())


def test_chrome_transport_retries_5xx_and_keeps_decoded_body_without_double_gzip():
    calls, waits = [], []
    def get(url, **kwargs):
        calls.append(kwargs)
        return response(url, 503 if len(calls) == 1 else 200, b"User-agent: *\nAllow: /\n",
                        {"content-type": "text/plain", "content-encoding": "gzip", "retry-after": "7"})
    async def sleep(seconds):
        waits.append(seconds)
    async def run():
        async with httpx.AsyncClient(transport=ChromeRobotsTransport(get, sleep=sleep)) as client:
            return await client.get(URL.replace("article", "robots.txt"), headers={"User-Agent": "VietMedBridge/0.2"})
    result = asyncio.run(run())
    assert result.content.startswith(b"User-agent:") and result.status_code == 200
    assert waits == [7] and len(calls) == 2
    assert all(c["impersonate"] == "chrome" and not c["follow_redirects"] for c in calls)


def test_recovery_resumes_without_refetching_and_keeps_complete_coverage(tmp_path):
    plan, calls = plan_for_ids(), Counter()
    def get(url, **kwargs):
        calls[url] += 1
        return response(url, body=b"User-agent: *\nAllow: /\n" if url.endswith("robots.txt") else BODY)
    first = asyncio.run(recover_remaining(plan, tmp_path, get, max_new_ids=1, sleep=no_sleep))
    assert first["official_ids"] == 6 and first["processed_remaining_ids"] == 1
    assert first["states"]["PENDING_RECOVERY"] == 2
    second = asyncio.run(recover_remaining(plan, tmp_path, get, sleep=no_sleep))
    assert second["processed_remaining_ids"] == 3
    assert second["states"]["ARTICLE_CANDIDATE_REVIEW"] == 3
    assert all(calls[f'https://source.test/article-{i}'] == 1 for i in (4, 5, 6))
    assert all(calls[f'https://source.test/article-{i}'] == 0 for i in (1, 2, 3))
    assert len(second["coverage_ledger"]) == 6
    record = next(r for r in second["records"] if r["doc_id"] == 4)
    doc = read_json(Path(second["output_dir"]) / record["assets"]["document.json"]["path"])["document"]
    assert doc["body_sha256"] == hashlib.sha256(BODY).hexdigest()
    before = dict(calls)
    asyncio.run(recover_remaining(plan, tmp_path, get, sleep=no_sleep))
    assert dict(calls) == before
    text_path = Path(second["output_dir"]) / record["assets"]["source.txt"]["path"]
    text_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed artifact"):
        asyncio.run(recover_remaining(plan, tmp_path, get, sleep=no_sleep))


@pytest.mark.parametrize("robots_status", [403, 429, 503])
def test_held_robots_never_calls_article_or_browser(tmp_path, robots_status):
    plan, urls = plan_for_ids(), []
    def get(url, **kwargs):
        urls.append(url)
        assert url.endswith("robots.txt")
        return response(url, robots_status, b"Unavailable")
    async def browser(*_args, **_kwargs):
        pytest.fail("Browser cannot ignore held robots")
    result = asyncio.run(recover_remaining(plan, tmp_path, get, browser_probe=browser, sleep=no_sleep))
    assert not result["states"].get("ARTICLE_CANDIDATE_REVIEW")
    assert len(result["coverage_ledger"]) == 6


def test_explicit_prior_disallow_is_preserved_for_policy_review(tmp_path):
    plan = plan_for_ids()
    plan["records"][2]["prior_robots_state"] = "ROBOTS_OK_DISALLOWED"
    fetched = []
    def get(url, **kwargs):
        fetched.append(url)
        return response(url, body=b"User-agent: *\nAllow: /\n" if url.endswith("robots.txt") else BODY)
    result = asyncio.run(recover_remaining(plan, tmp_path, get, sleep=no_sleep))
    assert result["states"]["ROBOTS_POLICY_REVIEW"] == 1
    assert "https://source.test/article-4" not in fetched


def test_http_short_shell_then_browser_dom_keeps_both_variants(tmp_path):
    plan = plan_for_ids()
    shell = b'<html><title>Page</title><body><article><p>Short source shell.</p></article></body></html>'
    def get(url, **kwargs):
        return response(url, body=b"User-agent: *\nAllow: /\n" if url.endswith("robots.txt") else shell)
    async def browser(url, _guard, **kwargs):
        assert kwargs["config"].allowed_hosts == ("source.test",)
        return ({"http_status": 200, "rendered_http_status": 200, "guard_errors": [], "final_url": url},
                {"response.html": shell, "rendered.html": BODY})
    result = asyncio.run(recover_remaining(plan, tmp_path, get, browser_probe=browser, max_new_ids=1, sleep=no_sleep))
    record = next(r for r in result["records"] if r["doc_id"] == 4)
    assert record["state"] == "ARTICLE_CANDIDATE_REVIEW" and record["capture_kind"] == "rendered_dom"
    assert record["source_method"] == "crawl4ai_browser"
    assert len(list((Path(result["output_dir"]) / "documents/4").rglob("*.json"))) == 3


def test_interrupted_id_resumes_even_if_new_response_bytes_change(tmp_path, monkeypatch):
    import vietmedbridge.stage_a_recovery as recovery
    plan, generation = plan_for_ids(), [1]
    def get(url, **kwargs):
        body = (b"User-agent: *\nAllow: /\n" if url.endswith("robots.txt") else
                BODY.replace(b"</html>", f"<!-- request {generation[0]} --></html>".encode()))
        return response(url, body=body)
    atomic = recovery.atomic_json
    def interrupted(path, value):
        if Path(path).parent.name == "markers":
            raise RuntimeError("Injected runtime interruption")
        return atomic(path, value)
    monkeypatch.setattr(recovery, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="runtime interruption"):
        asyncio.run(recover_remaining(plan, tmp_path, get, max_new_ids=1, sleep=no_sleep))
    monkeypatch.setattr(recovery, "atomic_json", atomic)
    generation[0] = 2
    result = asyncio.run(recover_remaining(plan, tmp_path, get, max_new_ids=1, sleep=no_sleep))
    assert result["processed_remaining_ids"] == 1
    record = next(r for r in result["records"] if r["doc_id"] == 4)
    assert record["state"] == "ARTICLE_CANDIDATE_REVIEW"
    assert len(list((Path(result["output_dir"]) / "assets/4").iterdir())) == 2


def test_drive_plan_checks_raw_ledger_and_reuses_captures_before_network(tmp_path):
    import csv
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_probe_review import fixture_probe, URL as captured_url
    from vietmedbridge.crawl import CrawlConfig, crawl_links

    root = tmp_path / "data"
    root.mkdir()
    pairs = {1: "https://source.test/ok", 3: "https://source.test/legacy",
             4: "https://source.test/missing", 583: captured_url}
    links = root / "stage_a.parquet"
    pq.write_table(pa.table({"id": list(pairs), "url": list(pairs.values())}), links)
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200 if request.url.path == "/ok" else 404,
                              content=BODY, headers={"content-type": "text/html"})
    result = asyncio.run(crawl_links(links, root / "crawl", run_name="stage-a-v2",
        config=CrawlConfig(per_host_delay=0, attempts=1), transport=httpx.MockTransport(handler), work_dir=tmp_path))
    signature = result["run_signature"]
    atomic_json(root / "reports/inventory/stage_a.json",
                {"files": {"stage_a_links": {"path": links.name, "sha256": sha256_file(links)}}})
    report = root / "reports/crawl_recovery/stage-a-v2"
    legacy = report / "experiments/6a5793cdc03be0cb"
    legacy.mkdir(parents=True)
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return sha256_file(path)
    failures = [{"doc_id": i, "url": pairs[i], "error": "http_404"} for i in (3, 4, 583)]
    failed_path = report / "failures_after_retry.csv"
    failed_sha = write_csv(failed_path, failures)
    triage = {"run_signature": signature, "successful": 1, "unresolved": 3,
              "failures_csv": failed_path.relative_to(root).as_posix(), "failures_sha256": failed_sha}
    atomic_json(report / "triage.json", triage)
    robots_sha = write_csv(legacy / "robots.csv", [
        {"doc_id": i, "url": pairs[i], "robots_state": "ROBOTS_OK_ALLOWED"} for i in (3, 4, 583)])
    body_path = legacy / "legacy.bin"
    body_path.write_bytes(BODY)
    legacy_sha = write_csv(legacy / "attempts.csv", [{"doc_id": 3, "url": pairs[3],
        "article_candidate": True, "asset_path": body_path.relative_to(root).as_posix(),
        "body_sha256": sha256_file(body_path)}])
    atomic_json(legacy / "experiment.json", {"source_run_signature": signature})
    atomic_json(legacy / "summary.json", {"robots_sha256": robots_sha,
        "attempts_sha256": legacy_sha, "article_candidates_needing_human_review": 1})
    probe, browser_sha = fixture_probe(tmp_path)
    browser_summary = read_json(probe / "summary.json")
    browser_summary["candidate_ids_needing_review"] = [583]
    atomic_json(probe / "summary.json", browser_summary)
    shutil.copytree(probe, report / "experiments/4c49d3358aa818b7")
    kwargs = dict(robots_sha256=robots_sha, legacy_sha256=legacy_sha,
                  browser_sha256=browser_sha, work_dir=tmp_path)
    plan, provenance, sources = load_remaining_plan(root, **kwargs)
    assert plan["baseline_http_captures"] == 1 and plan["remaining_ids"] == 1
    assert {r["doc_id"]: r["state"] for r in plan["records"]} == {
        3: "LEGACY_CANDIDATE_REVIEW", 4: "PENDING_RECOVERY", 583: "CACHED_BROWSER_REVIEW"}
    assert provenance["run_signature"] == signature and sources["legacy-3.bin"] == body_path
    # Even an updated CSV hash cannot override the official URL mapping.
    failures[0]["url"] += "-changed"
    triage["failures_sha256"] = write_csv(failed_path, failures)
    atomic_json(report / "triage.json", triage)
    with pytest.raises(ValueError, match="official Stage A"):
        load_remaining_plan(root, **kwargs)
