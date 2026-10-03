"""Versioned cells for the one-URL HTTP/browser diagnostic."""


def browser_cells(bootstrap, md, code):
    return [
        md("""
        # VietMedBridge — 01e: Chẩn đoán HTTP 403 bằng trình duyệt

        Chọn **runtime CPU mới**, chạy từ trên xuống. 01c/01d chưa thử browser;
        403 từ HTTP chưa chứng minh browser không tải được. Notebook này dùng
        Scrapling 0.4.15 + Chromium chạy JavaScript, profile mới. Mặc định thử
        **1 URL Long Châu** từ 11 URL lỗi 01d, kèm một lượt HTTP Chrome TLS.

        Lưu response HTML kể cả 403, DOM sau render, screenshot, headers đã lọc,
        robots decisions và hashes lên Drive. Có checkpoint từng URL/phương pháp.
        Guard được cài trước navigation, kiểm từng redirect; phạm vi host hẹp,
        giới hạn request và dừng khi gặp rate limit. Không tự nhập vào corpus.
        """),
        code(bootstrap.replace("code_lock.json", "browser_probe_code_lock.json")),
        md("""
        ## 1. Xác minh 11 URL lỗi và chọn một URL

        Giữ `N_URLS=1` cho lượt đầu. Sau khi kiểm HTML/screenshot của URL mẫu,
        nếu có bài thật, có thể đổi thành `11` và chạy lại cell này trở xuống.
        Checkpoint của URL đã xong được dùng lại; code/config mới tạo experiment mới.
        """),
        code("""
        import pandas as pd
        from vietmedbridge.artifacts import (
            code_fingerprint, digest_json, publish_file, read_json, sha256_file,
            utc_now, verify_file,
        )

        SOURCE_KEY = "3e6c01b92a7e3194"
        EXPECTED_ATTEMPTS_SHA256 = "def3ac4cf8beb4be7d950282951db76e18804bc705c3b5a63c83eb801623ea59"
        REPORT_DIR = DATA_ROOT / "reports/crawl_recovery/stage-a-v2"
        SOURCE_DIR = REPORT_DIR / "experiments" / SOURCE_KEY
        SOURCE = read_json(SOURCE_DIR / "summary.json")
        ATTEMPTS_PATH = SOURCE_DIR / "attempts.csv"
        if SOURCE["attempts_sha256"] != EXPECTED_ATTEMPTS_SHA256:
            raise ValueError("Summary 01d khác kết quả đã audit.")
        verify_file(ATTEMPTS_PATH, EXPECTED_ATTEMPTS_SHA256)
        INPUT = pd.read_csv(ATTEMPTS_PATH, dtype={"doc_id": "int64", "url": "string"})
        LEDGER = pd.read_csv(CHECKOUT / "reports/stage-a-v2-longchau-access-gap.csv",
                             dtype={"doc_id": "int64", "url": "string"})
        if len(INPUT) != 11 or INPUT["doc_id"].duplicated().any():
            raise ValueError("Input phải có 11 ID riêng biệt từ 01d.")
        if not INPUT["outcome"].eq("http_403").all():
            raise ValueError("Input không khớp 11 kết quả 403 đã audit.")
        if dict(zip(INPUT.doc_id, INPUT.url)) != dict(zip(LEDGER.doc_id, LEDGER.url)):
            raise ValueError("ID/URL khác ledger nguồn chính thức đã audit.")
        N_URLS = 1  # Sau khi kiểm bài mẫu, có thể tăng lên 11.
        PROBE_LABEL = "browser-v1"  # Đổi tên để tạo lượt thử mới nếu checkpoint cũ lỗi.
        if not 1 <= N_URLS <= 11:
            raise ValueError("N_URLS phải nằm trong 1–11.")
        SELECTED = INPUT.sort_values("doc_id").head(N_URLS)
        display(SELECTED[["doc_id", "url", "outcome"]])
        """),
        md("## 2. Cài Scrapling và Chromium — chỉ CPU"),
        code("""
        import importlib.metadata
        import subprocess
        import time
        from dataclasses import asdict
        from urllib.parse import urlsplit

        # Pin cả browser driver và HTTP transport của phép thử này.
        PINS = {"scrapling": "0.4.15", "playwright": "1.63.0", "curl_cffi": "0.16.3"}
        missing = []
        for package, version in PINS.items():
            try:
                if importlib.metadata.version(package) != version:
                    missing.append(package)
            except importlib.metadata.PackageNotFoundError:
                missing.append(package)
        if missing:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "scrapling[fetchers]==0.4.15", "playwright==1.63.0",
                            "curl_cffi==0.16.3"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
                       check=True)
        from scrapling.fetchers import AsyncDynamicSession, Fetcher as ScraplingFetcher
        from vietmedbridge.browser_probe import (
            BrowserProbeConfig, guarded_browser_probe, response_evidence,
        )
        from vietmedbridge.crawl import CrawlConfig, Fetcher
        from vietmedbridge.recovery import RecoveryConfig, guarded_http_fetch
        from vietmedbridge.text import EXPECTED_PARSE_ERRORS, extract_source
        from vietmedbridge.quality import document_quality

        BROWSER_CONFIG = BrowserProbeConfig()
        HTTP_CONFIG = RecoveryConfig(attempts=1, impersonate="chrome")
        VERSIONS = {p: importlib.metadata.version(p) for p in (
            "scrapling", "playwright", "curl_cffi", "browserforge", "patchright",
        )}
        MANIFEST = {
            "source_key": SOURCE_KEY, "source_attempts_sha256": EXPECTED_ATTEMPTS_SHA256,
            "code_sha256": code_fingerprint(), "versions": VERSIONS,
            "browser_config": asdict(BROWSER_CONFIG), "http_config": asdict(HTTP_CONFIG),
            "policy": "single-article-browser-probe-v1",
            "probe_label": PROBE_LABEL,
            "notebook_sha256": sha256_file(
        CHECKOUT / "archive/notebooks/stage-a-1000-recovery/01e_colab_browser_diagnosis.ipynb"
            ),
        }
        # Selection count is excluded so increasing 1 -> 11 resumes the same config.
        PROBE_KEY = digest_json(MANIFEST)[:16]
        PROBE_DIR = REPORT_DIR / "experiments" / PROBE_KEY
        MARKERS = PROBE_DIR / "markers"
        MARKERS.mkdir(parents=True, exist_ok=True)
        manifest_path = PROBE_DIR / "experiment.json"
        if manifest_path.exists() and digest_json(read_json(manifest_path)) != digest_json(MANIFEST):
            raise ValueError("Experiment config đã đổi.")
        if not manifest_path.exists():
            atomic_json(manifest_path, MANIFEST)
        print("Experiment:", PROBE_KEY, "| package versions:", VERSIONS)
        """),
        md("""
        ## 3. HTTP Chrome TLS và browser render, lưu checkpoint lên Drive

        `article_candidate` vẫn cần kiểm nội dung. Chỉ HTTP 200, nội dung đủ dài
        và có tiêu đề bài mới thành ứng viên; HTTP 403 luôn là lỗi, dù có HTML.
        Host tài nguyên ngoài phạm vi được ghi trong `request_log` để chẩn đoán.
        """),
        code("""
        def save_assets(doc_id, method, assets):
            saved = {}
            for name, body in assets.items():
                local = WORK_DIR / f"probe-{doc_id}-{method}-{name}"
                target = PROBE_DIR / "assets" / str(doc_id) / method / name
                local.write_bytes(body)
                try:
                    digest = publish_file(local, target)
                finally:
                    local.unlink(missing_ok=True)
                saved[name] = {"path": str(target.relative_to(DATA_ROOT)), "sha256": digest}
            return saved

        def assess(record, body, is_browser=False):
            record["article_candidate"] = False
            if record.get("http_status") != 200 or body is None:
                return
            if is_browser and (record.get("guard_errors") or record.get("rendered_http_status") != 200):
                return
            try:
                extracted = extract_source(body, "text/html")
                record.update(source_chars=len(extracted["source_text"]),
                              source_text_sha256=extracted["source_text_sha256"],
                              source_preview=extracted["source_text"][:500],
                              extracted_title=extracted["title"])
                record.update(document_quality(extracted["source_text"], title=extracted["title"],
                                               raw_bytes=len(body), content_type="text/html"))
                record["article_candidate"] = bool(
                    len(extracted["source_text"]) >= 300 and extracted["title"].strip()
                )
                if record["article_candidate"]:
                    record["outcome"] = "article_candidate_needs_review"
                    record["assets"].update(save_assets(record["doc_id"], record["method"],
                                                       {"extracted.txt": extracted["source_text"].encode("utf-8")}))
                else:
                    record["outcome"] = "short_or_untitled_text_review"
            except EXPECTED_PARSE_ERRORS as exc:
                record.update(outcome="extract_rejected", extract_error=str(exc)[:300])

        held_hosts = set()
        records = []
        for row in SELECTED.itertuples(index=False):
            for method in ("chrome_http", "dynamic_browser"):
                marker = MARKERS / f"{int(row.doc_id)}-{method}.json"
                if marker.exists():
                    saved = read_json(marker)
                    if saved["url"] != row.url or saved["method"] != method:
                        raise ValueError("Checkpoint khác URL hoặc phương pháp.")
                    for asset in saved.get("assets", {}).values():
                        verify_file(DATA_ROOT / asset["path"], asset["sha256"])
                    held_hosts.update(saved.get("held_hosts", []))
                    if saved.get("http_status") == 429:
                        held_hosts.add(urlsplit(row.url).hostname)
                    records.append(saved)
                    continue
                started = time.monotonic()
                record = {"doc_id": int(row.doc_id), "url": row.url, "method": method,
                          "started_at": utc_now(), "article_candidate": False, "assets": {}}
                if urlsplit(row.url).hostname in held_hosts:
                    record["outcome"] = "host_rate_limit_hold"
                else:
                    guard = Fetcher(CrawlConfig(concurrency=1, attempts=1,
                                    per_host_delay=3 if method == "chrome_http" else 0.1))
                    try:
                        if method == "chrome_http":
                            observations = []
                            def observed_get(url, **kwargs):
                                response = ScraplingFetcher.get(url, **kwargs)
                                body = bytes(response.body)
                                evidence = {"url": url, "http_status": int(response.status),
                                            **response_evidence(body, response.headers)}
                                name = f"response-{len(observations)}.html"
                                if len(body) <= HTTP_CONFIG.max_body_bytes:
                                    record["assets"].update(save_assets(row.doc_id, method, {name: body}))
                                observations.append(evidence)
                                return response
                            result, body = await guarded_http_fetch(
                                row.url, guard, observed_get, user_agent=guard.config.user_agent,
                                config=HTTP_CONFIG, held_hosts=held_hosts,
                            )
                            record.update(result, response_evidence=observations)
                            assess(record, body)
                        else:
                            async with AsyncDynamicSession(
                                headless=True, retries=1, google_search=False,
                                locale="vi-VN", additional_args={"service_workers": "block",
                                                               "accept_downloads": False},
                            ) as session:
                                result, assets = await guarded_browser_probe(
                                    session.context, row.url, guard, config=BROWSER_CONFIG,
                                )
                            # Preserve the notebook's method/checkpoint names.
                            result.pop("method", None)
                            record.update(result)
                            record["assets"].update(save_assets(row.doc_id, method, assets))
                            held_hosts.update(result.get("held_hosts", []))
                            assess(record, assets.get("rendered.html"), is_browser=True)
                    except Exception as exc:
                        record.update(outcome="probe_error", error_type=type(exc).__name__,
                                      error_message=str(exc)[:500])
                    finally:
                        await guard.client.aclose()
                record.update(elapsed_ms=round((time.monotonic() - started) * 1000),
                              finished_at=utc_now())
                atomic_json(marker, record)
                records.append(record)
                print(row.doc_id, method, record["outcome"], "| chars:", record.get("source_chars", 0))
        """),
        md("## 4. Tổng hợp và tải gói bằng chứng"),
        code("""
        import shutil
        from IPython.display import Image, display

        # Export all completed IDs in this experiment, including the first sample.
        ALL_RECORDS = [read_json(p) for p in sorted(MARKERS.glob("*.json"))]
        frame = pd.DataFrame(ALL_RECORDS)
        local_csv = WORK_DIR / "browser-probe-attempts.csv"
        local_csv.write_text(frame.to_csv(index=False), encoding="utf-8-sig")
        result_path = PROBE_DIR / "attempts.csv"
        attempts_hash = publish_file(local_csv, result_path)
        summary = {
            "source_experiment": SOURCE_KEY, "probe_experiment": PROBE_KEY,
            "completed_ids": int(frame["doc_id"].nunique()), "completed_methods": len(frame),
            "outcomes": frame["outcome"].value_counts().to_dict(),
            "candidate_ids_needing_review": sorted({r["doc_id"] for r in ALL_RECORDS if r.get("article_candidate")}),
            "attempts_csv": str(result_path.relative_to(DATA_ROOT)),
            "attempts_sha256": attempts_hash, "created_at": utc_now(),
        }
        atomic_json(PROBE_DIR / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        display(frame.reindex(columns=["doc_id", "method", "outcome", "http_status",
                                      "rendered_http_status", "source_chars", "page_title"]))
        for r in ALL_RECORDS[:2]:
            screenshot = r.get("assets", {}).get("screenshot.png")
            if screenshot:
                display(Image(filename=str(DATA_ROOT / screenshot["path"])))
        local_zip = Path(shutil.make_archive(str(WORK_DIR / f"browser-probe-{PROBE_KEY}"),
                                            "zip", root_dir=PROBE_DIR))
        zip_path = REPORT_DIR / local_zip.name
        publish_file(local_zip, zip_path)
        print("Gửi gói ZIP này để kiểm HTML/DOM/screenshot:", zip_path)
        DOWNLOAD_ZIP = True
        if DOWNLOAD_ZIP:
            from google.colab import files
            files.download(str(local_zip))
        """),
    ]
