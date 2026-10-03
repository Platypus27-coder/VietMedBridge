"""Colab recovery of all remaining Stage A IDs with a complete coverage ledger."""


def remaining_cells(bootstrap, md, code):
    return [
        md("""
        # VietMedBridge — 01g: Thu tiếp các URL Stage A còn thiếu

        Dùng **runtime CPU mới**, cùng Drive DATA_ROOT. Notebook tự đọc đủ
        1.000 ID, kiểm raw checkpoint và ledger sau retry, trích lại 11 browser
        captures sẵn có (bao gồm công việc của 01f), giữ 4 ứng viên 01c để review,
        rồi xử lý 68 ID còn lại bằng HTTP Chrome TLS → Crawl4AI browser.

        Có checkpoint từng ID; không tải lại 917 nguồn gốc hoặc 11 captures.
        Đọc robots bằng Chrome TLS, retry lỗi mạng/5xx có giới hạn. Giữ policy
        hold cho Disallow đã xác nhận, 401/403/429 hoặc robots chưa đọc được.
        Đầu ra là ứng viên review; notebook chưa gộp/sửa corpus Stage A.
        """),
        code(bootstrap.replace("code_lock.json", "remaining_recovery_code_lock.json")),
        md("## 1. Xác minh toàn bộ Stage A và chọn đúng nhóm còn thiếu"),
        code("""
        from vietmedbridge.stage_a_recovery import load_remaining_plan
        from vietmedbridge.artifacts import atomic_json, publish_file, sha256_file

        PLAN, PROVENANCE, SOURCE_FILES = load_remaining_plan(DATA_ROOT, work_dir=WORK_DIR)
        if len(PLAN["official_pairs"]) != 1000:
            raise ValueError("Notebook này dành cho mẫu Stage A 1.000 URL đã audit.")
        print("Tổng ID:", len(PLAN["official_pairs"]))
        print("HTTP gốc:", PLAN["baseline_http_captures"])
        print("Các ID sau retry:", PLAN["failed_ids"], "| cần thử tiếp:", PLAN["remaining_ids"])
        print("11 bài đã có được trích lại tự động; không cần chạy 01f trước.")
        """),
        md("## 2. Cài bộ HTTP/browser — CPU, không dùng LLM"),
        code("""
        import importlib.metadata
        import subprocess

        CRAWL4AI_COMMIT = "e5d2e786d1a101225f3f6a3e6fd344d76eeb13af"
        CRAWL4AI_URL = f"https://github.com/unclecode/crawl4ai/archive/{CRAWL4AI_COMMIT}.zip"
        os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(WORK_DIR / "crawl4ai")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            f"crawl4ai @ {CRAWL4AI_URL}", "scrapling[fetchers]==0.4.15",
            "playwright==1.63.0", "curl_cffi==0.16.3"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=True)
        from scrapling.fetchers import Fetcher as ScraplingFetcher
        from vietmedbridge.crawl4ai_probe import crawl4ai_browser_probe
        from vietmedbridge.stage_a_recovery import recover_remaining

        VERSIONS = {p: importlib.metadata.version(p) for p in (
            "crawl4ai", "scrapling", "playwright", "curl_cffi", "patchright",
        )}
        VERSIONS["crawl4ai_source_commit"] = CRAWL4AI_COMMIT
        print(VERSIONS)
        """),
        md("""
        ## 3. Thu tiếp nhóm còn thiếu và lưu từng ID

        Nếu runtime ngắt, chạy lại cùng notebook/cấu hình; ID đã ghi checkpoint
        được kiểm hash và dùng lại. MAX_NEW_IDS chỉ giới hạn công việc trong
        phiên hiện tại, không làm rơi ID khỏi ledger. Đổi RECOVERY_LABEL để tạo
        lượt retry mới khi cần thử lại lỗi tạm thời; giữ lượt cũ làm bằng chứng.
        """),
        code("""
        RECOVERY_LABEL = "remaining-v1"
        MAX_NEW_IDS = None  # None = xử lý đủ nhóm còn thiếu; có thể đặt 10 cho mỗi phiên.
        REPORT_DIR = DATA_ROOT / "reports/crawl_recovery/stage-a-v2"
        RESULT = await recover_remaining(
            PLAN, REPORT_DIR / "remaining", ScraplingFetcher.get,
            browser_probe=crawl4ai_browser_probe, provenance=PROVENANCE,
            versions=VERSIONS, label=RECOVERY_LABEL, max_new_ids=MAX_NEW_IDS,
        )
        RESULT_DIR = Path(RESULT["output_dir"])
        print(json.dumps({k: RESULT[k] for k in (
            "official_ids", "baseline_http_captures", "remaining_input_ids",
            "processed_remaining_ids", "states", "canonical_corpus_modified",
        )}, ensure_ascii=False, indent=2))
        """),
        md("## 4. Xuất ledger đủ 1.000 ID và gói bằng chứng để gửi kiểm tra"),
        code("""
        import shutil
        import pandas as pd
        from google.colab import files

        for name, source in SOURCE_FILES.items():
            name = "official_stage_a.parquet" if name == "official_stage_a" else name
            publish_file(source, RESULT_DIR / "inputs" / name)
        pd.DataFrame(RESULT["coverage_ledger"]).to_csv(
            RESULT_DIR / "coverage_1000.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(RESULT["records"]).to_csv(
            RESULT_DIR / "recovery_results.csv", index=False, encoding="utf-8-sig")
        artifacts = {p.relative_to(RESULT_DIR).as_posix(): sha256_file(p)
                     for p in RESULT_DIR.rglob("*") if p.is_file() and p.name != "export_hashes.json"}
        atomic_json(RESULT_DIR / "export_hashes.json", artifacts)
        display(pd.DataFrame(RESULT["records"])[["doc_id", "url", "state"]])
        local_zip = Path(shutil.make_archive(
            str(WORK_DIR / f"remaining-stage-a-{RESULT['experiment']}"), "zip", root_dir=RESULT_DIR))
        publish_file(local_zip, REPORT_DIR / local_zip.name)
        print("Gửi ZIP này để kiểm đủ URL và nội dung:", local_zip.name)
        files.download(str(local_zip))
        """),
    ]
