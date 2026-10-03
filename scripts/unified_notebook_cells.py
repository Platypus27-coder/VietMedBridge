"""Single user-facing crawl/recovery workflow for Notebook 01."""


def notebook_01_cells(bootstrap, md, code):
    return [
        md("""
        # VietMedBridge — 01: Crawl và recovery trong một notebook

        Notebook này có ba pha: `baseline` lấy URL theo shard/checkpoint; `retry_http`
        thử lại ID lỗi bằng HTTP có robots guard; `advanced_recovery` xác minh toàn bộ
        range rồi xử lý URL lỗi theo lô bằng Crawl4AI + Scrapling. Chạy lại Bootstrap
        trong runtime Colab mới và giữ nguyên định danh run để resume từ checkpoint.

        Mặc định là Stage A 1.000 URL. Mở rộng lần lượt qua gate 10k → 100k → 1m.
        Khi chuyển sang recovery, chỉ đổi `ACTION`; giữ nguyên input, range và `RUN_NAME`.
        Recovery chỉ chạy sau khi baseline của đúng range đã hoàn tất và khớp ID/URL gốc.
        Mọi pha dùng CPU, không cần GPU. Browser chỉ cài ở pha advanced khi còn ID cần xử lý.
        """),
        # Notebook 00's existing code lock may predate this notebook-only
        # orchestration file. Keep the package source identical so crawl
        # fingerprints remain resumable, while a dedicated lock retrieves it.
        code(bootstrap.replace("code_lock.json", "notebook01_code_lock.json")),
        md("## 1. Chọn pha, input và phạm vi theo vị trí dòng Parquet"),
        code("""
        from vietmedbridge.dataset import load_snapshot, parquet_path, validate_link_subset
        from vietmedbridge.crawl import CrawlConfig, crawl_links, completed_parts
        from vietmedbridge.artifacts import read_json, verify_file, sha256_file
        from vietmedbridge.gates import authorize_scale
        import pyarrow.parquet as pq

        SNAPSHOT = load_snapshot(DATA_ROOT)
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        ACTION = "baseline"   # baseline | retry_http | advanced_recovery
        MODE = "stage_a"      # stage_a | smoke | range
        START_ROW = 0          # vị trí dòng Parquet, không phải official doc_id
        STOP_ROW = 10000       # chỉ dùng ở mode range; None = hết file, cần full gate
        RANGE_RUN_NAME = "stage-b1-v2"  # Tên ổn định khi resume; đổi cho range/milestone mới
        WORKER_INDEX = 0
        WORKER_COUNT = 1       # workers phải có cùng range và chia shard khác nhau
        MAX_NEW_SHARDS = 10    # giới hạn mỗi phiên; None = hết range

        if ACTION not in {"baseline", "retry_http", "advanced_recovery"}:
            raise ValueError("ACTION phải là baseline, retry_http hoặc advanced_recovery.")
        if MODE == "stage_a":
            pilot = read_json(DATA_ROOT / "reports/inventory/stage_a.json")
            if pilot["corpus_sha256"] != SNAPSHOT["files"]["links_corpus.parquet"]["sha256"]:
                raise ValueError("Stage A sample không thuộc snapshot hiện tại.")
            INPUT_LINKS = DATA_ROOT / pilot["files"]["stage_a_links"]["path"]
            verify_file(INPUT_LINKS, pilot["files"]["stage_a_links"]["sha256"])
            validate_link_subset(INPUT_LINKS, OFFICIAL_LINKS, work_dir=WORK_DIR)
            RUN_NAME, START, STOP = "stage-a-v2", 0, None
        elif MODE == "smoke":
            audit = read_json(DATA_ROOT / "reports/dataset_audit.json")
            if audit["corpus_sha256"] != SNAPSHOT["files"]["links_corpus.parquet"]["sha256"]:
                raise ValueError("Audit/sample không thuộc snapshot; chạy lại notebook 00.")
            INPUT_LINKS = DATA_ROOT / audit["sample"]["path"]
            verify_file(INPUT_LINKS, audit["sample"]["sha256"])
            validate_link_subset(INPUT_LINKS, OFFICIAL_LINKS, work_dir=WORK_DIR)
            RUN_NAME, START, STOP = "smoke-v2", 0, None
        elif MODE == "range":
            INPUT_LINKS, RUN_NAME = OFFICIAL_LINKS, RANGE_RUN_NAME
            START, STOP = START_ROW, STOP_ROW
        else:
            raise ValueError("MODE phải là stage_a, smoke hoặc range.")

        input_rows = pq.ParquetFile(INPUT_LINKS).metadata.num_rows
        requested_rows = max(0, min(STOP if STOP is not None else input_rows, input_rows) - START)
        if ACTION in {"baseline", "retry_http"}:
            GOLDEN = read_json(DATA_ROOT / "reports/golden_latest.json")
            if GOLDEN["fixture_sha256"] != sha256_file(CHECKOUT / "tests/golden/cases.json"):
                raise ValueError("Golden fixtures đổi; chạy lại notebook 00.")
            STAGE = authorize_scale(DATA_ROOT, rows=requested_rows,
                corpus_sha256=SNAPSHOT["files"]["links_corpus.parquet"]["sha256"],
                golden_report=GOLDEN, chunking=PIPELINE_CONFIG["chunking"])
        else:
            STAGE = "post-baseline recovery"
        print("Action:", ACTION, "| run:", RUN_NAME, "| input:", INPUT_LINKS.name,
              "| physical row range:", START, STOP)
        """),
        md("""
        ## 2. Baseline hoặc retry HTTP

        `baseline` tiếp tục shard còn thiếu; `retry_http` chỉ thử lại ID hiện đang lỗi
        và ghi attempt mới, còn các raw thành công được giữ. Host delay, backoff và robots
        guard áp dụng cho mọi request. Pha advanced bỏ qua cell này.
        """),
        code("""
        if ACTION in {"baseline", "retry_http"}:
            CONFIG = CrawlConfig(**PIPELINE_CONFIG["crawler"])
            SUMMARY = await crawl_links(
                INPUT_LINKS, DATA_ROOT / "crawl", run_name=RUN_NAME, config=CONFIG,
                start=START, stop=STOP, worker_index=WORKER_INDEX, worker_count=WORKER_COUNT,
                retry_failed=(ACTION == "retry_http"), max_shards=MAX_NEW_SHARDS,
                work_dir=WORK_DIR,
                origin_corpus_sha256=SNAPSHOT["files"]["links_corpus.parquet"]["sha256"],
            )
            print(json.dumps(SUMMARY, ensure_ascii=False, indent=2))
        else:
            print("Pha advanced recovery: bỏ qua crawler baseline.")
        """),
        md("## 3. Coverage của baseline"),
        code("""
        import pandas as pd
        CRAWL_DIR = DATA_ROOT / "crawl" / RUN_NAME
        if (CRAWL_DIR / "summary.json").exists():
            CRAWL_SUMMARY = read_json(CRAWL_DIR / "summary.json")
            PARTS = completed_parts(CRAWL_DIR)
            failed_count = sum(len(part["failed_ids"]) for part in PARTS)
            recorded = sum(part["records"] for part in PARTS)
            print("Recorded:", recorded, "| requested:", CRAWL_SUMMARY["requested_input_records"],
                  "| range_complete:", CRAWL_SUMMARY["range_complete"], "| failures:", failed_count)
            display(pd.DataFrame([{"status": key, "count": value}
                                  for key, value in CRAWL_SUMMARY["statuses"].items()]))
        else:
            print("Chưa có crawl checkpoint cho run này.")
        """),
        md("""
        ## 4. Advanced recovery sau khi baseline xong

        Giữ nguyên `MODE`, range và `RUN_NAME` của baseline; chỉ đổi `ACTION` thành
        `advanced_recovery`. Preflight đối chiếu **mọi ID và URL** trong raw shards với
        đúng dòng của input Parquet, kiểm hash và yêu cầu đủ range. Nếu Colab bị ngắt,
        chạy lại notebook với cùng định danh: shard, batch và checkpoint từng ID sẽ resume.

        Crawl4AI và Scrapling cung cấp các HTTP/browser engines có guard. Chúng không
        bỏ qua robots Disallow hay policy chưa xác minh. Kết quả là capture candidate;
        canonical corpus không bị sửa và candidate vẫn cần human review.
        """),
        code("""
        if ACTION == "advanced_recovery":
            sys.path.insert(0, str(CHECKOUT / "scripts"))
            from crawl_recovery import prepare_crawl_recovery_source
            SOURCE = prepare_crawl_recovery_source(
                DATA_ROOT, INPUT_LINKS, RUN_NAME,
                expected_origin_corpus_sha256=SNAPSHOT["files"]["links_corpus.parquet"]["sha256"],
                work_dir=WORK_DIR,
            )
            print(json.dumps({k: SOURCE.get(k) for k in (
                "official_ids", "baseline_successes", "failed_ids", "pending_recovery_ids",
                "robots_policy_review_ids", "range_complete", "failures_path",
            )}, ensure_ascii=False, indent=2))
        else:
            print("Bỏ qua preflight; ACTION chưa phải advanced_recovery.")
        """),
        md("### 4.1. Cài Crawl4AI + Scrapling khi còn ID cần xử lý (CPU, không cần GPU)"),
        code("""
        ENGINES_NEEDED = ACTION == "advanced_recovery" and SOURCE["pending_recovery_ids"] > 0
        if ENGINES_NEEDED:
            import importlib.metadata
            import subprocess
            CRAWL4AI_VERSION = "0.9.4"
            os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(WORK_DIR / "crawl4ai")
            subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                f"crawl4ai=={CRAWL4AI_VERSION}", "scrapling[fetchers]==0.4.15",
                "playwright==1.63.0", "patchright==1.63.0", "playwright-stealth==2.0.3",
                "curl_cffi==0.16.3"], check=True)
            subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=True)
            subprocess.run([sys.executable, "-m", "patchright", "install", "chromium"], check=True)
            VERSIONS = {p: importlib.metadata.version(p) for p in (
                "crawl4ai", "scrapling", "playwright", "curl_cffi", "patchright", "playwright-stealth",
            )}
            VERSIONS["crawl4ai_release"] = CRAWL4AI_VERSION
            print(VERSIONS)
        else:
            VERSIONS = {}
            print("Không cần browser runtime ở pha này.")
        """),
        md("### 4.2. Chạy recovery theo batch; chạy lại cell để tiếp tục"),
        code("""
        if ACTION == "advanced_recovery":
            sys.path.insert(0, str(CHECKOUT / "scripts"))
            from crawl_recovery import recover_crawl_failures
            from vietmedbridge.advanced_recovery import AdvancedRecoveryRuntime

            RECOVERY_LABEL = "advanced-v1"  # Đổi khi chủ động mở experiment mới
            RECOVERY_BATCH_SIZE = 512
            MAX_BATCHES_THIS_SESSION = 8  # Giới hạn mỗi phiên Colab; rerun cell để tiếp tục
            MAX_NEW_IDS_PER_BATCH = None  # Hoàn tất batch hiện tại trước khi sang batch sau
            DOMAIN_PROFILES = {}  # Chỉ cấu hình từ bằng chứng nguồn; không dùng vượt robots
            RECOVERY_ROOT = DATA_ROOT / "reports" / "crawl_recovery" / RUN_NAME / "advanced"

            if ENGINES_NEEDED:
                async with AdvancedRecoveryRuntime(WORK_DIR / "recovery_browser",
                                                   profiles=DOMAIN_PROFILES) as ENGINES:
                    RESULT = await recover_crawl_failures(
                        SOURCE, RECOVERY_ROOT, ENGINES.get,
                        browser_probe=ENGINES.browser_probe, advanced_runtime=ENGINES,
                        versions=VERSIONS, label=RECOVERY_LABEL,
                        batch_size=RECOVERY_BATCH_SIZE,
                        max_batches_this_session=MAX_BATCHES_THIS_SESSION,
                        max_new_ids_per_batch=MAX_NEW_IDS_PER_BATCH,
                    )
            else:
                def unused_get(*_args, **_kwargs):
                    raise RuntimeError("Không có request nào được dự kiến ở policy-only recovery.")
                RESULT = await recover_crawl_failures(
                    SOURCE, RECOVERY_ROOT, unused_get, advanced_runtime=None,
                    versions=VERSIONS, label=RECOVERY_LABEL,
                    batch_size=RECOVERY_BATCH_SIZE,
                    max_batches_this_session=MAX_BATCHES_THIS_SESSION,
                )
            print(json.dumps({k: RESULT[k] for k in (
                "official_ids", "baseline_http_captures", "baseline_failed_ids",
                "recovery_complete", "evaluated_failed_ids", "pending_or_incomplete_failed_ids",
                "completed_batches", "batches_this_session", "failure_states",
                "failed_id_coverage_csv", "coverage_by_domain_csv", "unresolved_urls_csv",
                "candidate_review_csv", "canonical_corpus_modified",
            )}, ensure_ascii=False, indent=2))
        else:
            print("Bỏ qua advanced recovery.")
        """),
        md("### 4.3. Báo cáo coverage cần review"),
        code("""
        if ACTION == "advanced_recovery":
            display(pd.DataFrame([{"state": state, "ids": count}
                                  for state, count in RESULT["failure_states"].items()]))
            print("Coverage cho mọi failure ID:", RESULT["failed_id_coverage_csv"])
            print("Theo domain/trạng thái:", RESULT["coverage_by_domain_csv"])
            print("URL đang chờ xử lý hoặc human review:", RESULT["unresolved_urls_csv"])
            print("Article candidates:", RESULT["candidate_review_csv"])
            print("Summary:", Path(RESULT["output_dir"]) / "summary.json")
            if not RESULT["recovery_complete"]:
                print("Còn batch chưa hoàn tất; chạy lại cell 4.2 và 4.3 với cùng cấu hình.")
            else:
                print("Mọi failure ID đã được đánh giá; capture candidate vẫn cần human review.")
        else:
            print("Sau khi baseline mục tiêu hoàn tất, đặt ACTION=advanced_recovery để dùng pha này.")
        """),
        md("""
        ## Resume, scale và GPU

        Baseline: giữ cùng ACTION/input/range/RUN_NAME rồi chạy lại cell crawl; shard đã
        xong sẽ bỏ qua. `retry_http` chỉ thử lại ID lỗi. Để recovery, giữ nguyên định danh
        run và chạy lại đến khi `recovery_complete: true`. Sau review, dùng notebook 02–03
        để trích/chia đoạn, kiểm tra và freeze. Stage B/C cần evidence gate tương ứng.

        Không chạy hai runtime đồng thời trên cùng shard/recovery run. Mọi pha trong
        Notebook 01 dùng CPU; không cần GPU.
        """),
    ]
