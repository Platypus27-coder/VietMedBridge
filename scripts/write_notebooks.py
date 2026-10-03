"""Regenerate clean Colab notebooks from versioned cell sources."""

from pathlib import Path
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = '''
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

if not (3, 11) <= sys.version_info[:2] < (3, 14):
    raise RuntimeError(
        "Dùng Python 3.11–3.13. Trong Runtime > Change runtime type, "
        "chọn phiên bản runtime có Python trong khoảng này."
    )
if importlib.util.find_spec("google.colab") is None:
    raise RuntimeError("Notebook này được thiết kế cho Google Colab.")

from google.colab import drive
# Chạy lại cell này sau MỖI runtime mới/restart: package chỉ được cài trong phiên Colab.
drive.mount("/content/drive")

# Giữ cùng DATA_ROOT trong cả bốn notebook.
DATA_ROOT = Path("/content/drive/MyDrive/VietMedBridge/data")
WORK_DIR = Path("/content/vmb_work")  # temp files/spill ở ổ local của Colab
DATA_ROOT.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
CODE_REVISION = None  # Có thể đặt full commit SHA; mặc định dùng code_lock đã lưu.
EXPECTED_PIPELINE_API = 2
CHECKOUT = Path("/content/VietMedBridge")
lock_path = DATA_ROOT / "code_lock.json"
lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
if lock and lock.get("repo_url") != REPO_URL:
    raise ValueError("DATA_ROOT đang khóa vào một repo khác.")
if lock and lock.get("pipeline_api") != EXPECTED_PIPELINE_API and not CODE_REVISION:
    raise RuntimeError(
        "DATA_ROOT đang dùng code cũ. Để nâng lên data-v2, đặt CODE_REVISION='main' "
        "ở cell này rồi chạy lại; dùng tên crawl/build run mới. Raw cũ vẫn được giữ."
    )
reference = CODE_REVISION or lock.get("git_commit") or "main"
if not (CHECKOUT / ".git").exists():
    subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CHECKOUT)], check=True)
remote = subprocess.check_output(
    ["git", "-C", str(CHECKOUT), "remote", "get-url", "origin"], text=True
).strip()
if remote != REPO_URL:
    raise ValueError("Checkout hiện tại không thuộc repo VietMedBridge.")
subprocess.run(["git", "-C", str(CHECKOUT), "fetch", "--depth", "1", "origin", reference], check=True)
CODE_COMMIT = subprocess.check_output(
    ["git", "-C", str(CHECKOUT), "rev-parse", "FETCH_HEAD"], text=True
).strip()
if "vietmedbridge" in sys.modules and globals().get("_VMB_IMPORTED_COMMIT") != CODE_COMMIT:
    raise RuntimeError("Mã nguồn đổi trong runtime đã import package. Restart session rồi chạy lại notebook.")
subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "--detach", "FETCH_HEAD"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".[notebook]"],
    cwd=CHECKOUT,
    check=True,
)
PACKAGE_SOURCE = CHECKOUT / "src"
if not (PACKAGE_SOURCE / "vietmedbridge" / "__init__.py").is_file():
    raise RuntimeError(
        f"Repo checkout thiếu package source: {PACKAGE_SOURCE}; commit={CODE_COMMIT}. "
        "Kiểm tra lại git fetch/checkout ở output của cell Bootstrap."
    )
sys.path.insert(0, str(PACKAGE_SOURCE))
importlib.invalidate_caches()
from vietmedbridge.artifacts import atomic_json, runtime_versions
from vietmedbridge import PIPELINE_API_VERSION
if PIPELINE_API_VERSION != EXPECTED_PIPELINE_API:
    raise RuntimeError("Notebook và package API không khớp; chọn commit data-v2 và restart session.")
_VMB_IMPORTED_COMMIT = CODE_COMMIT
if not lock or CODE_REVISION:
    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, "pipeline_api": PIPELINE_API_VERSION})
PIPELINE_CONFIG = json.loads((CHECKOUT / "configs/data_pipeline.json").read_text())
atomic_json(DATA_ROOT / "runtime.json", {
    "git_commit": CODE_COMMIT, "python": sys.version, "packages": runtime_versions()
})
print("Code commit:", CODE_COMMIT)
import vietmedbridge
print("Package source:", Path(vietmedbridge.__file__).resolve())
print("Dữ liệu/checkpoint:", DATA_ROOT)
'''

# Recovery experiments use new code without changing the pinned stage-a-v2 code.
RECOVERY_BOOTSTRAP = BOOTSTRAP.replace("code_lock.json", "recovery_code_lock.json")
GUARDED_RECOVERY_BOOTSTRAP = BOOTSTRAP.replace(
    "code_lock.json", "guarded_recovery_code_lock.json"
)


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


def save(name, cells):
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:12]
    display_name = Path(name).name
    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {
            "colab": {"name": display_name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }
    path = ROOT / "notebooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT / "notebooks"))


def main():
    save("00_colab_dataset_audit.ipynb", [
        md("""
        # VietMedBridge — 00: Tải và kiểm tra ViBioMIR

        Chạy trên Google Colab với runtime CPU. Notebook tải hai file Parquet của một
        revision cố định, kiểm tra ID/schema và thống kê domain bằng DuckDB có disk spill.
        Dữ liệu cùng checkpoint được lưu vào Google Drive; cần cấp quyền mount Drive ở cell đầu.

        Dataset có hai cấu hình query và corpus. Split mang tên train là cách đóng gói
        trên Hub; query hiện chỉ chứa id/query, chưa có nhãn reference để tính F2.
        Giữ nguyên ID chính thức, kể cả khi ID không liên tục hoặc URL trùng nhau.
        """),
        code(BOOTSTRAP),
        md("## 1. Tải snapshot gắn revision và kiểm tra SHA-256"),
        code("""
        from vietmedbridge.dataset import DATASET_REVISION, snapshot_dataset, audit_dataset

        # Revision đã kiểm tra của AIGuruTinix/ViBioMIR. Nếu Hub xóa revision này,
        # package phân giải main, cảnh báo và vẫn khóa SHA thực tế trong manifest.
        SNAPSHOT = snapshot_dataset(DATA_ROOT, revision=DATASET_REVISION)
        print("Dataset revision:", SNAPSHOT["revision"])
        for name, entry in SNAPSHOT["files"].items():
            print(name, entry.get("rows"), entry["bytes"], entry["sha256"])
        """),
        md("""
        ## 2. Audit trực tiếp file Parquet

        Không coi số dòng trong dataset card là số dòng đã xác minh. Báo cáo ghi số
        dòng, số ID duy nhất, ID min/max, URL trùng, domain và độ dài query thực tế.
        Sample chọn vài ID đầu của các domain lớn để thử crawler, không dùng làm tập đánh giá.
        """),
        code("""
        REPORT = audit_dataset(DATA_ROOT, work_dir=WORK_DIR, sample_domains=30, per_domain=3)
        print(json.dumps({key: REPORT[key] for key in ("query", "corpus", "sample")}, ensure_ascii=False, indent=2))
        import pandas as pd
        display(pd.DataFrame(REPORT["top_domains"]))
        """),
        md("""
        ## Stage A: inventory, mẫu phân tầng và golden regression

        Mẫu ~1.000 URL cân bằng nhóm domain, định dạng và language hint từ URL.
        Đây là mẫu feasibility; tỷ lệ không trọng số chưa phải dự báo toàn corpus.
        **Kiểm chứng nhanh:** output của cell Bootstrap phải hiện `Code commit` và
        `Package source` dưới `/content/VietMedBridge/src/vietmedbridge`. Nếu setup
        dừng ở lệnh pip/git, xử lý lỗi hiển thị tại đó trước khi chạy cell này.
        Khi mở runtime mới hoặc vừa restart, chạy lại cell **Bootstrap** đầu notebook
        (mount Drive, git clone/fetch và pip install) trước khi chạy cell này.
        Golden suite dùng 11 fixture synthetic offline để kiểm tra trước pilot. Team cần
        gán expected assertions cho 100–500 nguồn thật và replay ở notebook 03 trước Stage B1.
        Đây là bước bootstrap golden thực tế từ nguồn đã crawl, chưa hoàn tất golden set của plan.
        """),
        code("""
        import importlib.util
        if importlib.util.find_spec("vietmedbridge") is None:
            raise RuntimeError(
                "Chưa setup package trong runtime này. Chạy cell Bootstrap đầu notebook "
                "để git clone/cập nhật VietMedBridge và cài package, rồi chạy lại cell này."
            )
        from vietmedbridge.inventory import create_stage_a_sample
        from vietmedbridge.golden import run_golden_suite
        from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer
        from vietmedbridge.artifacts import read_json, atomic_json

        STAGE_A = create_stage_a_sample(DATA_ROOT, work_dir=WORK_DIR, **PIPELINE_CONFIG["stage_a"])
        tokenizer_lock_path = DATA_ROOT / "tokenizer_lock.json"
        tokenizer_lock = read_json(tokenizer_lock_path) if tokenizer_lock_path.exists() else {}
        TOKENIZER, TOKENIZER_SPEC = load_bge_tokenizer(tokenizer_lock.get("revision"))
        atomic_json(tokenizer_lock_path, TOKENIZER_SPEC)
        GOLDEN = run_golden_suite(CHECKOUT / "tests/golden/cases.json", TOKENIZER, TOKENIZER_SPEC,
                                  ChunkConfig(**PIPELINE_CONFIG["chunking"]))
        atomic_json(DATA_ROOT / "reports/golden_latest.json", GOLDEN)
        print("Stage A rows:", STAGE_A["rows"], "| golden:", GOLDEN["passed"])
        display(pd.DataFrame(GOLDEN["cases"]))
        if not GOLDEN["passed"]:
            raise RuntimeError("Golden regression fail; chưa được crawl/scale.")
        """),
        md("""
        ## 3. Cách dùng load_dataset với đúng cấu hình

        Query nhỏ được tải bình thường; corpus dùng streaming và chỉ xem ba dòng.
        Corpus này là danh sách liên kết, chưa phải nội dung văn bản đã crawl.
        """),
        code("""
        from datasets import load_dataset
        from itertools import islice

        queries = load_dataset(
            "AIGuruTinix/ViBioMIR", "query", split="train",
            revision=SNAPSHOT["revision"], cache_dir="/content/hf_cache/datasets",
        )
        corpus_preview = load_dataset(
            "AIGuruTinix/ViBioMIR", "corpus", split="train",
            revision=SNAPSHOT["revision"], streaming=True,
        )
        print("Queries:", len(queries), queries.features)
        display(pd.DataFrame(islice(corpus_preview, 3)))
        """),
        md("""
        ## Kết quả để dùng ở notebook 01

        Google Drive/VietMedBridge/data/raw chứa snapshot nguồn có hash.
        reports/dataset_audit.json chứa kết quả audit; reports/domains.parquet chứa
        thống kê domain và reports/sample_links.parquet chứa mẫu thử crawler.

        Lần đầu code_lock.json lưu commit GitHub đang chạy. Các notebook sau dùng lại
        commit này. Khi chủ động nâng code, đặt CODE_REVISION và dùng tên crawl/build run mới.
        Nguồn: [ViBioMIR](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR).
        """),
    ])
    from unified_notebook_cells import notebook_01_cells
    save("01_colab_crawl_sources.ipynb", notebook_01_cells(BOOTSTRAP, md, code))
    save("experiments/stage-a-1000/01b_colab_recover_failed_urls.ipynb", [
        md("""
        # VietMedBridge — 01b: Phục hồi và phân loại URL lỗi Stage A

        Chạy sau notebook 01, trước notebook 02. Notebook này dùng đúng package
        đang khóa trong `code_lock.json` để giữ nguyên checkpoint `stage-a-v2`.
        Nó kiểm từng official ID/URL trong mẫu, retry một lượt các ID lỗi, rồi
        lưu danh sách chưa tải được và quy mô domain tương ứng trên toàn corpus.

        Retry vẫn tuân theo robots.txt. HTTP 403, robots_blocked và HTTP 404 cần
        tuyến truy cập được phép hoặc xác minh nguồn; số URL có outcome không phải
        số trang đã tải thành công. Chạy trên CPU.
        """),
        code(BOOTSTRAP),
        md("## 1. Kiểm checkpoint và đối chiếu mọi ID/URL trong mẫu"),
        code("""
        from collections import Counter
        from urllib.parse import urlsplit
        import pandas as pd
        import pyarrow.parquet as pq
        from vietmedbridge.artifacts import (
            atomic_json, code_fingerprint, read_json, sha256_file, utc_now, verify_file,
        )
        from vietmedbridge.crawl import (
            CrawlConfig, completed_parts, crawl_links, iter_raw_records, range_complete,
        )
        from vietmedbridge.dataset import load_snapshot

        RUN_NAME = "stage-a-v2"
        RUN_DIR = DATA_ROOT / "crawl" / RUN_NAME
        REPORT_DIR = DATA_ROOT / "reports" / "crawl_recovery" / RUN_NAME
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        RUN_INFO = read_json(RUN_DIR / "run.json")
        SNAPSHOT = load_snapshot(DATA_ROOT)
        PILOT = read_json(DATA_ROOT / "reports/inventory/stage_a.json")
        if PILOT["corpus_sha256"] != SNAPSHOT["files"]["links_corpus.parquet"]["sha256"]:
            raise ValueError("Stage A inventory không thuộc corpus snapshot hiện tại.")
        INPUT_LINKS = DATA_ROOT / PILOT["files"]["stage_a_links"]["path"]
        verify_file(INPUT_LINKS, PILOT["files"]["stage_a_links"]["sha256"])
        if sha256_file(INPUT_LINKS) != RUN_INFO["links_sha256"]:
            raise ValueError("Checkpoint và Stage A input không cùng file.")
        if RUN_INFO["origin_corpus_sha256"] != SNAPSHOT["files"]["links_corpus.parquet"]["sha256"]:
            raise ValueError("Checkpoint không thuộc corpus snapshot hiện tại.")
        if RUN_INFO["code_sha256"] != code_fingerprint():
            raise RuntimeError(
                "Checkpoint dùng code khác runtime hiện tại. Giữ code_lock của run gốc; "
                "không nâng CODE_REVISION trước khi phục hồi Stage A."
            )

        input_table = pq.read_table(INPUT_LINKS, columns=["id", "url"])
        EXPECTED = {int(row["id"]): row["url"] for row in input_table.to_pylist()}
        if len(EXPECTED) != input_table.num_rows:
            raise ValueError("Stage A input có ID trùng.")
        if RUN_INFO["requested_range"] != {"start": 0, "stop": input_table.num_rows}:
            raise ValueError("Run không bao phủ đúng toàn bộ Stage A input.")

        ERROR_ACTION = {
            "ConnectTimeout": "retry_later",
            "ReadTimeout": "retry_later",
            "robots_unavailable": "check_robots_then_retry",
            "robots_blocked": "authorized_api_or_permission",
            "robots_401": "review_source_access",
            "robots_403": "review_source_access",
            "robots_4xx_other": "review_source_access",
            "robots_429": "respect_retry_after_then_probe",
            "robots_5xx": "retry_later",
            "robots_connect_timeout": "retry_later",
            "robots_read_timeout": "retry_later",
            "robots_dns_error": "retry_later",
            "robots_ssl_error": "review_tls_then_retry",
            "robots_connection_error": "retry_later",
            "robots_redirect_limit": "review_robots_redirects",
            "robots_too_large": "review_robots_file",
            "http_403": "review_source_access",
            "http_404": "verify_moved_or_removed_source",
        }
        FAILURE_COLUMNS = [
            "doc_id", "url", "domain", "error", "http_status", "attempts", "next_action",
        ]

        def failure_frame():
            parts = completed_parts(RUN_DIR)
            if not range_complete(RUN_INFO, parts):
                raise ValueError("Crawl range còn thiếu shard; resume notebook 01 trước.")
            seen = set()
            failures = []
            for part in parts:
                for raw in iter_raw_records(RUN_DIR / part["raw_file"]):
                    doc_id = int(raw["doc_id"])
                    if doc_id in seen or EXPECTED.get(doc_id) != raw["url"]:
                        raise ValueError(f"ID/URL trùng hoặc lệch official input: {doc_id}")
                    seen.add(doc_id)
                    if raw["status"] != "ok":
                        error = raw.get("error", "unknown_error")
                        failures.append({
                            "doc_id": doc_id, "url": raw["url"],
                            "domain": (urlsplit(raw["url"]).hostname or "").lower(),
                            "error": error, "http_status": raw.get("http_status"),
                            "attempts": raw.get("attempts"),
                            "next_action": ERROR_ACTION.get(error, "manual_review"),
                        })
            if seen != set(EXPECTED):
                raise ValueError("Crawl ledger thiếu hoặc dư official ID.")
            return pd.DataFrame(failures, columns=FAILURE_COLUMNS).sort_values(
                ["error", "domain", "doc_id"], ignore_index=True,
            )

        BEFORE = failure_frame()
        BEFORE_PATH = REPORT_DIR / "failures_before_retry.csv"
        BEFORE.to_csv(BEFORE_PATH, index=False, encoding="utf-8-sig")
        print("Official Stage A IDs:", len(EXPECTED), "| successful:", len(EXPECTED) - len(BEFORE),
              "| unresolved:", len(BEFORE))
        display(BEFORE.groupby(["error", "domain"], dropna=False).size().rename("rows").reset_index())
        """),
        md("""
        ## 2. Retry đúng run hiện tại một lượt

        Package hiện tại retry mọi ID lỗi và giữ lại bản tải thành công. Một lượt
        giúp phát hiện nguồn vừa phục hồi, nhưng sẽ không tự giải quyết robots
        blocked hay quyền truy cập HTTP 403. Marker trên Drive tránh chạy lại
        lượt retry khi mở notebook lần nữa. Nếu Colab ngắt giữa chừng, chạy lại
        cell này; shard đã commit vẫn giữ nguyên và có thể được retry thêm một lần.
        """),
        code("""
        RETRY_MARKER = REPORT_DIR / "retry_once.json"
        if RETRY_MARKER.exists():
            if read_json(RETRY_MARKER)["run_signature"] != RUN_INFO["signature"]:
                raise ValueError("Retry marker không thuộc run hiện tại.")
            print("Đã có lượt retry hoàn tất:", RETRY_MARKER)
        elif BEFORE.empty:
            atomic_json(RETRY_MARKER, {"run_name": RUN_NAME,
                                       "run_signature": RUN_INFO["signature"],
                                       "before_failed": 0,
                                       "note": "No failed IDs to retry", "completed_at": utc_now()})
            print("Không có ID lỗi để retry.")
        else:
            RETRY_SUMMARY = await crawl_links(
                INPUT_LINKS, DATA_ROOT / "crawl", run_name=RUN_NAME,
                config=CrawlConfig(**RUN_INFO["config"]),
                start=RUN_INFO["row_range"]["start"], stop=RUN_INFO["row_range"]["stop"],
                worker_index=0, worker_count=1, retry_failed=True,
                max_shards=None, work_dir=WORK_DIR,
                origin_corpus_sha256=RUN_INFO["origin_corpus_sha256"],
            )
            if not RETRY_SUMMARY["range_complete"] or RETRY_SUMMARY["recorded_documents"] != len(EXPECTED):
                raise ValueError("Retry chưa ghi đủ outcome cho Stage A input.")
            atomic_json(RETRY_MARKER, {
                "run_name": RUN_NAME, "run_signature": RUN_INFO["signature"],
                "before_failed": len(BEFORE), "call": RETRY_SUMMARY["call"],
                "completed_at": utc_now(),
            })
            print("Sau retry:", RETRY_SUMMARY["statuses"])
        """),
        md("""
        ## 3. Xuất ledger còn lỗi và quy mô domain toàn corpus

        File `failures_after_retry.csv` giữ official ID và URL gốc cho mọi nguồn
        chưa lấy được nội dung. `affected_domains_in_corpus.csv` đếm chính xác
        số URL của các domain này trong inventory toàn corpus; số lỗi trên mẫu
        không được dùng làm tỷ lệ dự báo không trọng số cho toàn corpus.
        """),
        code("""
        AFTER = failure_frame()
        AFTER_PATH = REPORT_DIR / "failures_after_retry.csv"
        AFTER.to_csv(AFTER_PATH, index=False, encoding="utf-8-sig")
        INVENTORY = DATA_ROOT / PILOT["files"]["url_inventory"]["path"]
        verify_file(INVENTORY, PILOT["files"]["url_inventory"]["sha256"])
        affected_domains = sorted(AFTER["domain"].dropna().unique().tolist())
        import duckdb
        with duckdb.connect() as con:
            con.execute("SET memory_limit = '512MB'")
            con.execute("SET temp_directory = ?", [str(WORK_DIR)])
            con.execute("SET threads = 2")
            if affected_domains:
                domains = con.execute(
                    '''SELECT domain, count(*) AS official_urls
                       FROM read_parquet(?)
                       WHERE domain IN (SELECT unnest(?))
                       GROUP BY domain ORDER BY official_urls DESC, domain''',
                    [str(INVENTORY), affected_domains],
                ).df()
            else:
                domains = pd.DataFrame(columns=["domain", "official_urls"])
        domains_path = REPORT_DIR / "affected_domains_in_corpus.csv"
        domains.to_csv(domains_path, index=False, encoding="utf-8-sig")
        counts = {str(k): int(v) for k, v in Counter(AFTER["error"]).items()}
        atomic_json(REPORT_DIR / "triage.json", {
            "run_name": RUN_NAME, "run_signature": RUN_INFO["signature"],
            "official_stage_a_ids": len(EXPECTED), "successful": len(EXPECTED) - len(AFTER),
            "unresolved": len(AFTER), "errors": counts,
            "failures_csv": str(AFTER_PATH.relative_to(DATA_ROOT)),
            "failures_sha256": sha256_file(AFTER_PATH),
            "affected_domains_csv": str(domains_path.relative_to(DATA_ROOT)),
            "affected_domains_sha256": sha256_file(domains_path),
            "created_at": utc_now(),
        })
        print("Successful:", len(EXPECTED) - len(AFTER), "| unresolved:", len(AFTER))
        print("Retry result CSV:", AFTER_PATH)
        print("Affected domain counts:", domains_path)
        display(AFTER.groupby(["error", "domain", "next_action"], dropna=False).size()
                .rename("rows").reset_index())
        display(domains)
        """),
        md("""
        Sau cell cuối, chạy notebook 02 trên cùng `stage-a-v2` và cùng code lock
        để đánh giá chất lượng phần đã tải. Những ID chưa lấy được nội dung đi vào
        failures/ledger của build, không biến mất. Trước khi mở rộng corpus, xử lý
        các domain bị chặn qua API/bulk được phép hoặc quyền truy cập từ nguồn;
        không coi `range_complete` là 100% tải được nội dung.
        """),
    ])
    save("experiments/stage-a-1000/01c_colab_robots_and_scrapling_pilot.ipynb", [
        md("""
        # VietMedBridge — 01c: Chẩn đoán robots và thử Scrapling có kiểm soát

        Chạy trong **runtime Colab CPU mới** sau 01b. Notebook này đọc đúng
        `failures_after_retry.csv`, kiểm official ID/URL, phân loại lại robots,
        rồi thử HTTP kiểu browser chỉ khi robots cho phép. Chromium tạm dừng
        vì route guard của Scrapling 0.4.15 chưa fail-closed. Mỗi ID/method có
        marker riêng trên Drive để resume.

        Mã mới được khóa trong `recovery_code_lock.json`; không đổi
        `code_lock.json` hoặc checkpoint `stage-a-v2`. Kết quả là experiment
        cần human review, chưa tự động nhập vào corpus. GPU không cần.
        """),
        code(RECOVERY_BOOTSTRAP),
        md("## 1. Khóa input và xác minh official ID/URL"),
        code("""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vietmedbridge.artifacts import (
            atomic_json, code_fingerprint, digest_json, read_json, sha256_file, utc_now,
        )
        from vietmedbridge.dataset import parquet_path, validate_link_subset

        REPORT_DIR = DATA_ROOT / "reports/crawl_recovery/stage-a-v2"
        TRIAGE = read_json(REPORT_DIR / "triage.json")
        FAILURES_PATH = DATA_ROOT / TRIAGE["failures_csv"]
        if sha256_file(FAILURES_PATH) != TRIAGE["failures_sha256"]:
            raise ValueError("Báo cáo sau retry đã đổi; chạy lại notebook 01b hoặc dùng run mới.")
        RUN_INFO = read_json(DATA_ROOT / "crawl/stage-a-v2/run.json")
        if TRIAGE["run_signature"] != RUN_INFO["signature"]:
            raise ValueError("Báo cáo và crawl run không cùng signature.")
        FAILURES = pd.read_csv(FAILURES_PATH, dtype={"doc_id": "int64", "url": "string", "error": "string"})
        if len(FAILURES) != TRIAGE["unresolved"] or FAILURES["doc_id"].duplicated().any():
            raise ValueError("Số ID lỗi hoặc uniqueness không khớp triage.json.")
        subset_path = WORK_DIR / "recovery_official_subset.parquet"
        pq.write_table(pa.table({
            "id": pa.array(FAILURES["doc_id"].tolist(), type=pa.int64()),
            "url": pa.array(FAILURES["url"].tolist(), type=pa.string()),
        }), subset_path)
        validate_link_subset(subset_path, parquet_path(DATA_ROOT, "links_corpus.parquet"), work_dir=WORK_DIR)
        EXPERIMENT_KEY = digest_json({
            "source_run": RUN_INFO["signature"], "failures_sha256": TRIAGE["failures_sha256"],
            "code_sha256": code_fingerprint(), "scrapling_version": "0.4.15",
            "policy": "robots-review-v1", "max_bytes": 16 * 1024 * 1024,
        })[:16]
        EXPERIMENT_DIR = REPORT_DIR / "experiments" / EXPERIMENT_KEY
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = EXPERIMENT_DIR / "experiment.json"
        experiment = {
            "key": EXPERIMENT_KEY, "source_run_signature": RUN_INFO["signature"],
            "failures_sha256": TRIAGE["failures_sha256"], "code_sha256": code_fingerprint(),
            "code_commit": CODE_COMMIT, "scrapling_version": "0.4.15",
            "unresolved_input_ids": len(FAILURES),
        }
        if manifest_path.exists() and read_json(manifest_path) != experiment:
            raise ValueError("Experiment marker không khớp cấu hình hiện tại.")
        if not manifest_path.exists():
            atomic_json(manifest_path, experiment)
        print("Input unresolved:", len(FAILURES), "| experiment:", EXPERIMENT_DIR)
        display(FAILURES.groupby("error", dropna=False).size().rename("ids").reset_index())
        affected = REPORT_DIR / "affected_domains_in_corpus.csv"
        if affected.exists():
            if sha256_file(affected) != TRIAGE["affected_domains_sha256"]:
                raise ValueError("Domain report đã đổi so với triage.json.")
            display(pd.read_csv(affected))
        """),
        md("""
        ## 2. Phân loại lại robots cho tất cả URL còn lỗi

        Phân loại là probe `/robots.txt`, chưa tải trang đích. Kết quả là quan
        sát tại thời điểm hiện tại; `robots_blocked` lịch sử không được tự động
        đưa sang bước recovery dù probe mới khác trước.
        """),
        code("""
        from vietmedbridge.crawl import CrawlConfig, Fetcher

        ROBOTS_DIR = EXPERIMENT_DIR / "robots"
        ROBOTS_DIR.mkdir(exist_ok=True)

        async def probe_robots():
            fetcher = Fetcher(CrawlConfig(concurrency=1, per_host_delay=1, attempts=1))
            try:
                for row in FAILURES.itertuples(index=False):
                    marker = ROBOTS_DIR / f"{int(row.doc_id)}.json"
                    if marker.exists():
                        saved = read_json(marker)
                        if saved["url"] != row.url:
                            raise ValueError(f"Checkpoint URL không khớp: {row.doc_id}")
                        continue
                    decision = await fetcher.robots_decision(row.url)
                    atomic_json(marker, {
                        "doc_id": int(row.doc_id), "url": row.url,
                        "original_error": row.error, **decision.record(),
                    })
            finally:
                await fetcher.client.aclose()

        await probe_robots()
        ROBOTS = pd.DataFrame([read_json(ROBOTS_DIR / f"{int(i)}.json")
                               for i in FAILURES["doc_id"]])
        if len(ROBOTS) != len(FAILURES):
            raise ValueError("Thiếu robots probe cho một official ID.")
        display(ROBOTS.groupby(["original_error", "robots_state", "robots_allowed"],
                               dropna=False).size().rename("ids").reset_index())
        """),
        md("""
        ## 3. Thử Scrapling HTTP trên nguồn được phép

        Pin Scrapling 0.4.15; không dùng Spider robots fail-open của thư viện.
        Bộ resolver VietMedBridge ở cell trước là cổng bắt buộc. Chỉ thử các URL
        403/timeout hoặc `robots_unavailable` mà probe mới xác nhận được phép.
        HTTP 200 chỉ tạo **article candidate**, phải kiểm nội dung và human review.
        """),
        code("""
        import importlib.metadata
        import subprocess
        import time
        from vietmedbridge.artifacts import publish_file
        from vietmedbridge.text import EXPECTED_PARSE_ERRORS, extract_source
        from vietmedbridge.quality import document_quality

        if importlib.metadata.packages_distributions().get("scrapling") is None or (
            importlib.metadata.version("scrapling") != "0.4.15"
        ):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "scrapling[fetchers]==0.4.15"], check=True)
        from scrapling.fetchers import Fetcher as ScraplingFetcher

        HTTP_DIR = EXPERIMENT_DIR / "scrapling_http"
        ASSET_DIR = EXPERIMENT_DIR / "assets"
        HTTP_DIR.mkdir(exist_ok=True)
        ASSET_DIR.mkdir(exist_ok=True)
        MAX_HTTP_CASES = None  # None = mọi 403/timeout/robots_unavailable đã được phép
        MAX_BYTES = 16 * 1024 * 1024
        USER_AGENT = "VietMedBridge/0.2 (+https://github.com/Platypus27-coder/VietMedBridge)"

        def assess_and_archive(response, *, doc_id, method, capture_kind):
            body = response.body
            result = {
                "http_status": int(response.status), "final_url": str(response.url),
                "content_type": response.headers.get("content-type", ""),
                "body_bytes": len(body), "capture_kind": capture_kind,
                "article_candidate": False,
            }
            if len(body) > MAX_BYTES:
                return {**result, "outcome": "body_too_large"}
            local_asset = WORK_DIR / f"recovery-{doc_id}-{method}.bin"
            local_asset.write_bytes(body)
            target = ASSET_DIR / local_asset.name
            try:
                result["body_sha256"] = publish_file(local_asset, target)
            finally:
                local_asset.unlink(missing_ok=True)
            result["asset_path"] = str(target.relative_to(DATA_ROOT))
            if response.status != 200:
                return {**result, "outcome": f"http_{response.status}"}
            try:
                extracted = extract_source(body, result["content_type"])
            except EXPECTED_PARSE_ERRORS as exc:
                return {**result, "outcome": "extract_rejected", "extract_error": str(exc)[:300]}
            quality = document_quality(extracted["source_text"], title=extracted["title"],
                                       raw_bytes=len(body), content_type=result["content_type"])
            result.update({
                "source_text_sha256": extracted["source_text_sha256"],
                "source_chars": len(extracted["source_text"]),
                "source_preview": extracted["source_text"][:500],
                "quality_tier": quality["quality_tier"],
                "quality_flags": quality["quality_flags"],
                "article_candidate": len(extracted["source_text"]) >= 80,
                "outcome": "article_candidate" if len(extracted["source_text"]) >= 80 else "short_text_review",
            })
            return result

        eligible_errors = {"http_403", "ReadTimeout", "ConnectTimeout", "robots_unavailable"}
        candidates = FAILURES.merge(ROBOTS[["doc_id", "robots_state", "robots_allowed"]], on="doc_id")
        candidates = candidates[candidates["error"].isin(eligible_errors) & candidates["robots_allowed"]]
        candidates = candidates.sort_values(["error", "doc_id"])
        if MAX_HTTP_CASES is not None:
            candidates = candidates.head(MAX_HTTP_CASES)
        print("Allowed recovery candidates:", len(candidates))

        for row in candidates.itertuples(index=False):
            marker = HTTP_DIR / f"{int(row.doc_id)}.json"
            if marker.exists():
                if read_json(marker)["url"] != row.url:
                    raise ValueError(f"Recovery checkpoint URL không khớp: {row.doc_id}")
                continue
            started = time.monotonic()
            record = {"doc_id": int(row.doc_id), "url": row.url,
                      "original_error": row.error, "robots_state": row.robots_state,
                      "method": "scrapling_http", "started_at": utc_now()}
            try:
                response = ScraplingFetcher.get(
                    row.url, timeout=45, retries=0, stealthy_headers=True,
                    follow_redirects=False, headers={"User-Agent": USER_AGENT},
                )
                record.update(assess_and_archive(response, doc_id=row.doc_id,
                                                 method="http", capture_kind="http_entity"))
            except Exception as exc:
                record.update(outcome="fetch_error", error_type=type(exc).__name__,
                              error_message=str(exc)[:300])
            record.update(elapsed_ms=round((time.monotonic() - started) * 1000),
                          finished_at=utc_now())
            atomic_json(marker, record)
        print("HTTP experiment markers:", len(list(HTTP_DIR.glob("*.json"))))
        """),
        md("""
        ## 4. Chromium tạm dừng

        Scrapling 0.4.15 nuốt lỗi trong `page_setup`, nên route guard có thể
        không được cài mà browser vẫn tiếp tục điều hướng. Chưa bật Chromium
        cho recovery cho tới khi có adapter fail-closed được kiểm thử.
        """),
        code("""
        BROWSER_DIR = EXPERIMENT_DIR / "browser"
        BROWSER_DIR.mkdir(exist_ok=True)
        print("Browser recovery chưa bật; dùng notebook 01d cho HTTP retry có robots guard.")
        """),
        md("## 5. Xuất báo cáo để review, không sửa Stage A checkpoint"),
        code("""
        import json

        attempts = [read_json(path) for folder in (HTTP_DIR, BROWSER_DIR)
                    for path in sorted(folder.glob("*.json"))]
        result_path = EXPERIMENT_DIR / "attempts.csv"
        pd.DataFrame(attempts).to_csv(result_path, index=False, encoding="utf-8-sig")
        robots_path = EXPERIMENT_DIR / "robots.csv"
        ROBOTS.to_csv(robots_path, index=False, encoding="utf-8-sig")
        summary = {
            "input_unresolved": len(FAILURES),
            "robots_states": ROBOTS["robots_state"].value_counts().to_dict(),
            "robots_allowed": int(ROBOTS["robots_allowed"].sum()),
            "attempts": len(attempts),
            "article_candidates_needing_human_review": sum(bool(r.get("article_candidate")) for r in attempts),
            "attempts_csv": str(result_path.relative_to(DATA_ROOT)),
            "attempts_sha256": sha256_file(result_path),
            "robots_csv": str(robots_path.relative_to(DATA_ROOT)),
            "robots_sha256": sha256_file(robots_path),
            "created_at": utc_now(),
        }
        atomic_json(EXPERIMENT_DIR / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        preview_columns = ["doc_id", "original_error", "method", "outcome",
                           "article_candidate", "elapsed_ms"]
        display(pd.DataFrame(attempts).reindex(columns=preview_columns))
        print("Gửi cho tôi robots.csv và attempts.csv để audit. Chưa merge vào stage-a-v2.")
        """),
    ])

    save("experiments/stage-a-1000/01d_colab_guarded_recovery.ipynb", [
        md("""
        # VietMedBridge — 01d: Thử lại có robots guard và pacing

        Chạy trong **Colab CPU mới** sau 01c. Notebook này đọc experiment 01c đã
        khóa bằng SHA-256, chỉ thử lại 11 URL chưa tạo article candidate. Mỗi
        request, redirect và retry đều kiểm robots và áp dụng Crawl-delay hoặc
        Request-rate. Không bật browser, không vượt Disallow/robots lỗi, không
        tự nhập kết quả vào Stage A. Mỗi ID có marker trên Drive để resume.
        """),
        code(GUARDED_RECOVERY_BOOTSTRAP),
        md("## 1. Kiểm input 01c và lập danh sách 11 URL"),
        code("""
        import pandas as pd
        from vietmedbridge.artifacts import (
            atomic_json, code_fingerprint, digest_json, read_json, sha256_file, utc_now,
        )

        SOURCE_KEY = "6a5793cdc03be0cb"
        EXPECTED_ATTEMPTS_SHA256 = "09f3edd03c459b8791bb819805656e01626a837eec898c2a1d46130af5de5244"
        EXPECTED_ROBOTS_SHA256 = "637738e9641c7b8cfa668b9f30ea547e9de664fe8a9204458e6836bd023b3a27"
        REPORT_DIR = DATA_ROOT / "reports/crawl_recovery/stage-a-v2"
        SOURCE_DIR = REPORT_DIR / "experiments" / SOURCE_KEY
        SOURCE = read_json(SOURCE_DIR / "summary.json")
        SOURCE_EXPERIMENT = read_json(SOURCE_DIR / "experiment.json")
        TRIAGE = read_json(REPORT_DIR / "triage.json")
        if SOURCE_EXPERIMENT["source_run_signature"] != TRIAGE["run_signature"]:
            raise ValueError("01c và 01b không cùng Stage A run.")
        if SOURCE["attempts_sha256"] != EXPECTED_ATTEMPTS_SHA256 or SOURCE["robots_sha256"] != EXPECTED_ROBOTS_SHA256:
            raise ValueError("01c summary khác experiment đã báo; xác minh key và hash trước khi chạy.")
        ATTEMPTS_PATH = DATA_ROOT / SOURCE["attempts_csv"]
        ROBOTS_PATH = DATA_ROOT / SOURCE["robots_csv"]
        if sha256_file(ATTEMPTS_PATH) != EXPECTED_ATTEMPTS_SHA256 or sha256_file(ROBOTS_PATH) != EXPECTED_ROBOTS_SHA256:
            raise ValueError("CSV 01c đã đổi hoặc tải thiếu.")
        ATTEMPTS = pd.read_csv(ATTEMPTS_PATH, dtype={"doc_id": "int64", "url": "string"})
        ROBOTS = pd.read_csv(ROBOTS_PATH, dtype={"doc_id": "int64", "url": "string"})
        if len(ROBOTS) != SOURCE["input_unresolved"] or ROBOTS["doc_id"].duplicated().any():
            raise ValueError("robots.csv không đủ 83 official IDs hoặc có ID trùng.")
        if len(ATTEMPTS) != SOURCE["attempts"] or ATTEMPTS[["doc_id", "method"]].duplicated().any():
            raise ValueError("attempts.csv không đủ 15 thử nghiệm hoặc có ID/method trùng.")
        ATTEMPTS["article_candidate"] = ATTEMPTS["article_candidate"].astype(str).str.lower().eq("true")
        ROBOTS["robots_allowed"] = ROBOTS["robots_allowed"].astype(str).str.lower().eq("true")
        official = ROBOTS.set_index("doc_id")
        for row in ATTEMPTS.itertuples(index=False):
            if row.doc_id not in official.index or row.url != official.loc[row.doc_id, "url"]:
                raise ValueError(f"ID/URL trong attempts không khớp robots: {row.doc_id}")
            if not bool(official.loc[row.doc_id, "robots_allowed"]):
                raise ValueError(f"01c đã thử URL không được phép: {row.doc_id}")
        if int(ATTEMPTS["article_candidate"].sum()) != SOURCE["article_candidates_needing_human_review"]:
            raise ValueError("Số article candidate khác summary.json.")
        REMAINING = ATTEMPTS[~ATTEMPTS["article_candidate"]].copy().sort_values("doc_id")
        print("01c attempts:", len(ATTEMPTS), "| cần thử lại:", len(REMAINING))
        display(REMAINING.reindex(columns=[
            "doc_id", "url", "original_error", "outcome", "http_status", "error_type",
        ]).fillna(""))
        print("4 article candidates từ 01c vẫn cần human review; notebook này không fetch lại chúng.")
        """),
        md("## 2. Thử Scrapling HTTP với robots mới, pacing và redirect guard"),
        code("""
        import importlib.metadata
        import subprocess
        import time
        from urllib.parse import urlsplit
        from vietmedbridge.artifacts import publish_file
        from vietmedbridge.crawl import CrawlConfig, Fetcher
        from vietmedbridge.recovery import RecoveryConfig, guarded_http_fetch
        from vietmedbridge.text import EXPECTED_PARSE_ERRORS, extract_source
        from vietmedbridge.quality import document_quality

        if importlib.metadata.packages_distributions().get("scrapling") is None or (
            importlib.metadata.version("scrapling") != "0.4.15"
        ):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "scrapling[fetchers]==0.4.15"], check=True)
        from scrapling.fetchers import Fetcher as ScraplingFetcher

        FOLLOWUP_KEY = digest_json({
            "source_key": SOURCE_KEY, "attempts_sha256": EXPECTED_ATTEMPTS_SHA256,
            "code_sha256": code_fingerprint(), "policy": "guarded-http-v1",
            "scrapling_version": "0.4.15",
        })[:16]
        FOLLOWUP_DIR = REPORT_DIR / "experiments" / FOLLOWUP_KEY
        MARKERS_DIR = FOLLOWUP_DIR / "markers"
        ASSETS_DIR = FOLLOWUP_DIR / "assets"
        MARKERS_DIR.mkdir(parents=True, exist_ok=True)
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = FOLLOWUP_DIR / "experiment.json"
        manifest = {"source_key": SOURCE_KEY, "source_attempts_sha256": EXPECTED_ATTEMPTS_SHA256,
                    "code_commit": CODE_COMMIT, "code_sha256": code_fingerprint(),
                    "policy": "guarded-http-v1", "input_ids": len(REMAINING)}
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError("Follow-up checkpoint khác cấu hình hiện tại.")
        if not manifest_path.exists():
            atomic_json(manifest_path, manifest)

        USER_AGENT = "VietMedBridge/0.2 (+https://github.com/Platypus27-coder/VietMedBridge)"
        guard = Fetcher(CrawlConfig(concurrency=1, per_host_delay=3, attempts=1))
        settings = RecoveryConfig(attempts=2, max_redirects=5, max_body_bytes=16 * 1024 * 1024)
        held_hosts = set()
        try:
            for row in REMAINING.itertuples(index=False):
                marker = MARKERS_DIR / f"{int(row.doc_id)}.json"
                if marker.exists():
                    saved = read_json(marker)
                    if saved["url"] != row.url or saved["source_attempt_outcome"] != row.outcome:
                        raise ValueError(f"Marker khác URL/outcome gốc: {row.doc_id}")
                    if saved.get("outcome") in ("rate_limited_retry_later", "server_retry_later", "host_rate_limit_hold"):
                        held_hosts.add(urlsplit(saved.get("final_url", row.url)).hostname)
                    continue
                started = time.monotonic()
                record = {"doc_id": int(row.doc_id), "url": row.url,
                          "source_attempt_outcome": row.outcome,
                          "method": "guarded_scrapling_http", "started_at": utc_now(),
                          "article_candidate": False}
                if urlsplit(row.url).hostname in held_hosts:
                    record["outcome"] = "host_rate_limit_hold"
                else:
                    result, body = await guarded_http_fetch(
                        row.url, guard, ScraplingFetcher.get,
                        user_agent=USER_AGENT, config=settings, held_hosts=held_hosts,
                    )
                    record.update(result)
                    if body is not None:
                        local_asset = WORK_DIR / f"guarded-recovery-{int(row.doc_id)}.bin"
                        local_asset.write_bytes(body)
                        target = ASSETS_DIR / local_asset.name
                        try:
                            record["body_sha256"] = publish_file(local_asset, target)
                        finally:
                            local_asset.unlink(missing_ok=True)
                        record["asset_path"] = str(target.relative_to(DATA_ROOT))
                        try:
                            extracted = extract_source(body, record["content_type"])
                            quality = document_quality(
                                extracted["source_text"], title=extracted["title"],
                                raw_bytes=len(body), content_type=record["content_type"],
                            )
                            record.update(
                                source_text_sha256=extracted["source_text_sha256"],
                                source_chars=len(extracted["source_text"]),
                                source_preview=extracted["source_text"][:500],
                                quality_tier=quality["quality_tier"],
                                quality_flags=quality["quality_flags"],
                                article_candidate=len(extracted["source_text"]) >= 80,
                            )
                            record["outcome"] = ("article_candidate" if record["article_candidate"]
                                                 else "short_text_review")
                        except EXPECTED_PARSE_ERRORS as exc:
                            record.update(outcome="extract_rejected", extract_error=str(exc)[:300])
                record.update(elapsed_ms=round((time.monotonic() - started) * 1000),
                              finished_at=utc_now())
                atomic_json(marker, record)
                print(row.doc_id, record["outcome"])
        finally:
            await guard.client.aclose()
        print("Follow-up experiment:", FOLLOWUP_DIR)
        """),
        md("## 3. Xuất kết quả; không nhập vào Stage A"),
        code("""
        records = [read_json(MARKERS_DIR / f"{int(i)}.json") for i in REMAINING["doc_id"]]
        result_path = FOLLOWUP_DIR / "attempts.csv"
        pd.DataFrame(records).to_csv(result_path, index=False, encoding="utf-8-sig")
        summary = {"source_experiment": SOURCE_KEY, "input_non_candidates": len(REMAINING),
                   "outcomes": pd.Series([r["outcome"] for r in records]).value_counts().to_dict(),
                   "new_article_candidates_needing_human_review": sum(bool(r.get("article_candidate")) for r in records),
                   "attempts_csv": str(result_path.relative_to(DATA_ROOT)),
                   "attempts_sha256": sha256_file(result_path), "created_at": utc_now()}
        atomic_json(FOLLOWUP_DIR / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        display(pd.DataFrame(records).reindex(columns=[
            "doc_id", "source_attempt_outcome", "outcome", "http_status",
            "error_type", "article_candidate", "elapsed_ms",
        ]).fillna(""))
        print("Gửi attempts.csv và summary.json của experiment này để audit nội dung.")
        """),
    ])

    from browser_probe_cells import browser_cells
    save("experiments/stage-a-1000/01e_colab_browser_diagnosis.ipynb", browser_cells(BOOTSTRAP, md, code))
    from recovery_review_cells import review_cells
    save("experiments/stage-a-1000/01f_colab_review_recovered_articles.ipynb", review_cells(BOOTSTRAP, md, code))
    from remaining_recovery_cells import remaining_cells
    save("experiments/stage-a-1000/01g_colab_recover_remaining_stage_a.ipynb", remaining_cells(BOOTSTRAP, md, code))

    save("02_colab_extract_and_chunk.ipynb", [
        md("""
        # VietMedBridge — 02: Trích văn bản và chia đoạn có provenance

        Chạy sau notebook 01. Trích HTML/XML/PDF thành source_text, rồi tạo child
        và parent bằng tokenizer BGE-M3 fast. Chỉ tải tokenizer; runtime CPU đủ.
        Mỗi span được kiểm tra bằng lát cắt source_text[start_char:end_char].

        source_text là văn bản nguồn đã trích từ snapshot; offset tính bằng ký tự
        Unicode của chuỗi Python, không phải vị trí byte trong HTML/PDF.
        retrieval_text được chuẩn hóa riêng. Tham số 180/40/512 là cấu hình khởi đầu
        để thử nghiệm, chưa phải cấu hình đã tối ưu Chunk F2.
        """),
        code(BOOTSTRAP),
        md("## 1. Chọn crawl run và ghim revision tokenizer"),
        code("""
        from vietmedbridge.artifacts import read_json, atomic_json
        from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer
        from vietmedbridge.build import build_corpus
        from vietmedbridge.crawl import completed_parts

        CRAWL_RUN = "stage-a-v2"
        BUILD_RUN = "stage-a-data-v2"
        MAX_NEW_SHARDS = None
        TOKENIZER_REVISION = None
        tokenizer_lock_path = DATA_ROOT / "tokenizer_lock.json"
        tokenizer_lock = read_json(tokenizer_lock_path) if tokenizer_lock_path.exists() else {}
        requested_revision = TOKENIZER_REVISION or tokenizer_lock.get("revision")
        TOKENIZER, TOKENIZER_SPEC = load_bge_tokenizer(requested_revision)
        if not tokenizer_lock or TOKENIZER_REVISION:
            atomic_json(tokenizer_lock_path, TOKENIZER_SPEC)
        print("Tokenizer:", TOKENIZER_SPEC)
        print("Complete raw shards:", len(completed_parts(DATA_ROOT / "crawl" / CRAWL_RUN)))
        """),
        md("""
        ## 2. Tạo Parquet cho documents, children, parents và failures

        Parser HTML dùng trafilatura; XML xử lý abstract/body; PDF lấy text theo thứ
        tự trang. PDF scan không có text và trang challenge bị ghi lỗi để xử lý tiếp.
        Các ID chính thức có cùng nội dung vẫn được giữ riêng.
        Section được nhận diện khi heading nguồn khớp chính xác một dòng trong source_text;
        nếu thiếu thì dùng body section và quality flag. Child ưu tiên paragraph/sentence,
        parent giữ trong section khi có thể. OCR và parser chuyên biệt từng domain để sau audit.
        """),
        code("""
        from vietmedbridge.dataset import parquet_path
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        CHUNK_CONFIG = ChunkConfig(**PIPELINE_CONFIG["chunking"])
        BUILD = build_corpus(
            DATA_ROOT / "crawl" / CRAWL_RUN, DATA_ROOT / "processed",
            TOKENIZER, TOKENIZER_SPEC, run_name=BUILD_RUN, config=CHUNK_CONFIG,
            work_dir=WORK_DIR, max_shards=MAX_NEW_SHARDS, official_links=OFFICIAL_LINKS,
        )
        print(json.dumps({
            key: BUILD[key] for key in (
                "run_name", "snapshot_sha256", "counts", "selected_shards", "available_crawl_shards"
            )
        }, ensure_ascii=False, indent=2))
        """),
        md("## 3. Xem một tài liệu và kiểm tra span đã lưu"),
        code("""
        import pyarrow.parquet as pq
        from vietmedbridge.build import verify_spans

        BUILD_DIR = DATA_ROOT / "processed" / BUILD_RUN
        document_preview = None
        for part in BUILD["parts"]:
            documents = pq.ParquetFile(BUILD_DIR / part["files"]["documents"]["path"])
            if documents.metadata.num_rows:
                document_preview = next(documents.iter_batches(batch_size=1)).to_pylist()[0]
                break
        if document_preview is not None:
            doc_id = document_preview["doc_id"]
            children = pq.read_table(
                BUILD_DIR / part["files"]["children"]["path"],
                filters=[("doc_id", "=", doc_id)],
            ).to_pylist()
            parents = pq.read_table(
                BUILD_DIR / part["files"]["parents"]["path"],
                filters=[("doc_id", "=", doc_id)],
            ).to_pylist()
            verify_spans(document_preview, children, parents)
            print("Verified doc_id:", doc_id, "| title:", document_preview["title"])
            print("Language hint:", document_preview["language"], document_preview["language_method"])
            if children:
                print("Child source text:", children[0]["text"][:1500])
            if parents:
                print("Parent source text:", parents[0]["text"][:2500])
        else:
            print("Chưa có tài liệu trích được. Xem failures và crawl summary.")
        """),
        md("""
        ## 4. Các artifact dùng cho giai đoạn retrieval sau này

        Chỉ đọc danh sách file trong build.json/snapshot manifest. Không glob toàn bộ
        processed folder, vì nơi này giữ cả các attempt cũ để rollback.
        documents giữ source_text và hash; children/parents giữ offset cùng raw source ID.
        failures giữ cả lỗi crawl và lỗi parser. Language chỉ là hint phục vụ audit,
        chưa được dùng làm hard gate.

        Notebook chưa tạo embedding/index và chưa tính F2. Query công khai hiện không
        có reference labels; BGE-token windows cũng chưa thay thế scorer chính thức.
        Raw corpus và artifact được lưu trên Drive, không đưa vào GitHub.
        Chạy notebook 03 để validate toàn snapshot, dedup xuyên shard và freeze candidate.
        """),
    ])


    save("03_colab_validate_and_freeze.ipynb", [
        md("""
        # VietMedBridge — 03: Kiểm tra toàn snapshot, audit và freeze candidate

        Chạy sau notebook 02, runtime CPU. Validator kiểm official ID/URL, uniqueness,
        foreign keys, source hash, exact offsets và bảo toàn mọi input qua documents/failures.
        Dedup reducer đọc tất cả shard được manifest chọn; không xóa official IDs.
        """),
        code(BOOTSTRAP),
        md("## 1. Chọn build và chạy lại golden với tokenizer đã ghim"),
        code("""
        from vietmedbridge.artifacts import read_json, atomic_json
        from vietmedbridge.dataset import load_snapshot, parquet_path
        from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer, parent_for_child
        from vietmedbridge.golden import run_golden_suite
        from vietmedbridge.health import health_report, freeze_candidate

        CRAWL_RUN = "stage-a-v2"
        BUILD_RUN = "stage-a-data-v2"
        BUILD_DIR = DATA_ROOT / "processed" / BUILD_RUN
        SNAPSHOT = load_snapshot(DATA_ROOT)
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        TOKENIZER, TOKENIZER_SPEC = load_bge_tokenizer(read_json(DATA_ROOT / "tokenizer_lock.json")["revision"])
        CHUNK_CONFIG = ChunkConfig(**read_json(BUILD_DIR / "config.json")["chunks"])
        GOLDEN = run_golden_suite(CHECKOUT / "tests/golden/cases.json", TOKENIZER, TOKENIZER_SPEC, CHUNK_CONFIG)
        atomic_json(DATA_ROOT / "reports/golden_latest.json", GOLDEN)
        if not GOLDEN["passed"]:
            raise RuntimeError("Golden fail; chưa được freeze candidate.")
        """),
        md("## 2. Health report, global dedup và kiểm tra nguồn bằng HTML"),
        code("""
        HEALTH = health_report(BUILD_DIR, official_links=OFFICIAL_LINKS,
                               crawl_dir=DATA_ROOT / "crawl" / CRAWL_RUN,
                               work_dir=WORK_DIR, audit_size=PIPELINE_CONFIG["audit_size"])
        print(json.dumps({key: HEALTH[key] for key in ("coverage", "documents", "chunks", "dedup")}, ensure_ascii=False, indent=2))
        import pandas as pd
        from IPython.display import HTML, display
        display(pd.DataFrame(HEALTH["by_domain"]))
        display(pd.DataFrame(HEALTH["failure_reasons"]))
        display(HTML((BUILD_DIR / HEALTH["files"]["audit"]["path"]).read_text()))
        print("Golden real-document candidates (chưa có expected assertions):",
              BUILD_DIR / HEALTH["files"]["golden_candidates"]["path"])
        """),
        md("""
        ## 3. Freeze snapshot candidate

        FROZEN_CANDIDATE là snapshot bất biến để xây index/benchmark. Nó chưa được
        PROMOTED: cần retrieval với qrels/reference spans, ngân sách đo thật và human audit.
        Corpus query hiện không có gold labels. LOW quality vẫn nằm trong corpus eligible.
        """),
        code("""
        CANDIDATE = freeze_candidate(BUILD_DIR, HEALTH, golden_report=GOLDEN)
        print(CANDIDATE["state"], CANDIDATE["candidate_manifest_sha256"])
        print(CANDIDATE["promotion"])
        print("Canonical aliases:", BUILD_DIR / HEALTH["files"]["canonical_aliases"]["path"])
        print("Representation aliases:", BUILD_DIR / HEALTH["files"]["representation_aliases"]["path"])
        """),
        md("""
        ## 4. Team ghi nhận milestone sau khi kiểm tra mẫu nguồn

        Copy golden_candidates.json thành reviewed_golden.json trên Drive. Với mỗi nguồn,
        kiểm tra raw/source rồi điền expected_key_snippets, đổi annotation_status thành REVIEWED.
        Có thể bổ sung expected_title_contains, expected_language, expected_min_text_length,
        expected_chunk_count_range. Không tự lấy text hiện tại làm đáp án rồi coi là human review.
        Replay cần 100–500 nguồn; 11 fixture synthetic không thay cho bộ golden này.
        Chỉ bật AUDIT_APPROVED sau khi người review đã xem health report/source audit và golden thật pass.
        Stage A được duyệt mở đường tới Stage B1 10k. B2/C/full còn cần báo cáo retrieval
        thật và budget; không tự đánh dấu pass khi chưa chạy. Chỉ một runtime ghi build/report.
        """),
        code("""
        from vietmedbridge.gates import record_milestone, stage_for_rows
        from vietmedbridge.golden import run_reviewed_golden

        AUDIT_APPROVED = False
        REVIEWER = ""  # tên người trong team đã thực hiện audit
        REVIEWED_GOLDEN_PATH = DATA_ROOT / "reports/reviewed_golden.json"
        REVIEWED_GOLDEN = None
        if REVIEWED_GOLDEN_PATH.exists():
            REVIEWED_GOLDEN = run_reviewed_golden(
                REVIEWED_GOLDEN_PATH, DATA_ROOT / "crawl" / CRAWL_RUN, TOKENIZER, TOKENIZER_SPEC,
                config=CHUNK_CONFIG,
            )
            atomic_json(DATA_ROOT / "reports/reviewed_golden_latest.json", REVIEWED_GOLDEN)
            print("Reviewed golden passed:", REVIEWED_GOLDEN["passed"])
        if AUDIT_APPROVED:
            MILESTONE = record_milestone(
                DATA_ROOT, CANDIDATE, stage=stage_for_rows(CANDIDATE["counts"]["input_records"]),
                corpus_sha256=SNAPSHOT["files"]["links_corpus.parquet"]["sha256"],
                reviewer=REVIEWER, approved=True, reviewed_golden_report=REVIEWED_GOLDEN,
            )
            print(MILESTONE)
        else:
            print("Candidate đã freeze; milestone human review đang chờ team.")
        """),
    ])


if __name__ == "__main__":
    main()
