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
        "có thể chọn Runtime Version 2026.07 (Python 3.12)."
    )
if importlib.util.find_spec("google.colab") is None:
    raise RuntimeError("Notebook này được thiết kế cho Google Colab.")

from google.colab import drive
drive.mount("/content/drive")

# Giữ cùng DATA_ROOT trong cả ba notebook.
DATA_ROOT = Path("/content/drive/MyDrive/VietMedBridge/data")
WORK_DIR = Path("/content/vmb_work")  # temp files/spill ở ổ local của Colab
DATA_ROOT.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
CODE_REVISION = None  # Có thể đặt full commit SHA; mặc định dùng code_lock đã lưu.
CHECKOUT = Path("/content/VietMedBridge")
lock_path = DATA_ROOT / "code_lock.json"
lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
if lock and lock.get("repo_url") != REPO_URL:
    raise ValueError("DATA_ROOT đang khóa vào một repo khác.")
reference = CODE_REVISION or lock.get("git_commit") or "main"
if not (CHECKOUT / ".git").exists():
    subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CHECKOUT)], check=True)
remote = subprocess.check_output(
    ["git", "-C", str(CHECKOUT), "remote", "get-url", "origin"], text=True
).strip()
if remote != REPO_URL:
    raise ValueError("Checkout hiện tại không thuộc repo VietMedBridge.")
subprocess.run(["git", "-C", str(CHECKOUT), "fetch", "--depth", "1", "origin", reference], check=True)
subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "--detach", "FETCH_HEAD"], check=True)
CODE_COMMIT = subprocess.check_output(
    ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True
).strip()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(CHECKOUT) + "[notebook]"],
    check=True,
)
from vietmedbridge.artifacts import atomic_json, runtime_versions
if not lock or CODE_REVISION:
    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT})
atomic_json(DATA_ROOT / "runtime.json", {
    "git_commit": CODE_COMMIT, "python": sys.version, "packages": runtime_versions()
})
print("Code commit:", CODE_COMMIT)
print("Dữ liệu/checkpoint:", DATA_ROOT)
'''


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


def save(name, cells):
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:12]
    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }
    path = ROOT / "notebooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.name)


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

        # Revision đã kiểm tra của AIGuruTinix/ViBioMIR. Đổi revision sẽ tạo snapshot mới.
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
    save("01_colab_crawl_sources.ipynb", [
        md("""
        # VietMedBridge — 01: Crawl nguồn có checkpoint

        Chạy sau notebook 00. Mặc định chỉ crawl mẫu domain nhỏ.
        File raw chứa bytes HTTP đã giải nén content-encoding, hash, URL gốc,
        URL sau redirect và trạng thái cho từng ID. Mỗi shard hoàn thành được lưu
        vào Drive; khi Colab ngắt, chạy lại cell để tiếp tục từ checkpoint.

        Dữ liệu có hàng triệu URL. Cần xem domain audit và chất lượng sample trước
        khi chọn phạm vi lớn; các nguồn như PubMed/PMC có thể cần bulk/API adapter
        để thu thập hiệu quả. Crawler HTTP ở đây là nền tảng khởi đầu.
        """),
        code(BOOTSTRAP),
        md("## 1. Chọn input và phạm vi theo vị trí dòng Parquet"),
        code("""
        from vietmedbridge.dataset import load_snapshot, parquet_path, validate_link_subset
        from vietmedbridge.crawl import CrawlConfig, crawl_links, completed_parts
        from vietmedbridge.artifacts import read_json, verify_file

        SNAPSHOT = load_snapshot(DATA_ROOT)
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        MODE = "smoke"       # "smoke" hoặc "range"
        START_ROW = 0        # vị trí dòng, KHÔNG phải official doc_id
        STOP_ROW = 5000     # chỉ dùng ở mode range; None = hết file
        WORKER_INDEX = 0
        WORKER_COUNT = 1     # workers phải có cùng range và xử lý shard khác nhau
        MAX_NEW_SHARDS = 10  # số shard mới mỗi lần chạy; None = hết range
        RETRY_FAILED = False

        if MODE == "smoke":
            audit = read_json(DATA_ROOT / "reports/dataset_audit.json")
            if audit["corpus_sha256"] != SNAPSHOT["files"]["links_corpus.parquet"]["sha256"]:
                raise ValueError("Audit/sample không thuộc snapshot đang dùng; chạy lại notebook 00.")
            INPUT_LINKS = DATA_ROOT / audit["sample"]["path"]
            verify_file(INPUT_LINKS, audit["sample"]["sha256"])
            validate_link_subset(INPUT_LINKS, OFFICIAL_LINKS, work_dir=WORK_DIR)
            RUN_NAME = "smoke-v1"
            START, STOP = 0, None
        elif MODE == "range":
            INPUT_LINKS = OFFICIAL_LINKS
            RUN_NAME = "corpus-v1"
            START, STOP = START_ROW, STOP_ROW
        else:
            raise ValueError("MODE phải là smoke hoặc range.")

        print("Run:", RUN_NAME, "| input:", INPUT_LINKS.name, "| row range:", START, STOP)
        """),
        md("""
        ## 2. Crawl theo shard

        Giữ cùng shard_size, range và cấu hình khi resume. Đổi range, cấu hình hoặc input phải
        dùng RUN_NAME mới. Mỗi host có giới hạn tốc độ; 429/5xx được retry có backoff.
        Robots bị chặn, lỗi tải, file quá lớn và các lỗi khác được lưu vào ledger.
        Một shard ghi đủ trạng thái cho mọi ID mới được coi là hoàn thành.
        """),
        code("""
        CONFIG = CrawlConfig(
            concurrency=4, shard_size=128, per_host_delay=1.0,
            timeout_seconds=30, max_bytes=16 * 1024 * 1024,
            attempts=3, respect_robots=True,
        )
        SUMMARY = await crawl_links(
            INPUT_LINKS, DATA_ROOT / "crawl", run_name=RUN_NAME, config=CONFIG,
            start=START, stop=STOP, worker_index=WORKER_INDEX, worker_count=WORKER_COUNT,
            retry_failed=RETRY_FAILED, max_shards=MAX_NEW_SHARDS, work_dir=WORK_DIR,
            origin_corpus_sha256=SNAPSHOT["files"]["links_corpus.parquet"]["sha256"],
        )
        print(json.dumps(SUMMARY, ensure_ascii=False, indent=2))
        """),
        md("""
        ## 3. Kiểm tra coverage và các ID lỗi

        Số recorded_documents bên dưới chỉ là phần đã crawl trong run.
        Sample thành công không có nghĩa đã thu thập đủ corpus chính thức.
        """),
        code("""
        import pandas as pd
        PARTS = completed_parts(DATA_ROOT / "crawl" / RUN_NAME)
        from itertools import islice
        failed_count = sum(len(part["failed_ids"]) for part in PARTS)
        failed_preview = list(islice(
            ({"part": part["part"], "doc_id": doc_id}
             for part in PARTS for doc_id in part["failed_ids"]), 100,
        ))
        print("Official corpus rows:", SNAPSHOT["files"]["links_corpus.parquet"]["rows"])
        print("Recorded:", SUMMARY["recorded_documents"], "| errors:", failed_count)
        display(pd.DataFrame([{"status": key, "count": value} for key, value in SUMMARY["statuses"].items()]))
        if failed_preview:
            display(pd.DataFrame(failed_preview))
        """),
        md("""
        ## Resume và retry

        Chạy lại cell 2 để xử lý các shard còn thiếu trong range. Đặt RETRY_FAILED=True
        rồi chạy lại cell 1–2 để chỉ tải lại ID thất bại trong shard đã hoàn thành.
        Các nguồn thành công được giữ lại, và raw attempt cũ vẫn tồn tại.
        Không chạy hai runtime ghi cùng shard; dùng WORKER_INDEX/WORKER_COUNT để chia việc.
        Sau sample, kiểm tra tỷ lệ lỗi theo domain trước khi mở rộng phạm vi.
        """),
    ])
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

        CRAWL_RUN = "smoke-v1"  # đổi thành corpus-v1 khi xử lý run lớn
        BUILD_RUN = "canonical-v1"
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
        Parent là cửa sổ theo token trong cùng tài liệu; phân tách section chuyên biệt
        và OCR chưa được triển khai ở giai đoạn này.
        """),
        code("""
        CHUNK_CONFIG = ChunkConfig(
            child_tokens=180, overlap_tokens=40, parent_tokens=512, snap_sentences=True,
        )
        BUILD = build_corpus(
            DATA_ROOT / "crawl" / CRAWL_RUN, DATA_ROOT / "processed",
            TOKENIZER, TOKENIZER_SPEC, run_name=BUILD_RUN, config=CHUNK_CONFIG,
            work_dir=WORK_DIR, max_shards=MAX_NEW_SHARDS,
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
        """),
    ])


if __name__ == "__main__":
    main()
