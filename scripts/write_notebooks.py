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
    save("01_colab_crawl_sources.ipynb", [
        md("""
        # VietMedBridge — 01: Crawl nguồn có checkpoint

        Chạy sau notebook 00. Mặc định crawl mẫu Stage A phân tầng ~1.000 URL.
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
        from vietmedbridge.artifacts import read_json, verify_file, sha256_file
        from vietmedbridge.gates import authorize_scale
        import pyarrow.parquet as pq

        SNAPSHOT = load_snapshot(DATA_ROOT)
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        MODE = "stage_a"     # "stage_a", "smoke" hoặc "range"
        START_ROW = 0        # vị trí dòng, KHÔNG phải official doc_id
        STOP_ROW = 10000    # chỉ dùng ở mode range; None = hết file (full gate bắt buộc)
        WORKER_INDEX = 0
        WORKER_COUNT = 1     # workers phải có cùng range và xử lý shard khác nhau
        MAX_NEW_SHARDS = 10  # số shard mới mỗi lần chạy; None = hết range
        RETRY_FAILED = False

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
                raise ValueError("Audit/sample không thuộc snapshot đang dùng; chạy lại notebook 00.")
            INPUT_LINKS = DATA_ROOT / audit["sample"]["path"]
            verify_file(INPUT_LINKS, audit["sample"]["sha256"])
            validate_link_subset(INPUT_LINKS, OFFICIAL_LINKS, work_dir=WORK_DIR)
            RUN_NAME = "smoke-v2"
            START, STOP = 0, None
        elif MODE == "range":
            INPUT_LINKS = OFFICIAL_LINKS
            RUN_NAME = "stage-b1-v2"
            START, STOP = START_ROW, STOP_ROW
        else:
            raise ValueError("MODE phải là stage_a, smoke hoặc range.")

        GOLDEN = read_json(DATA_ROOT / "reports/golden_latest.json")
        if GOLDEN["fixture_sha256"] != sha256_file(CHECKOUT / "tests/golden/cases.json"):
            raise ValueError("Golden fixtures đổi; chạy lại notebook 00.")
        input_rows = pq.ParquetFile(INPUT_LINKS).metadata.num_rows
        requested_rows = max(0, min(STOP if STOP is not None else input_rows, input_rows) - START)
        STAGE = authorize_scale(DATA_ROOT, rows=requested_rows,
                                corpus_sha256=SNAPSHOT["files"]["links_corpus.parquet"]["sha256"],
                                golden_report=GOLDEN, chunking=PIPELINE_CONFIG["chunking"])

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
        CONFIG = CrawlConfig(**PIPELINE_CONFIG["crawler"])
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
        Chạy notebook 02–03 để kiểm tra, tạo health report và review milestone.
        Stage B/C cần evidence gate của giai đoạn trước. Không đặt STOP_ROW=None
        khi chưa có benchmark retrieval và budget đã đo.
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
