"""Generate the isolated CPU handoff for the ten completed new BTC sources."""
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    setup = dedent('''
        #@title 1. Colab CPU — dùng code/runtime lock của intake đã hoàn tất
        import importlib
        import json
        import os
        import re
        import subprocess
        import sys
        from pathlib import Path

        print("Bắt đầu freeze data mới trên CPU; không nạp model embedding.", flush=True)
        if not (3, 11) <= sys.version_info[:2] < (3, 14):
            raise RuntimeError("Dùng Colab Python 3.11–3.13.")
        from google.colab import drive
        drive.mount("/content/drive")
        PROJECT = Path("/content/drive/MyDrive/VietMedBridge")
        if not (PROJECT / "data_staging/btc-new-71be0e12-20261019-v1").is_dir():
            PROJECT = Path("/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge")
        PROJECT = PROJECT.resolve()
        PARENT = PROJECT / "data_staging/btc-new-71be0e12-20261019-v1"
        lock = json.loads((PARENT / "intake_code_lock.json").read_text())
        pinned = json.loads((PARENT / "intake_runtime_lock.json").read_text())
        REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
        if lock.get("repo_url") != REPO_URL or not re.fullmatch(r"[0-9a-f]{40}", lock.get("commit", "")):
            raise ValueError("Thiếu/sai code lock của intake; không tự chọn code khác.")
        CHECKOUT = Path("/content/VietMedBridge-freeze-new")
        WORK_DIR = Path("/content/vmb_freeze_new_work")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        if not (CHECKOUT / ".git").is_dir():
            subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CHECKOUT)], check=True)
        remote = subprocess.check_output(["git", "-C", str(CHECKOUT), "remote", "get-url", "origin"], text=True).strip()
        if remote != REPO_URL:
            raise ValueError("Checkout thuộc repo khác.")
        commit = lock["commit"]
        if "vietmedbridge" in sys.modules and globals().get("_FREEZE_NEW_COMMIT") != commit:
            raise RuntimeError("Dùng một phiên Colab CPU riêng cho notebook này; checkpoint Drive giữ nguyên.")
        print("Nạp producer đã khóa:", commit, flush=True)
        subprocess.run(["git", "-C", str(CHECKOUT), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "--detach", "FETCH_HEAD"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(CHECKOUT)], check=True)
        packages = [name+"=="+version for name, version in pinned.items() if name != "vietmedbridge"]
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)
        sys.path.insert(0, str(CHECKOUT / "src"))
        importlib.invalidate_caches()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HOME"] = "/content/hf_cache_freeze_new"
        _FREEZE_NEW_COMMIT = commit
        print("Checkpoint:", PARENT / "freeze_batch_v1", flush=True)
    ''').strip()
    helper = (ROOT / "scripts/new_data_freeze.py").read_text(encoding="utf-8")
    prepare = helper + '\n' + dedent('''
        from vietmedbridge.artifacts import verify_file
        from vietmedbridge.chunks import load_bge_tokenizer, ChunkConfig
        from vietmedbridge.golden import run_golden_suite
        from huggingface_hub import hf_hub_download
        FREEZER = NewDataFreeze(PROJECT, WORK_DIR)
        LAZY_FREEZE = {}
        SCOPE = read_json(PARENT / "full_intake_scope.json")
        LINK_SPECS = SCOPE["base_plan"]["snapshots"]

        def load_tokenizer(item):
            if "tokenizer" not in LAZY_FREEZE:
                LAZY_FREEZE["tokenizer"] = load_bge_tokenizer(item["tokenizer"]["revision"])
            return LAZY_FREEZE["tokenizer"]

        def load_links(item):
            checksum = item["official_links_sha256"]
            if checksum not in LAZY_FREEZE:
                matches = [(key, spec) for key, spec in LINK_SPECS.items() if spec["sha256"] == checksum]
                if len(matches) != 1:
                    raise ValueError("Không xác định duy nhất official links của build.")
                snapshot, spec = matches[0]
                source = PARENT / "references" / (snapshot[:16] + "-links.parquet")
                if not source.is_file():
                    source = Path(hf_hub_download("AIGuruTinix/ViBioMIR", spec["filename"],
                        repo_type="dataset", revision=spec["revision"]))
                LAZY_FREEZE[checksum] = copy_verified(source, WORK_DIR / "references" / (checksum+".parquet"), checksum)
            return LAZY_FREEZE[checksum]

        def load_golden(item, tokenizer, spec):
            if "golden" not in LAZY_FREEZE:
                LAZY_FREEZE["golden"] = run_golden_suite(CHECKOUT / "tests/golden/cases.json",
                    tokenizer, spec, ChunkConfig(**item["chunking"]))
                if not LAZY_FREEZE["golden"]["passed"]:
                    raise ValueError("Golden regression chưa đạt; giữ nguyên data đã import.")
            return LAZY_FREEZE["golden"]

        completed = sum(FREEZER.completed(item) is not None for item in FREEZER.builds)
        print(f"Nguồn mới: {len(FREEZER.builds)} | freeze/input đã xong: {completed}/10", flush=True)
        print("Đọc receipt JSON; chưa mở corpus hoặc nạp tokenizer/model.", flush=True)
    ''')
    run = dedent('''
        #@title 3. Freeze và tạo input: tiếp tục build/part còn thiếu, lưu Drive từng build
        import threading
        import time
        started = time.monotonic()
        stopped = threading.Event()
        def show_progress():
            while not stopped.wait(30):
                print(f"CPU freeze heartbeat | {time.monotonic()-started:.0f}s | "
                      + getattr(FREEZER, "phase", "receipt preflight"), flush=True)
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        try:
            available_mb = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                                    if line.startswith("MemAvailable:"))) // 1024
            memory_mb = min(2048, max(64, available_mb // 4))
            RESULT = FREEZER.run(load_tokenizer=load_tokenizer, load_links=load_links,
                load_golden=load_golden, memory_mb=memory_mb)
        finally:
            stopped.set()
            progress_thread.join(timeout=1)
        print(json.dumps({k: RESULT[k] for k in ("state", "completed_builds", "total_builds", "documents",
            "children", "input_parts", "unique_dense_inputs_per_build_sum", "embedding_complete",
            "submission_candidate_changed")}, ensure_ascii=False, indent=2), flush=True)
        print("Handoff:", FREEZER.root / "freeze_batch.json", flush=True)
        print("CPU freeze/input hoàn tất. GPU embedding và cập nhật index là bước tiếp theo.", flush=True)
    ''').strip()
    book = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(dedent('''
        # 03 — Freeze 10 nguồn data mới và chuẩn bị input embedding (CPU)

        **Một phiên Colab CPU riêng → Run all. Cấu hình đã đặt sẵn.**
        Nhận 00000–00007 + retained và 00008 đã `INTAKE_COMPLETE` trong `data_staging`.
        Không chạy lại crawl/import/chunk. Không đổi candidate hoặc model cache của luồng submit.

        Lần đầu mỗi build: đọc payload đã xử lý một lần từ Drive sang scratch của Colab,
        kiểm integrity trên bản local, freeze và tạo input text theo part 4.096 dòng trên Drive.
        SQL một luồng, tối đa 2 GiB; giữ tối đa một build ở scratch, giải phóng bản staging
        sau khi receipt trên Drive hoàn tất. Không tải corpus về máy cá nhân.

        Retry: build hoàn tất chỉ đọc JSON. Build dở dùng frozen candidate và các input part
        đã lưu; runtime mới vẫn cần staging nguồn của build đang dở. Payload SHA được kiểm khi dùng.
        Không chạy hai phiên notebook này cùng lúc.

        Đích: `data_staging/btc-new-71be0e12-20261019-v1/freeze_batch_v1/freeze_batch.json`.
        Kết thúc ở **`NEW_DATA_INPUTS_READY`**: chưa embedding/index/submit và chưa xác nhận
        phủ đủ toàn bộ ID BTC. Tổng input của các build có thể trùng text hash giữa build.
    ''').strip()), nb.v4.new_code_cell(setup), nb.v4.new_code_cell('#@title 2. Nhận 10 checkpoint đã hoàn tất\n'+prepare), nb.v4.new_code_cell(run)])
    book.metadata = {"colab": {"name": "03_colab_freeze_new_data.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}
    return book


if __name__ == "__main__":
    target = ROOT / "notebooks/03_colab_freeze_new_data.ipynb"
    nb.write(notebook(), target)
    print(target)
