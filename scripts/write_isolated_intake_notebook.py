"""Generate the one CPU notebook for additions while submission stays frozen."""
from write_notebooks import ROOT, code, md, save

RESUME_TOOLS = (ROOT / "scripts/intake_resume.py").read_text(encoding="utf-8")


def main():
    save("02_colab_import_new_data.ipynb", [
        md("""
        # 02 — Chỉ nhập URL thuộc dataset BTC mới (CPU)

        **Chọn CPU → Run all.** Dùng một phiên cho notebook này. Luồng submission
        vẫn chạy riêng với candidate hiện tại; notebook này không đổi active
        candidate, snapshot, model cache hoặc khóa code của luồng đó.

        Lượt mặc định `btc-new-71be0e12-20261019-v1`: chỉ sáu frontier mới
        00000–00005, gắn với dataset BTC revision `71be0e12`. Không nhập shard
        thuộc snapshot cũ. Chọn đúng phiên bản tích lũy
        đã audit, kể cả TAR lồng TAR. Đọc nguồn qua Drive API, không cần tạo
        shortcut AI Guru hay tải toàn bộ TAR về /content.

        Checkpoint và kết quả: `VietMedBridge/data_staging/btc-new-71be0e12-20261019-v1/`.
        Ngắt phiên: mở lại cùng notebook, cùng folder, Run all; từng source/shard
        hoàn tất được giữ. Shard đang ghi dở sẽ chạy lại. Đây là import, kiểm tra
        và chia chunk, chưa chạy embedding hay ghép vào submission.
        """),
        code(r'''
        #@title 1. Kết nối Drive và cài code riêng cho lượt nhập mới
        import importlib
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        if not (3, 11) <= sys.version_info[:2] < (3, 14):
            raise RuntimeError("Dùng runtime Colab Python 3.11–3.13.")
        print("Bắt đầu CPU intake; không nạp model embedding.", flush=True)
        from google.colab import drive
        drive.mount("/content/drive")

        # Chỉ sửa nếu project của team ở vị trí khác; tự tìm shortcut dùng chung.
        PROJECT = Path("/content/drive/MyDrive/VietMedBridge")
        if not (PROJECT / "data").is_dir():
            shared = Path("/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge")
            if (shared / "data").is_dir():
                PROJECT = shared
            else:
                raise FileNotFoundError("Chưa tìm thấy folder VietMedBridge dùng chung. Thêm shortcut VietMedBridge vào My Drive.")
        CAMPAIGN = "btc-new-71be0e12-20261019-v1"
        INTAKE_ROOT = PROJECT.resolve() / "data_staging" / CAMPAIGN
        INTAKE_ROOT.mkdir(parents=True, exist_ok=True)
        WORK_DIR = Path("/content/vmb_intake_work")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        CHECKOUT = Path("/content/VietMedBridge-intake")
        REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
        lock_path = INTAKE_ROOT / "intake_code_lock.json"
        lock = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
        if lock and lock.get("repo_url") != REPO_URL:
            raise ValueError("Intake code lock belongs to another repository.")
        if not (CHECKOUT / ".git").is_dir():
            subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CHECKOUT)], check=True)
        remote = subprocess.check_output(["git", "-C", str(CHECKOUT), "remote", "get-url", "origin"], text=True).strip()
        if remote != REPO_URL:
            raise ValueError("Checkout intake thuộc repo khác.")
        reference = lock.get("commit") or "main"
        print("Tải code intake:", reference, flush=True)
        subprocess.run(["git", "-C", str(CHECKOUT), "fetch", "--depth", "1", "origin", reference], check=True)
        commit = subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "FETCH_HEAD"], text=True).strip()
        if "vietmedbridge" in sys.modules and globals().get("_INTAKE_COMMIT") != commit:
            raise RuntimeError("Restart session để nạp đúng code intake; checkpoint trên Drive vẫn giữ.")
        subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "--detach", commit], check=True)
        print("Cài dependencies CPU…", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(CHECKOUT)], check=True)
        runtime_path = INTAKE_ROOT / "intake_runtime_lock.json"
        if runtime_path.is_file():
            pinned = json.loads(runtime_path.read_text())
            packages = [name+"=="+version for name, version in pinned.items() if name != "vietmedbridge"]
            subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)
        sys.path.insert(0, str(CHECKOUT / "src"))
        importlib.invalidate_caches()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HOME"] = "/content/hf_cache_intake"
        from vietmedbridge.artifacts import atomic_json, read_json, runtime_versions
        from vietmedbridge.isolated_intake import intake_root
        if INTAKE_ROOT != intake_root(PROJECT / "data", CAMPAIGN):
            raise ValueError("Intake output root không tách biệt khỏi submission.")
        if not lock:
            atomic_json(lock_path, {"repo_url": REPO_URL, "commit": commit})
        if not runtime_path.is_file():
            atomic_json(runtime_path, runtime_versions())
        _INTAKE_COMMIT = commit
        print("Code:", commit, "\nCheckpoint:", INTAKE_ROOT, flush=True)
        '''),
        code('#@title 2. Nhận checkpoint trước; chỉ nạp tài nguyên khi cần xử lý phần thiếu\n' + RESUME_TOOLS + '\n' + r'''
import json
from huggingface_hub import hf_hub_download
from vietmedbridge.artifacts import verify_file, publish_file, digest_json
from vietmedbridge.isolated_intake import DriveSource, fetch_reference
from vietmedbridge.chunks import load_bge_tokenizer, ChunkConfig
from vietmedbridge.golden import run_golden_suite

def validate_new_btc_plan(plan):
    required = "5586f423916005c11ffa8afb96895017d672af51d68aee8a88d28dd2e0dc39c6"
    if (plan.get("required_snapshot_id") != required
            or set(plan["snapshots"]) != {required}
            or not plan["sources"]
            or any(source["snapshot_id"] != required
                   or source["mode"] != "NEW_SNAPSHOT_SOURCE"
                   for source in plan["sources"])):
        raise ValueError("Lượt này chỉ nhận URL thuộc snapshot BTC mới 71be0e12; không nhập nguồn legacy.")

PLAN = read_json(CHECKOUT / "configs/intake_ai_guru_20261019.json")
validate_new_btc_plan(PLAN)
CONFIG = ChunkConfig(**PLAN["chunking"])
TOKENIZER_SPEC = {"model_id": "BAAI/bge-m3", "revision": PLAN["tokenizer_revision"], "special_tokens": False}
CONTROLLER = IntakeResume(PLAN, root=INTAKE_ROOT, work_dir=WORK_DIR,
    tokenizer_spec=TOKENIZER_SPEC, chunk_config=CONFIG, shard_size=2048)
CHECKPOINT_STATUS = CONTROLLER.status()
print("Checkpoint:", CHECKPOINT_STATUS["completed_sources"], "/", CHECKPOINT_STATUS["total_sources"],
      "nguồn đã xong | chỉ đọc receipt nhỏ, chưa tải lại dữ liệu cũ", flush=True)
LAZY_RESOURCES = {}

def load_reader():
    if "reader" not in LAZY_RESOURCES:
        from google.colab import auth
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        auth.authenticate_user()
        credentials, _ = google.auth.default()
        LAZY_RESOURCES["reader"] = DriveSource(AuthorizedSession(credentials))
    return LAZY_RESOURCES["reader"]

def load_baseline():
    if "baseline" not in LAZY_RESOURCES:
        reference = PLAN["baseline"]
        target = INTAKE_ROOT / "references/baseline_aliases.parquet"
        if not target.is_file():
            fetch_reference(load_reader(), reference, target)
        print("Cần mapping cũ để chọn phần mới; chỉ đưa mapping lên ổ local một lần.", flush=True)
        LAZY_RESOURCES["baseline"] = intake_core._local_mapping(target, reference["sha256"], WORK_DIR)
    return LAZY_RESOURCES["baseline"]

def load_links(snapshot_id):
    key = "links-"+snapshot_id
    if key not in LAZY_RESOURCES:
        spec = PLAN["snapshots"][snapshot_id]
        target = INTAKE_ROOT / "references" / (snapshot_id[:16]+"-links.parquet")
        if not target.is_file():
            print("Tải official links:", spec["revision"], flush=True)
            downloaded = Path(hf_hub_download("AIGuruTinix/ViBioMIR", spec["filename"],
                repo_type="dataset", revision=spec["revision"]))
            verify_file(downloaded, spec["sha256"])
            publish_file(downloaded, target)
        LAZY_RESOURCES[key] = intake_core._local_mapping(target, spec["sha256"], WORK_DIR)
    return LAZY_RESOURCES[key]

def load_tokenizer():
    if "tokenizer" not in LAZY_RESOURCES:
        tokenizer, spec = load_bge_tokenizer(PLAN["tokenizer_revision"])
        golden = run_golden_suite(CHECKOUT / "tests/golden/cases.json", tokenizer, spec, CONFIG)
        atomic_json(INTAKE_ROOT / "golden.json", golden)
        if not golden["passed"]:
            raise RuntimeError("Golden failed; chưa chia chunk mới.")
        LAZY_RESOURCES["tokenizer"] = (tokenizer, spec)
    return LAZY_RESOURCES["tokenizer"]
        '''),
        code(r'''
        #@title 3. Chỉ tiếp tục nguồn thiếu, lưu checkpoint từng shard và từng nguồn

        def continue_intake():
            return CONTROLLER.run(load_reader=load_reader, load_links=load_links,
                load_baseline=load_baseline, load_tokenizer=load_tokenizer)

        try:
            RESULT = continue_intake()
        except OSError as error:
            if error.errno not in (103, 107):
                raise
            print("Drive ngắt kết nối; gắn lại một lần rồi tiếp tục checkpoint intake.", flush=True)
            drive.mount("/content/drive", force_remount=True)
            RESULT = continue_intake()
        print(json.dumps(RESULT, ensure_ascii=False, indent=2))
        print("Hoàn tất lượt nhập CPU. Giữ nguyên luồng submission đang chạy.")
        print("Bàn giao lượt data mới:", INTAKE_ROOT / "intake_status.json")
        '''),
        md("""
        `INTAKE_COMPLETE` nghĩa là mọi ID trong lượt nguồn đã có kết quả:
        `ADDED_VALID_DOCUMENT`, `REUSED_EXISTING_DOCUMENT` hoặc `PIPELINE_FAILURE`.
        URL lỗi được giữ trong `sources/<job>/source_outcomes.parquet` và failure
        shards; trạng thái này không có nghĩa là crawl thành công 100%.

        `SKIP_COMPLETED_SOURCE`: bỏ qua nguồn xong, chỉ đọc receipt/trạng thái file.
        `REUSE_SELECTION`: tiếp tục nguồn dở, không chọn lại URL hoặc nạp mapping cũ.
        `RESUME_PARTS`: giữ shard đã commit; chỉ source đang dở cần kiểm tra checksum.
        `REUSE_VALIDATED_BUILD`: build đã qua integrity, chỉ hoàn tất bước bàn giao.
        Mapping được kiểm SHA khi thực sự sử dụng; receipt không thay thế
        một lần audit lại toàn bộ payload. Không nạp catalog/BM25 hoặc embedding cũ.

        Gửi `intake_status.json` sau khi xong để kiểm tra số thực tế. Bước kế tiếp
        là freeze/chuẩn bị index cho **lượt thêm riêng**, rồi chỉ embedding text
        chưa có; ghép vào phiên submission kế tiếp sau khi ZIP hiện tại hoàn tất.
        """),
    ])


if __name__ == "__main__":
    main()
