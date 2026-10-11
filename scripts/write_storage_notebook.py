"""One CPU-only storage utility; no data preprocessing or model inference."""
from write_notebooks import code, md, save


def main():
    save("tools/drive_hf_storage.ipynb",[
        md('''
        # VietMedBridge — Drive → Hugging Face storage

        Chạy **Colab CPU**. Drive chỉ được đọc; plan/progress lưu ở ổ tạm.
        Repo đích mặc định `platypus123/biobridgeir-checkpoints`, yêu cầu Private.
        Thêm Colab Secret **HF_TOKEN**, bật Notebook access; token có quyền ghi
        đúng dataset repo. Không dán token vào code/chat. Chỉ một người chuyển
        dữ liệu và công bố snapshot ở một thời điểm; dừng writer đang sửa candidate.

        Run all: thống kê active candidate, source shards, inputs, CPU catalog và
        BGE/Qwen blocks được tham chiếu → upload theo batch → xác minh → công bố
        HF_STORAGE_VERIFIED. Mặc định ngân sách snapshot 90 GB, không phải xác nhận
        quota còn trống của tài khoản; kiểm tra Storage của HF trước khi chạy.
        Upload ngắt: Run all lại; objects đã commit được kiểm và bỏ qua, kể cả
        runtime mới. Incoming TAR, các build cũ và model weights không nằm trong
        snapshot này. File trên Drive giữ nguyên; chuyển xong chưa tự giải phóng
        quota Drive. Không có GPU/model inference trong notebook này.
        '''),
        code('''
        import importlib, json, os, subprocess, sys
        from pathlib import Path
        from google.colab import drive, userdata

        DATA_ROOT = Path("/content/drive/MyDrive/VietMedBridge/data")
        HF_STORAGE_REPO = "platypus123/biobridgeir-checkpoints"
        MAX_SNAPSHOT_GB = 90  # ngân sách do team chọn; kiểm quota HF thực tế
        INCLUDE_SOURCE_SHARDS = True  # giữ cả dữ liệu đã xử lý để dùng các bước sau
        WORK_DIR = Path("/content/vmb_storage")
        CHECKOUT = Path("/content/VietMedBridge")
        REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
        drive.mount("/content/drive")
        if not (DATA_ROOT / "active_data_candidate.json").is_file():
            raise RuntimeError("DATA_ROOT chưa đúng thư mục data đã chạy notebook 03.")
        WORK_DIR.mkdir(parents=True,exist_ok=True)
        try:
            token = userdata.get("HF_TOKEN")
        except Exception:
            raise RuntimeError("Thêm Colab Secret HF_TOKEN có quyền ghi dataset repo; bật Notebook access.") from None
        if not token:
            raise RuntimeError("HF_TOKEN đang rỗng.")
        os.environ["HF_TOKEN"] = token
        if not (CHECKOUT / ".git").is_file() and not (CHECKOUT / ".git").is_dir():
            subprocess.run(["git","clone","--no-checkout",REPO_URL,str(CHECKOUT)],check=True)
        if subprocess.check_output(["git","-C",str(CHECKOUT),"remote","get-url","origin"],text=True).strip() != REPO_URL:
            raise RuntimeError("Checkout không thuộc VietMedBridge.")
        subprocess.run(["git","-C",str(CHECKOUT),"fetch","--depth","1","origin","main"],check=True)
        CODE_COMMIT = subprocess.check_output(["git","-C",str(CHECKOUT),"rev-parse","FETCH_HEAD"],text=True).strip()
        if "vietmedbridge" in sys.modules and globals().get("_VMB_STORAGE_COMMIT") != CODE_COMMIT:
            raise RuntimeError("Code đã đổi; Restart session rồi chạy lại notebook.")
        subprocess.run(["git","-C",str(CHECKOUT),"checkout","--detach","FETCH_HEAD"],check=True)
        subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","-e",".[notebook]"],cwd=CHECKOUT,check=True)
        sys.path.insert(0,str(CHECKOUT / "src"))
        importlib.invalidate_caches()
        _VMB_STORAGE_COMMIT = CODE_COMMIT
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        print("Code commit:",CODE_COMMIT)
        print("Source:",DATA_ROOT,"| Destination:",HF_STORAGE_REPO)
        '''),
        md('''
        ## Kiểm kê và chuyển snapshot

        Dung lượng vượt MAX_SNAPSHOT_GB thì dừng trước upload. Con số này là
        dung lượng objects của snapshot hiện tại; các snapshot cũ và repo khác
        cũng dùng quota tài khoản. Không tự nâng gói hoặc thay visibility của
        repo đang Public. Source/input/model hashes giữ nguyên khi chuyển.
        '''),
        code('''
        from vietmedbridge.hub_storage import plan_active_storage, storage_summary, upload_storage
        from vietmedbridge.artifacts import atomic_json

        PLAN = plan_active_storage(DATA_ROOT,CHECKOUT,include_sources=INCLUDE_SOURCE_SHARDS)
        atomic_json(WORK_DIR / "storage_plan.json",PLAN)
        print(json.dumps(storage_summary(PLAN),ensure_ascii=False,indent=2))
        STORAGE = upload_storage(DATA_ROOT,PLAN,HF_STORAGE_REPO,work_dir=WORK_DIR,
            max_snapshot_bytes=int(MAX_SNAPSHOT_GB * 1e9),api=api)
        if STORAGE["state"] != "HF_STORAGE_VERIFIED":
            raise RuntimeError("Chưa upload: snapshot vượt ngân sách. Xem thống kê và quota HF thực tế trước khi đổi ngân sách.")
        '''),
        md('''
        ## Giữ thông tin snapshot, rồi chạy notebook 04

        Sau HF_STORAGE_VERIFIED, dùng **04_colab_retrieval_baseline.ipynb** bản
        mới trên Colab GPU. Cell 1: `USE_HF_STORAGE=True`, giữ repo bên trên,
        `HF_STORAGE_PROFILE="bge_submit"`. Có thể điền HF_STORAGE_REVISION bằng
        revision in bên dưới để khóa đúng bản này. Mặc định lấy snapshot mới nhất
        rồi khóa commit trong phiên. Chỉ cần HF_TOKEN có quyền đọc repo.

        04 tự cấu hình baseline BGE + BM25, ghi checkpoint/ZIP vào ổ local Colab;
        không mount/ghi Drive. BGE đủ và catalog phù hợp runtime là điều kiện để
        submit. Với profile này không tải lại source shards hay Qwen.
        Giữ phiên Colab đến khi tải ZIP: checkpoint local mất khi runtime bị thu hồi.
        File locator nhỏ dưới đây giúp tìm lại snapshot; không chứa token.
        '''),
        code('''
        from google.colab import files
        print("HF_STORAGE_REVISION =",repr(STORAGE["revision"]))
        print("Snapshot:",STORAGE["snapshot"])
        files.download(str(WORK_DIR / "hf-storage-locator.json"))
        '''),
    ])


if __name__ == "__main__":
    main()
