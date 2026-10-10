"""Generate one account-independent Colab/Kaggle transfer and GPU notebook."""
import json
from pathlib import Path

from write_notebooks import code, md, save


def main():
    cuda_runtime = json.loads((Path(__file__).resolve().parents[1] / "configs/kaggle_runtime.json").read_text())
    save("04_kaggle_embedding_worker.ipynb", [
        md('''
        # 04 — Chuyển embedding sang Kaggle, rồi nhập lại Drive

        **Trước tiên:** notebook 04 chính chạy CPU đến `CPU_PREPARATION_COMPLETE`.
        Không đổi hoặc dừng phiên CPU đang chạy để mở notebook này.

        Cùng một notebook này dùng ở hai nơi:

        1. **Colab CPU:** giữ `KAGGLE_NOTEBOOK_OUTPUT=""`, Run all để xuất model inputs
           và cache hợp lệ từ Drive thành một **private Kaggle Dataset**. Chỉ cần
           thêm Colab Secret `KAGGLE_JSON`: nội dung file `kaggle.json` lấy từ
           Kaggle Settings → API → Legacy API Credentials. Bật quyền notebook đọc
           Secret. Username tự lấy từ Secret của người chạy; không gửi token vào chat.
        2. **Kaggle:** import notebook này, giữ Private, Add Input dataset vừa xuất, bật Internet
           và GPU **T4 x2**, rồi **Save Version → Save & Run All**. Chạy hai tài khoản:
           cùng `KAGGLE_ACCOUNTS=2`, `GPUS_PER_ACCOUNT=2`; tài khoản thứ nhất đặt
           `KAGGLE_ACCOUNT_ID=0`, tài khoản thứ hai đặt `KAGGLE_ACCOUNT_ID=1`.
           Notebook tự gán worker toàn nhóm: 0–1 và 2–3. Hai bên chạy cùng lúc.
        3. **Colab CPU:** sau khi Kaggle version hoàn tất và có Outputs, điền
           `KAGGLE_NOTEBOOK_OUTPUT="owner-a/notebook-a, owner-b/notebook-b"`, Run all. Notebook
           kiểm checksum, model, source text và từng vector row trước khi nhập Drive.
           Sau đó mới tiếp tục notebook 04 chính trên Colab GPU.

        Người nào cũng chạy được bằng Secret tài khoản của mình và quyền đọc Drive.
        Nếu dùng private dataset/output của người khác, chủ sở hữu cần cấp quyền
        truy cập trên Kaggle. Không chia sẻ API key cho nhau.
        Xuất **một dataset chung** rồi cấp quyền đọc cho tài khoản thứ hai; cả hai
        Add Input đúng dataset đó. Gói khóa số tài khoản/GPU để không chia lệch.
        Chạy một tài khoản thì đặt `KAGGLE_ACCOUNTS=1` trước khi xuất gói.
        `RUNTIME_PLATFORM="auto"` nhận diện runtime đang chạy, không dựa vào thư mục
        `/kaggle/input`. Có thể chọn rõ `"colab"` hoặc `"kaggle"` tại đầu cell 1.
        Runtime GPU được khóa riêng, không lấy bản PyTorch CPU của máy xuất gói.
        Gói cũ khóa `+cpu` và chưa có seed vectors được chuyển sang CUDA ngay tại
        ổ tạm Kaggle, giữ nguyên input/checksum/commit/team. Không upload lại dataset.
        Nếu phiên đã import torch CPU, mở phiên mới bằng Save & Run All.

        Phiên Kaggle tự dừng trước mốc 10 giờ tính từ cell đầu hoặc khi gần 18 GB
        output. Mỗi part hoàn tất có receipt/checksum. Đây là **checkpoint một phần**,
        chưa phải toàn bộ 700k đã embedding và chưa xuất submission. Giữ các model
        BGE-M3 + Qwen3-Embedding-8B đúng config của hệ thống, không đổi model cho nhanh.
        Chỉ các đoạn thiếu được encode; query LLM/reranker chạy ở coordinator sau.

        `/kaggle/working` cần được lưu thành notebook version thành công mới lấy
        được từ Colab. Không cố chờ quota ngắt cứng. Có thể Add Input Outputs của
        phiên Kaggle trước để resume; version mới giữ cả checkpoint trước đó.
        ''') ,
        md("## 1. Cấu hình và bootstrap — tự nhận Colab hoặc Kaggle"),
        code('''
        import importlib
        import json
        import os
        import re
        import subprocess
        import sys
        import time
        from IPython import get_ipython
        from pathlib import Path

        SESSION_STARTED = globals().get("SESSION_STARTED", time.time())
        DATA_ROOT = Path("/content/drive/MyDrive/VietMedBridge/data")
        KAGGLE_NOTEBOOK_OUTPUT = ""  # Nhập về Drive: owner-a/notebook-a, owner-b/notebook-b
        KAGGLE_ACCOUNTS = 2  # Giống nhau trên cả hai tài khoản và lúc xuất input
        KAGGLE_ACCOUNT_ID = 0  # Tài khoản thứ nhất: 0; tài khoản thứ hai: 1
        GPUS_PER_ACCOUNT = 2  # Chọn T4 x2 ở cả hai tài khoản
        SESSION_HOURS = 10.0
        RUNTIME_PLATFORM = "auto"  # "auto", "colab" hoặc "kaggle"
        REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"

        def resolve_runtime_platform(requested):
            if requested not in {"auto", "colab", "kaggle"}:
                raise ValueError('RUNTIME_PLATFORM phải là "auto", "colab" hoặc "kaggle".')
            if requested != "auto":
                return requested
            # A downloaded dataset may create /kaggle/input outside Kaggle.
            shell_module = type(get_ipython()).__module__
            if shell_module.startswith("google.colab") or os.environ.get("COLAB_RELEASE_TAG"):
                return "colab"
            if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
                return "kaggle"
            raise RuntimeError('Không nhận diện được runtime. Đặt RUNTIME_PLATFORM="colab" hoặc "kaggle" ở đầu cell 1.')

        IS_KAGGLE = resolve_runtime_platform(RUNTIME_PLATFORM) == "kaggle"
        print("Platform:", "Kaggle" if IS_KAGGLE else "Colab CPU")
        CHECKOUT = Path("/kaggle/temp/VietMedBridge") if IS_KAGGLE else Path("/content/VietMedBridge-kaggle")
        WORK_DIR = Path("/kaggle/temp/vmb") if IS_KAGGLE else Path("/content/vmb_kaggle")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if not (3, 11) <= sys.version_info[:2] < (3, 14):
            raise RuntimeError("Cần Python 3.11–3.13.")
        if (type(KAGGLE_ACCOUNTS) is not int or KAGGLE_ACCOUNTS < 1
            or type(KAGGLE_ACCOUNT_ID) is not int or not 0 <= KAGGLE_ACCOUNT_ID < KAGGLE_ACCOUNTS
            or type(GPUS_PER_ACCOUNT) is not int or GPUS_PER_ACCOUNT < 1):
            raise ValueError("Kiểm tra KAGGLE_ACCOUNTS, KAGGLE_ACCOUNT_ID và GPUS_PER_ACCOUNT.")

        if IS_KAGGLE:
            available = {}
            for path in sorted(Path("/kaggle/input").glob("**/job.json")):
                saved = json.loads(path.read_text())
                if saved.get("schema") == "portable-embeddings-v1" and all(
                    (path.parent / entry["asset"]).is_file() for entry in saved.get("files", [])
                ) and saved.get("files"):
                    available[saved["manifest_sha256"]] = path.parent
            if len(available) != 1:
                raise RuntimeError(f"Kaggle: tìm thấy {len(available)} input job hợp lệ. Add Input đúng một dataset job VietMedBridge; có thể thêm Outputs cũ để resume.")
            JOB_DIR = next(iter(available.values()))
            JOB_HEADER = json.loads((JOB_DIR / "job.json").read_text())
            if JOB_HEADER.get("repo_url") != REPO_URL or not re.fullmatch(r"[0-9a-f]{40}", JOB_HEADER.get("code_commit", "")):
                raise ValueError("Job chưa khóa đúng repo/commit.")
            if JOB_HEADER.get("team") != {"accounts": KAGGLE_ACCOUNTS, "gpu_count": GPUS_PER_ACCOUNT}:
                raise ValueError("Số tài khoản/GPU không khớp input chung. Xuất gói bằng notebook mới trên Colab CPU với cùng cấu hình team.")
            reference = JOB_HEADER["code_commit"]
        else:
            from google.colab import drive
            drive.mount("/content/drive")
            if not (DATA_ROOT / "active_data_candidate.json").is_file():
                raise RuntimeError("DATA_ROOT chưa trỏ đúng data đã chuẩn bị. Sửa duy nhất DATA_ROOT ở đầu cell này.")
            reference = "main"

        if not (CHECKOUT / ".git").exists():
            subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CHECKOUT)], check=True)
        if subprocess.check_output(["git", "-C", str(CHECKOUT), "remote", "get-url", "origin"], text=True).strip() != REPO_URL:
            raise ValueError("Checkout thuộc repo khác.")
        subprocess.run(["git", "-C", str(CHECKOUT), "fetch", "--depth", "1", "origin", reference], check=True)
        CODE_COMMIT = subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "FETCH_HEAD"], text=True).strip()
        if "vietmedbridge" in sys.modules and globals().get("_VMB_KAGGLE_COMMIT") != CODE_COMMIT:
            raise RuntimeError("Restart session vì code đã thay đổi; không trộn package đang import.")
        subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "--detach", "FETCH_HEAD"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-e",
                        ".[notebook,retrieval,strong]", "kagglehub>=0.3.13,<2", "kaggle>=1.8,<3"], cwd=CHECKOUT, check=True)
        sys.path.insert(0, str(CHECKOUT / "src"))
        importlib.invalidate_caches()
        _VMB_KAGGLE_COMMIT = CODE_COMMIT
        from vietmedbridge.portable_embeddings import (
            checked, embedding_runtime, export_kaggle_job, install_result, launch_kaggle,
            runtime_install_commands, validate_job, _copy_checked,
        )
        from vietmedbridge.artifacts import atomic_json, digest_json, read_json

        def repair_cpu_export_job(job_dir, checkout, work_dir):
            # Keep compatibility with the worker commit pinned in existing datasets.
            source = Path(job_dir)
            original = validate_job(read_json(source / "job.json"), checkout)
            if not original["runtime"]["torch"].endswith("+cpu"):
                return source, original
            if original.get("seeds") or any(entry["path"].startswith("model_cache/") for entry in original["files"]):
                raise ValueError("Gói CPU có seed vectors: không được đổi runtime của vectors đã tạo. Cần xuất gói CUDA riêng.")
            target_runtime = __CUDA_RUNTIME__
            runtime_install_commands(target_runtime)  # validate before copying any files
            payload = {key: value for key, value in original.items() if key != "manifest_sha256"}
            payload["runtime"] = target_runtime
            payload["runtime_repair"] = {"policy": "cpu-export-to-cuda-v1",
                "source_job": original["manifest_sha256"], "source_runtime": original["runtime"]}
            repaired = {**payload, "manifest_sha256": digest_json(payload)}
            validate_job(repaired, checkout)
            destination = Path(work_dir) / ("cuda-job-" + repaired["manifest_sha256"])
            header = destination / "job.json"
            if header.exists() and read_json(header) != repaired:
                raise ValueError("Local CUDA job differs; restart session with the original input dataset.")
            for entry in repaired["files"]:
                _copy_checked(source / entry["asset"], destination / entry["asset"], entry["sha256"])
            atomic_json(header, repaired)  # publish only after every original payload verifies
            return destination, repaired

        if IS_KAGGLE:
            JOB_DIR, JOB_HEADER = repair_cpu_export_job(JOB_DIR, CHECKOUT, WORK_DIR)
            commands = runtime_install_commands(JOB_HEADER["runtime"])
            if commands and "torch" in sys.modules:
                raise RuntimeError("Restart session trước khi đổi torch. Save & Run All từ runtime mới.")
            for command in commands:
                subprocess.run(command, check=True)
            importlib.invalidate_caches()
            import torch
            print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda,
                  "| CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
            if embedding_runtime() != JOB_HEADER["runtime"]:
                raise RuntimeError("Runtime sau cài đặt không khớp job. Save & Run All từ phiên mới.")
            if not torch.cuda.is_available() or torch.cuda.device_count() != GPUS_PER_ACCOUNT:
                raise RuntimeError("PyTorch chưa dùng đủ GPU yêu cầu. Chọn T4 x2 và Save & Run All từ phiên mới; không giảm số worker.")
            if JOB_HEADER.get("runtime_repair"):
                print("CPU export repaired locally → CUDA; original dataset and all input checksums preserved.")
            print("Kaggle job:", JOB_HEADER["manifest_sha256"], "| inputs:", JOB_HEADER["input_count"])
        print("Platform:", "Kaggle" if IS_KAGGLE else "Colab CPU", "| code:", CODE_COMMIT)
        '''.replace("__CUDA_RUNTIME__", repr(cuda_runtime))),
        md("## 2. Xuất dữ liệu / chạy hai GPU / nhập checkpoint — tự chọn theo nền tảng"),
        code('''
        if IS_KAGGLE:
            RESULT = launch_kaggle(JOB_DIR, CHECKOUT, output=Path("/kaggle/working/vmb_checkpoints"),
                                  scratch=WORK_DIR / "data", session_started=SESSION_STARTED, hours=SESSION_HOURS,
                                  accounts=KAGGLE_ACCOUNTS, account_id=KAGGLE_ACCOUNT_ID, gpu_count=GPUS_PER_ACCOUNT)
            print(json.dumps({k: RESULT[k] for k in ("state", "processes", "assignment", "updated_at", "scope")}
                             | {"completed_parts": len(RESULT["receipts"])}, ensure_ascii=False, indent=2))
            print("Chờ Save & Run All hoàn tất để Outputs được lưu. Sau đó nhập Outputs về Drive bằng Colab CPU.")
        else:
            from google.colab import userdata
            try:
                credentials = json.loads(userdata.get("KAGGLE_JSON"))
            except Exception:
                raise RuntimeError("Thêm Colab Secret KAGGLE_JSON chứa nội dung kaggle.json, bật Notebook access, rồi chạy lại.") from None
            owner, api_key = credentials.get("username"), credentials.get("key")
            if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", owner) or not api_key:
                raise ValueError("KAGGLE_JSON phải có username và key hợp lệ.")
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = owner, api_key
            del credentials, api_key
            if KAGGLE_NOTEBOOK_OUTPUT:
                handles = [h.strip() for h in KAGGLE_NOTEBOOK_OUTPUT.split(",")]
                if len(set(handles)) != len(handles) or any(not re.fullmatch(
                    r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[0-9]+)?", handle) for handle in handles):
                    raise ValueError("Điền các username/notebook-slug khác nhau, cách nhau bởi dấu phẩy; có thể thêm /version.")
                outputs = []
                for handle in handles:
                    download = WORK_DIR / "outputs" / digest_json(handle)[:16]
                    download.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["kaggle", "kernels", "output", handle, "-p", str(download), "-o"], check=True)
                    manifests = list(download.glob("**/result-manifest.json"))
                    if len(manifests) != 1:
                        raise RuntimeError("Mỗi Outputs phải có đúng một result-manifest.json; kiểm tra version đã lưu thành công.")
                    outputs.append(manifests[0].parent)
                if len({read_json(path / "job.json")["manifest_sha256"] for path in outputs}) != 1:
                    raise ValueError("Hai Outputs thuộc hai input job khác nhau. Chạy cùng dataset chung.")
                account_ids = [read_json(path / "result-manifest.json")["assignment"]["account_id"]
                               for path in outputs if read_json(path / "result-manifest.json").get("assignment") is not None]
                if len(set(account_ids)) != len(account_ids):
                    raise ValueError("Các Outputs trùng KAGGLE_ACCOUNT_ID. Chọn một version mới nhất cho mỗi tài khoản; hai tài khoản phải là 0 và 1.")
                for handle, path in zip(handles, outputs, strict=True):
                    print("Nhập Outputs:", handle)
                    print(json.dumps(install_result(path, DATA_ROOT, CHECKOUT), ensure_ascii=False, indent=2))
                print("Đã nhập checkpoint. Các worker Colab 04 dùng lại parts này; dừng mọi worker Kaggle trước khi chia lại TEAM_SIZE.")
            else:
                from vietmedbridge.portable_embeddings import cuda_embedding_runtime
                runtime_lock = DATA_ROOT / "retrieval_embedding_runtime_lock.json"
                runtime = cuda_embedding_runtime(CHECKOUT,
                    checked(read_json(runtime_lock))["runtime"] if runtime_lock.exists() else None)
                destination = WORK_DIR / ("inputs-" + digest_json([CODE_COMMIT, runtime, KAGGLE_ACCOUNTS, GPUS_PER_ACCOUNT,
                                            read_json(DATA_ROOT / "active_data_candidate.json")])[:16])
                job = export_kaggle_job(DATA_ROOT, CHECKOUT, destination, code_commit=CODE_COMMIT, runtime=runtime,
                                        accounts=KAGGLE_ACCOUNTS, gpu_count=GPUS_PER_ACCOUNT)
                print("Input files:", len(job["files"]), "| GB:", round(sum(f["bytes"] for f in job["files"])/1e9, 2),
                      "| seed blocks:", len(job["seeds"]))
                import kagglehub
                pointer = DATA_ROOT / "retrieval/kaggle_exports" / (owner + "-" + job["manifest_sha256"][:16] + ".json")
                if pointer.exists():
                    handle = read_json(pointer)["dataset_handle"]
                else:
                    # A new unique handle: never update an unrelated/public dataset.
                    handle = owner + "/vmb-embed-" + job["manifest_sha256"][:12] + "-" + str(time.time_ns())
                    kagglehub.dataset_upload(handle, str(destination))
                    atomic_json(pointer, {"dataset_handle": handle, "job": job["manifest_sha256"],
                                          "code_commit": CODE_COMMIT, "state": "UPLOADED_VERIFY_PENDING"})
                for attempt in range(6):
                    try:
                        remote_job = Path(kagglehub.dataset_download(handle, path="job.json", force_download=True))
                        break
                    except Exception:
                        if attempt == 5:
                            raise RuntimeError("Kaggle chưa cho tải job.json. Run all lại để kiểm dataset đã upload, không upload bản mới.") from None
                        time.sleep(min(30, 5 * (attempt + 1)))
                if read_json(remote_job) != job:
                    raise ValueError("Uploaded Kaggle job differs; do not run it.")
                atomic_json(pointer, {"dataset_handle": handle, "job": job["manifest_sha256"], "code_commit": CODE_COMMIT})
                print("Private dataset đã kiểm:", "https://www.kaggle.com/datasets/" + handle)
                print("Cấu hình chung:", job["team"], "| Cấp quyền đọc dataset này cho thành viên chạy tài khoản thứ hai.")
                print("Tài khoản thứ nhất KAGGLE_ACCOUNT_ID=0; tài khoản thứ hai KAGGLE_ACCOUNT_ID=1.")
                print("Kaggle: import notebook này → Add Input dataset trên → Internet ON → T4 x2 → Save Version / Save & Run All.")
        '''),
    ])


if __name__ == "__main__":
    main()
