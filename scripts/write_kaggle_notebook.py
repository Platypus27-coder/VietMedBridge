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
           và GPU **T4 x2**, rồi **Save Version → Save & Run All**. Mặc định một tài
           khoản chạy cả hai GPU: `KAGGLE_ACCOUNTS=1`, `KAGGLE_ACCOUNT_ID=0`,
           `GPUS_PER_ACCOUNT=2`. Notebook tự gán hai worker 0–1.
        3. **Colab CPU:** sau khi Kaggle version hoàn tất và có Outputs, điền
           `KAGGLE_NOTEBOOK_OUTPUT="owner/notebook"`, Run all. Nếu nhập lượt cũ
           của hai tài khoản, ghi cả hai notebook, cách nhau dấu phẩy. Notebook
           kiểm trạng thái/manifest của Outputs trước khi tải vector, rồi kiểm
           checksum, model, source text và từng vector row trước khi nhập Drive.
           Nếu còn thiếu, tự xuất một dataset tiếp tục chỉ giao phần thiếu từ checkpoint
           Drive và chuyển sang một tài khoản; dùng dataset đó cho lượt Kaggle kế tiếp.
           Khi cả BGE và Qwen đủ,
           mới tiếp tục notebook 04 chính trên Colab GPU.

        Người nào cũng chạy được bằng Secret tài khoản của mình và quyền đọc Drive.
        Nếu dùng private dataset/output của người khác, chủ sở hữu cần cấp quyền
        truy cập trên Kaggle. Không chia sẻ API key cho nhau.
        Dataset continuation mới khóa một tài khoản và hai GPU. Không Add Input
        dataset cũ đã khóa hai tài khoản; Colab CPU xuất dataset mới từ checkpoint Drive.
        `RUNTIME_PLATFORM="auto"` nhận diện runtime đang chạy, không dựa vào thư mục
        `/kaggle/input`. Có thể chọn rõ `"colab"` hoặc `"kaggle"` tại đầu cell 1.
        Runtime GPU được khóa riêng, không lấy bản PyTorch CPU của máy xuất gói.
        Gói cũ khóa `+cpu` và chưa có seed vectors được chuyển sang CUDA ngay tại
        ổ tạm Kaggle, giữ nguyên input/checksum/commit/team. Không upload lại dataset.
        Kiểm tra CUDA và launcher chạy bằng tiến trình Python mới, tách khỏi
        torch đã import trong kernel notebook. Không hot-reload torch trong kernel.

        Phiên Kaggle tự dừng trước mốc 10 giờ tính từ cell đầu hoặc khi gần 18 GB
        output. Mỗi part hoàn tất có receipt/checksum. Đây là **checkpoint một phần**,
        chưa phải toàn bộ 700k đã embedding và chưa xuất submission. Giữ các model
        BGE-M3 + Qwen3-Embedding-8B đúng config của hệ thống, không đổi model cho nhanh.
        Chỉ các đoạn thiếu được encode; query LLM/reranker chạy ở coordinator sau.
        Một tài khoản giảm thao tác phối hợp nhưng không tăng tổng tốc độ GPU.
        Giới hạn output buộc lưu Outputs và nhập Drive sau mỗi lượt, rồi xuất
        dataset continuation mới cho lượt sau.

        `/kaggle/working` cần được lưu thành notebook version thành công mới lấy
        được từ Colab. Không cố chờ quota ngắt cứng. Có thể Add Input Outputs của
        phiên Kaggle trước để resume theo cách cũ; version mới giữ cả checkpoint trước đó.
        Với nhiều triệu văn bản, ưu tiên nhập Drive rồi dùng **continuation dataset**:
        nó bỏ qua phần đã có, không mang vector cũ vào Outputs và tránh đầy giới hạn
        dung lượng vì checkpoint tích lũy. Một phiên kết thúc không có nghĩa cả hai
        model đã xong; xem `account_coverage`, `worker_status` và coverage trên Drive.
        ''') ,
        md("## 1. Cấu hình và bootstrap — tự nhận Colab hoặc Kaggle"),
        code('''
        import importlib
        import json
        import os
        import re
        import signal
        import subprocess
        import sys
        import time
        from IPython import get_ipython
        from pathlib import Path

        SESSION_STARTED = globals().get("SESSION_STARTED", time.time())
        DATA_ROOT = Path("/content/drive/MyDrive/VietMedBridge/data")
        KAGGLE_NOTEBOOK_OUTPUT = ""  # Nhập về Drive: owner/notebook (lượt cũ hai acc: hai handles)
        KAGGLE_ACCOUNTS = 1  # Một tài khoản dùng cả hai GPU
        KAGGLE_ACCOUNT_ID = 0
        GPUS_PER_ACCOUNT = 2  # Chọn T4 x2
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
                    continuation_file = path.parent / "continuation-plan.json"
                    continuation_id = json.loads(continuation_file.read_text()).get("manifest_sha256") if continuation_file.is_file() else None
                    available[(saved["manifest_sha256"], continuation_id)] = path.parent
            if len(available) != 1:
                raise RuntimeError(f"Kaggle: tìm thấy {len(available)} input job/plan hợp lệ. Add Input đúng một dataset job VietMedBridge. Với continuation, bỏ dataset gốc và các plan cũ; chỉ giữ dataset tiếp tục mới nhất.")
            JOB_DIR = next(iter(available.values()))
            JOB_HEADER = json.loads((JOB_DIR / "job.json").read_text())
            if JOB_HEADER.get("repo_url") != REPO_URL or not re.fullmatch(r"[0-9a-f]{40}", JOB_HEADER.get("code_commit", "")):
                raise ValueError("Job chưa khóa đúng repo/commit.")
            if JOB_HEADER.get("team") != {"accounts": KAGGLE_ACCOUNTS, "gpu_count": GPUS_PER_ACCOUNT}:
                raise ValueError("Số tài khoản/GPU không khớp input chung. Xuất gói bằng notebook mới trên Colab CPU với cùng cấu hình team.")
            reference = JOB_HEADER["code_commit"]
            continuation_header = JOB_DIR / "continuation-plan.json"
            if continuation_header.is_file():
                header = json.loads(continuation_header.read_text())
                if (header.get("job") != JOB_HEADER["manifest_sha256"] or
                    not re.fullmatch(r"[0-9a-f]{40}", header.get("worker_code_commit", ""))):
                    raise ValueError("Continuation plan chưa khóa đúng job/worker commit.")
                # Transport may advance; validate_job still locks both model source hashes.
                reference = header["worker_code_commit"]
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
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-e",
                        ".[notebook,retrieval,strong]", "kagglehub>=0.3.13,<2", "kaggle>=1.8,<3"], cwd=CHECKOUT, check=True)
        sys.path.insert(0, str(CHECKOUT / "src"))
        importlib.invalidate_caches()
        _VMB_KAGGLE_COMMIT = CODE_COMMIT
        from vietmedbridge.portable_embeddings import (
            checked, export_kaggle_job, install_result,
            runtime_install_commands, validate_job, _copy_checked,
        )
        from vietmedbridge.artifacts import atomic_json, digest_json, read_json

        def fresh_kaggle_python(checkout, source, arguments, *, capture_output=False):
            # A new executable cannot inherit the notebook's imported CPU torch.
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(checkout) / "src") + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            command = [sys.executable, "-u", "-c", source, json.dumps(arguments)]
            process = subprocess.Popen(command, cwd=checkout, env=env, text=True,
                start_new_session=os.name == "posix",
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None)
            try:
                stdout, stderr = process.communicate()
            except BaseException:
                # Stop the launcher AND its GPU workers when the cell is interrupted.
                def stop_group(force):
                    try:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                        elif force:
                            process.kill()
                        else:
                            process.terminate()
                    except ProcessLookupError:
                        pass
                stop_group(False)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    stop_group(True)
                    process.wait(timeout=10)
                raise
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if result.returncode and capture_output:
                print(result.stdout or "", end="")
                print(result.stderr or "", end="", file=sys.stderr)
            result.check_returncode()
            return result

        def configure_kaggle_cuda(runtime, gpu_count, checkout):
            commands = runtime_install_commands(runtime)
            if commands:
                print("Installing locked GPU runtime; CUDA will run in fresh Python processes.", flush=True)
            for command in commands:
                subprocess.run(command, check=True)
            importlib.invalidate_caches()
            probe_source = (
                "import importlib.metadata, json, torch\\n"
                "runtime = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'bitsandbytes')}\\n"
                "available = torch.cuda.is_available()\\n"
                "count = torch.cuda.device_count()\\n"
                "devices = []\\n"
                "if available:\\n"
                "    for i in range(count):\\n"
                "        x = torch.ones(1, device=f'cuda:{i}').add_(1)\\n"
                "        assert x.item() == 2, 'CUDA kernel smoke check failed'\\n"
                "        devices.append(torch.cuda.get_device_name(i))\\n"
                "        del x\\n"
                "print(json.dumps({'runtime': runtime, 'cuda_build': torch.version.cuda, "
                "'cuda_available': available, 'gpu_count': count, 'devices': devices}))\\n"
            )
            process = fresh_kaggle_python(checkout, probe_source, {}, capture_output=True)
            probe = json.loads(process.stdout.strip().splitlines()[-1])
            print("PyTorch:", probe["runtime"]["torch"], "| CUDA build:", probe["cuda_build"],
                  "| CUDA available:", probe["cuda_available"], "| GPUs:", probe["gpu_count"], flush=True)
            if probe["runtime"] != runtime:
                raise RuntimeError("GPU subprocess runtime không khớp job sau khi cài đặt.")
            if not probe["cuda_available"] or probe["gpu_count"] != gpu_count:
                raise RuntimeError("GPU subprocess chưa thấy đủ GPU. Chọn T4 x2; không giảm số worker.")
            return probe

        def launch_kaggle_fresh(job_dir, checkout, *, output, scratch, session_started, hours,
                                accounts, account_id, gpu_count):
            arguments = {"job_dir": str(job_dir), "checkout": str(checkout), "output": str(output),
                "scratch": str(scratch), "session_started": session_started, "hours": hours,
                "accounts": accounts, "account_id": account_id, "gpu_count": gpu_count}
            source = (
                "import json, sys\\n"
                "from vietmedbridge.portable_embeddings import launch_kaggle\\n"
                "launch_kaggle(**json.loads(sys.argv[1]))\\n"
            )
            fresh_kaggle_python(checkout, source, arguments)
            result = checked(read_json(Path(output) / "result-manifest.json"))
            if result["job"] != read_json(Path(job_dir) / "job.json")["manifest_sha256"]:
                raise ValueError("GPU subprocess output belongs to another job.")
            return result

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
            validate_job(JOB_HEADER, CHECKOUT)
            if (JOB_DIR / "continuation-plan.json").is_file():
                from vietmedbridge.kaggle_continuation import assignment_progress, load_plan
                from vietmedbridge.portable_embeddings import kaggle_assignment
                units_entry = next(entry for entry in JOB_HEADER["files"]
                    if entry["path"] == f"processed/{JOB_HEADER['build_run']}/index_inputs/units.json")
                JOB_INPUTS = checked(read_json(JOB_DIR / units_entry["asset"]))
                PLAN = load_plan(JOB_DIR, JOB_HEADER, JOB_INPUTS)
                assignment = kaggle_assignment(accounts=KAGGLE_ACCOUNTS, account_id=KAGGLE_ACCOUNT_ID,
                                               gpu_count=GPUS_PER_ACCOUNT)
                print("Remaining assignment:", json.dumps(assignment_progress(JOB_INPUTS, assignment, PLAN, [])))
            CUDA_PROBE = configure_kaggle_cuda(JOB_HEADER["runtime"], GPUS_PER_ACCOUNT, CHECKOUT)
            if JOB_HEADER.get("runtime_repair"):
                print("CPU export repaired locally → CUDA; original dataset and all input checksums preserved.")
            print("Kaggle job:", JOB_HEADER["manifest_sha256"], "| inputs:", JOB_HEADER["input_count"])
        print("Platform:", "Kaggle" if IS_KAGGLE else "Colab CPU", "| code:", CODE_COMMIT)
        '''.replace("__CUDA_RUNTIME__", repr(cuda_runtime))),
        md("## 2. Xuất dữ liệu / chạy hai GPU / nhập checkpoint — tự chọn theo nền tảng"),
        code('''
        if IS_KAGGLE:
            RESULT = launch_kaggle_fresh(JOB_DIR, CHECKOUT, output=Path("/kaggle/working/vmb_checkpoints"),
                                  scratch=WORK_DIR / "data", session_started=SESSION_STARTED, hours=SESSION_HOURS,
                                  accounts=KAGGLE_ACCOUNTS, account_id=KAGGLE_ACCOUNT_ID, gpu_count=GPUS_PER_ACCOUNT)
            print(json.dumps({k: RESULT[k] for k in ("state", "processes", "assignment", "updated_at", "scope")}
                             | {"completed_parts": len(RESULT["receipts"]),
                                "account_coverage": RESULT.get("account_coverage"),
                                "worker_status": RESULT.get("worker_status")}, ensure_ascii=False, indent=2))
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
            from vietmedbridge.kaggle_continuation import find_imported_job, upload_continuation
            if KAGGLE_NOTEBOOK_OUTPUT:
                from kaggle.api.kaggle_api_extended import KaggleApi
                from vietmedbridge.kaggle_outputs import download_kaggle_outputs, output_handles
                handles = output_handles(KAGGLE_NOTEBOOK_OUTPUT)
                api = KaggleApi()
                api.authenticate()
                outputs = download_kaggle_outputs(api, handles, WORK_DIR)
                if len({read_json(path / "job.json")["manifest_sha256"] for path in outputs}) != 1:
                    raise ValueError("Các Outputs thuộc các input job khác nhau. Nhập mỗi job theo lượt riêng.")
                account_ids = [read_json(path / "result-manifest.json")["assignment"]["account_id"]
                               for path in outputs if read_json(path / "result-manifest.json").get("assignment") is not None]
                if len(set(account_ids)) != len(account_ids):
                    raise ValueError("Các Outputs trùng KAGGLE_ACCOUNT_ID. Chọn một version mới nhất cho mỗi tài khoản.")
                for handle, path in zip(handles, outputs, strict=True):
                    print("Nhập Outputs:", handle)
                    report = install_result(path, DATA_ROOT, CHECKOUT)
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                if not report["corpus_embeddings_complete"]:
                    upload_continuation(DATA_ROOT, CHECKOUT, WORK_DIR / "continuations",
                        read_json(outputs[0] / "job.json"), owner=owner, worker_code_commit=CODE_COMMIT,
                        accounts=KAGGLE_ACCOUNTS, gpu_count=GPUS_PER_ACCOUNT)
                else:
                    print("Cả BGE và Qwen đã đủ. Tiếp tục notebook 04 chính; giữ checkpoint đã nhập.")
            elif (previous_job := find_imported_job(DATA_ROOT, CHECKOUT,
                    accounts=KAGGLE_ACCOUNTS, gpu_count=GPUS_PER_ACCOUNT)) is not None:
                from vietmedbridge.portable_embeddings import embedding_coverage
                if all(value["complete"] for value in embedding_coverage(DATA_ROOT, previous_job).values()):
                    print("Cả BGE và Qwen đã đủ. Tiếp tục notebook 04 chính; không xuất lại embedding job.")
                else:
                    upload_continuation(DATA_ROOT, CHECKOUT, WORK_DIR / "continuations",
                        previous_job, owner=owner, worker_code_commit=CODE_COMMIT,
                        accounts=KAGGLE_ACCOUNTS, gpu_count=GPUS_PER_ACCOUNT)
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
                print("Cấu hình Kaggle:", job["team"], "| KAGGLE_ACCOUNT_ID=0.")
                print("Kaggle: import notebook này → Add Input dataset trên → Internet ON → T4 x2 → Save Version / Save & Run All.")
        '''),
    ])


if __name__ == "__main__":
    main()
