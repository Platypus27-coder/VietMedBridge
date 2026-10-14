"""Generate the two-phase Colab notebook: all legacy data -> first ZIP -> strong ZIP."""
from pathlib import Path
from textwrap import dedent
import nbformat as nb


def code(value):
    return nb.v4.new_code_cell(dedent(value).strip()+"\n")


def notebook():
    cells = [nb.v4.new_markdown_cell(dedent('''
    # Submit từ toàn bộ dữ liệu đã xử lý — giữ URL cũ

    **Chạy trên Colab, chọn T4 GPU → Run all.** Notebook tự khôi phục checkpoint Drive,
    tiếp tục phần chuẩn bị CPU còn thiếu, rồi mới nạp model để inference.
    Có thể chuẩn bị trên CPU trước để tiết kiệm quota GPU: khi báo
    `CPU_PREPARATION_COMPLETE`, đổi sang T4 GPU và Run all cùng notebook.
    Đổi runtime mất biến/ổ tạm; các checkpoint đã publish trên Drive vẫn được giữ và tự nhận lại.
    Chỉ cần chỉnh `DATA_ROOT_PATH` nếu Drive của tài khoản đang chạy dùng đường dẫn khác.

    Giữ URL thuộc snapshot gốc, kể cả URL bị BTC loại trong bản dataset mới, theo yêu cầu
    của Sếp. Không xóa dữ liệu gốc, không crawl/extract/chunk lại. Các query và ID gắn
    đúng snapshot nguồn; lượt này **chưa xác nhận tương thích phạm vi BTC mới**.

    Trên GPU: bổ sung BGE còn thiếu → lưu ZIP BGE/BM25 đủ 1.200 query trước →
    dịch EN/ZH + PICO/subqueries/HyDE + document dense + Qwen rerank hai tầng +
    source parents → ZIP nâng cấp. Qwen child dense dùng khi cache đầy đủ;
    checkpoint Qwen chưa đủ vẫn được giữ. Đây chưa phải hoàn tất toàn bộ master plan.
    Mỗi model ≤15B. Chưa có nhãn độc lập để fine-tune/hiệu chỉnh hoặc cam kết điểm.

    Mỗi thời điểm chỉ **một phiên notebook này** ghi cùng DATA_ROOT. Các stage, vector,
    model input, query và ZIP lưu bền trên Drive. Nếu bị ngắt, Run all cùng notebook
    và cấu hình để tiếp tục checkpoint. Khi đã có ZIP baseline, có thể nộp thử trong
    lúc lượt strong còn đang chạy; ZIP mạnh hơn chỉ xuất sau khi đủ 1.200 query.
    ''').strip()), code('''
    #@title 1. Drive, code và môi trường (tự nhận CPU/GPU)
    import hashlib
    import importlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import subprocess
    import sys

    DATA_ROOT_PATH = "/content/drive/MyDrive/VietMedBridge/data" #@param {type:"string"}
    DATA_ROOT = Path(DATA_ROOT_PATH)
    WORK_DIR = Path("/content/vmb_submit_work")
    CHECKOUT = Path("/content/VietMedBridge-submit")
    REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
    WORKFLOW_VERSION = "all-available-legacy-20261014-v3"
    os.environ["HF_HOME"] = "/content/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    WORK_DIR.mkdir(parents=True,exist_ok=True)

    if not Path("/var/colab/hostname").exists():
        raise RuntimeError("Notebook này chạy trên Google Colab. Kaggle dùng notebook embedding worker riêng.")
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    try:
        (DATA_ROOT / "active_data_candidate.json").read_bytes()
    except OSError as error:
        if error.errno != 107:
            raise
        print("Drive preflight: remount trước khi đọc code lock hoặc nạp model.",flush=True)
        drive.mount("/content/drive",force_remount=True)
        (DATA_ROOT / "active_data_candidate.json").read_bytes()
    if not (DATA_ROOT / "active_data_candidate.json").is_file():
        raise RuntimeError("DATA_ROOT chưa trỏ tới data đã freeze. Chỉnh DATA_ROOT ở đầu cell, không tạo data rỗng.")
    print("DATA_ROOT:", DATA_ROOT,flush=True)
    # A lock for this workflow, separate from old notebook/data-processing locks.
    code_lock = DATA_ROOT / "retrieval" / (WORKFLOW_VERSION+"-code-lock.json")
    locked = json.loads(code_lock.read_text()) if code_lock.exists() else None
    if locked and (locked.get("repo") != REPO_URL or not re.fullmatch(r"[0-9a-f]{40}",locked.get("commit",""))):
        raise ValueError("Submission code lock không hợp lệ.")
    reference = locked["commit"] if locked else "origin/main"
    def command(args):
        print("Setup:"," ".join(map(str,args)),flush=True)
        subprocess.run(list(map(str,args)),check=True)
    if not (CHECKOUT / ".git").is_dir():
        command(["git","clone",REPO_URL,CHECKOUT])
    command(["git","-C",CHECKOUT,"fetch","origin"])
    command(["git","-C",CHECKOUT,"checkout","--detach",reference])
    COMMIT = subprocess.check_output(["git","-C",str(CHECKOUT),"rev-parse","HEAD"],text=True).strip()
    if not (CHECKOUT / "src/vietmedbridge/submission_workflow.py").is_file():
        raise RuntimeError("Checkout chưa có workflow submission mới. Dùng notebook cùng bản code đã push.")
    # Detect the physical GPU before importing torch/transformers.
    GPU_SESSION = False
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True)
        GPU_SESSION = probe.returncode == 0 and bool(probe.stdout.strip())
        print("Physical GPUs:",probe.stdout.strip() or "none",flush=True)
    runtime = None
    runtime_lock = DATA_ROOT / "retrieval_embedding_runtime_lock.json"
    if GPU_SESSION:
        if not runtime_lock.is_file():
            raise RuntimeError("Thiếu runtime lock của vector đã nhập. Khôi phục retrieval_embedding_runtime_lock.json trước khi inference.")
        lock = json.loads(runtime_lock.read_text())
        payload = {k:v for k,v in lock.items() if k != "manifest_sha256"}
        checksum = hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        if checksum != lock.get("manifest_sha256"):
            raise ValueError("Embedding runtime lock đã thay đổi.")
        runtime = lock["runtime"]
        if (set(runtime) != {"torch","transformers","bitsandbytes"}
            or any(not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*",v) for v in runtime.values())
            or not re.search(r"\\+cu[0-9]+$",runtime["torch"])):
            raise ValueError("Runtime lock cần CUDA torch, transformers và bitsandbytes hợp lệ.")
        for name,expected in lock["model_code"].items():
            if Path(name).name != name or hashlib.sha256((CHECKOUT / "src/vietmedbridge" / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Model producer code khác cache đã tính; dừng trước khi nạp model.")
        if "torch" in sys.modules and sys.modules["torch"].__version__ != runtime["torch"]:
            raise RuntimeError("Runtime đang giữ torch phiên bản cũ. Restart session rồi Run all; giữ nguyên Drive/checkpoint.")
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
        if torch_version != runtime["torch"]:
            suffix = re.search(r"\\+(cu[0-9]+)$",runtime["torch"])[1]
            command([sys.executable,"-m","pip","install","torch=="+runtime["torch"],"--index-url","https://download.pytorch.org/whl/"+suffix])
            command([sys.executable,"-m","pip","uninstall","-y","torchvision","torchaudio"])
        constraints = WORK_DIR / "runtime-constraints.txt"
        constraints.write_text("".join(f"{name}=={version}\\n" for name,version in runtime.items()))
        command([sys.executable,"-m","pip","install","-c",constraints,"-e",str(CHECKOUT)+"[retrieval,strong]"])
    else:
        command([sys.executable,"-m","pip","install","-e",CHECKOUT,
            "numpy>=1.26,<3","faiss-cpu==1.12.0","pyvi==0.1.1","jieba==0.42.1",
            "scikit-learn==1.7.2","sklearn-crfsuite==0.5.0"])
    sys.path.insert(0,str(CHECKOUT / "src"))
    import vietmedbridge
    if Path(vietmedbridge.__file__).resolve() != (CHECKOUT / "src/vietmedbridge/__init__.py").resolve():
        raise RuntimeError("Kernel đang giữ package của checkout khác. Restart session rồi Run all.")
    importlib.invalidate_caches()
    for module_name in ("vietmedbridge.index_inputs","vietmedbridge.official_scope","vietmedbridge.submission_workflow"):
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    from vietmedbridge.artifacts import atomic_json, read_json, sha256_file, verify_file
    if not locked:
        atomic_json(code_lock,{"repo":REPO_URL,"commit":COMMIT,"workflow":WORKFLOW_VERSION})
    try:
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass # Public models do not require a token.
    if GPU_SESSION:
        import torch
        if not torch.cuda.is_available() or torch.__version__ != runtime["torch"]:
            raise RuntimeError("CUDA torch chưa hoạt động. Restart session rồi Run all sau khi setup đã cài xong.")
        print("CUDA:",torch.__version__,torch.cuda.get_device_name(0),flush=True)
    print("Code:",COMMIT,"| package:",vietmedbridge.__file__,flush=True)
    print("Phase:","GPU inference" if GPU_SESSION else "CPU corpus/catalog preparation",flush=True)
    '''), code('''
    #@title 2. Ghép toàn bộ build đã xử lý, giữ URL cũ (CPU)
    from vietmedbridge.submission_workflow import ensure_submission_preparation
    PREPARATION = ensure_submission_preparation(DATA_ROOT,CHECKOUT,WORK_DIR,gpu_session=GPU_SESSION)
    print(json.dumps(PREPARATION,ensure_ascii=False,indent=2),flush=True)
    if not GPU_SESSION:
        print("CPU xong: đổi runtime sang T4 GPU rồi Run all cùng notebook. Không cần đổi worker_id hoặc ACTION.",flush=True)
    '''), code('''
    #@title 3. BGE phần còn thiếu và ZIP đầu tiên đủ 1.200 query (GPU)
    from vietmedbridge.submission_workflow import run_current_baseline
    BASELINE = None
    if GPU_SESSION:
        BASELINE = run_current_baseline(DATA_ROOT,CHECKOUT,WORK_DIR,code_commit=COMMIT)
        print(json.dumps(BASELINE["ready"],ensure_ascii=False,indent=2),flush=True)
        if BASELINE["ready"]["state"] != "READY_FOR_MANUAL_UPLOAD":
            raise RuntimeError("Baseline chưa đủ query/index: Run all cùng cấu hình để resume. Chưa nạp Qwen reranker.")
        print("ĐÃ CÓ ZIP ĐẦU TIÊN trên Drive. Cell cuối có thể tải bản này trong lúc chuẩn bị bản mạnh hơn.",flush=True)
    else:
        print("GPU phase skipped on CPU.")
    '''), code('''
    #@title 4. Bản nâng cấp: dịch + expansion + document dense + Qwen rerank + ZIP (GPU)
    from vietmedbridge.submission_workflow import run_current_submission
    STRONG = None
    if GPU_SESSION:
        STRONG = run_current_submission(DATA_ROOT,CHECKOUT,WORK_DIR,code_commit=COMMIT)
        print(json.dumps(STRONG["ready"],ensure_ascii=False,indent=2),flush=True)
        print("Diagnostics:",STRONG["diagnostics_path"],flush=True)
        if STRONG["ready"]["state"] != "READY_FOR_MANUAL_UPLOAD":
            print("Bản strong chưa hoàn tất; baseline ZIP vẫn còn trên Drive. Run all cùng notebook để tiếp tục checkpoint.",flush=True)
    else:
        print("GPU phase skipped on CPU.")
    '''), code('''
    #@title 5. Tải ZIP mới nhất đã kiểm tra (chạy cell này riêng cũng được)
    from google.colab import files
    latest = DATA_ROOT / "retrieval/latest_submission.json"
    if latest.is_file():
        submission = read_json(latest)
        if submission["candidate_manifest_sha256"] != PREPARATION["candidate_manifest_sha256"]:
            raise ValueError("ZIP thuộc candidate khác; chưa tải nhầm kết quả cũ.")
        verify_file(submission["zip_path"],submission["zip_sha256"])
        print(json.dumps(submission,ensure_ascii=False,indent=2),flush=True)
        print("Upload submission.zip lên cổng BTC. URL cũ vẫn được giữ; chưa có official score cho lượt này.",flush=True)
        files.download(submission["zip_path"])
    else:
        print("Chưa có ZIP hoàn tất. CPU preparation không phải submission; tiếp tục pha GPU.",flush=True)
    ''')]
    return nb.v4.new_notebook(cells=cells,metadata={"colab":{"name":"04_colab_submit_available_data.ipynb"},
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"}})


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "notebooks/04_colab_submit_available_data.ipynb"
    nb.write(notebook(),target)
    print(target)
