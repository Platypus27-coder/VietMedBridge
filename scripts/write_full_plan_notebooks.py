"""Generate the full inference notebook and the supervised training notebook."""
import textwrap
from write_notebooks import BOOTSTRAP, code, md, save

BOOT = (BOOTSTRAP
    .replace("code_lock.json", "retrieval_code_lock.json")
    .replace("runtime.json", "retrieval_runtime.json")
    .replace('.[notebook]', '.[notebook,retrieval,strong]')
    .replace('"install", "-q", "-e"', '"install", "--disable-pip-version-check", "-e"')
    .replace("Giữ cùng DATA_ROOT trong cả bốn notebook.", "Giữ DATA_ROOT đã dùng ở notebook 00–03.")
    .replace('reference = CODE_REVISION or lock.get("git_commit") or "main"',
        '''RETRIEVAL_WORKFLOW_API = "full-master-plan-strong-v3-per-model-15b"
upgrade = lock.get("workflow_api") != RETRIEVAL_WORKFLOW_API
reference = CODE_REVISION or ("main" if upgrade else lock.get("git_commit")) or "main"''')
    .replace('if not lock or CODE_REVISION:\n    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, "pipeline_api": PIPELINE_API_VERSION})',
        'if not lock or CODE_REVISION or upgrade:\n'
        '    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, '
        '"pipeline_api": PIPELINE_API_VERSION, "workflow_api": RETRIEVAL_WORKFLOW_API})'))

# Only present after a verified Kaggle import. Pin producer versions so native
# Colab workers reuse imported content maps instead of silently changing identity.
BOOT = BOOT.replace('PIPELINE_CONFIG = json.loads', '''embedding_runtime_lock = DATA_ROOT / "retrieval_embedding_runtime_lock.json"
if embedding_runtime_lock.exists():
    from vietmedbridge.portable_embeddings import checked, runtime_install_commands
    pinned_runtime = checked(json.loads(embedding_runtime_lock.read_text()))["runtime"]
    install_commands = runtime_install_commands(pinned_runtime)
    if install_commands and "torch" in sys.modules:
        raise RuntimeError("Restart session trước khi cài runtime embedding đã khóa từ Kaggle.")
    for command in install_commands:
        subprocess.run(command, check=True)
    importlib.invalidate_caches()
PIPELINE_CONFIG = json.loads''')


HF_BOOT = '''
import importlib, json, os, subprocess, sys
from pathlib import Path
from google.colab import userdata

if not (3,11) <= sys.version_info[:2] < (3,14):
    raise RuntimeError("Dùng Colab Python 3.11–3.13.")
DATA_ROOT = Path("/content/vmb_data")  # dữ liệu/outputs ở ổ local, không dùng quota Drive
WORK_DIR = Path("/content/vmb_work")
RESULTS_ROOT = Path("/content/vmb_results")
WORK_DIR.mkdir(parents=True,exist_ok=True)
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    token = userdata.get("HF_TOKEN")
except Exception:
    raise RuntimeError("Thêm Colab Secret HF_TOKEN có quyền đọc repo checkpoints, bật Notebook access.") from None
if not token:
    raise RuntimeError("HF_TOKEN đang rỗng.")
os.environ["HF_TOKEN"] = token
REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
CODE_REVISION = None
EXPECTED_PIPELINE_API = 2
CHECKOUT = Path("/content/VietMedBridge")
if not (CHECKOUT / ".git").exists():
    subprocess.run(["git","clone","--no-checkout",REPO_URL,str(CHECKOUT)],check=True)
if subprocess.check_output(["git","-C",str(CHECKOUT),"remote","get-url","origin"],text=True).strip() != REPO_URL:
    raise ValueError("Checkout không thuộc VietMedBridge.")
subprocess.run(["git","-C",str(CHECKOUT),"fetch","--depth","1","origin",CODE_REVISION or "main"],check=True)
CODE_COMMIT = subprocess.check_output(["git","-C",str(CHECKOUT),"rev-parse","FETCH_HEAD"],text=True).strip()
if "vietmedbridge" in sys.modules and globals().get("_VMB_IMPORTED_COMMIT") != CODE_COMMIT:
    raise RuntimeError("Code đã đổi trong runtime; Restart session rồi chạy lại notebook.")
subprocess.run(["git","-C",str(CHECKOUT),"checkout","--detach","FETCH_HEAD"],check=True)
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","-e",".[notebook,retrieval,strong]"],cwd=CHECKOUT,check=True)
sys.path.insert(0,str(CHECKOUT / "src"))
importlib.invalidate_caches()
from huggingface_hub import HfApi
from vietmedbridge import PIPELINE_API_VERSION
from vietmedbridge.artifacts import atomic_json, runtime_versions
from vietmedbridge.hub_storage import restore_storage
if PIPELINE_API_VERSION != EXPECTED_PIPELINE_API:
    raise RuntimeError("Pipeline API khác phiên bản notebook.")
HF_STAGED = restore_storage(HF_STORAGE_REPO,data_root=DATA_ROOT,work_dir=WORK_DIR,
    profile=HF_STORAGE_PROFILE,revision=HF_STORAGE_REVISION,api=HfApi(token=token),checkout=CHECKOUT)
from vietmedbridge.portable_embeddings import checked, runtime_install_commands
embedding_runtime_lock = DATA_ROOT / "retrieval_embedding_runtime_lock.json"
if embedding_runtime_lock.exists():
    pinned_runtime = checked(json.loads(embedding_runtime_lock.read_text()))["runtime"]
    commands = runtime_install_commands(pinned_runtime)
    if commands and "torch" in sys.modules:
        raise RuntimeError("Restart session trước khi cài runtime embedding đã khóa từ Kaggle.")
    for command in commands:
        subprocess.run(command,check=True)
    importlib.invalidate_caches()
PIPELINE_CONFIG = json.loads((CHECKOUT / "configs/data_pipeline.json").read_text())
atomic_json(DATA_ROOT / "retrieval_code_lock.json",{"repo_url":REPO_URL,"git_commit":CODE_COMMIT,
    "pipeline_api":PIPELINE_API_VERSION,"workflow_api":"full-master-plan-strong-v19-ready-resume"})
atomic_json(DATA_ROOT / "retrieval_runtime.json",runtime_versions())
_VMB_IMPORTED_COMMIT = CODE_COMMIT
print("HF snapshot:",HF_STAGED["snapshot"],"| revision:",HF_STAGED["revision"])
print("DATA_ROOT:",DATA_ROOT,"| outputs ở ổ tạm Colab; tải ZIP trước khi kết thúc runtime.")
print("Code commit:",CODE_COMMIT)
'''


DRIVE_CACHED_BOOT = '''
import importlib, json, os, subprocess, sys
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive", readonly=True)
DATA_ROOT = Path(DRIVE_DATA_ROOT)
if not (DATA_ROOT / "active_data_candidate.json").is_file():
    shared = sorted(Path("/content/drive/.shortcut-targets-by-id").glob("*/VietMedBridge/data/active_data_candidate.json"))
    if DATA_ROOT == Path("/content/drive/MyDrive/VietMedBridge/data") and len(shared) == 1:
        DATA_ROOT = shared[0].parent
    else:
        raise RuntimeError(f"Không thấy active candidate tại {DATA_ROOT}. Điền DRIVE_DATA_ROOT đúng thư mục data đã xử lý.")
WORK_DIR = Path("/content/vmb_work")
RESULTS_ROOT = Path("/content/vmb_results")
WORK_DIR.mkdir(parents=True,exist_ok=True)
RESULTS_ROOT.mkdir(parents=True,exist_ok=True)
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
REPO_URL = "https://github.com/Platypus27-coder/VietMedBridge.git"
CODE_REVISION = None  # main mới nhất; không đổi code trong một runtime đang chạy
EXPECTED_PIPELINE_API = 2
CHECKOUT = Path("/content/VietMedBridge")
if not (CHECKOUT / ".git").exists():
    subprocess.run(["git","clone","--no-checkout",REPO_URL,str(CHECKOUT)],check=True)
if subprocess.check_output(["git","-C",str(CHECKOUT),"remote","get-url","origin"],text=True).strip() != REPO_URL:
    raise ValueError("Checkout không thuộc VietMedBridge.")
subprocess.run(["git","-C",str(CHECKOUT),"fetch","--depth","1","origin",CODE_REVISION or "main"],check=True)
CODE_COMMIT = subprocess.check_output(["git","-C",str(CHECKOUT),"rev-parse","FETCH_HEAD"],text=True).strip()
if "vietmedbridge" in sys.modules and globals().get("_VMB_IMPORTED_COMMIT") != CODE_COMMIT:
    raise RuntimeError("Code đã đổi trong runtime; Restart session rồi chạy lại notebook.")
subprocess.run(["git","-C",str(CHECKOUT),"checkout","--detach","FETCH_HEAD"],check=True)
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","-e",".[notebook,retrieval,strong]"],cwd=CHECKOUT,check=True)
sys.path.insert(0,str(CHECKOUT / "src"))
importlib.invalidate_caches()
from vietmedbridge import PIPELINE_API_VERSION
from vietmedbridge.artifacts import atomic_json, read_json, runtime_versions
from vietmedbridge.portable_embeddings import checked, runtime_install_commands, validate_runtime_lock, resources_relative
from vietmedbridge.scale_benchmark import load_handoff
from vietmedbridge.scale_baseline import imported_bge_manifest
if PIPELINE_API_VERSION != EXPECTED_PIPELINE_API:
    raise RuntimeError("Pipeline API khác phiên bản notebook.")
embedding_runtime_lock = DATA_ROOT / "retrieval_embedding_runtime_lock.json"
BUILD_DIR, CANDIDATE, INDEX_INPUTS = load_handoff(DATA_ROOT)
if CANDIDATE["counts"]["input_records"] < MIN_EXPECTED_INPUT_RECORDS:
    raise RuntimeError(f"Active candidate chỉ có {CANDIDATE['counts']['input_records']:,} input IDs; cần ít nhất {MIN_EXPECTED_INPUT_RECORDS:,}. Kiểm tra đúng DATA_ROOT/candidate 700k; không tự chạy lại union.")
FULL_CONFIG = read_json(CHECKOUT / "configs/retrieval_full.json")
if embedding_runtime_lock.is_file():
    pinned_runtime = checked(read_json(embedding_runtime_lock))["runtime"]
    runtime_source = "Drive Kaggle import lock"
else:
    # A verified imported BGE manifest records its producer's Torch and
    # Transformers versions. If the small lock file was omitted during a
    # storage copy, recover the pinned producer runtime only when those
    # identities match; never guess from the current Colab environment.
    pinned_runtime = read_json(CHECKOUT / "configs/kaggle_runtime.json")
    validate_runtime_lock(pinned_runtime)
    bge_config_path = DATA_ROOT / resources_relative(CANDIDATE, FULL_CONFIG) / "bge" / "config.json"
    if not bge_config_path.is_file():
        raise RuntimeError("Thiếu runtime lock và manifest BGE. Khôi phục đúng Kaggle Outputs hoặc snapshot Drive trước khi chạy retrieval.")
    bge_encoder = read_json(bge_config_path).get("encoder", {})
    if any(bge_encoder.get(name) != pinned_runtime[name] for name in ("torch", "transformers")):
        raise RuntimeError("Thiếu runtime lock và phiên bản producer trong BGE cache không khớp configs/kaggle_runtime.json. Nhập lại đúng Kaggle Outputs để khôi phục lock; không tự đổi runtime.")
    runtime_source = "verified BGE manifest + repository-pinned Kaggle runtime"
commands = runtime_install_commands(pinned_runtime)
if commands and "torch" in sys.modules:
    raise RuntimeError("Restart session trước khi cài runtime embedding đã khóa từ Kaggle.")
for command in commands:
    subprocess.run(command,check=True)
importlib.invalidate_caches()
subprocess.run([sys.executable,"-c", "import torch; print('PyTorch:',torch.__version__,'| CUDA:',torch.cuda.is_available()); assert torch.cuda.is_available(), 'Chọn Colab T4 GPU; PyTorch phải là bản CUDA.'"],check=True)
_, BGE_CACHE = imported_bge_manifest(DATA_ROOT,CANDIDATE,INDEX_INPUTS,FULL_CONFIG)
if BGE_CACHE is None or BGE_CACHE["state"] != "COMPLETE":
    raise RuntimeError("BGE cache của active candidate chưa đủ. Không tự encode lại corpus.")
if any(BGE_CACHE["encoder"].get(name) != pinned_runtime[name] for name in ("torch", "transformers")):
    raise RuntimeError("Producer runtime trong manifest BGE không khớp runtime đã chọn; dừng trước khi nạp model.")
PIPELINE_CONFIG = read_json(CHECKOUT / "configs/data_pipeline.json")
atomic_json(RESULTS_ROOT / "runtime.json",{"git_commit":CODE_COMMIT,"source":str(DATA_ROOT),"embedding_runtime_source":runtime_source,
    "source_read_only":True,"packages":runtime_versions()})
_VMB_IMPORTED_COMMIT = CODE_COMMIT
print("Drive source (read only):",DATA_ROOT)
print("Embedding runtime:",runtime_source)
print("Corpus:",json.dumps(CANDIDATE["counts"],ensure_ascii=False))
print(f"BGE cache: {len(BGE_CACHE['parts'])}/{len(INDEX_INPUTS['parts'])} parts | {INDEX_INPUTS['input_count']:,} inputs")
print("Checkpoint/ZIP:",RESULTS_ROOT,"| tải ZIP trước khi kết thúc runtime.")
print("Code commit:",CODE_COMMIT)
'''


DRIVE_PREFLIGHT = '''
def _vmb_mount_data_root(readonly=False):
    # Probe real file contents: an existing mount directory can still be disconnected.
    import errno
    from google.colab import drive
    mountpoint = "/content/drive"
    for attempt in range(2):
        try:
            drive.mount(mountpoint, force_remount=bool(attempt), readonly=readonly)
            root = Path(DRIVE_DATA_ROOT)
            try:
                active_bytes = (root / "active_data_candidate.json").read_bytes()
            except FileNotFoundError:
                shared = sorted(Path(mountpoint, ".shortcut-targets-by-id").glob(
                    "*/VietMedBridge/data/active_data_candidate.json"))
                if root != Path("/content/drive/MyDrive/VietMedBridge/data") or len(shared) != 1:
                    raise RuntimeError(f"Không thấy active candidate tại {root}. Kiểm tra đúng tài khoản Drive và DRIVE_DATA_ROOT; không tạo thư mục data rỗng.") from None
                root = shared[0].parent
                active_bytes = (root / "active_data_candidate.json").read_bytes()
            active = json.loads(active_bytes)
            print("Drive đã đọc được:", root)
            print("Active candidate:", active.get("candidate_manifest_sha256"),
                "| build:", active.get("build_run"))
            return root
        except OSError as error:
            disconnected = error.errno in (107, errno.ENOTCONN) or "transport endpoint is not connected" in str(error).lower()
            if not disconnected:
                raise
            if attempt:
                raise RuntimeError("Drive vẫn mất kết nối sau khi gắn lại. Giữ runtime và chạy lại cell Bootstrap khi Drive truy cập được; chưa nạp model hay xóa checkpoint.") from error
            print("Drive mất kết nối (Errno 107); tự gắn lại một lần trước khi đọc checkpoint.", flush=True)
'''


def retrieval_bootstrap():
    settings = '''PREPARE_ONLY = False  #@param {type:"boolean"}
INFERENCE_MODE = "full"  # mặc định chạy đủ các nhánh, dùng lại checkpoint hợp lệ
TEAM_SIZE = 1  # một acc chạy tiếp toàn bộ phần thiếu
TEAM_WORKER_ID = None  # coordinator: tiếp tục embeddings rồi inference và xuất submission
RUN_CACHED_BASELINE_FROM_DRIVE = False  # mặc định: full architecture trên frozen corpus
RUN_QWEN_CORPUS_ONLY = False  # True: tiếp tục checkpoint Qwen corpus; False: chạy hệ thống full
CACHED_INFERENCE_MODE = "cached_strong"  # dịch + rerank + parents; "bge_bm25_cached": bản nhanh chỉ BGE/BM25
DOCUMENT_DENSE_ENABLED = True  # thêm BGE title/opening embeddings; False: bỏ nhánh này và ghi rõ trong báo cáo
QWEN_DENSE_MODE = "if_complete"  # chỉ áp dụng cached_strong; chế độ full luôn yêu cầu đủ Qwen
VECTOR_BACKEND = "faiss_ivf_sq8"  # ANN lưu theo shard; "qdrant": server; "exact": quét đầy đủ theo yêu cầu
ANN_NPROBE = 64  # tự tăng tối đa 512 nếu recall sample < 0.95
QDRANT_URL = ""  # chỉ điền khi VECTOR_BACKEND="qdrant"; không dùng Qdrant Python local mode cho 700k
DRIVE_DATA_ROOT = "/content/drive/MyDrive/VietMedBridge/data"
MIN_EXPECTED_INPUT_RECORDS = 700244  # chặn chạy nhầm candidate 100k cũ; tăng khi có batch mới
USE_HF_STORAGE = False  # True sau HF_STORAGE_VERIFIED; khi đó tắt RUN_CACHED_BASELINE_FROM_DRIVE
HF_STORAGE_REPO = "platypus123/biobridgeir-checkpoints"
HF_STORAGE_PROFILE = "bge_submit"  # chỉ tải metadata/catalog/BGE để lấy điểm trước; "all": toàn snapshot
HF_STORAGE_REVISION = None  # có thể khóa full commit SHA đã nhận từ notebook chuyển storage
'''
    legacy = (BOOT.replace("full-master-plan-strong-v3-per-model-15b","full-master-plan-strong-v19-ready-resume")
        .replace('drive.mount("/content/drive")', 'DATA_ROOT = _vmb_mount_data_root()')
        .replace('DATA_ROOT = Path("/content/drive/MyDrive/VietMedBridge/data")\n', '')
        .replace('DATA_ROOT.mkdir(parents=True, exist_ok=True)\n', ''))
    cached_boot = DRIVE_CACHED_BOOT.replace('drive.mount("/content/drive", readonly=True)\nDATA_ROOT = Path(DRIVE_DATA_ROOT)',
        'DATA_ROOT = _vmb_mount_data_root(readonly=True)')
    vector_boot = '''
import os
if VECTOR_BACKEND == "qdrant":
    if not QDRANT_URL:
        raise ValueError("Điền QDRANT_URL của server trước khi chọn backend qdrant. Mặc định FAISS ANN không cần server.")
    os.environ["VMB_QDRANT_URL"] = QDRANT_URL
    try:
        from google.colab import userdata
        os.environ["VMB_QDRANT_API_KEY"] = userdata.get("QDRANT_API_KEY") or ""
    except Exception:
        pass  # server riêng có thể không yêu cầu API key; không in token
'''
    gpu_check = '''
if not PREPARE_ONLY:
    subprocess.run([sys.executable, "-c",
        "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count()); "
        "assert torch.cuda.is_available(), 'Chọn Runtime > Change runtime type > T4 GPU rồi Run all. Chưa nạp model hoặc encode corpus.'"], check=True)
print("Notebook 04:", "CPU preparation only" if PREPARE_ONLY else "GPU full pipeline",
    "| team:", TEAM_SIZE, "| worker:", TEAM_WORKER_ID)
'''
    return (settings + '\nimport json\nfrom pathlib import Path\n' + DRIVE_PREFLIGHT + vector_boot + '\nif RUN_CACHED_BASELINE_FROM_DRIVE and USE_HF_STORAGE:\n'
        '    raise ValueError("Chọn đúng một nguồn Drive hoặc HF.")\n'
        'if RUN_CACHED_BASELINE_FROM_DRIVE:\n' + textwrap.indent(cached_boot.strip(),"    ")
        + "\nelif USE_HF_STORAGE:\n" + textwrap.indent(HF_BOOT.strip(),"    ")
        + "\nelse:\n" + textwrap.indent(legacy.strip(),"    ") + '\n' + gpu_check)


def main():
    save("04_colab_retrieval_baseline.ipynb", [
        md('''
        # VietMedBridge — 04: Full retrieval system trên frozen corpus

        Plan chính: `R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md`.
        **Mặc định chạy full architecture:** `RUN_CACHED_BASELINE_FROM_DRIVE=False`,
        `USE_HF_STORAGE=False`, `INFERENCE_MODE="full"`, `PREPARE_ONLY=False`,
        `TEAM_SIZE=1`, `TEAM_WORKER_ID=None`. **Chọn Colab T4 GPU → Run all**
        để tiếp tục candidate đã chuẩn bị, chạy phần thiếu rồi inference/xuất ZIP.
        Bootstrap tự clone/cài code; không cần chèn cell sửa lỗi hoặc chọn ACTION.

        **Chế độ cache để lấy baseline sớm (tùy chọn):** đặt
        `RUN_CACHED_BASELINE_FROM_DRIVE=True`. `CACHED_INFERENCE_MODE="cached_strong"`
        dùng BGE child cache đã đủ,
        BM25 VI/EN/ZH, Qwen 4B dịch/expansion, BGE document dense, Qwen 8B rerank
        document và child/MaxP, source parents 512/640 và LCS dedup.
        `QWEN_DENSE_MODE="if_complete"` không chờ phần Qwen corpus còn thiếu;
        subqueries/HyDE dùng BGE khi Qwen dense chưa đủ. Không encode lại child corpus.
        Document dense **tạo vectors mới** cho title/opening của mỗi document;
        `VECTOR_BACKEND="faiss_ivf_sq8"` dựng chỉ mục ANN nén từ vectors hiện có,
        checkpoint từng shard trong RESULTS_ROOT/vector_indexes; không encode lại child corpus.
        Các shard cũ được dùng lại khi parts tương ứng giữ nguyên ở batch mới.
        ANN kiểm recall trên 8 query thực so với exact top-20 toàn corpus, tối thiểu 0.95;
        tự tăng nprobe nếu cần, dừng nếu vẫn không đạt. Đây không phải đánh giá relevance/F2.
        Candidate ANN được chấm lại bằng vector gốc trước RRF/rerank.
        Chấm lại toàn bộ query pool trong một lượt đọc tuần tự vector parts,
        checkpoint từng part; không tải lại corpus riêng cho mỗi query.
        Qdrant server đã được nối: chọn `VECTOR_BACKEND="qdrant"`, điền `QDRANT_URL`,
        API key đặt trong Secret `QDRANT_API_KEY`. Vectors/HNSW nằm trên disk của server;
        upsert có acknowledgement và count chính xác, dùng lại collections của shard cũ.
        Chưa có server thì giữ FAISS ANN. FAISS là chỉ mục portable, không gọi là vector DB server.
        đặt `DOCUMENT_DENSE_ENABLED=False` nếu cần bỏ chi phí này, báo cáo ghi DISABLED.
        Nhánh dịch và rerank vẫn cần GPU và thời gian; không phải đổi tên baseline thành full.
        Muốn xuất bản BGE/BM25 trước: `CACHED_INFERENCE_MODE="bge_bm25_cached"`.
        Chọn **Colab T4 GPU**, Run all. Đọc active candidate, BGE đã nhập và
        catalog BM25 đã hoàn tất trên Drive; không chờ upload HF.
        Baseline chỉ encode 1.200 query. Cached strong thêm query variants và
        document representations nếu bật; cả hai khôi phục catalog và cache
        vectors vào ổ local khi cần. Không crawl/chunk/encode lại child corpus
        hoặc ghi thêm vào Drive đang đầy. Thiếu catalog/BGE phù hợp thì dừng.
        Checkpoint và ZIP ở `/content/vmb_results`, dùng lại khi chạy lại trong
        cùng runtime. Runtime bị thu hồi sẽ mất checkpoint local; corpus vẫn ở
        Drive. Tải ZIP và bản checkpoint ở cuối notebook trước khi ngắt phiên.
        Backup cached strong có thêm document vectors mới nên có thể vài GB.

        **Tiếp tục lượt hiện tại:** giữ nguyên cấu hình mặc định, chọn **GPU,
        T4 trở lên**, Run all. Catalog CPU và vector parts hợp lệ được dùng lại;
        không chạy lại 00–03. Các dòng `reused` là kiểm tra/dùng lại checkpoint,
        không phải encode lại. Chỉ phần thiếu hoặc không vượt qua kiểm tra mới tính lại.
        **Chỉ khi có data mới chờ ghép:** đặt `PREPARE_ONLY=True` ở cell Bootstrap,
        chạy trên CPU đến `CPU_PREPARATION_COMPLETE`, rồi trở về False trên GPU.
        Trước khi nạp model, stage kiểm tra Drive đọc được active candidate.
        Nếu Drive rớt với Errno 107, thử gắn lại một lần rồi kiểm tra candidate
        vẫn giữ nguyên trước khi tiếp tục từ checkpoint. Nếu vẫn lỗi, dừng rõ
        DRIVE_UNAVAILABLE; giữ runtime, gắn lại Drive và chạy lại cell xử lý.
        Model cache trong /content vẫn giữ khi chỉ remount Drive. Checkpoint
        có hash hợp lệ được dùng lại; phần chưa công bố có thể phải tính lại.

        **Nộp baseline ngay sau khi nhập BGE từ Kaggle:** chọn GPU, đặt
        Trong chế độ Drive cached chỉ đổi `CACHED_INFERENCE_MODE="bge_bm25_cached"`.
        Với bootstrap full cũ, đặt `PREPARE_ONLY=False`, `INFERENCE_MODE="bge_bm25_cached"`, `TEAM_SIZE=1`,
        `TEAM_WORKER_ID=None`. Dùng toàn bộ BGE của active candidate và catalog
        BM25 đã chuẩn bị; chỉ encode queries, tìm kiếm, chọn source chunks và
        xuất ZIP 1.200 query. Thiếu BGE thì dừng WAITING, không encode lại corpus.
        Phần Qwen đã chạy được giữ nguyên. Đây là baseline để đo điểm trước;
        `INFERENCE_MODE="full"` vẫn chạy kiến trúc đầy đủ mô tả dưới đây.

        **Drive đầy, dữ liệu đã chuyển lên HF:** chạy utility
        `tools/drive_hf_storage.ipynb` trên CPU đến HF_STORAGE_VERIFIED. Trong
        cell 1 notebook này đặt `RUN_CACHED_BASELINE_FROM_DRIVE=False`,
        `USE_HF_STORAGE=True`, giữ repo đã chuyển,
        `HF_STORAGE_PROFILE="bge_submit"`; chọn Colab GPU và Run all.
        Tự tải metadata/catalog/BGE về /content/vmb_data rồi dùng
        `CACHED_INFERENCE_MODE` như trên. Không mount hoặc ghi Drive; Qwen và source
        shards không cần tải cho lượt cached. Checkpoint inference nằm ổ tạm; giữ runtime đến lúc
        tải ZIP. Profile "all" tải toàn snapshot để chạy các bước khác khi đủ disk.

        Nếu 03 đã publish active_data_candidate.json, tự nhận candidate đó.
        Corpus vượt giới hạn pilot (2.000 documents hoặc 50.000 children) chạy
        **đầy đủ các nhánh trên disk**: BGE child/document dense + Qwen embedding
        + BM25 VI/EN/ZH với query dịch → RRF → Qwen document/child MaxP reranking
        → source parents 512/640 → LCS dedup → ZIP đủ 1.200 query.
        Tự nhận active candidate của 03, gồm team-100k-data-v1 hiện có.
        Khi có batch mới, 03 tự ghi candidate cũ + mới vào lineage trên Drive;
        coordinator 04 tự ghép chúng trước inference, không cần sửa danh sách code.
        Coordinator chuẩn bị trên CPU một lần trước khi ba worker bắt đầu trên GPU.
        Cache vectors gắn với nội dung/model, giữ official ID/alias riêng. Data mới
        dùng lại vectors của nội dung cũ; vectors baseline 100k được kiểm và đăng ký
        vào cache, không copy thêm một ma trận corpus. Cache không xóa được khi còn
        manifest tham chiếu. Model đổi hoặc source text đổi thì tính phần tương ứng.
        Query translations/expansions cũng được cache độc lập với corpus.
        Mỗi block vector, scored stage và query có checkpoint. Dense search lưu
        top-k giữa lượt, ngắt phiên rồi resume thay vì quét lại toàn bộ.

        **Chạy một người (đã đặt sẵn):** PREPARE_ONLY=False, TEAM_SIZE=1,
        TEAM_WORKER_ID=None, Run all trên GPU. Cấu hình nằm ở đầu cell Bootstrap;
        cell xử lý không ghi đè lựa chọn đó.
        **Team 3 người:** PREPARE_ONLY=False, TEAM_SIZE=3 ở cả ba người,
        TEAM_WORKER_ID lần lượt 0/1/2.
        Mỗi người chỉ encode input parts được giao, ghi vector blocks và mapping
        riêng theo phần. Sau đó một người đặt TEAM_WORKER_ID=None để tổng hợp và
        chạy các bước còn lại; coordinator báo WAITING nếu thiếu phần.
        Khi ngắt, giữ cùng TEAM_SIZE/ID, source và runtime package versions.
        Các phiên cùng truy cập đúng DATA_ROOT đã chia sẻ; dừng phiên solo đang
        ghi cùng view trước khi chia việc. Mỗi ID chỉ chạy một phiên.
        Đây là phân công embedding, chưa ghép một job QLoRA thành nhiều GPU.
        Dùng tài nguyên được cấp phù hợp với quy định nền tảng.

        Full architecture có chi phí thực: Qwen embedding 8B trên 100k được ngoại
        suy ~24 giờ T4 từ mẫu 64 đoạn, chưa gồm các stage khác và không bảo đảm
        thời gian. Thử giới hạn MAX_NEW_* nếu cần; không giả trạng thái đã chạy.
        Full pretrained chạy được trước nhãn. Fine-tune/dev cutoff chỉ chạy khi
        có labels review; adapter từ 05 được kiểm và dùng cho cả corpus lớn.
        Corpus mới giữ adapter hợp lệ nhưng ngưỡng cũ cần dev recalibration.

        Kiến trúc full cho pilot nhỏ: **BGE-M3 child/document dense + Qwen3-Embedding-8B + BM25 VI/EN/ZH → weighted RRF
        → Qwen3-Reranker-8B document → local child retrieval/MaxP rerank
        → exact-source parent 512/640 → LCS dedup → ZIP đủ 1.200 query**.
        Qwen3-4B dịch query; PICO-lite/subqueries/HyDE chỉ additive cho query phức tạp.
        Các số/Latin entities/constraint cues được kiểm; original query luôn giữ.
        Sparse dùng PyVI, Jieba, CJK n-grams và title/heading/body/alias fields.

        Sếp đã xác nhận với BTC: **≤15B cho từng model**, không tính tổng model.
        BGE-M3 0,568B; Qwen embedding 7,567B; query LLM 4,022B; reranker 8,189B.
        Gate tính adapter vào base model tương ứng, trước quantization.
        Hai model 8B dùng **NF4 4-bit**.
        Models nạp lần lượt. Đây là lựa chọn tài nguyên
        cho T4, chưa benchmark tương đương fp16. Weights/cache nằm ổ local Colab;
        vector parts, translation/expansion và scored stages checkpoint trên Drive.
        Khi Colab ngắt, Run all lại cùng cấu hình; không chạy hai runtime cùng run.

        Với corpus nhỏ: chạy full inference và xuất submission như trước.
        Có code train/dev mining, QLoRA, checkpoint selection, dev F2 cutoff và
        ablation trong module/05. Bộ nhãn pilot 32/8/8 là AI-assisted, chưa được
        người kiểm định; mining đang thiếu hard negatives đã review nên chưa fine-tune.
        04 dùng pretrained models và cutoff chưa tune; không giả lập fine-tune hoặc điểm.
        Nếu đã chạy 05 với nhãn độc lập hợp lệ, 04 tự đọc adapter/dev policy đã chọn,
        tạo namespace -ft- mới. Nguồn/candidate cũ và các kết quả baseline được giữ.

        Với pilot: reuse verified corpus/document vectors từ cả 04 và 05. Chỉ encode
        inputs còn thiếu. Batch Qwen tự chọn theo VRAM và giảm nếu OOM.
        Theo dõi runtime_profile.json/execution_policy.json trong thư mục run;
        mỗi embedding pass cũng ghi thời gian encode, số parts mới/đã cache.
        Model revisions của full pilot giữ nguyên.
        '''),
        md("## 1. Bootstrap, kiểm tra Drive/GPU, clone và cài dependencies"),
        code(retrieval_bootstrap()),
        md('''
        ## 2. Full system / phần embedding được giao — CPU/GPU lần lượt

        Mặc định `PREPARE_ONLY=False`: chạy full trên một acc GPU, dùng lại catalog,
        embeddings và các stage đã checkpoint. Chạy tiếp phần thiếu, sau đó xuất ZIP.
        Chọn `PREPARE_ONLY=True` ở Bootstrap khi chỉ cần ghép/freeze/chuẩn bị inputs
        và catalog BM25 trên CPU; chế độ đó dừng ở CPU_PREPARATION_COMPLETE.
        Nếu Colab ngắt sau khi ghép đủ IDs, bản DATA_VALIDATED được xác minh và
        dùng lại, không chọn/xuất union lại. union_progress.json lưu bước cuối/lỗi;
        RUNNING có thể là phiên đang chạy hoặc bị ngắt cứng, không phải chứng nhận hoàn tất.
        Sau chuẩn bị CPU, đổi PREPARE_ONLY=False ở runtime GPU; giữ cùng DATA_ROOT.

        Giữ MAX_NEW_*=None để chạy hết. Đổi model/data/policy cần RUN_NAME mới;
        các run cũ không bị ghi đè. Vector BGE của stage-a-retrieval-v1 được kiểm
        hash/order/model trước reuse. Qwen vectors có corpus/query identities riêng.

        Sau checks nguồn/snapshot/query, GPU chạy phần thiếu:
        BGE child/document → Qwen4B query understanding → Qwen8B embedding → Qwen8B rerank.
        Run v3-per-model-15b tách khỏi v1/v2; không trộn Qwen vectors 0.6B và 8B.
        Giữ DATA_ROOT; corpus và BGE vectors hợp lệ vẫn được tận dụng.
        Runtime lưu từng phần; lỗi mạng/OOM giữ checkpoints đã hoàn tất.
        Với candidate lớn, toàn bộ architecture chạy trên disk. TEAM_SIZE/ID
        phân công corpus embeddings; coordinator chạy query LLM và reranking.
        MAX_NEW_* giới hạn mỗi phiên, mặc định None chạy hết.
        Ngắt runtime thì mở cùng notebook và Run all để resume vector/query.
        Catalog CPU giữ bản local đã dựng xong, lưu Drive thành các phần gzip
        64 MiB có checksum trước khi ghi COMPLETE. Nếu mất file catalog cũ,
        tự khôi phục từ bản local/các phần hợp lệ; thiếu cả hai thì dựng lại
        riêng catalog từ frozen data. Retry upload cùng runtime dùng lại SQLite
        local và các phần đã lưu. Nếu runtime mất trước khi dựng xong SQLite,
        bước dựng catalog vẫn phải chạy lại. Mỗi worker ID chỉ có một runtime đang ghi;
        chỉ một coordinator công bố manifest và submission.
        Glossary tùy chọn data/labels/medical_aliases.json cần reviewed=true,
        entries=[{aliases:[...], source:"..."}]; không có thì alias field rỗng,
        biomedical Latin/acronym tokens vẫn được giữ.
        '''),
        code('''
        import json
        from vietmedbridge.full_plan_runtime import run_full_pipeline
        from vietmedbridge.scale_benchmark import load_handoff
        from vietmedbridge.full_scale_runtime import run_scale_full_pipeline, prepare_scale_cpu_resources, resume_qwen_corpus_embeddings
        from vietmedbridge.scale_baseline import run_large_baseline
        from vietmedbridge.artifacts import read_json, digest_json
        from vietmedbridge.index_inputs import read_candidate_lineage
        RESULTS_ROOT = globals().get("RESULTS_ROOT", WORK_DIR.parent / "vmb_results")
        VECTOR_SEARCH_OPTIONS = {"backend":globals().get("VECTOR_BACKEND","faiss_ivf_sq8"),
            "nprobe":globals().get("ANN_NPROBE",64)}

        BUILD_RUN = "stage-a-data-v3-laodong"
        CANDIDATE_NAME = "candidate-1cd220a4be956d5a.json"
        RUN_NAME = "stage-a-full-plan-v3-per-model-15b"
        EMBEDDING_CACHE_RUN = "stage-a-retrieval-v1"
        MAX_NEW_EMBEDDING_PARTS = None
        MAX_NEW_TRANSLATIONS = None
        MAX_NEW_QUERIES = None
        # Use Bootstrap settings; keep the same defaults when this cell is run alone.
        PREPARE_ONLY = globals().get("PREPARE_ONLY", False)
        INFERENCE_MODE = globals().get("INFERENCE_MODE", "full")
        TEAM_SIZE = globals().get("TEAM_SIZE", 1)
        TEAM_WORKER_ID = globals().get("TEAM_WORKER_ID", None)
        QWEN_ONLY_MODE = globals().get("RUN_QWEN_CORPUS_ONLY",False)

        DRIVE_CACHED_MODE = globals().get("RUN_CACHED_BASELINE_FROM_DRIVE",False)
        CACHED_STORAGE_MODE = DRIVE_CACHED_MODE or (globals().get("USE_HF_STORAGE") and HF_STORAGE_PROFILE == "bge_submit")
        if CACHED_STORAGE_MODE:
            PREPARE_ONLY = False
            INFERENCE_MODE = globals().get("CACHED_INFERENCE_MODE","cached_strong")
            TEAM_SIZE, TEAM_WORKER_ID = 1, None
        if QWEN_ONLY_MODE:
            if CACHED_STORAGE_MODE:
                raise ValueError("Qwen-only resume dùng nguồn Drive full, không dùng cached/HF bge_submit mode.")
            PREPARE_ONLY = False
            TEAM_SIZE, TEAM_WORKER_ID = 1, None

        if INFERENCE_MODE not in ("full", "bge_bm25_cached", "cached_strong"):
            raise ValueError("INFERENCE_MODE phải là full, cached_strong hoặc bge_bm25_cached.")
        if INFERENCE_MODE != "full" and (TEAM_SIZE != 1 or TEAM_WORKER_ID is not None):
            raise ValueError("Cached inference dùng một coordinator: TEAM_SIZE=1, TEAM_WORKER_ID=None.")
        print("Inference mode:",INFERENCE_MODE,"| document dense:",globals().get("DOCUMENT_DENSE_ENABLED",True),
            "| Qwen dense:","require" if INFERENCE_MODE == "full" else globals().get("QWEN_DENSE_MODE","if_complete"),
            "| team:",TEAM_SIZE,"| worker:",TEAM_WORKER_ID)

        # Notebook 03 appends each newly frozen candidate to this Drive manifest.
        # The coordinator composes the old corpus plus new batch automatically.
        LINEAGE_SOURCES = [] if QWEN_ONLY_MODE or INFERENCE_MODE != "full" else read_candidate_lineage(DATA_ROOT)
        if PREPARE_ONLY and TEAM_WORKER_ID is not None:
            raise ValueError("Chuẩn bị CPU dùng TEAM_WORKER_ID=None; GPU workers đặt PREPARE_ONLY=False.")
        if len(LINEAGE_SOURCES) > 1:
            if not PREPARE_ONLY:
                raise ValueError("Data đang chờ ghép. Chạy PREPARE_ONLY=True trên CPU trước khi dùng GPU.")
            if TEAM_WORKER_ID is not None:
                raise ValueError("Có data mới chờ ghép. Coordinator chạy 04 một lần với TEAM_WORKER_ID=None trước workers.")
            from vietmedbridge.corpus_union import compose_candidates
            from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer
            from vietmedbridge.dataset import parquet_path
            from vietmedbridge.golden import run_golden_suite
            CORPUS_SOURCES = [(item["build_run"], item["candidate_name"]) for item in LINEAGE_SOURCES]
            lineage_key = digest_json([item["candidate_manifest_sha256"] for item in LINEAGE_SOURCES])
            CORPUS_UNION_RUN = f"cumulative-{lineage_key[:16]}"
            print("Tự ghép các candidate đã freeze:", json.dumps(CORPUS_SOURCES, ensure_ascii=False))
            first_config = read_json(DATA_ROOT / "processed" / CORPUS_SOURCES[0][0] / "config.json")
            tokenizer, tokenizer_spec = load_bge_tokenizer(first_config["tokenizer"]["revision"])
            golden = run_golden_suite(CHECKOUT / "tests/golden/cases.json", tokenizer,
                tokenizer_spec, ChunkConfig(**first_config["chunks"]))
            print(compose_candidates(DATA_ROOT, CORPUS_SOURCES, tokenizer, run_name=CORPUS_UNION_RUN,
                golden_report=golden, official_links=parquet_path(DATA_ROOT,"links_corpus.parquet"), work_dir=WORK_DIR))

        LARGE_CANDIDATE = False
        if PREPARE_ONLY and not (DATA_ROOT / "active_data_candidate.json").is_file():
            raise ValueError("Chưa có frozen candidate active. Kiểm tra DATA_ROOT và hoàn tất notebook 03 trước 04.")
        if (DATA_ROOT / "active_data_candidate.json").is_file():
            active = read_json(DATA_ROOT / "active_data_candidate.json")
            _, candidate, _ = load_handoff(DATA_ROOT)
            BUILD_RUN, CANDIDATE_NAME = active["build_run"], active["candidate_name"]
            limits = read_json(CHECKOUT / "configs/retrieval_full.json")["pilot_limits"]
            LARGE_CANDIDATE = (candidate["counts"]["documents"] > limits["max_documents"]
                or candidate["counts"]["children"] > limits["max_children"])
            RUN_NAME = BUILD_RUN + "-full-plan-v1"
        if QWEN_ONLY_MODE:
            status = resume_qwen_corpus_embeddings(DATA_ROOT,CHECKOUT,WORK_DIR)
            RESULT = {"status":status,"ready":{"state":status["state"],"query_count":0},
                "samples":[],"diagnostics_path":status.get("checkpoint","")}
            READY = RESULT["ready"]
        elif PREPARE_ONLY:
            status = prepare_scale_cpu_resources(DATA_ROOT, CHECKOUT, WORK_DIR) if LARGE_CANDIDATE else {
                "state":"CPU_PREPARATION_COMPLETE", "build_run":BUILD_RUN,
                "scope":"Frozen pilot inputs ready; set PREPARE_ONLY=False on GPU for inference"}
            RESULT = {"status":status,"ready":{"state":status["state"],"query_count":0},
                "samples":[],"diagnostics_path":str(DATA_ROOT / "retrieval/cpu_preparation")}
            READY = RESULT["ready"]
        elif INFERENCE_MODE == "bge_bm25_cached":
            storage_options = {"output_root":RESULTS_ROOT,"read_only_source":True} if DRIVE_CACHED_MODE else {}
            RESULT = run_large_baseline(DATA_ROOT, CHECKOUT, code_commit=CODE_COMMIT,
                work_dir=WORK_DIR, reuse_content_only=True, max_new_queries=MAX_NEW_QUERIES,
                vector_search_options=VECTOR_SEARCH_OPTIONS,**storage_options)
            READY = RESULT["ready"]
        elif INFERENCE_MODE == "cached_strong":
            RESULT = run_scale_full_pipeline(DATA_ROOT, CHECKOUT, code_commit=CODE_COMMIT,
                work_dir=WORK_DIR, execution_mode="cached_strong", output_root=RESULTS_ROOT,
                second_dense_mode=globals().get("QWEN_DENSE_MODE","if_complete"),
                document_dense_enabled=globals().get("DOCUMENT_DENSE_ENABLED",True),
                vector_search_options=VECTOR_SEARCH_OPTIONS,
                max_new_embedding_parts=MAX_NEW_EMBEDDING_PARTS,
                max_new_translations=MAX_NEW_TRANSLATIONS, max_new_queries=MAX_NEW_QUERIES)
            READY = RESULT["ready"]
        elif LARGE_CANDIDATE:
            RESULT = run_scale_full_pipeline(DATA_ROOT, CHECKOUT, code_commit=CODE_COMMIT,
                work_dir=WORK_DIR, worker_id=TEAM_WORKER_ID, workers=TEAM_SIZE,
                vector_search_options=VECTOR_SEARCH_OPTIONS,
                max_new_embedding_parts=MAX_NEW_EMBEDDING_PARTS,
                max_new_translations=MAX_NEW_TRANSLATIONS, max_new_queries=MAX_NEW_QUERIES)
            READY = RESULT["ready"]
        else:
            RESULT = run_full_pipeline(DATA_ROOT, CHECKOUT, code_commit=CODE_COMMIT,
                work_dir=WORK_DIR, build_run=BUILD_RUN, candidate_name=CANDIDATE_NAME,
                run_name=RUN_NAME, embedding_cache_run=EMBEDDING_CACHE_RUN,
                max_new_embedding_parts=MAX_NEW_EMBEDDING_PARTS,
                max_new_translations=MAX_NEW_TRANSLATIONS, max_new_queries=MAX_NEW_QUERIES)
            READY = RESULT["ready"]
        print(json.dumps(RESULT["status"], ensure_ascii=False, indent=2))
        print(json.dumps(READY, ensure_ascii=False, indent=2))
        '''),
        md('''
        ## 3. Xem source samples và báo cáo — CPU

        Chunks bên dưới là source slices. Với corpus lớn, full_plan_status.json
        ghi các model/stages thực thi, corpus, cache và adapter/dev calibration;
        coordinator-profile.json và worker-<id>-profile.json ghi thời gian các stage resource. Với full pilot, diagnostics.json
        có language coverage, branch contributions, output counts và logit distributions. Không có nhãn
        thì không tính F2 giả; `labels/retrieval_reference.json` chỉ được chấm local
        khi `reviewed=true` và vẫn là proxy theo plan, không phải điểm BTC.
        full_plan_status.json phân biệt code đã thực thi với
        supervised/scale gates còn cần dữ liệu và đo đạc. Corpus nhỏ và chất lượng
        nguồn vẫn giới hạn recall.
        '''),
        code('''
        for sample in RESULT["samples"]:
            print("\\nQuery", sample["query"]["id"], sample["query"]["query"])
            prediction = sample["prediction"]
            print("Documents:", prediction["relevant_docs"])
            print("First chunk:", prediction["relevant_chunks"][0]["chunk_text"][:800]
                  if prediction["relevant_chunks"] else "EMPTY")
        print("Diagnostics:", RESULT["diagnostics_path"])
        '''),
        md('''
        ## 4. Tải ZIP và submit thử để lấy điểm thật

        ZIP chỉ chứa results.json ở root, đủ 1.200 official IDs; kiểm source/schema
        và checksum trước download. Manifest/model/runtime evidence nằm ngoài ZIP.
        Mỗi ZIP hợp lệ tự được đóng băng vào `releases/<release_id>` cùng manifest
        và run contract; đây là bản lưu/rollback local, chưa được xem là quality pass.
        Upload ZIP lên Dashboard BTC; score_feedback.json giữ submission ID và
        điểm thật, không bị reset khi resume. Chưa có official score trước upload.
        '''),
        code('''
        from google.colab import files
        from vietmedbridge.artifacts import verify_file
        if READY.get("release_id"):
            print("Release đã lưu:", READY["release_id"], "| manifest:", READY["release_manifest"])
        if READY["state"] == "CPU_PREPARATION_COMPLETE":
            print("Chuẩn bị CPU hoàn tất. Chuyển runtime GPU, đặt PREPARE_ONLY=False.")
            print("Team 3 người: TEAM_SIZE=3, TEAM_WORKER_ID=0/1/2; coordinator sau đó dùng None.")
        elif READY["state"] in ("QWEN_CORPUS_COMPLETE","QWEN_CORPUS_CHECKPOINTED_PARTIAL"):
            print("Qwen corpus checkpoint:",READY["state"])
            print("Đủ Qwen: đặt RUN_QWEN_CORPUS_ONLY=False, PREPARE_ONLY=False để chạy full inference.")
        elif READY["state"] != "READY_FOR_MANUAL_UPLOAD" or READY["query_count"] != 1200:
            print("Đã lưu checkpoint:", READY)
            print("Run all lại cùng cấu hình để hoàn tất trước download.")
        else:
            verify_file(READY["zip_path"], READY["zip_sha256"])
            files.download(READY["zip_path"])
        '''),
        md('''
        ## 5. Ghi điểm dashboard hoặc chọn lại một release — tùy chọn

        Sau khi nộp thủ công và BTC hiện kết quả, bật `RECORD_BTC_SCORE`, điền
        submission ID, phase, tài khoản, giờ có timezone, bảy metric và hypothesis.
        Event được lưu trong `DATA_ROOT/submission_journal` và gắn vào ZIP hash.
        Đây chỉ là nhật ký dữ liệu Sếp nhập, không đọc hay sửa Dashboard.

        Có thể chọn lại một ZIP đã đóng băng trong release folder của run hiện tại
        bằng `ROLLBACK_RELEASE_ID`; notebook xác minh hash trước khi tải ZIP đó.
        '''),
        code('''
        from pathlib import Path
        import json
        from google.colab import files
        from vietmedbridge.release_registry import record_dashboard_result, select_release, verify_release

        RECORD_BTC_SCORE = False  #@param {type:"boolean"}
        BTC_SUBMISSION_ID = ""  #@param {type:"string"}
        BTC_PHASE = "public"  #@param ["public", "private"]
        BTC_ACCOUNT = ""  #@param {type:"string"}
        BTC_SUBMITTED_AT = ""  #@param {type:"string"}
        BTC_HYPOTHESIS = ""  #@param {type:"string"}
        BTC_FINAL_SCORE = 0.0  #@param {type:"number"}
        BTC_DOCUMENT_F2 = 0.0  #@param {type:"number"}
        BTC_CHUNK_F2 = 0.0  #@param {type:"number"}
        BTC_DOCUMENT_PRECISION = 0.0  #@param {type:"number"}
        BTC_DOCUMENT_RECALL = 0.0  #@param {type:"number"}
        BTC_CHUNK_PRECISION = 0.0  #@param {type:"number"}
        BTC_CHUNK_RECALL = 0.0  #@param {type:"number"}
        BTC_METRICS = {"final_score": BTC_FINAL_SCORE, "document_f2": BTC_DOCUMENT_F2,
            "chunk_f2": BTC_CHUNK_F2, "document_precision": BTC_DOCUMENT_PRECISION,
            "document_recall": BTC_DOCUMENT_RECALL, "chunk_precision": BTC_CHUNK_PRECISION,
            "chunk_recall": BTC_CHUNK_RECALL}
        if RECORD_BTC_SCORE:
            if not READY.get("release_manifest"):
                raise RuntimeError("Không có release manifest gắn với ZIP đang mở.")
            event = record_dashboard_result(READY["release_manifest"], DATA_ROOT / "submission_journal",
                submission_id=BTC_SUBMISSION_ID, phase=BTC_PHASE, submitted_at=BTC_SUBMITTED_AT,
                metrics=BTC_METRICS, hypothesis=BTC_HYPOTHESIS, account=BTC_ACCOUNT)
            print(json.dumps(event, ensure_ascii=False, indent=2))
        else:
            print("Chưa ghi điểm. Sau khi dashboard có kết quả, bật RECORD_BTC_SCORE và điền số liệu.")

        ROLLBACK_RELEASE_ID = ""  #@param {type:"string"}
        ROLLBACK_REASON = ""  #@param {type:"string"}
        if ROLLBACK_RELEASE_ID:
            run_dir = Path(READY["zip_path"]).parent.parent
            release_path = run_dir / "releases" / ROLLBACK_RELEASE_ID / "release.json"
            release = verify_release(release_path)
            pointer = select_release(release_path, DATA_ROOT / "submission_active_release.json",
                reason=ROLLBACK_REASON)
            print("Đã chọn release để tải:", json.dumps(pointer, ensure_ascii=False, indent=2))
            files.download(str(release_path.parent / "submission.zip"))
        '''),
        md('''
        ## 6. Lưu checkpoint inference về máy — tùy chọn

        Với chế độ cached, cell này gom query embeddings, dense search, scored
        stages, predictions, submission và document vectors mới (nếu có).
        Không gom child corpus vectors trên Drive, catalog hoặc model weights.
        Backup cached strong có thể vài GB. Chạy sau khi đã dừng cell 2.
        Muốn resume ở runtime mới: upload ZIP vào `/content`, giải nén vào
        `/content/vmb_results`, giữ nguyên code/model/candidate/policy.
        Không tự đồng bộ lên Drive hoặc HF đang đầy/bị giới hạn.
        '''),
        code('''
        if globals().get("CACHED_STORAGE_MODE",False):
            import shutil
            from google.colab import files
            checkpoint_zip = shutil.make_archive("/content/vmb-inference-checkpoints","zip",RESULTS_ROOT)
            print("Checkpoint backup:",checkpoint_zip)
            files.download(checkpoint_zip)
        '''),
    ])
    save("05_colab_supervised_training.ipynb", [
        md('''
        # VietMedBridge — 05: Source → nhãn đã duyệt → hard negatives → QLoRA → dev F2

        Giai đoạn supervised trong master §§13/27/39/56. V7 tự nhận active
        candidate của 03, dùng disk full index/cache chung với 04 cho corpus lớn.
        Source review chọn pool có giới hạn; hard negatives vẫn được truy hồi trên
        toàn candidate hiện tại. Adapter đã chọn được 04 large/full dùng trực tiếp.
         **04 chạy pretrained và
        submit được ngay**, không cần chạy 05 trước khi có nhãn. Chọn runtime GPU
        mới; L4/A100 thuận tiện hơn cho training. Qwen8B nạp NF4, batch=1,
        gradient checkpointing/accumulation.

        Chưa có nhãn: notebook tự tách các nhóm source/đoạn trùng trước, chọn tối
        đa 256 source samples train, 40 dev và 40 held-out. Qwen 4B tạo câu hỏi VI
        **chỉ cho train**, có exact evidence quote, checkpoint từng mẫu.
        Dev/held-out để trống câu hỏi cho team viết độc lập trên các nhóm source
        đã reserve. Không thêm teacher mới. Model outputs luôn là draft.

        Notebook tải source_review.json để team duyệt: chỉ sửa review fields,
        ACCEPT/REJECT + reviewer + query + exact evidence_quote. Dev/held-out yêu cầu
        independently_written=true sau khi người review tự viết câu hỏi.
        Lưu file đã duyệt về đúng review_path trên Drive và Run all lại.
        Human review: tối thiểu 32 train/8 dev/8 held-out được accept, hết PENDING mới xuất labels.
        Không tự bật reviewed hoặc exhaustive_chunks cho model predictions.

        Nếu Sếp giao Codex duyệt bộ pilot: file có review_mode=AI_ASSISTED_PILOT,
        reviewer_type=AI và nguồn gốc ủy quyền được ghi rõ. Bộ pilot vẫn cần đủ
        32 train/8 dev/8 held-out; các mẫu PENDING còn lại được để dành, không
        biến thành nhãn âm tính. Dev/held-out ghi AI_REVIEWED_PILOT, không giả
        independently_written=true hoặc nhãn human gold. Upload file
        source_review_ai_pilot.json vào /content bằng tab Files của Colab rồi
        Run all. Cell chuẩn bị tự giữ backup và đưa bản duyệt vào đúng Drive path.

        Train/dev tách theo query, IDs và nội dung không trùng 1.200 contest queries.
        Mine 1 positive + 7 negatives, ưu tiên top ranks, không lấy predictions làm gold.
        QLoRA checkpoint giữ optimizer/RNG để resume. Đánh giá dev NDCG/MRR rồi
        chạy finalists qua full pipeline và CPU F2 cutoff sweep. Chọn adapter theo
        dev F2. Chạy dev ablations khi bật tùy chọn, rồi chấm held-out với adapter/cutoff đã
        freeze; không chọn model bằng held-out score và không promote corpus.
        Sau đó 05 replay parent windows 256/384/512/640/768 và merge 512→768 từ
        scored query checkpoints trên CPU (không nạp model hay gọi GPU lại). Dev
        chọn duy nhất một biến thể; nếu có held-out độc lập, chỉ biến thể đó được
        chấm thêm một lần và lưu chênh lệch so với parent policy gốc. Kết quả không
        tự thay cấu hình inference. Báo cáo audit ghi recall theo tầng, ngôn ngữ,
        độ dài gold và loại lỗi source/candidate/parent/cutoff.
        04 tự nhận selected_adapter/calibrated_policy trong namespace mới.

        V7: corpus embeddings được tìm và kiểm hash giữa 04/05 và các experiment;
        không tính lại corpus chỉ vì đổi tập query. Mining cũ được kiểm và replay
        trước khi nạp GPU. Mining mới dùng retrieval rộng, không chạy cascade 8B
        để tạo file chờ nhãn. Qwen reranker 8B vẫn dùng cho QLoRA/dev/held-out/04.
        Batch inference theo VRAM, có OOM backoff. Final adapter trùng checkpoint
        cuối được đánh giá một lần. Tám research ablations được để sau kết quả
        đầu tiên; API run_training_workflow(..., run_ablations=True) vẫn hỗ trợ.
        Thời gian từng bước ghi training/<run>/runtime_profile.json; từng embedding
        pass ghi runtime_profile.json bên cạnh parts. Các source/weights/hash cũ giữ nguyên.
        '''),
        md("## 1. Bootstrap — CPU"),
        code(BOOT.replace("retrieval_code_lock.json", "training_code_lock.json")
            .replace("retrieval_runtime.json", "training_runtime.json")
            .replace("full-master-plan-strong-v3-per-model-15b", "full-master-plan-supervised-v8-output-audit")),
        md("## 2. Chuẩn bị nhãn — GPU chỉ khi còn thiếu draft train"),
        code('''
        import json
        from pathlib import Path
        from vietmedbridge.training_data import prepare_training_data, install_ai_pilot_review
        AI_REVIEW_UPLOAD = Path("/content/source_review_ai_pilot.json")
        if AI_REVIEW_UPLOAD.is_file():
            print(json.dumps(install_ai_pilot_review(AI_REVIEW_UPLOAD, DATA_ROOT), ensure_ascii=False, indent=2))
        PREPARATION = prepare_training_data(DATA_ROOT, CHECKOUT, max_new_samples=None)
        print(json.dumps(PREPARATION, ensure_ascii=False, indent=2))
        if PREPARATION["state"] == "WAITING_FOR_SOURCE_QUERY_REVIEW":
            from google.colab import files
            files.download(PREPARATION["review_path"])
            print("Duyệt file, lưu lại đúng review_path trên Drive, rồi Run all lại.")
        '''),
        md('''
        ## 3. Hybrid mining → duyệt hard negatives → QLoRA — GPU

        Sau khi duyệt source queries, retriever chạy thật trên các query độc lập.
        Chưa đủ positives/7 reviewed negatives mỗi query thì tải review.json chứa
        source candidates/ranks. Team đánh POSITIVE/NEGATIVE/SKIP và reviewer;
        category có thể để trống (unclassified). Không suy diễn missing = negative.
        Lưu file về đúng review_path trên Drive rồi Run all lại. Vectors và scored
        stages được reuse; dữ liệu training thay đổi có checkpoint namespace mới.

        Nếu Sếp giao Codex duyệt, upload review_ai_pilot.json vào /content;
        cell sau tự nhập vào đúng review_path, giữ backup và quyết định team.
        Giữ checkpoint/run cũ. Khi còn chờ review, không cần giữ runtime GPU.
        '''),
        code('''
        from vietmedbridge.training_workflow import run_training_workflow
        from vietmedbridge.negative_review import install_ai_candidate_review
        NEGATIVE_REVIEW_UPLOAD = Path("/content/review_ai_pilot.json")
        if NEGATIVE_REVIEW_UPLOAD.is_file():
            print(json.dumps(install_ai_candidate_review(NEGATIVE_REVIEW_UPLOAD, DATA_ROOT), ensure_ascii=False, indent=2))
        if PREPARATION["state"] == "READY_FOR_HARD_NEGATIVE_REVIEW":
            TRAINING_STATUS = run_training_workflow(DATA_ROOT, CHECKOUT,
                work_dir=WORK_DIR, run_name="stage-a-qlora-v3-per-model-15b")
        else:
            TRAINING_STATUS = {"state": "WAITING_FOR_SOURCE_QUERY_REVIEW", "fine_tuned": False}
        print(json.dumps(TRAINING_STATUS, ensure_ascii=False, indent=2))
        if TRAINING_STATUS.get("review_path"):
            from google.colab import files
            files.download(TRAINING_STATUS["review_path"])
        '''),
        md('''
        Dừng ở bước review nghĩa là cần nhãn nguồn hoặc hard negatives; chưa có model fine-tune
        hoặc F2 được chứng minh. Dev F2, ablation và held-out đều là local proxy,
        chưa phải score BTC. Embedding fine-tune/sharded dense có API trong package,
        hướng dẫn ở docs/FULL_PLAN.md. Sau khi chọn adapter hợp lệ, mở 04 và Run all;
        không chỉnh ACTION hoặc xóa run pretrained. Nhãn/config đổi cần run mới.
        '''),
    ])


if __name__ == "__main__":
    main()
