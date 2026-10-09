"""Generate the multi-archive 02/03 Colab intake and freeze workflow."""
from write_notebooks import BOOTSTRAP, code, md, save

BOOT = (BOOTSTRAP.replace("code_lock.json","data_processing_code_lock.json")
    .replace("runtime.json","data_processing_runtime.json")
    .replace('reference = CODE_REVISION or lock.get("git_commit") or "main"',
        'DATA_WORKFLOW_API = "external-extraction-import-v7-parallel-workers"\n'
        'upgrade = lock.get("workflow_api") != DATA_WORKFLOW_API\n'
        'reference = CODE_REVISION or ("main" if upgrade else lock.get("git_commit")) or "main"')
    .replace('if not lock or CODE_REVISION:\n    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, "pipeline_api": PIPELINE_API_VERSION})',
        'if not lock or CODE_REVISION or upgrade:\n'
        '    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, '
        '"pipeline_api": PIPELINE_API_VERSION, "workflow_api": DATA_WORKFLOW_API})'))
BOOT_FREEZE = BOOT.replace(
    'external-extraction-import-v7-parallel-workers',
    'external-extraction-import-v11-freeze-drive-aliases',
)


def write_data_notebooks():
    save("02_colab_extract_and_chunk.ipynb",[
        md('''
        # VietMedBridge — 02: Nhập cả batch crawl → source + parent/child chunks

        **Runtime CPU, Run all.** `EXTERNAL_SOURCE` có thể là một file `.tar` hoặc
        thư mục batch chứa nhiều file `vibiomir_shard_*.tar`. Notebook nhập hết các
        archive trong thư mục theo thứ tự, mỗi archive có checkpoint/build riêng.
        Để tiếp tục sau restart, Run all lại với cùng batch folder.
        Tên build tự sinh ổn định theo từng archive; `BUILD_RUN` chỉ cần khi muốn
        đặt prefix riêng cho batch.
        Hỗ trợ `vibiomir_shard_*.tar`, thư mục đã giải nén và `.tar.parts`.
        Frontier bên trong tar có thể mang số shard khác `00000`.
        Bản chia phần được tự ghép trên ổ local Colab và kiểm SHA-256 đúng bản gốc.
        Tái sử dụng văn bản EXTRACT_SUCCESS; không tải lại các website.
        Archive và snapshot BTC phải cùng SHA-256. Map bằng official URL, giữ toàn bộ
        official ID/alias và một outcome cho từng ID. ID thiếu dòng crawl/extraction
        được ghi thành failure với reason rõ ràng và đưa vào `external_coverage_gaps.csv`
        để truy hồi sau; URL mapping sai vẫn chặn import và có báo cáo riêng.
        Nguồn lỗi/challenge/redirect về trang chủ được ghi failures; LOW quality được giữ kèm flags.

        Mặc định trỏ tới batch mới `data/incoming/team-crawl-archives-2026-10-08`.
        Có thể điền đường dẫn thư mục batch khác; không điền lại một archive cũ.
        Sau khi ghép (nếu cần), copy bốn file metadata
        từ archive sang ổ local Colab để đọc; raw archive vẫn giữ làm nguồn kiểm tra.
        Metadata binding và exact-source spans không chứng nhận chất lượng y khoa.

        Ghi theo shard 2.048 ID, checkpoint trên Drive. Chế độ batch chạy tuần tự một runtime.
        Để chia tải, dừng batch runner hiện tại rồi đặt RUN_MODE="worker", WORKER_ID
        khác nhau ở mỗi acc và EXTERNAL_SOURCE trỏ tới đúng một tar; để BUILD_RUN trống.
        Cùng một worker ID có thể chạy tiếp archive kế tiếp, manifest worker sẽ tích lũy.
        Chỉ chạy notebook 03 sau khi mọi worker xong và chọn BATCH_SOURCE="workers".
        Không chạy batch mode đồng thời với workers.
        Không tạo embedding, không cần GPU. Sau COMPLETE, chạy notebook 03.
        '''),code(BOOT),md("## 1. Chọn nguồn và tokenizer"),code('''
        from vietmedbridge.artifacts import read_json, atomic_json
        from vietmedbridge.dataset import load_snapshot, parquet_path
        from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer
        from vietmedbridge.external_import import (
            external_build_run_name, import_external_corpus, list_external_source_batches,
        )

        INPUT_KIND = "external" #@param ["external", "crawl"]
        EXTERNAL_SOURCE = "" #@param {type:"string"}
        BUILD_RUN = "" #@param {type:"string"}
        RUN_MODE = "batch" #@param ["batch", "worker"]
        WORKER_ID = "" #@param {type:"string"}
        CRAWL_RUN = "stage-a-v2"  # chỉ dùng với INPUT_KIND="crawl"
        SHARD_SIZE = 2048
        MAX_NEW_SHARDS = None  # None = xử lý hết; giữ cùng config khi resume

        SNAPSHOT = load_snapshot(DATA_ROOT)
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        tokenizer_lock_path = DATA_ROOT / "tokenizer_lock.json"
        lock = read_json(tokenizer_lock_path) if tokenizer_lock_path.exists() else {}
        TOKENIZER, TOKENIZER_SPEC = load_bge_tokenizer(lock.get("revision"))
        if not lock:
            atomic_json(tokenizer_lock_path, TOKENIZER_SPEC)
        CHUNK_CONFIG = ChunkConfig(**PIPELINE_CONFIG["chunking"])
        if INPUT_KIND == "external":
            if RUN_MODE == "worker" and not EXTERNAL_SOURCE.strip():
                raise ValueError("Worker mode cần EXTERNAL_SOURCE là đường dẫn đúng một file .tar.")
            if RUN_MODE == "worker" and BUILD_RUN.strip():
                raise ValueError("Worker mode phải để BUILD_RUN trống để tự resume đúng checkpoint.")
            source_selector = EXTERNAL_SOURCE.strip() or str(
                DATA_ROOT / "incoming" / "team-crawl-archives-2026-10-08")
            if not Path(source_selector).exists():
                raise FileNotFoundError(f"Không tìm thấy batch folder/archive: {source_selector}")
            EXTERNAL_SOURCES = list_external_source_batches(
                DATA_ROOT, source_selector)
            if not EXTERNAL_SOURCES:
                raise FileNotFoundError(
                    "Không tìm thấy archive trong folder. EXTERNAL_SOURCE phải là file .tar "
                    "hoặc folder chứa các archive .tar.")
            if RUN_MODE == "worker" and len(EXTERNAL_SOURCES) != 1:
                raise ValueError("Worker mode cần EXTERNAL_SOURCE trỏ tới đúng một file .tar.")
            if RUN_MODE not in {"batch", "worker"}:
                raise ValueError("RUN_MODE phải là batch hoặc worker.")
            if RUN_MODE == "worker":
                import re
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", WORKER_ID.strip()):
                    raise ValueError("WORKER_ID chỉ gồm chữ/số/_/-, dài tối đa 32 ký tự.")
                prior_run_names = set()
                manifests = [DATA_ROOT / "active_data_batch.json",
                    DATA_ROOT / "worker_batches" / f"{WORKER_ID.strip()}.json"]
                for manifest_path in manifests:
                    if manifest_path.is_file():
                        prior = read_json(manifest_path)
                        prior_run_names.update(item.get("build_run") for item in prior.get("builds", [])
                            if item.get("source_path") == str(EXTERNAL_SOURCES[0]) and item.get("build_run"))
                if len(prior_run_names) > 1:
                    raise ValueError("Checkpoint manifests disagree on this archive's BUILD_RUN.")
                if prior_run_names:
                    BUILD_RUNS = [next(iter(prior_run_names))]
                else:
                    BUILD_RUNS = [external_build_run_name(
                        EXTERNAL_SOURCES[0], tokenizer_spec=TOKENIZER_SPEC,
                        chunking=CHUNK_CONFIG, shard_size=SHARD_SIZE)]
            elif BUILD_RUN.strip() and len(EXTERNAL_SOURCES) == 1:
                BUILD_RUNS = [BUILD_RUN.strip()]
            else:
                import re
                prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", BUILD_RUN.strip()).strip("-_.")
                BUILD_RUNS = []
                for source in EXTERNAL_SOURCES:
                    auto_name = external_build_run_name(
                        source, tokenizer_spec=TOKENIZER_SPEC, chunking=CHUNK_CONFIG,
                        shard_size=SHARD_SIZE)
                    if prefix:
                        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(source).stem).strip("-_.")[:48]
                        auto_name = f"{prefix}-{label}-{auto_name.rsplit('-', 1)[-1]}"
                    BUILD_RUNS.append(auto_name)
            print(f"Batch sources: {len(EXTERNAL_SOURCES)} archive(s)")
            print(f"Total archive bytes: {sum(path.stat().st_size for path in EXTERNAL_SOURCES if path.is_file()):,}")
            for source, run_name in zip(EXTERNAL_SOURCES, BUILD_RUNS):
                size = source.stat().st_size if source.is_file() else 0
                print(f"  {source.name} | {size:,} bytes | build={run_name}")
        elif INPUT_KIND != "crawl":
            raise ValueError("INPUT_KIND phải là external hoặc crawl.")
        elif not BUILD_RUN.strip():
            BUILD_RUN = "team-100k-data-v1"
        if INPUT_KIND == "crawl":
            print("Build:", BUILD_RUN, "| tokenizer:", TOKENIZER_SPEC)
        '''),md("## 2. Nhập/chia chunk theo shard — CPU"),code('''
        from vietmedbridge.artifacts import utc_now
        import re
        BATCH_PATH = DATA_ROOT / "active_data_batch.json"
        if INPUT_KIND == "external":
            if RUN_MODE == "worker":
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", WORKER_ID.strip()):
                    raise ValueError("WORKER_ID chỉ gồm chữ/số/_/-, dài tối đa 32 ký tự.")
                worker_path = DATA_ROOT / "worker_batches" / f"{WORKER_ID.strip()}.json"
                worker_path.parent.mkdir(parents=True, exist_ok=True)
                BATCH = read_json(worker_path) if worker_path.exists() else {
                    "schema_version": 1, "input_kind": "external", "worker_id": WORKER_ID.strip(),
                    "state": "IMPORTING", "builds": []}
                if BATCH.get("worker_id") != WORKER_ID.strip():
                    raise ValueError("Worker manifest identity mismatch.")
                BATCH["state"] = "IMPORTING"
                for source, run_name in zip(EXTERNAL_SOURCES, BUILD_RUNS):
                    source_path = str(source)
                    matching = [item for item in BATCH["builds"]
                                if item.get("source_path") == source_path]
                    if matching and any(item.get("build_run") != run_name for item in matching):
                        raise ValueError("Archive đã có build khác trong worker manifest.")
                    if matching:
                        item = matching[0]
                        item["state"] = "IMPORTING"
                    else:
                        item = {"source_path": source_path, "build_run": run_name, "state": "IMPORTING"}
                        BATCH["builds"].append(item)
                    BATCH["updated_at"] = utc_now()
                    atomic_json(worker_path, BATCH)
                    BUILD = import_external_corpus(
                        source, OFFICIAL_LINKS, DATA_ROOT / "processed", TOKENIZER, TOKENIZER_SPEC,
                        run_name=run_name, config=CHUNK_CONFIG, shard_size=SHARD_SIZE,
                        max_shards=MAX_NEW_SHARDS, work_dir=WORK_DIR)
                    item.update({"state": "COMPLETE" if BUILD["selected_range_complete"] else "INCOMPLETE",
                        "signature": BUILD["signature"], "snapshot_sha256": BUILD["snapshot_sha256"],
                        "requested_input_records": BUILD["requested_input_records"], "counts": BUILD["counts"]})
                    BATCH["updated_at"] = utc_now()
                    atomic_json(worker_path, BATCH)
                    print(f"Worker {WORKER_ID}: {source.name} -> {item['state']}")
                    if not BUILD["selected_range_complete"]:
                        raise RuntimeError("Archive chưa đủ shard. Run all lại với cùng worker ID/source để resume.")
                BATCH["state"] = "COMPLETE" if all(
                    item.get("state") == "COMPLETE" for item in BATCH["builds"]) else "IMPORTING"
                BATCH["updated_at"] = utc_now()
                atomic_json(worker_path, BATCH)
                BUILDS = [{"build_run": item["build_run"], "build": read_json(
                    DATA_ROOT / "processed" / item["build_run"] / "build.json")}
                    for item in BATCH["builds"] if item.get("state") == "COMPLETE"]
                BUILD_RUN = BUILD_RUNS[-1]
                BUILD = read_json(DATA_ROOT / "processed" / BUILD_RUN / "build.json")
                print("Worker checkpoint:", worker_path)
            else:
                previous_batch = read_json(BATCH_PATH) if BATCH_PATH.exists() else {}
                previous_by_key = {
                    (item.get("source_path"), item.get("build_run")): item
                    for item in previous_batch.get("builds", [])
                }
                batch_builds = []
                for source, run_name in zip(EXTERNAL_SOURCES, BUILD_RUNS):
                    key = (str(source), run_name)
                    prior = previous_by_key.get(key, {})
                    batch_builds.append({**prior, "source_path": str(source), "build_run": run_name,
                                         "state": "PENDING"})
                BATCH = {"schema_version": 1, "input_kind": "external", "state": "IMPORTING",
                         "builds": batch_builds, "updated_at": None}
                atomic_json(BATCH_PATH, BATCH)
                for index, item in enumerate(BATCH["builds"], 1):
                    item["state"] = "IMPORTING"
                    atomic_json(BATCH_PATH, BATCH)
                    BUILD = import_external_corpus(
                        Path(item["source_path"]), OFFICIAL_LINKS, DATA_ROOT / "processed", TOKENIZER, TOKENIZER_SPEC,
                        run_name=item["build_run"], config=CHUNK_CONFIG, shard_size=SHARD_SIZE,
                        max_shards=MAX_NEW_SHARDS, work_dir=WORK_DIR)
                    item.update({"state": "COMPLETE" if BUILD["selected_range_complete"] else "INCOMPLETE",
                        "signature": BUILD["signature"], "snapshot_sha256": BUILD["snapshot_sha256"],
                        "requested_input_records": BUILD["requested_input_records"], "counts": BUILD["counts"]})
                    atomic_json(BATCH_PATH, BATCH)
                    print(f"Archive {index}/{len(BATCH['builds'])}: {item['source_path']} -> {item['state']}")
                    if not BUILD["selected_range_complete"]:
                        raise RuntimeError("Batch chưa đủ shard. Run all lại với cùng folder để resume.")
                BATCH["state"] = "COMPLETE"
                BATCH["updated_at"] = utc_now()
                atomic_json(BATCH_PATH, BATCH)
                BUILDS = [{"build_run": item["build_run"], "build": read_json(
                    DATA_ROOT / "processed" / item["build_run"] / "build.json")} for item in BATCH["builds"]]
                BUILD_RUN = BUILDS[-1]["build_run"]
                BUILD = BUILDS[-1]["build"]
        else:
            if RUN_MODE != "batch":
                raise ValueError("Worker mode only supports external archives.")
            from vietmedbridge.build import build_corpus
            BUILD = build_corpus(DATA_ROOT / "crawl" / CRAWL_RUN, DATA_ROOT / "processed",
                TOKENIZER, TOKENIZER_SPEC, run_name=BUILD_RUN, config=CHUNK_CONFIG,
                max_shards=MAX_NEW_SHARDS, official_links=OFFICIAL_LINKS, work_dir=WORK_DIR)
            BUILDS = [{"build_run": BUILD_RUN, "build": BUILD}]
            BATCH = {"schema_version": 1, "input_kind": "crawl", "state":
                "COMPLETE" if BUILD["selected_range_complete"] else "INCOMPLETE",
                "builds": [{"build_run": BUILD_RUN, "state": "COMPLETE" if BUILD["selected_range_complete"] else "INCOMPLETE",
                            "signature": BUILD["signature"], "snapshot_sha256": BUILD["snapshot_sha256"]}]}
            atomic_json(BATCH_PATH, BATCH)
        BUILD_DIR = DATA_ROOT / "processed" / BUILD_RUN
        TOTAL_COUNTS = {key: sum(item["build"].get("counts", {}).get(key, 0) for item in BUILDS)
                        for key in sorted({name for item in BUILDS for name in item["build"].get("counts", {})})}
        print(json.dumps({"state": BATCH["state"], "archives": len(BUILDS),
            "requested_input_records": sum(item["build"].get("requested_input_records", 0) for item in BUILDS),
            "counts": TOTAL_COUNTS, "build_runs": [item["build_run"] for item in BUILDS]},
            ensure_ascii=False, indent=2))
        for item in BUILDS:
            build_dir = DATA_ROOT / "processed" / item["build_run"]
            source_audit = item["build"].get("source_audit")
            if source_audit:
                print(item["build_run"], "input audit:", source_audit["state"])
                print("Coverage gaps:", json.dumps(source_audit["coverage_gaps"], ensure_ascii=False))
                gap_artifact = source_audit["coverage_gaps"]["artifact"]
                if gap_artifact:
                    print("Gap review file:", build_dir / gap_artifact["path"])
        if BATCH["state"] == "COMPLETE":
            if INPUT_KIND == "external" and RUN_MODE == "worker":
                print("Worker hoàn tất. Chờ mọi worker xong, rồi chạy notebook 03 với BATCH_SOURCE='workers'.")
            else:
                atomic_json(DATA_ROOT / "active_data_build.json", {
                    "build_run": BUILD_RUN, "crawl_run": BUILD.get("crawl_run"),
                    "signature": BUILD["signature"], "snapshot_sha256": BUILD["snapshot_sha256"],
                    "input_kind": BUILD.get("input_kind", "CRAWL"), "state": BUILD["state"]})
                print("Hoàn tất toàn batch. Chạy notebook 03 trên CPU.")
        else:
            raise RuntimeError("Input range chưa hoàn tất. Run all cùng config để resume.")
        '''),md("## 3. Kiểm tra một tài liệu và thời gian thực đo"),code('''
        import pyarrow.parquet as pq
        from vietmedbridge.build import verify_spans
        for build_item in BUILDS:
            build_dir = DATA_ROOT / "processed" / build_item["build_run"]
            current_build = build_item["build"]
            found_sample = False
            for part in current_build["parts"]:
                path = build_dir / part["files"]["documents"]["path"]
                if pq.ParquetFile(path).metadata.num_rows == 0:
                    continue
                doc = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
                children = pq.read_table(build_dir / part["files"]["children"]["path"], filters=[("doc_id", "=", doc["doc_id"])]).to_pylist()
                parents = pq.read_table(build_dir / part["files"]["parents"]["path"], filters=[("doc_id", "=", doc["doc_id"])]).to_pylist()
                verify_spans(doc, children, parents)
                print("Source:", build_item["build_run"], doc["doc_id"], doc["title"], "| children:", len(children))
                print(children[0]["text"][:800] if children else "NO CHILDREN")
                found_sample = True
                break
            timing = build_dir / "import_runtime.json"
            if timing.exists():
                print(build_item["build_run"], json.dumps(read_json(timing), ensure_ascii=False, indent=2))
            if not found_sample:
                print(build_item["build_run"], "no parsed document in sample scan")
        print("Failures được bảo toàn trong các shard, cùng ID BTC.")
        '''),md('''
        Outcome đủ range bao gồm nguồn lỗi, không có nghĩa 100% URL đã có nội dung tốt.
        Notebook 03 kiểm toàn snapshot, freeze candidate và chuẩn bị các phần input
        embedding trên disk. Notebook 04 sẽ đo tài nguyên trên mẫu nhỏ trước corpus lớn.
        ''')])
    save("03_colab_validate_and_freeze.ipynb",[
        md('''
        # VietMedBridge — 03: Kiểm toàn bộ dữ liệu → freeze → chia input embedding

        **Runtime CPU, chạy sau notebook 02.** Đã đặt sẵn chế độ 3 worker,
        đọc mọi archive trong `DATA_ROOT/incoming`, gồm `.tar.parts` và các file tar.
        Giữ Bootstrap với DATA_ROOT đã dùng ở 02 và CODE_REVISION=None.
        Ba tài khoản chọn FREEZE_WORKER_ID lần lượt 0, 1, 2 trên form rồi Run all.
        Mỗi worker xử lý một tập build riêng, checkpoint riêng, không ghi manifest tổng.
        Chỉ một acc chạy `FREEZE_MODE="coordinator"` sau khi các worker COMPLETE.
        Kiểm mọi official ID/URL, source hash, offsets, parent/child và bảo toàn IDs lỗi.
        Global dedup giữ toàn bộ aliases; input model giống hệt chỉ cần encode một lần.
        Chuẩn bị input embedding thành các file nhỏ trên disk, chưa nạp model GPU.

        Mỗi archive được freeze/checkpoint riêng; batch resume tiếp build chưa xong.
        Candidate freeze là mốc integrity, không tự duyệt relevance hoặc human QA.
        Source audit/golden thật vẫn cần team review; không tạo nhãn thi.
        '''),code(BOOT_FREEZE),md("## 1. Đọc build hoàn tất và chạy regression của chunker"),code('''
        #@title Phân công worker — chọn FREEZE_WORKER_ID: 0, 1 hoặc 2
        from vietmedbridge.artifacts import read_json, atomic_json, digest_json, code_fingerprint
        from vietmedbridge.dataset import load_snapshot, parquet_path
        from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer
        from vietmedbridge.golden import run_golden_suite
        from vietmedbridge.index_inputs import publish_data_handoff
        from vietmedbridge.external_import import list_external_source_batches
        sys.path.insert(0, str(CHECKOUT / "scripts"))
        from freeze_workers import (
            freeze_batch_signature, freeze_worker_manifest_path,
            partition_freeze_builds, verify_freeze_workers,
            write_freeze_worker_manifest, collect_completed_build_refs,
            freeze_build_for_notebook, audit_freeze_batch, verify_freeze_record,
        )

        BUILD_RUN_OVERRIDE = None  # dùng để chủ động đọc build cũ
        BATCH_SOURCE = "workers" #@param ["active_batch", "workers"]
        WORKER_EXPECTED_SOURCE = "" #@param {type:"string"}
        FREEZE_MODE = "worker" #@param ["serial", "worker", "coordinator"]
        FREEZE_TEAM_SIZE = 3 #@param {type:"integer"}
        FREEZE_WORKER_ID = 0 #@param {type:"integer"}
        if FREEZE_MODE != "serial" and (BATCH_SOURCE != "workers" or BUILD_RUN_OVERRIDE):
            raise ValueError("Parallel freeze requires BATCH_SOURCE='workers' and no BUILD_RUN_OVERRIDE.")
        batch_path = DATA_ROOT / "active_data_batch.json"
        if BUILD_RUN_OVERRIDE:
            build_refs = [{"build_run": BUILD_RUN_OVERRIDE}]
            BATCH = {"schema_version": 1, "state": "COMPLETE", "builds": build_refs}
        elif BATCH_SOURCE == "workers":
            expected_source = WORKER_EXPECTED_SOURCE.strip() or str(DATA_ROOT / "incoming")
            expected_paths = [str(path.resolve()) for path in list_external_source_batches(
                DATA_ROOT, expected_source)]
            if not expected_paths:
                raise FileNotFoundError(f"Không tìm thấy archive nguồn: {expected_source}")
            prior = read_json(batch_path) if batch_path.exists() else {}
            worker_dir = DATA_ROOT / "worker_batches"
            worker_files = sorted(worker_dir.glob("*.json")) if worker_dir.exists() else []
            workers = [(path.name, read_json(path)) for path in worker_files]
            build_refs = collect_completed_build_refs(expected_paths, prior, workers, data_root=DATA_ROOT)
            BATCH = {"schema_version": 1, "input_kind": "external", "state": "COMPLETE",
                "builds": build_refs, "assembled_from_workers": [path.name for path in worker_files]}
            if FREEZE_MODE == "serial":
                atomic_json(batch_path, BATCH)
            print(f"Đã hợp nhất {len(worker_files)} worker manifest; đủ {len(build_refs)} archive.")
        else:
            if batch_path.exists():
                BATCH = read_json(batch_path)
            else:
                active_path = DATA_ROOT / "active_data_build.json"
                active = read_json(active_path) if active_path.exists() else {}
                if active.get("build_run"):
                    BATCH = {"schema_version": 1, "state": "COMPLETE", "builds": [{
                        "build_run": active["build_run"], "snapshot_sha256": active.get("snapshot_sha256")}]}
                else:
                    BATCH = {}
            if BATCH.get("state") != "COMPLETE" or not BATCH.get("builds"):
                raise RuntimeError("Notebook 02 chưa hoàn tất toàn batch. Chạy 02 trước.")
            build_refs = BATCH["builds"]
        ACTIVE_BUILDS = []
        for item in build_refs:
            run_name = item["build_run"]
            build_dir = DATA_ROOT / "processed" / run_name
            build = read_json(build_dir / "build.json")
            if not build.get("selected_range_complete"):
                raise RuntimeError(f"Build {run_name} còn shard thiếu; resume notebook 02.")
            if item.get("snapshot_sha256") and item["snapshot_sha256"] != build["snapshot_sha256"]:
                raise ValueError(f"Batch pointer và snapshot {run_name} không khớp.")
            config = read_json(build_dir / "config.json")
            ACTIVE_BUILDS.append({"build_run": run_name, "build_dir": build_dir,
                                  "build": build, "config": config, "batch_item": item})
        BUILD_RUN = ACTIVE_BUILDS[-1]["build_run"]
        BUILD_DIR = ACTIVE_BUILDS[-1]["build_dir"]
        BUILD = ACTIVE_BUILDS[-1]["build"]
        SNAPSHOT = load_snapshot(DATA_ROOT)
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        CONFIG = ACTIVE_BUILDS[0]["config"]
        for item in ACTIVE_BUILDS[1:]:
            if item["config"]["tokenizer"] != CONFIG["tokenizer"] or item["config"]["chunks"] != CONFIG["chunks"]:
                raise ValueError("Các archive có tokenizer/chunk policy khác nhau; không được gộp âm thầm.")
        TOKENIZER, TOKENIZER_SPEC = load_bge_tokenizer(CONFIG["tokenizer"]["revision"])
        CHUNK_CONFIG = ChunkConfig(**CONFIG["chunks"])
        GOLDEN = run_golden_suite(CHECKOUT / "tests/golden/cases.json", TOKENIZER, TOKENIZER_SPEC, CHUNK_CONFIG)
        if not GOLDEN["passed"]:
            raise RuntimeError("Golden regression fail; chưa freeze candidate.")
        if FREEZE_MODE != "worker":
            atomic_json(DATA_ROOT / "reports/golden_latest.json", GOLDEN)
        print("Batch builds:", len(ACTIVE_BUILDS), "| requested IDs:",
              sum(item["build"]["requested_input_records"] for item in ACTIVE_BUILDS))
        print("Freeze mode:", FREEZE_MODE, "| team size:", FREEZE_TEAM_SIZE)
        '''),md("## 2. Kiểm, freeze và chuẩn bị input toàn batch — CPU"),code('''
        HANDOFFS = []
        AUDIT_RECORDS = []
        def freeze_one(item):
            return freeze_build_for_notebook(item, data_root=DATA_ROOT,
                official_links=OFFICIAL_LINKS, tokenizer=TOKENIZER,
                golden_report=GOLDEN, work_dir=WORK_DIR,
                audit_size=PIPELINE_CONFIG["audit_size"])

        if FREEZE_MODE == "worker":
            if type(FREEZE_TEAM_SIZE) is not int or FREEZE_TEAM_SIZE < 1:
                raise ValueError("FREEZE_TEAM_SIZE must be a positive integer.")
            freeze_refs = [{"build_run":item["build_run"],
                "source_path":item["batch_item"].get("source_path"),
                "snapshot_sha256":item["build"]["snapshot_sha256"]} for item in ACTIVE_BUILDS]
            freeze_code_sha = code_fingerprint()
            freeze_signature = freeze_batch_signature(freeze_refs,
                code_sha256=freeze_code_sha, team_size=FREEZE_TEAM_SIZE)
            assigned = partition_freeze_builds(freeze_refs,
                team_size=FREEZE_TEAM_SIZE, worker_id=FREEZE_WORKER_ID)
            assigned_runs = [item["build_run"] for item in assigned]
            if not assigned_runs:
                raise ValueError("This worker has no assigned builds; reduce FREEZE_TEAM_SIZE.")
            worker_path = freeze_worker_manifest_path(DATA_ROOT,
                batch_signature=freeze_signature, worker_id=FREEZE_WORKER_ID)
            completed = {}
            if worker_path.is_file():
                previous = read_json(worker_path)
                previous_payload = {key:value for key,value in previous.items()
                                    if key != "manifest_sha256"}
                if (previous.get("manifest_sha256") != digest_json(previous_payload)
                    or previous.get("batch_signature") != freeze_signature
                    or previous.get("worker_id") != FREEZE_WORKER_ID
                    or previous.get("team_size") != FREEZE_TEAM_SIZE
                    or previous.get("assigned_build_runs") != assigned_runs):
                    raise ValueError("Freeze worker checkpoint does not match this assignment.")
                completed = previous.get("completed", {})
            write_freeze_worker_manifest(worker_path, batch_signature=freeze_signature,
                code_sha256=freeze_code_sha, team_size=FREEZE_TEAM_SIZE,
                worker_id=FREEZE_WORKER_ID, assigned_build_runs=assigned_runs,
                completed=completed, state="RUNNING")
            by_run = {item["build_run"]:item for item in ACTIVE_BUILDS}
            for index, run_name in enumerate(assigned_runs, 1):
                candidate, inputs, record = freeze_one(by_run[run_name])
                completed[run_name] = record
                write_freeze_worker_manifest(worker_path, batch_signature=freeze_signature,
                    code_sha256=freeze_code_sha, team_size=FREEZE_TEAM_SIZE,
                    worker_id=FREEZE_WORKER_ID, assigned_build_runs=assigned_runs,
                    completed=completed, state="RUNNING")
                print(f"Worker {FREEZE_WORKER_ID}: {index}/{len(assigned_runs)} {run_name} COMPLETE",
                      flush=True)
            write_freeze_worker_manifest(worker_path, batch_signature=freeze_signature,
                code_sha256=freeze_code_sha, team_size=FREEZE_TEAM_SIZE,
                worker_id=FREEZE_WORKER_ID, assigned_build_runs=assigned_runs,
                completed=completed, state="COMPLETE")
            print(json.dumps({"state":"FREEZE_WORKER_COMPLETE",
                "worker_id":FREEZE_WORKER_ID, "team_size":FREEZE_TEAM_SIZE,
                "assigned_builds":assigned_runs, "checkpoint":str(worker_path)},
                ensure_ascii=False, indent=2))
        elif FREEZE_MODE == "coordinator":
            freeze_refs = [{"build_run":item["build_run"],
                "source_path":item["batch_item"].get("source_path"),
                "snapshot_sha256":item["build"]["snapshot_sha256"]} for item in ACTIVE_BUILDS]
            freeze_code_sha = code_fingerprint()
            freeze_signature = freeze_batch_signature(freeze_refs,
                code_sha256=freeze_code_sha, team_size=FREEZE_TEAM_SIZE)
            records = verify_freeze_workers(DATA_ROOT, freeze_refs,
                batch_signature=freeze_signature, code_sha256=freeze_code_sha,
                team_size=FREEZE_TEAM_SIZE)
            COVERAGE = audit_freeze_batch(DATA_ROOT, freeze_refs, records, work_dir=WORK_DIR)
            for item, record in zip(ACTIVE_BUILDS, records):
                candidate, inputs = record["candidate"], record["inputs"]
                handoff = publish_data_handoff(DATA_ROOT, item["build_run"], candidate, inputs)
                item["batch_item"].update({"state":"FROZEN",
                    "candidate_name":record["candidate_name"],
                    "candidate_manifest_sha256":record["candidate_manifest_sha256"],
                    "index_inputs_manifest_sha256":record["index_inputs_manifest_sha256"]})
                HANDOFFS.append(handoff)
            BATCH["freeze_state"] = "COMPLETE"
            BATCH["coverage_report_sha256"] = COVERAGE["report_sha256"]
            atomic_json(batch_path, BATCH)
            print(json.dumps({**COVERAGE, "state":"BATCH_FROZEN", "builds":len(HANDOFFS),
                "documents":sum(item["documents"] for item in HANDOFFS),
                "children":sum(item["children"] for item in HANDOFFS),
                "lineage_state":read_json(DATA_ROOT / "candidate_lineage.json")["state"],
                "lineage_candidates":len(read_json(DATA_ROOT / "candidate_lineage.json")["sources"])},
                ensure_ascii=False, indent=2))
            print("Coordinator đã xác minh checkpoint của mọi worker và công bố batch. Tiếp theo chạy notebook 04.")
            print("Coverage report:", DATA_ROOT / "reports/freeze_batch_coverage.json")
        else:
            for index, item in enumerate(ACTIVE_BUILDS, 1):
                candidate, inputs, record = freeze_one(item)
                verified = verify_freeze_record(DATA_ROOT, item["build_run"],
                    {"snapshot_sha256":item["build"]["snapshot_sha256"]}, record, code_fingerprint())
                HANDOFF = publish_data_handoff(DATA_ROOT, item["build_run"], candidate, inputs)
                item["batch_item"].update({"state":"FROZEN", "candidate_name":record["candidate_name"],
                    "candidate_manifest_sha256":record["candidate_manifest_sha256"],
                    "index_inputs_manifest_sha256":record["index_inputs_manifest_sha256"]})
                atomic_json(batch_path, BATCH)
                HANDOFFS.append(HANDOFF)
                AUDIT_RECORDS.append(verified)
                print(f"Freeze {index}/{len(ACTIVE_BUILDS)}: {item['build_run']} | documents="
                      f"{candidate['counts']['documents']:,} | children={candidate['counts']['children']:,}")
            audit_refs = [{"build_run":item["build_run"], "snapshot_sha256":item["build"]["snapshot_sha256"]}
                          for item in ACTIVE_BUILDS]
            COVERAGE = audit_freeze_batch(DATA_ROOT, audit_refs, AUDIT_RECORDS, work_dir=WORK_DIR)
            BATCH["freeze_state"] = "COMPLETE"
            BATCH["coverage_report_sha256"] = COVERAGE["report_sha256"]
            atomic_json(batch_path, BATCH)
            print(json.dumps({**COVERAGE, "state":"BATCH_FROZEN", "builds":len(HANDOFFS),
                "documents":sum(item["documents"] for item in HANDOFFS),
                "children":sum(item["children"] for item in HANDOFFS),
                "lineage_state":read_json(DATA_ROOT / "candidate_lineage.json")["state"],
                "lineage_candidates":len(read_json(DATA_ROOT / "candidate_lineage.json")["sources"])},
                ensure_ascii=False, indent=2))
            print("Model input parts và checkpoints đã lưu riêng trong từng build. Tiếp theo chạy notebook 04.")
            print("Coverage report:", DATA_ROOT / "reports/freeze_batch_coverage.json")
        '''),md('''
        ### Chạy song song trên 3 tài khoản

        Cả ba tài khoản dùng cùng DATA_ROOT, cùng BATCH_SOURCE/WORKER_EXPECTED_SOURCE,
        FREEZE_TEAM_SIZE=3 và FREEZE_MODE="worker"; đặt FREEZE_WORKER_ID lần lượt
        0, 1, 2. Các worker nhận 3/2/2 build theo thứ tự ổn định và chỉ ghi vào
        thư mục build riêng cùng `data/freeze_workers/<batch-signature>/worker-N.json`.
        Mỗi build xong được checkpoint; chạy lại đúng worker ID để resume.
        Không chạy hai runtime cùng worker ID.

        Khi cả ba manifest báo COMPLETE, một tài khoản đổi sang
        FREEZE_MODE="coordinator", giữ FREEZE_TEAM_SIZE=3 và chạy lại notebook.
        Coordinator xác minh đủ build/candidate/index-input hashes rồi mới cập nhật
        manifest tổng, candidate lineage và active pointer. Không chạy coordinator
        khi còn worker đang làm. Notebook này vẫn chạy CPU; embedding dùng GPU ở 04.

        Candidate lớn không được nạp vào catalog RAM của pilot. Notebook 04 sẽ
        đọc một mẫu từ các input parts để đo BGE/Qwen/reranker và ghi chi phí,
        trước khi triển khai inference/index toàn corpus. Nhãn/relevance score không
        được suy từ throughput hoặc các kiểm tra integrity này.
        ''')])


if __name__ == "__main__":
    write_data_notebooks()
