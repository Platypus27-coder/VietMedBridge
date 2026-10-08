"""Update the existing 02/03 entrypoints for the team's 100k handoff."""
from write_notebooks import BOOTSTRAP, code, md, save

BOOT = (BOOTSTRAP.replace("code_lock.json","data_processing_code_lock.json")
    .replace("runtime.json","data_processing_runtime.json")
    .replace('reference = CODE_REVISION or lock.get("git_commit") or "main"',
        'DATA_WORKFLOW_API = "external-extraction-import-v5-cumulative-candidate-lineage"\n'
        'upgrade = lock.get("workflow_api") != DATA_WORKFLOW_API\n'
        'reference = CODE_REVISION or ("main" if upgrade else lock.get("git_commit")) or "main"')
    .replace('if not lock or CODE_REVISION:\n    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, "pipeline_api": PIPELINE_API_VERSION})',
        'if not lock or CODE_REVISION or upgrade:\n'
        '    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, '
        '"pipeline_api": PIPELINE_API_VERSION, "workflow_api": DATA_WORKFLOW_API})'))


def write_data_notebooks():
    save("02_colab_extract_and_chunk.ipynb",[
        md('''
        # VietMedBridge — 02: Nhập bản crawl → source + parent/child chunks

        **Runtime CPU, Run all.** Chọn nguồn ở form của cell cấu hình; nếu có nhiều
        archive, notebook hiện danh sách để chọn. Tên BUILD_RUN tự sinh ổn định theo
        archive và cấu hình, có thể nhập tay vào form để resume một run cũ.
        Hỗ trợ `vibiomir_shard_*.tar`, thư mục đã giải nén và `.tar.parts`.
        Frontier bên trong tar có thể mang số shard khác `00000`.
        Bản chia phần được tự ghép trên ổ local Colab và kiểm SHA-256 đúng bản gốc.
        Tái sử dụng văn bản EXTRACT_SUCCESS; không tải lại các website.
        Archive và snapshot BTC phải cùng SHA-256. Map bằng official URL, giữ toàn bộ
        official ID/alias và một outcome cho từng ID. ID thiếu dòng crawl/extraction
        được ghi thành failure với reason rõ ràng và đưa vào `external_coverage_gaps.csv`
        để truy hồi sau; URL mapping sai vẫn chặn import và có báo cáo riêng.
        Nguồn lỗi/challenge/redirect về trang chủ được ghi failures; LOW quality được giữ kèm flags.

        Mặc định tìm archive trong `VietMedBridge/data/incoming`,
        `VietMedBridge/data/data_temp` hoặc `VietMedBridge/data_temp` trên Drive.
        Đặt EXTERNAL_SOURCE nếu archive nằm nơi khác. Sau khi ghép (nếu cần), copy bốn file metadata
        từ archive sang ổ local Colab để đọc; raw archive vẫn giữ làm nguồn kiểm tra.
        Metadata binding và exact-source spans không chứng nhận chất lượng y khoa.

        Ghi theo shard 2.048 ID, checkpoint trên Drive. Khi ngắt, Run all với cùng
        đường dẫn input/BUILD_RUN/config để tiếp tục. Một runtime ghi một build.
        Không tạo embedding, không cần GPU. Sau COMPLETE, chạy notebook 03.
        '''),code(BOOT),md("## 1. Chọn nguồn và tokenizer"),code('''
        from vietmedbridge.artifacts import read_json, atomic_json
        from vietmedbridge.dataset import load_snapshot, parquet_path
        from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer
        from vietmedbridge.external_import import (
            external_build_run_name, find_external_source, import_external_corpus,
            list_external_sources,
        )

        INPUT_KIND = "external" #@param ["external", "crawl"]
        EXTERNAL_SOURCE = "" #@param {type:"string"}
        BUILD_RUN = "" #@param {type:"string"}
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
            if EXTERNAL_SOURCE.strip():
                SOURCE = find_external_source(DATA_ROOT, EXTERNAL_SOURCE.strip())
            else:
                sources = list_external_sources(DATA_ROOT)
                if not sources:
                    SOURCE = None
                elif len(sources) == 1:
                    SOURCE = sources[0]
                else:
                    print("Tìm thấy các nguồn crawl:")
                    for index, path in enumerate(sources, 1):
                        size = path.stat().st_size if path.is_file() else 0
                        print(f"  {index}. {path}" + (f" ({size:,} bytes)" if size else ""))
                    choice = input("Nhập số thứ tự nguồn muốn xử lý: ").strip()
                    try:
                        selected = int(choice) - 1
                    except ValueError as exc:
                        raise ValueError("Hãy nhập một số thứ tự trong danh sách nguồn.") from exc
                    if not 0 <= selected < len(sources):
                        raise ValueError("Số thứ tự nguồn nằm ngoài danh sách.")
                    SOURCE = sources[selected]
            if SOURCE is None:
                raise FileNotFoundError(
                    "Chưa thấy archive trên Drive. Điền đường dẫn vào trường EXTERNAL_SOURCE "
                    "hoặc thư mục giải nén. Vị trí mặc định: "
                    + str(DATA_ROOT / "incoming/vibiomir_shard_00000.tar"))
            if not BUILD_RUN.strip():
                BUILD_RUN = external_build_run_name(
                    SOURCE, tokenizer_spec=TOKENIZER_SPEC, chunking=CHUNK_CONFIG,
                    shard_size=SHARD_SIZE)
            print("Existing crawl:", SOURCE)
        elif INPUT_KIND != "crawl":
            raise ValueError("INPUT_KIND phải là external hoặc crawl.")
        elif not BUILD_RUN.strip():
            BUILD_RUN = "team-100k-data-v1"
        print("Build:", BUILD_RUN, "| tokenizer:", TOKENIZER_SPEC)
        '''),md("## 2. Nhập/chia chunk theo shard — CPU"),code('''
        if INPUT_KIND == "external":
            BUILD = import_external_corpus(
                SOURCE, OFFICIAL_LINKS, DATA_ROOT / "processed", TOKENIZER, TOKENIZER_SPEC,
                run_name=BUILD_RUN, config=CHUNK_CONFIG, shard_size=SHARD_SIZE,
                max_shards=MAX_NEW_SHARDS, work_dir=WORK_DIR)
        else:
            from vietmedbridge.build import build_corpus
            BUILD = build_corpus(DATA_ROOT / "crawl" / CRAWL_RUN, DATA_ROOT / "processed",
                TOKENIZER, TOKENIZER_SPEC, run_name=BUILD_RUN, config=CHUNK_CONFIG,
                max_shards=MAX_NEW_SHARDS, official_links=OFFICIAL_LINKS, work_dir=WORK_DIR)
        BUILD_DIR = DATA_ROOT / "processed" / BUILD_RUN
        print(json.dumps({k: BUILD[k] for k in (
            "state", "counts", "selected_shards", "requested_input_records", "selected_range_complete"
        )}, ensure_ascii=False, indent=2))
        if BUILD.get("source_audit"):
            print("Input audit:", BUILD["source_audit"]["state"])
            print("Coverage gaps:", json.dumps(BUILD["source_audit"]["coverage_gaps"], ensure_ascii=False))
            gap_artifact = BUILD["source_audit"]["coverage_gaps"]["artifact"]
            if gap_artifact:
                print("Gap review file:", BUILD_DIR / gap_artifact["path"])
        if BUILD["selected_range_complete"]:
            atomic_json(DATA_ROOT / "active_data_build.json", {
                "build_run": BUILD_RUN, "crawl_run": BUILD.get("crawl_run"),
                "signature": BUILD["signature"], "snapshot_sha256": BUILD["snapshot_sha256"],
                "input_kind": BUILD.get("input_kind", "CRAWL"), "state": BUILD["state"]})
            print("Hoàn tất input range. Chạy notebook 03 trên CPU.")
        else:
            raise RuntimeError("Import chưa đủ range. Run all cùng config để tiếp tục trước notebook 03.")
        '''),md("## 3. Kiểm tra một tài liệu và thời gian thực đo"),code('''
        import pyarrow.parquet as pq
        from vietmedbridge.build import verify_spans
        for part in BUILD["parts"]:
            path = BUILD_DIR / part["files"]["documents"]["path"]
            if pq.ParquetFile(path).metadata.num_rows == 0:
                continue
            doc = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
            children = pq.read_table(BUILD_DIR / part["files"]["children"]["path"], filters=[("doc_id", "=", doc["doc_id"])]).to_pylist()
            parents = pq.read_table(BUILD_DIR / part["files"]["parents"]["path"], filters=[("doc_id", "=", doc["doc_id"])]).to_pylist()
            verify_spans(doc, children, parents)
            print("Source:", doc["doc_id"], doc["title"], "| children:", len(children))
            print(children[0]["text"][:800] if children else "NO CHILDREN")
            break
        timing = BUILD_DIR / "import_runtime.json"
        if timing.exists():
            print(json.dumps(read_json(timing), ensure_ascii=False, indent=2))
        print("Failures được bảo toàn trong các shard, cùng ID BTC.")
        '''),md('''
        Outcome đủ range bao gồm nguồn lỗi, không có nghĩa 100% URL đã có nội dung tốt.
        Notebook 03 kiểm toàn snapshot, freeze candidate và chuẩn bị các phần input
        embedding trên disk. Notebook 04 sẽ đo tài nguyên trên mẫu nhỏ trước corpus lớn.
        ''')])
    save("03_colab_validate_and_freeze.ipynb",[
        md('''
        # VietMedBridge — 03: Kiểm toàn bộ dữ liệu → freeze → chia input embedding

        **Runtime CPU, Run all sau notebook 02.** Tự lấy đúng build từ active_data_build.json.
        Kiểm mọi official ID/URL, source hash, offsets, parent/child và bảo toàn IDs lỗi.
        Global dedup giữ toàn bộ aliases; input model giống hệt chỉ cần encode một lần.
        Chuẩn bị input embedding thành các file nhỏ trên disk, chưa nạp model GPU.

        Candidate freeze là mốc dữ liệu có integrity, không tự duyệt relevance hoặc
        human QA. Source audit/golden thật vẫn cần team review; không tạo nhãn thi.
        '''),code(BOOT),md("## 1. Đọc build hoàn tất và chạy regression của chunker"),code('''
        from vietmedbridge.artifacts import read_json, atomic_json
        from vietmedbridge.dataset import load_snapshot, parquet_path
        from vietmedbridge.chunks import ChunkConfig, load_bge_tokenizer
        from vietmedbridge.golden import run_golden_suite
        from vietmedbridge.health import health_report, freeze_candidate
        from vietmedbridge.index_inputs import prepare_index_inputs, publish_data_handoff

        BUILD_RUN_OVERRIDE = None  # dùng để chủ động đọc build cũ
        active_path = DATA_ROOT / "active_data_build.json"
        active = read_json(active_path) if active_path.exists() else {}
        BUILD_RUN = BUILD_RUN_OVERRIDE or active.get("build_run", "stage-a-data-v2-restored")
        BUILD_DIR = DATA_ROOT / "processed" / BUILD_RUN
        BUILD = read_json(BUILD_DIR / "build.json")
        if not BUILD.get("selected_range_complete"):
            raise RuntimeError("Build còn thiếu shard. Resume notebook 02 trước.")
        if not BUILD_RUN_OVERRIDE and active and active["snapshot_sha256"] != BUILD["snapshot_sha256"]:
            raise ValueError("Active build pointer và snapshot không khớp.")
        SNAPSHOT = load_snapshot(DATA_ROOT)
        OFFICIAL_LINKS = parquet_path(DATA_ROOT, "links_corpus.parquet")
        CONFIG = read_json(BUILD_DIR / "config.json")
        TOKENIZER, TOKENIZER_SPEC = load_bge_tokenizer(CONFIG["tokenizer"]["revision"])
        CHUNK_CONFIG = ChunkConfig(**CONFIG["chunks"])
        GOLDEN = run_golden_suite(CHECKOUT / "tests/golden/cases.json", TOKENIZER, TOKENIZER_SPEC, CHUNK_CONFIG)
        if not GOLDEN["passed"]:
            raise RuntimeError("Golden regression fail; chưa freeze candidate.")
        atomic_json(DATA_ROOT / "reports/golden_latest.json", GOLDEN)
        print("Build:", BUILD_RUN, "| requested IDs:", BUILD["requested_input_records"])
        '''),md("## 2. Health report và source audit — CPU"),code('''
        CRAWL_DIR = DATA_ROOT / "crawl" / BUILD["crawl_run"] if BUILD.get("crawl_run") else None
        HEALTH = health_report(BUILD_DIR, official_links=OFFICIAL_LINKS, crawl_dir=CRAWL_DIR,
            work_dir=WORK_DIR, audit_size=PIPELINE_CONFIG["audit_size"])
        print(json.dumps({k: HEALTH[k] for k in ("coverage", "documents", "chunks", "dedup")}, ensure_ascii=False, indent=2))
        from IPython.display import HTML, display
        display(HTML((BUILD_DIR / HEALTH["files"]["audit"]["path"]).read_text()))
        print("Human QA pending:", BUILD_DIR / HEALTH["files"]["golden_candidates"]["path"])
        '''),md("## 3. Freeze và chia input index theo phần — CPU"),code('''
        CANDIDATE = freeze_candidate(BUILD_DIR, HEALTH, golden_report=GOLDEN)
        CANDIDATE_NAME = f"candidate-{CANDIDATE['candidate_manifest_sha256'][:16]}.json"
        INPUTS = prepare_index_inputs(BUILD_DIR, CANDIDATE_NAME, TOKENIZER,
            work_dir=WORK_DIR, part_size=4096)
        HANDOFF = publish_data_handoff(DATA_ROOT, BUILD_RUN, CANDIDATE, INPUTS)
        print(json.dumps(HANDOFF, ensure_ascii=False, indent=2))
        print("Model input parts:", BUILD_DIR / "index_inputs/units.json")
        print("Tiếp theo: notebook 04 tự nhận candidate để benchmark GPU có giới hạn.")
        '''),md('''
        Candidate lớn không được nạp vào catalog RAM của pilot. Notebook 04 sẽ
        đọc một mẫu từ các input parts để đo BGE/Qwen/reranker và ghi chi phí,
        trước khi triển khai inference/index toàn corpus. Nhãn/relevance score không
        được suy từ throughput hoặc các kiểm tra integrity này.
        ''')])


if __name__ == "__main__":
    write_data_notebooks()
