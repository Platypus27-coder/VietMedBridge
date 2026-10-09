"""Generate the full inference notebook and the supervised training notebook."""
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


def main():
    save("04_colab_retrieval_baseline.ipynb", [
        md('''
        # VietMedBridge — 04: Full retrieval system trên frozen corpus

        Plan chính: `R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md`.
        Lần đầu hoặc khi ghép data mới: chọn **CPU**, giữ `PREPARE_ONLY=True`, Run all.
        Chờ `CPU_PREPARATION_COMPLETE` rồi chuyển sang **GPU, T4 trở lên**,
        đặt `PREPARE_ONLY=False` để chạy embeddings/inference. Giữ DATA_ROOT cũ;
        không chạy lại 00–03 và không chọn ACTION. Bootstrap tự clone/cài package.

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

        **Chạy một người:** sau chuẩn bị CPU, PREPARE_ONLY=False, TEAM_SIZE=1,
        TEAM_WORKER_ID=None, Run all trên GPU.
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
        ablation trong module/05. **Hiện chưa có train/dev labels** nên 04 dùng
        pretrained models và cutoff chưa tune; không giả lập fine-tune hoặc điểm.
        Nếu đã chạy 05 với nhãn độc lập hợp lệ, 04 tự đọc adapter/dev policy đã chọn,
        tạo namespace -ft- mới. Nguồn/candidate cũ và các kết quả baseline được giữ.

        Với pilot: reuse verified corpus/document vectors từ cả 04 và 05. Chỉ encode
        inputs còn thiếu. Batch Qwen tự chọn theo VRAM và giảm nếu OOM.
        Theo dõi runtime_profile.json/execution_policy.json trong thư mục run;
        mỗi embedding pass cũng ghi thời gian encode, số parts mới/đã cache.
        Model revisions của full pilot giữ nguyên.
        '''),
        md("## 1. Bootstrap, mount Drive, clone và cài dependencies — CPU"),
        code(BOOT.replace("full-master-plan-strong-v3-per-model-15b", "full-master-plan-strong-v11-catalog-recovery")),
        md('''
        ## 2. Full system / phần embedding được giao — CPU/GPU lần lượt

        Mặc định `PREPARE_ONLY=True`: chạy trên CPU để ghép/freeze/chuẩn bị inputs
        và catalog BM25. Dừng ở CPU_PREPARATION_COMPLETE, không nạp model weights.
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
        from vietmedbridge.full_scale_runtime import run_scale_full_pipeline, prepare_scale_cpu_resources
        from vietmedbridge.artifacts import read_json, digest_json
        from vietmedbridge.index_inputs import read_candidate_lineage

        BUILD_RUN = "stage-a-data-v3-laodong"
        CANDIDATE_NAME = "candidate-1cd220a4be956d5a.json"
        RUN_NAME = "stage-a-full-plan-v3-per-model-15b"
        EMBEDDING_CACHE_RUN = "stage-a-retrieval-v1"
        MAX_NEW_EMBEDDING_PARTS = None
        MAX_NEW_TRANSLATIONS = None
        MAX_NEW_QUERIES = None
        PREPARE_ONLY = True  # CPU: ghép/freeze/inputs/BM25; GPU workers/inference: False
        TEAM_SIZE = 3  # team hiện có 3 người; đặt 1 nếu chạy một mình
        TEAM_WORKER_ID = None  # None: tổng hợp/full inference; thành viên: 0, 1 hoặc 2

        # Notebook 03 appends each newly frozen candidate to this Drive manifest.
        # The coordinator composes the old corpus plus new batch automatically.
        LINEAGE_SOURCES = read_candidate_lineage(DATA_ROOT)
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
        if PREPARE_ONLY:
            status = prepare_scale_cpu_resources(DATA_ROOT, CHECKOUT, WORK_DIR) if LARGE_CANDIDATE else {
                "state":"CPU_PREPARATION_COMPLETE", "build_run":BUILD_RUN,
                "scope":"Frozen pilot inputs ready; set PREPARE_ONLY=False on GPU for inference"}
            RESULT = {"status":status,"ready":{"state":status["state"],"query_count":0},
                "samples":[],"diagnostics_path":str(DATA_ROOT / "retrieval/cpu_preparation")}
            READY = RESULT["ready"]
        elif LARGE_CANDIDATE:
            RESULT = run_scale_full_pipeline(DATA_ROOT, CHECKOUT, code_commit=CODE_COMMIT,
                work_dir=WORK_DIR, worker_id=TEAM_WORKER_ID, workers=TEAM_SIZE,
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
        thì không tính F2 giả. full_plan_status.json phân biệt code đã thực thi với
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
        Upload ZIP lên Dashboard BTC; score_feedback.json giữ submission ID và
        điểm thật, không bị reset khi resume. Chưa có official score trước upload.
        '''),
        code('''
        from google.colab import files
        from vietmedbridge.artifacts import verify_file
        if READY["state"] == "CPU_PREPARATION_COMPLETE":
            print("Chuẩn bị CPU hoàn tất. Chuyển runtime GPU, đặt PREPARE_ONLY=False.")
            print("Team 3 người: TEAM_SIZE=3, TEAM_WORKER_ID=0/1/2; coordinator sau đó dùng None.")
        elif READY["state"] != "READY_FOR_MANUAL_UPLOAD" or READY["query_count"] != 1200:
            print("Đã lưu checkpoint:", READY)
            print("Run all lại cùng cấu hình để hoàn tất trước download.")
        else:
            verify_file(READY["zip_path"], READY["zip_sha256"])
            files.download(READY["zip_path"])
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
            .replace("full-master-plan-strong-v3-per-model-15b", "full-master-plan-supervised-v7-disk-full-system")),
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
