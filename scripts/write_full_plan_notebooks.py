"""Generate the full inference notebook and the supervised training notebook."""
from write_notebooks import BOOTSTRAP, code, md, save

BOOT = (BOOTSTRAP
    .replace("code_lock.json", "retrieval_code_lock.json")
    .replace("runtime.json", "retrieval_runtime.json")
    .replace('.[notebook]', '.[notebook,retrieval,strong]')
    .replace('"install", "-q", "-e"', '"install", "--disable-pip-version-check", "-e"')
    .replace("Giữ cùng DATA_ROOT trong cả bốn notebook.", "Giữ DATA_ROOT đã dùng ở notebook 00–03.")
    .replace('reference = CODE_REVISION or lock.get("git_commit") or "main"',
        '''RETRIEVAL_WORKFLOW_API = "full-master-plan-strong-v2-15b"
upgrade = lock.get("workflow_api") != RETRIEVAL_WORKFLOW_API
reference = CODE_REVISION or ("main" if upgrade else lock.get("git_commit")) or "main"''')
    .replace('if not lock or CODE_REVISION:\n    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, "pipeline_api": PIPELINE_API_VERSION})',
        'if not lock or CODE_REVISION or upgrade:\n'
        '    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, '
        '"pipeline_api": PIPELINE_API_VERSION, "workflow_api": RETRIEVAL_WORKFLOW_API})'))


def main():
    save("04_colab_retrieval_baseline.ipynb", [
        md('''
        # VietMedBridge — 04: Kiến trúc đầy đủ → submission 1.200 query

        Plan chính: `R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md`.
        Chọn **runtime GPU mới, T4 trở lên**, rồi Run all. Giữ DATA_ROOT cũ;
        không chạy lại 00–03 và không chọn ACTION. Bootstrap tự clone/cài package.

        Kiến trúc: **BGE-M3 + Qwen3-Embedding-0.6B + BM25 VI/EN/ZH → weighted RRF
        → Qwen3-Reranker-8B document → local child retrieval/MaxP rerank
        → exact-source parent 512/640 → LCS dedup → ZIP đủ 1.200 query**.
        Qwen3-4B dịch query; PICO-lite/subqueries/HyDE chỉ additive cho query phức tạp.
        Các số/Latin entities/constraint cues được kiểm; original query luôn giữ.
        Sparse dùng PyVI, Jieba, CJK n-grams và title/heading/body/alias fields.

        Tổng bốn model **13.374.547.456 parameters**, trước lượng tử hóa.
        Gate kiểm tổng **≤15B** cho mọi model đã dùng, kể cả chạy lần lượt;
        adapter được cộng thêm khi fine-tune. Đây là giới hạn tổng Sếp yêu cầu,
        chưa phải xác nhận BTC giải thích rule theo tổng.
        Qwen embedding 0.6B dùng fp16; reranker 8B dùng **NF4 4-bit**.
        Models nạp lần lượt. Đây là lựa chọn tài nguyên
        cho T4, chưa benchmark tương đương fp16. Weights/cache nằm ổ local Colab;
        vector parts, translation/expansion và scored stages checkpoint trên Drive.
        Khi Colab ngắt, Run all lại cùng cấu hình; không chạy hai runtime cùng run.

        Dùng corpus hiện có (864 documents/9.076 children), chưa cần full corpus.
        Có code train/dev mining, QLoRA, checkpoint selection, dev F2 cutoff và
        ablation trong module/05. **Hiện chưa có train/dev labels** nên 04 dùng
        pretrained models và cutoff chưa tune; không giả lập fine-tune hoặc điểm.
        Nếu đã chạy 05 với nhãn độc lập hợp lệ, 04 tự đọc adapter/dev policy đã chọn,
        tạo namespace -ft- mới. Nguồn/candidate cũ và các kết quả baseline được giữ.
        '''),
        md("## 1. Bootstrap, mount Drive, clone và cài dependencies — CPU"), code(BOOT),
        md('''
        ## 2. Chạy kiến trúc đầy đủ — CPU/GPU lần lượt

        Giữ MAX_NEW_*=None để chạy hết. Đổi model/data/policy cần RUN_NAME mới;
        các run cũ không bị ghi đè. Vector BGE của stage-a-retrieval-v1 được kiểm
        hash/order/model trước reuse. Qwen vectors có corpus/query identities riêng.

        Sau checks nguồn/snapshot/query, GPU chạy phần thiếu:
        BGE (nếu cần) → Qwen4B query understanding → Qwen0.6B embedding → Qwen8B rerank.
        Run v2-15b tách khỏi v1; Qwen vectors cũ của 8B không được reuse.
        Giữ DATA_ROOT; corpus và BGE vectors hợp lệ vẫn được tận dụng.
        Runtime lưu từng phần; lỗi mạng/OOM giữ checkpoints đã hoàn tất.
        Glossary tùy chọn data/labels/medical_aliases.json cần reviewed=true,
        entries=[{aliases:[...], source:"..."}]; không có thì alias field rỗng,
        biomedical Latin/acronym tokens vẫn được giữ.
        '''),
        code('''
        import json
        from vietmedbridge.full_plan_runtime import run_full_pipeline

        BUILD_RUN = "stage-a-data-v3-laodong"
        CANDIDATE_NAME = "candidate-1cd220a4be956d5a.json"
        RUN_NAME = "stage-a-full-plan-v2-15b"
        EMBEDDING_CACHE_RUN = "stage-a-retrieval-v1"
        MAX_NEW_EMBEDDING_PARTS = None
        MAX_NEW_TRANSLATIONS = None
        MAX_NEW_QUERIES = None

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

        Chunks bên dưới là source slices. diagnostics.json có language coverage,
        branch contributions, output counts và logit distributions. Không có nhãn
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
        if READY["state"] != "READY_FOR_MANUAL_UPLOAD" or READY["query_count"] != 1200:
            raise RuntimeError("Hoàn tất pipeline và đủ 1.200 query trước download.")
        verify_file(READY["zip_path"], READY["zip_sha256"])
        files.download(READY["zip_path"])
        '''),
    ])
    save("05_colab_supervised_training.ipynb", [
        md('''
        # VietMedBridge — 05: Hard negatives → QLoRA → dev F2 chọn checkpoint

        Giai đoạn supervised trong master §§13/27/39/56. **04 chạy pretrained và
        submit được ngay**, không cần chạy 05 trước khi có nhãn. Chọn runtime GPU
        mới; L4/A100 thuận tiện hơn cho training. Qwen8B nạp NF4, batch=1,
        gradient checkpointing/accumulation.

        Hai file trong cùng Drive: data/labels/retrieval_train.json và
        data/labels/retrieval_dev.json. Schema: {reviewed:true, split:"train" hoặc
        "dev", exhaustive_chunks:true, queries:[{id:..., query:"...",
        relevant_docs:[...], relevant_chunks:[{doc_id:...,chunk_text:"source đã review"}],
        negative_categories:{child_id:"same_document_wrong_chunk|same_disease_wrong_intervention|same_entity_wrong_context|cross_language_false_friend|high_lexical_overlap"}}]}.
        Khi gold không exhaustive, chỉ dùng negative_child_ids đã review. Category
        thiếu giữ unclassified; không giả vờ đã kiểm semantic taxonomy.

        Train/dev tách theo query, IDs và nội dung không trùng 1.200 contest queries.
        Mine 1 positive + 7 negatives, ưu tiên top ranks, không lấy predictions làm gold.
        QLoRA checkpoint giữ optimizer/RNG để resume. Đánh giá dev NDCG/MRR rồi
        chạy finalists qua full pipeline và CPU F2 cutoff sweep. Chọn adapter theo
        dev F2, giữ trạng thái held-out pending; không promote corpus.
        04 tự nhận selected_adapter/calibrated_policy trong namespace mới.
        '''),
        md("## 1. Bootstrap — CPU"),
        code(BOOT.replace("retrieval_code_lock.json", "training_code_lock.json").replace("retrieval_runtime.json", "training_runtime.json")),
        md("## 2. Chạy supervised workflow — GPU khi đã có nhãn hợp lệ"),
        code('''
        import json
        from vietmedbridge.training_workflow import run_training_workflow
        TRAINING_STATUS = run_training_workflow(DATA_ROOT, CHECKOUT,
            work_dir=WORK_DIR, run_name="stage-a-qlora-v2-15b")
        print(json.dumps(TRAINING_STATUS, ensure_ascii=False, indent=2))
        '''),
        md('''
        Không có nhãn: WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS, không tải
        model để giả lập training. Gold/negative pool thiếu: giữ mining holds để
        review. Ablation/embedding fine-tune/sharded dense có API trong package,
        hướng dẫn ở docs/FULL_PLAN.md. Sau khi chọn adapter hợp lệ, mở 04 và Run all;
        không chỉnh ACTION hoặc xóa run pretrained. Nhãn/config đổi cần run mới.
        '''),
    ])


if __name__ == "__main__":
    main()
