"""Generate only notebook 04, with the plan-aligned retrieval cascade."""
from write_notebooks import BOOTSTRAP, code, md, save


RETRIEVAL_BOOTSTRAP = (BOOTSTRAP
    .replace("code_lock.json", "retrieval_code_lock.json")
    .replace("runtime.json", "retrieval_runtime.json")
    .replace('.[notebook]', '.[notebook,retrieval]')
    .replace("Giữ cùng DATA_ROOT trong cả bốn notebook.", "Giữ DATA_ROOT đã dùng ở notebook 00–03.")
    .replace('reference = CODE_REVISION or lock.get("git_commit") or "main"',
        '''RETRIEVAL_WORKFLOW_API = "competition-pilot-e2e-v1"
upgrade = lock.get("workflow_api") != RETRIEVAL_WORKFLOW_API
reference = CODE_REVISION or ("main" if upgrade else lock.get("git_commit")) or "main"''')
    .replace('if not lock or CODE_REVISION:\n    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, "pipeline_api": PIPELINE_API_VERSION})',
        'if not lock or CODE_REVISION or upgrade:\n'
        '    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, '
        '"pipeline_api": PIPELINE_API_VERSION, "workflow_api": RETRIEVAL_WORKFLOW_API})'))


def main():
    save("04_colab_retrieval_baseline.ipynb", [
        md('''
        # VietMedBridge — 04: End-to-end → submission đủ 1.200 query

        Chọn **Runtime > Change runtime type > GPU (T4 trở lên)** rồi Run all.
        Plan chính: `R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md`.
        Dùng dữ liệu đã có để lấy điểm baseline theo §§26/55; không chờ full corpus.
        Đây là bản cascade theo master plan: BGE-M3 + BM25 VI/EN/ZH → weighted RRF
        → document rerank → child MaxP rerank → parent source → token LCS dedup.
        LLM Qwen3-4B chỉ dịch query, không sinh nội dung submission. Chưa fine-tune.

        Một notebook chạy trọn đến ZIP. Mặc định Run all chạy HẾT 1.200 query;
        không bắt buộc canary và không cần nhãn để xuất submission.
        Bootstrap nâng code lock retrieval lên workflow end-to-end một lần;
        các notebook data 00–03 giữ nguyên. Khi nâng code, dùng **runtime mới**.
        Vector và query checkpoints nằm Drive; chạy lại từ đầu khi Colab ngắt.
        Không chạy hai runtime ghi cùng RUN_NAME. Kết quả cũ được giữ để so sánh.

        Corpus vẫn chỉ có 864 tài liệu trong pilot 1.000 URL. Kiến trúc đúng hơn chưa
        chứng minh F2 tốt hơn; cần dev labels và score BTC. Không dùng notebook này
        để index thẳng 100k/full corpus khi chưa có benchmark sharded/ANN.
        '''),
        md("## 1. Bootstrap — CPU"), code(RETRIEVAL_BOOTSTRAP),
        md('''
        ## 2. Candidate và run

        Giữ mặc định nếu dùng candidate của team. `stage-a-retrieval-v2.1` lưu kết quả
        mới; `stage-a-retrieval-v1` chỉ được đọc để tái sử dụng BGE vectors đã xác nhận.
        Nếu cache không có/không đủ thì encode trong run mới. Cache sai checksum,
        thứ tự hoặc policy sẽ báo lỗi, không tự dùng lại. Không sửa/xóa manifest.
        V2.1 siết validator tên thuốc/Latin entities và intolerance; dùng namespace
        mới để không trộn translation/score checkpoints đã sinh bằng validator v2.
        Giữ MAX_NEW_*=None để chạy hết và xuất ZIP. Các biến giới hạn chỉ dùng khi
        chủ động muốn chia phiên; không cần sửa mặc định để chạy end-to-end.
        Query đã hoàn tất được giữ lại. Đổi model/data/policy cần RUN_NAME mới.
        '''),
        code('''
        import re
        import time
        import torch
        from transformers import AutoTokenizer
        from vietmedbridge.artifacts import read_json, digest_json
        from vietmedbridge.dataset import parquet_path
        from vietmedbridge.retrieval_data import load_catalog, load_queries
        from vietmedbridge.competition_pilot import MASTER_PLAN, bind_pilot_run

        BUILD_RUN = "stage-a-data-v3-laodong"
        CANDIDATE_NAME = "candidate-1cd220a4be956d5a.json"
        RUN_NAME = "stage-a-retrieval-v2.1"
        EMBEDDING_CACHE_RUN = "stage-a-retrieval-v1"
        MAX_NEW_EMBEDDING_PARTS = None
        MAX_NEW_TRANSLATIONS = None
        MAX_NEW_QUERIES = None
        REFERENCE_LABELS_PATH = DATA_ROOT / "labels/retrieval_reference.json"
        for value in (BUILD_RUN, CANDIDATE_NAME, RUN_NAME, EMBEDDING_CACHE_RUN):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
                raise ValueError("Tên run/candidate không được chứa đường dẫn.")
        if not torch.cuda.is_available():
            raise RuntimeError("Chọn GPU trong Runtime > Change runtime type.")
        CONFIG = read_json(CHECKOUT / "configs/retrieval_cascade.json")
        BUILD_DIR = DATA_ROOT / "processed" / BUILD_RUN
        RUN_DIR = DATA_ROOT / "retrieval" / RUN_NAME
        CACHE_DIR = DATA_ROOT / "retrieval" / EMBEDDING_CACHE_RUN
        candidate_path = BUILD_DIR / CANDIDATE_NAME
        preview = read_json(candidate_path)
        tokenizer_spec = preview["golden"]["tokenizer"]
        if any(tokenizer_spec[k] != CONFIG["dense"][k] for k in ("model_id", "revision")):
            raise ValueError("Dense tokenizer phải khớp tokenizer frozen của candidate.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec["model_id"],
            revision=tokenizer_spec["revision"], use_fast=True, trust_remote_code=False)
        CATALOG = load_catalog(BUILD_DIR, CANDIDATE_NAME, tokenizer, **CONFIG["pilot_limits"])
        SNAPSHOT = read_json(DATA_ROOT / "raw/snapshot.json")
        if SNAPSHOT["files"]["links_corpus.parquet"]["sha256"] != CATALOG.build_config["official_links_sha256"]:
            raise ValueError("Candidate không thuộc corpus snapshot hiện tại.")
        QUERIES = load_queries(parquet_path(DATA_ROOT, "query.parquet"), expected_count=CONFIG["expected_queries"])
        QUERY_UNITS = [{"id":q["id"], "text":q["query"]} for q in QUERIES]
        PILOT_CONTRACT = bind_pilot_run(RUN_DIR, plan_path=CHECKOUT / MASTER_PLAN,
            catalog=CATALOG, queries=QUERIES, config=CONFIG, code_commit=CODE_COMMIT)
        print("GPU:", torch.cuda.get_device_name(0))
        print(json.dumps({"candidate": CATALOG.identity, "documents": len(CATALOG.documents),
            "children": len(CATALOG.children), "dense_inputs": len(CATALOG.units),
            "queries": len(QUERIES), "run": str(RUN_DIR)}, ensure_ascii=False, indent=2))
        '''),
        md('''
        ## 3. Kiểm cache BGE-M3; encode phần thiếu — GPU khi cần encode

        Model/revision, inputs/order, part-size, pooling và mọi checksum/marker phải khớp.
        Cache giữ đúng identity/runtime của phiên đã sinh vectors; không gán runtime
        mới cho vectors cũ. Nếu reuse thành công thì không nạp model embedding.
        BGE-M3 encode normalized CLS, không truncate hoặc thêm query instruction.
        '''),
        code('''
        from vietmedbridge.retrieval_cache import reuse_bge_cache
        from vietmedbridge.embeddings import embed_units, embedding_matrix
        from vietmedbridge.retrieval_models import TorchDenseEncoder

        corpus_cache = reuse_bge_cache(CACHE_DIR / "corpus_embeddings", CATALOG.units,
            CONFIG["dense"], part_size=CONFIG["embedding"]["part_size"])
        query_cache = reuse_bge_cache(CACHE_DIR / "query_embeddings", QUERY_UNITS,
            CONFIG["dense"], part_size=CONFIG["embedding"]["part_size"])
        if corpus_cache is not None and query_cache is not None:
            CORPUS_VECTORS, CORPUS_EMBEDDINGS = corpus_cache
            QUERY_VECTORS, QUERY_EMBEDDINGS = query_cache
            if CORPUS_EMBEDDINGS["encoder"] != QUERY_EMBEDDINGS["encoder"]:
                raise ValueError("Corpus/query cache phải dùng cùng encoder identity.")
            atomic_json(RUN_DIR / "embedding_reuse.json", {"source_run": EMBEDDING_CACHE_RUN,
                "corpus_manifest_sha256": CORPUS_EMBEDDINGS["manifest_sha256"],
                "query_manifest_sha256": QUERY_EMBEDDINGS["manifest_sha256"],
                "git_commit_this_call": CODE_COMMIT, "reencoded": False})
            print("Đã kiểm và tái sử dụng corpus/query vectors:", EMBEDDING_CACHE_RUN)
        else:
            dense = TorchDenseEncoder(CONFIG["dense"])
            started = time.monotonic()
            try:
                CORPUS_EMBEDDINGS = embed_units(CATALOG.units, dense, RUN_DIR / "corpus_embeddings",
                    work_dir=WORK_DIR, max_new_parts=MAX_NEW_EMBEDDING_PARTS, **CONFIG["embedding"])
                QUERY_EMBEDDINGS = embed_units(QUERY_UNITS, dense, RUN_DIR / "query_embeddings",
                    work_dir=WORK_DIR, max_new_parts=MAX_NEW_EMBEDDING_PARTS, **CONFIG["embedding"])
                atomic_json(RUN_DIR / "dense_runtime.json", {"git_commit": CODE_COMMIT,
                    "model": dense.identity, "gpu": torch.cuda.get_device_name(0),
                    "seconds_this_call": time.monotonic() - started, "oom_backoffs_this_call": dense.oom_backoffs})
            finally:
                dense.close()
                del dense
            CORPUS_VECTORS = embedding_matrix(RUN_DIR / "corpus_embeddings", CORPUS_EMBEDDINGS)
            QUERY_VECTORS = embedding_matrix(RUN_DIR / "query_embeddings", QUERY_EMBEDDINGS)
        '''),
        md('''
        ## 4. LLM dịch query VI → EN/ZH — GPU

        Qwen3-4B-Instruct-2507, public weights/pinned revision, khoảng 4B parameters.
        Nạp fp16 từng query sau khi giải phóng embedding model. Dịch toàn bộ câu hỏi,
        giữ original/entities/constraints; kiểm số, acronym, Latin entities, dấu so sánh và vài constraint
        cues. Branch không đạt kiểm tra bị tắt; original VI vẫn dùng dense và sparse.
        Các kiểm tra này chưa bảo đảm dịch đúng ngữ nghĩa. Raw output và rejection
        reasons được giữ theo từng query để team review. Chưa có HyDE/subquery.
        '''),
        code('''
        from vietmedbridge.translation_model import TorchQueryTranslator
        from vietmedbridge.query_translation import translate_queries, cached_translations

        cached = cached_translations(QUERIES, CONFIG["translation"], RUN_DIR / "translations")
        if cached is not None:
            TRANSLATIONS, TRANSLATION_REPORT = cached
            TRANSLATOR_IDENTITY = read_json(RUN_DIR / "translations/config.json")["translator"]
            print("Reuse verified translations; không nạp LLM lại.")
        else:
            translator = TorchQueryTranslator(CONFIG["translation"])
            TRANSLATOR_IDENTITY = dict(translator.identity)
            started = time.monotonic()
            try:
                TRANSLATIONS, TRANSLATION_REPORT = translate_queries(QUERIES, translator,
                    RUN_DIR / "translations", max_new_queries=MAX_NEW_TRANSLATIONS)
                atomic_json(RUN_DIR / "translation_runtime.json", {"git_commit": CODE_COMMIT,
                    "model": translator.identity, "gpu": torch.cuda.get_device_name(0),
                    "seconds_this_call": time.monotonic() - started})
            finally:
                translator.close()
                del translator
        print(json.dumps(TRANSLATION_REPORT, indent=2))
        print("Cascade sẽ xử lý các query đã dịch, resume các query còn thiếu ở phiên sau.")
        '''),
        md('''
        ## 5. Document indices và weighted RRF — CPU

        Dense MaxP gộp child hits thành document. BM25 postings dùng Lucene-style IDF,
        chạy ba nhánh VI/EN/ZH theo ngôn ngữ suy ra từ nội dung, giữ unknown trong
        mỗi nhánh. CJK unigram/bigram; VI dùng Unicode word tokens, chưa phải Lucene
        language analyzer/word segmentation. Weighted RRF lấy tối đa 200 documents.

        `index/quarantine.json` ghi nguồn homepage redirect/error/encoded payload
        bị giữ lại khỏi retrieval; frozen data và official IDs vẫn nguyên vẹn để phục hồi.
        Không loại theo độ ngắn/quality tier đơn thuần. Sidebar chưa chắc đã sạch.
        '''),
        code('''
        from vietmedbridge.retrieval_cascade import CascadeIndex, CascadeConfig, predict_cascade

        RETRIEVAL_CONFIG = CascadeConfig(**CONFIG["retrieval"])
        INDEX = CascadeIndex(CATALOG, CORPUS_VECTORS, CORPUS_EMBEDDINGS,
            RUN_DIR / "index", tokenizer, work_dir=WORK_DIR)
        print("Eligible documents:", len(INDEX.doc_ids), "Held:", len(INDEX.exclusions))
        print("Language counts:", {lang:sum(INDEX.languages[d] == lang for d in INDEX.doc_ids)
                                       for lang in ("vi", "en", "zh", "unknown")})
        print("Index signature:", INDEX.manifest["signature"])
        '''),
        md('''
        ## 6. Document rerank → child rerank → parent — GPU

        Document representation = title + top-2 child hits, chia token budget để giữ
        cả hai hits. Rerank tối đa 200 documents, lấy 30 documents để tìm local dense
        + BM25 + global dense rank, quota 5 children/doc. Child dài dùng sliding MaxP
        theo budget của query/passage, sau đó mở parent nguyên văn. Mặc định document/
        chunk caps là 10/8; token LCS/union >=0.8 dedup trong cùng document.

        Lưu TẤT CẢ pair inputs và scores cho hai stage, có checksum theo query.
        Ngắt sau document stage sẽ không rerank lại stage đó. Luồng này tốn GPU hơn v1.
        Raw logits không phải xác suất; score floors/margins mặc định chưa bật vì chưa
        có dev labels. Có thể thử rerank_fusion='rrf' ở run khác để ablate kiểu Stage 2.
        '''),
        code('''
        from vietmedbridge.retrieval_models import TorchReranker

        reranker = TorchReranker(CONFIG["reranker"])
        RERANKER_IDENTITY = dict(reranker.identity)
        started = time.monotonic()
        try:
            RECORDS, PREDICTION_REPORT = predict_cascade(INDEX, QUERIES, QUERY_VECTORS,
                reranker, RUN_DIR / "queries", query_embedding_manifest=QUERY_EMBEDDINGS,
                translations=TRANSLATIONS, translation_signature=TRANSLATION_REPORT["signature"],
                config=RETRIEVAL_CONFIG,
                batch_size=CONFIG["embedding"]["batch_size"], max_new_queries=MAX_NEW_QUERIES)
            atomic_json(RUN_DIR / "reranker_runtime.json", {"git_commit": CODE_COMMIT,
                "model": reranker.identity, "gpu": torch.cuda.get_device_name(0),
                "seconds_this_call": time.monotonic() - started,
                "oom_backoffs_this_call": reranker.oom_backoffs})
        finally:
            reranker.close()
            del reranker
        print(json.dumps(PREDICTION_REPORT, indent=2))
        if PREDICTION_REPORT["state"] != "COMPLETE":
            raise RuntimeError("Query còn thiếu: chạy lại từ đầu để resume trước khi xuất submission.")
        '''),
        md('''
        ## 7. Validate → ZIP → manifest → file ghi điểm — CPU

        Kiểm đủ 1.200 official queries, source slices và IDs trước export.
        ZIP chứa đúng results.json ở root. ready_to_submit.json ghi đường dẫn/hash;
        score_feedback.json giữ chỗ để ghi submission ID/điểm thật BTC sau upload.
        Không tự tạo điểm và không đòi nhãn trước khi xuất ZIP.

        Nếu REFERENCE_LABELS_PATH tồn tại, đánh giá proxy theo plan SAU export;
        file nhãn lỗi/unreviewed được ghi vào report nhưng không chặn submission.
        Nhãn tùy chọn: {"reviewed":true, "queries":[{"id":..., "relevant_docs":[...],
        "relevant_chunks":[{"doc_id":...,"chunk_text":"..."}]}]}.
        Không tune trên contest/test queries. Fine-tune cần train/dev gold độc lập,
        không phải điều kiện để lấy score pretrained đầu tiên theo master plan.
        '''),
        code('''
        from vietmedbridge.competition_pilot import finish_pilot

        EXPORT, READY = finish_pilot(RECORDS, QUERIES, CATALOG, PREDICTION_REPORT,
            RUN_DIR, contract=PILOT_CONTRACT, tokenizer=tokenizer, work_dir=WORK_DIR,
            reference_labels_path=REFERENCE_LABELS_PATH,
            evidence={"git_commit": CODE_COMMIT, "dense": CORPUS_EMBEDDINGS["encoder"],
                "translation_llm": TRANSLATOR_IDENTITY, "reranker": RERANKER_IDENTITY,
                "index": INDEX.manifest, "retrieval_config": CONFIG["retrieval"],
                "inference_scope": "COLAB_GPU_PRETRAINED_MODELS"})
        print(json.dumps(READY, ensure_ascii=False, indent=2))
        print("Nộp ZIP này trên Dashboard BTC. Gửi lại điểm/submission ID để so sánh baseline.")
        for q, record in zip(QUERIES[:3], RECORDS[:3]):
            p = record["prediction"]
            print("\\nQuery", q["id"], q["query"])
            print("Documents:", p["relevant_docs"])
            print("First chunk:", p["relevant_chunks"][0]["chunk_text"][:500] if p["relevant_chunks"] else "EMPTY")
        '''),
        md('''
        ## 8. Tải ZIP và submit thử để lấy điểm

        Cell dưới tải submission.zip đã validate. Sếp upload file này trong
        My Submissions trên Dashboard BTC, rồi gửi điểm/submission ID cho tôi.
        Drive giữ ZIP + manifest + score_feedback.json để gắn đúng điểm với đúng run.
        Khi Colab ngắt, chọn GPU và Run all lại cùng RUN_NAME; checkpoint sẽ resume.
        '''),
        code('''
        from google.colab import files
        from vietmedbridge.artifacts import verify_file
        if PREDICTION_REPORT["state"] != "COMPLETE" or "READY" not in globals():
            raise RuntimeError("Hoàn tất 1.200 queries và cell export trước khi tải ZIP.")
        if READY["prediction_signature"] != PREDICTION_REPORT["signature"]:
            raise ValueError("ZIP/report không cùng prediction run.")
        verify_file(READY["zip_path"], READY["zip_sha256"])
        files.download(READY["zip_path"])
        '''),
    ])


if __name__ == "__main__":
    main()
