"""Generate only notebook 04, with the plan-aligned retrieval cascade."""
from write_notebooks import BOOTSTRAP, code, md, save


RETRIEVAL_BOOTSTRAP = (BOOTSTRAP
    .replace("code_lock.json", "retrieval_code_lock.json")
    .replace("runtime.json", "retrieval_runtime.json")
    .replace('.[notebook]', '.[notebook,retrieval]')
    .replace("Giữ cùng DATA_ROOT trong cả bốn notebook.", "Giữ DATA_ROOT đã dùng ở notebook 00–03.")
    .replace('reference = CODE_REVISION or lock.get("git_commit") or "main"',
        '''RETRIEVAL_WORKFLOW_API = "document-child-parent-cascade-v2.1"
upgrade = lock.get("workflow_api") != RETRIEVAL_WORKFLOW_API
reference = CODE_REVISION or ("main" if upgrade else lock.get("git_commit")) or "main"''')
    .replace('if not lock or CODE_REVISION:\n    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, "pipeline_api": PIPELINE_API_VERSION})',
        'if not lock or CODE_REVISION or upgrade:\n'
        '    atomic_json(lock_path, {"repo_url": REPO_URL, "git_commit": CODE_COMMIT, '
        '"pipeline_api": PIPELINE_API_VERSION, "workflow_api": RETRIEVAL_WORKFLOW_API})'))


def main():
    save("04_colab_retrieval_baseline.ipynb", [
        md('''
        # VietMedBridge — 04: Truy hồi document → child → parent

        Chọn **Runtime > Change runtime type > GPU (T4 trở lên)** rồi Run all.
        Đây là bản cascade theo master plan: BGE-M3 + BM25 VI/EN/ZH → weighted RRF
        → document rerank → child MaxP rerank → parent source → token LCS dedup.
        LLM Qwen3-4B chỉ dịch query, không sinh nội dung submission. Chưa fine-tune.

        Một notebook cho retrieval. Bootstrap nâng code lock retrieval lên v2 một lần;
        các notebook data 00–03 giữ nguyên. Khi nâng từ v1, dùng **runtime mới**.
        Vector và query checkpoints nằm Drive; chạy lại từ đầu khi Colab ngắt.
        Không chạy hai runtime ghi cùng RUN_NAME. Kết quả cũ được giữ để so sánh.

        Corpus vẫn chỉ có 864 tài liệu trong pilot 1.000 URL. Kiến trúc đúng hơn chưa
        chứng minh F2 tốt hơn; cần dev labels và score BTC. Không dùng notebook này
        để index thẳng 100k/full corpus khi chưa có benchmark sharded/ANN.
        '''),
        md("## 1. Bootstrap — CPU"), code(RETRIEVAL_BOOTSTRAP),
        md('''
        ## 2. Candidate và run

        Giữ mặc định nếu dùng candidate của team. `stage-a-retrieval-v2` lưu kết quả
        mới; `stage-a-retrieval-v1` chỉ được đọc để tái sử dụng BGE vectors đã xác nhận.
        Nếu cache không có/không đủ thì encode trong run mới. Cache sai checksum,
        thứ tự hoặc policy sẽ báo lỗi, không tự dùng lại. Không sửa/xóa manifest.
        V2.1 siết validator tên thuốc/Latin entities và intolerance; dùng namespace
        mới để không trộn translation/score checkpoints đã sinh bằng validator v2.
        MAX_NEW_* có thể giới hạn một phiên; None chạy hết. Để chạy canary 25 queries,
        đặt MAX_NEW_TRANSLATIONS=25 và MAX_NEW_QUERIES=25. Các phiên sau tăng dần
        hoặc đặt None; query đã hoàn tất được giữ lại. Đổi policy cần RUN_NAME mới.
        '''),
        code('''
        import re
        import time
        import torch
        from transformers import AutoTokenizer
        from vietmedbridge.artifacts import read_json, digest_json
        from vietmedbridge.dataset import parquet_path
        from vietmedbridge.retrieval_data import load_catalog, load_queries

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
        ## 7. Đánh giá nếu có reference labels — CPU

        File labels tùy chọn: {"reviewed":true, "queries":[
        {"id":..., "relevant_docs":[...], "relevant_chunks":[{"doc_id":...,"chunk_text":"..."}]}]}.
        Nhãn phải được review. Cell này chỉ đánh giá, không tune trên 1.200 contest
        queries. Cutoff sweep có API riêng cho một dev query set độc lập và đã chấm
        candidates; không dùng chính contest query set làm dev.
        Scorer local suy ra từ plan: cùng doc, LCS/BGE-M3 >=40% reference tokens;
        macro F2 theo query, gộp Doc/Chunk F2. Chưa phải scorer chính thức BTC.
        Không tự đổi submission/cutoffs dựa trên reference scores.
        Nếu chưa có nhãn thì ghi NOT_EVALUATED, không tạo nhãn từ chính prediction.
        '''),
        code('''
        from vietmedbridge.retrieval_eval import evaluate_predictions, evaluate_candidate_recall

        if REFERENCE_LABELS_PATH.is_file():
            references = read_json(REFERENCE_LABELS_PATH)
            if references.get("reviewed") is not True:
                raise ValueError("Reference labels cần được review trước khi đánh giá.")
            EVALUATION = evaluate_predictions([r["prediction"] for r in RECORDS],
                                              references["queries"], tokenizer)
            EVALUATION["candidate_recall"] = evaluate_candidate_recall(RECORDS,
                references["queries"], CATALOG, tokenizer)
            atomic_json(RUN_DIR / "evaluation/reference_report.json", EVALUATION)
            print(json.dumps({k:v for k,v in EVALUATION.items() if k != "per_query"}, indent=2))
            print("Proxy score theo plan; không dùng contest labels để tune cutoff.")
        else:
            print("NOT_EVALUATED_NO_REFERENCE_LABELS — cutoff đang là baseline chưa tune.")
        '''),
        md('''
        ## 8. Kiểm source/ID/schema và xuất ZIP — CPU

        Chỉ xuất khi đủ 1.200 query; chunks là exact frozen source slices, không phải
        văn bản LLM sinh. ZIP chứa đúng results.json ở root; evidence nằm ngoài ZIP.
        Đây vẫn là submission pilot, chưa đủ corpus và chưa chứng minh relevance/F2.
        '''),
        code('''
        from vietmedbridge.submission import export_submission

        EXPORT = export_submission(RECORDS, QUERIES, CATALOG, PREDICTION_REPORT,
            RUN_DIR / "submission", expected_count=CONFIG["expected_queries"], work_dir=WORK_DIR,
            evidence={"git_commit": CODE_COMMIT, "dense": CORPUS_EMBEDDINGS["encoder"],
                "translation_llm": TRANSLATOR_IDENTITY, "reranker": RERANKER_IDENTITY,
                "index": INDEX.manifest, "retrieval_config": CONFIG["retrieval"]})
        print(json.dumps(EXPORT, ensure_ascii=False, indent=2))
        print("ZIP:", RUN_DIR / "submission/submission.zip")
        for q, record in zip(QUERIES[:3], RECORDS[:3]):
            p = record["prediction"]
            print("\\nQuery", q["id"], q["query"])
            print("Documents:", p["relevant_docs"])
            print("First chunk:", p["relevant_chunks"][0]["chunk_text"][:500] if p["relevant_chunks"] else "EMPTY")
        '''),
        md("## 9. Tải submission pilot; tự upload lên BTC"),
        code('''
        from google.colab import files
        files.download(str(RUN_DIR / "submission/submission.zip"))
        '''),
    ])


if __name__ == "__main__":
    main()
