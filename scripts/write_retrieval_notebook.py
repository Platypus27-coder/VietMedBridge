"""Generate only notebook 04; preserve the existing 00–03 data workflow."""
from write_notebooks import BOOTSTRAP, code, md, save


RETRIEVAL_BOOTSTRAP = (BOOTSTRAP
    .replace("code_lock.json", "retrieval_code_lock.json")
    .replace("runtime.json", "retrieval_runtime.json")
    .replace('.[notebook]', '.[notebook,retrieval]')
    .replace("Giữ cùng DATA_ROOT trong cả bốn notebook.", "Giữ cùng DATA_ROOT đã dùng ở notebook 00–03."))


def main():
    save("04_colab_retrieval_baseline.ipynb", [
        md("""
        # VietMedBridge — 04: Embedding → hybrid retrieval → reranker → submission

        Chạy **Runtime > Change runtime type > GPU (T4 trở lên)**, rồi Run all.
        Notebook dùng candidate đã freeze; không cần crawl/chia chunk lại.
        Mặc định đọc bản team đã xử lý `stage-a-data-v3-laodong`: 864 tài liệu,
        9.076 child chunks. Đây là baseline pretrained trên corpus pilot; chưa fine-tune.

        Một notebook cho toàn bộ bước retrieval. Bootstrap tự clone/cài package và
        dùng code lock riêng. Checkpoint embedding theo lô, query theo từng ID được
        lưu Drive; khi runtime ngắt, chạy lại từ đầu với cùng cấu hình và RUN_NAME.
        Không chạy hai runtime ghi cùng RUN_NAME.

        File submission bao phủ đủ 1.200 query, nhưng chỉ tìm trên corpus pilot.
        Kết quả không chứng minh coverage toàn corpus hay đạt relevance gate/F2.
        """),
        md("## 1. Bootstrap — mount Drive, clone repo, cài thư viện"),
        code(RETRIEVAL_BOOTSTRAP),
        md("""
        ## 2. Chọn candidate và cấu hình

        Với dữ liệu team đã freeze, giữ mặc định bên dưới. Nếu dùng candidate khác,
        sửa BUILD_RUN/CANDIDATE_NAME đúng manifest từ notebook 03 và chọn RUN_NAME mới.
        Không cần đổi code lock của data pipeline. Khi chủ động nâng retrieval code,
        đặt CODE_REVISION trong Bootstrap và chọn RUN_NAME mới.

        MAX_NEW_EMBEDDING_PARTS/MAX_NEW_QUERIES=None chạy hết. Có thể đặt số nhỏ để
        giới hạn phiên; nếu chưa hoàn tất, notebook dừng trước bước xuất submission.
        Giữ part_size và model revision khi resume; có thể giảm batch_size nếu cần.
        """),
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
        RUN_NAME = "stage-a-retrieval-v1"
        MAX_NEW_EMBEDDING_PARTS = None
        MAX_NEW_QUERIES = None
        for value in (BUILD_RUN, CANDIDATE_NAME, RUN_NAME):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
                raise ValueError("Tên run/candidate phải là một filename, không chứa đường dẫn.")
        if not torch.cuda.is_available():
            raise RuntimeError("Chọn GPU trong Runtime > Change runtime type rồi chạy lại.")
        CONFIG = read_json(CHECKOUT / "configs/retrieval_baseline.json")
        BUILD_DIR = DATA_ROOT / "processed" / BUILD_RUN
        RUN_DIR = DATA_ROOT / "retrieval" / RUN_NAME
        candidate_path = BUILD_DIR / CANDIDATE_NAME
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Không thấy {candidate_path}. Chọn candidate thực tế từ notebook 03.")
        preview = read_json(candidate_path)
        tokenizer_spec = preview["golden"]["tokenizer"]
        if tokenizer_spec["model_id"] != CONFIG["dense"]["model_id"] or tokenizer_spec["revision"] != CONFIG["dense"]["revision"]:
            raise ValueError("Dense tokenizer phải khớp revision của candidate; không tự đổi representation.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec["model_id"],
            revision=tokenizer_spec["revision"], use_fast=True, trust_remote_code=False)
        CATALOG = load_catalog(BUILD_DIR, CANDIDATE_NAME, tokenizer, **CONFIG["pilot_limits"])
        SNAPSHOT = read_json(DATA_ROOT / "raw/snapshot.json")
        if SNAPSHOT["files"]["links_corpus.parquet"]["sha256"] != CATALOG.build_config["official_links_sha256"]:
            raise ValueError("Candidate không thuộc official corpus snapshot hiện tại.")
        QUERIES = load_queries(parquet_path(DATA_ROOT, "query.parquet"), expected_count=CONFIG["expected_queries"])
        QUERY_UNITS = [{"id": q["id"], "text": q["query"]} for q in QUERIES]
        print("GPU:", torch.cuda.get_device_name(0))
        print(json.dumps({"candidate": CATALOG.identity, "documents": len(CATALOG.documents),
            "children": len(CATALOG.children), "unique_dense_inputs": len(CATALOG.units),
            "queries": len(QUERIES), "checkpoint": str(RUN_DIR)}, ensure_ascii=False, indent=2))
        '''),
        md("""
        ## 3. BGE-M3 embedding — cần GPU

        Encode mỗi representation duy nhất một lần, giữ aliases tới mọi official ID.
        CLS pooling và L2 normalization; không thêm instruction cho query. Input dense
        giữ budget 512 token gồm special tokens theo dense builder của candidate.
        Checkpoint mặc định 256 inputs/lô, checksum và marker chỉ ghi khi lô hoàn tất.
        Model tự giảm batch khi CUDA OOM. Runtime mới chỉ encode lô còn thiếu.
        """),
        code('''
        from vietmedbridge.embeddings import embed_units, embedding_matrix
        from vietmedbridge.retrieval_models import TorchDenseEncoder

        dense = TorchDenseEncoder(CONFIG["dense"])
        started = time.monotonic()
        try:
            CORPUS_EMBEDDINGS = embed_units(CATALOG.units, dense, RUN_DIR / "corpus_embeddings",
                work_dir=WORK_DIR, max_new_parts=MAX_NEW_EMBEDDING_PARTS, **CONFIG["embedding"])
            if CORPUS_EMBEDDINGS["state"] != "COMPLETE":
                raise RuntimeError("Embedding corpus còn thiếu: chạy lại cell này để resume rồi chạy tiếp.")
            QUERY_EMBEDDINGS = embed_units(QUERY_UNITS, dense, RUN_DIR / "query_embeddings",
                work_dir=WORK_DIR, max_new_parts=MAX_NEW_EMBEDDING_PARTS, **CONFIG["embedding"])
            if QUERY_EMBEDDINGS["state"] != "COMPLETE":
                raise RuntimeError("Embedding query còn thiếu: chạy lại cell này để resume rồi chạy tiếp.")
            atomic_json(RUN_DIR / "dense_runtime.json", {"git_commit": CODE_COMMIT,
                "model": dense.identity, "gpu": torch.cuda.get_device_name(0),
                "seconds_this_call": time.monotonic() - started, "oom_backoffs_this_call": dense.oom_backoffs})
        finally:
            dense.close()
            del dense
        CORPUS_VECTORS = embedding_matrix(RUN_DIR / "corpus_embeddings", CORPUS_EMBEDDINGS)
        QUERY_VECTORS = embedding_matrix(RUN_DIR / "query_embeddings", QUERY_EMBEDDINGS)
        print("Corpus vectors:", CORPUS_VECTORS.shape, "| query vectors:", QUERY_VECTORS.shape)
        '''),
        md("""
        ## 4. BM25 + FAISS exact cosine index — CPU

        BM25 dùng Unicode words và CJK unigrams/bigrams. Dense search dùng IndexFlatIP
        trên vectors đã chuẩn hóa; RRF gộp top-100 dense và top-100 sparse cho mỗi query.
        Đây là index pilot có giới hạn RAM, chưa thay thế benchmark ANN cho 100k/full corpus.
        """),
        code('''
        from vietmedbridge.retrieval import HybridIndex, RetrievalConfig, predict_queries

        INDEX = HybridIndex(CATALOG, CORPUS_VECTORS, CORPUS_EMBEDDINGS,
                            RUN_DIR / "index", work_dir=WORK_DIR)
        RETRIEVAL_CONFIG = RetrievalConfig(**CONFIG["retrieval"])
        print("Indexed unique representations:", INDEX.dense.ntotal)
        print("Index signature:", INDEX.manifest["signature"])
        '''),
        md("""
        ## 5. BGE reranker → parent context — cần GPU

        Rerank tối đa 40 query/child pairs bằng pretrained `bge-reranker-v2-m3`.
        Hai model được nạp lần lượt để tiết kiệm VRAM. Chọn top-10 documents và
        tối đa 8 parent chunks, tối đa 2 chunks/document; không bắt số docs bằng số chunks.
        Tham số này là baseline để đo, chưa được tune theo nhãn. Chunk output là
        nguyên văn source slice; model chỉ cho điểm, không sinh văn bản.

        Mỗi query có checkpoint riêng. Đổi query/model/selection policy cần RUN_NAME mới.
        """),
        code('''
        from vietmedbridge.retrieval_models import TorchReranker

        reranker = TorchReranker(CONFIG["reranker"])
        started = time.monotonic()
        RERANKER_IDENTITY = dict(reranker.identity)
        try:
            RECORDS, PREDICTION_REPORT = predict_queries(INDEX, QUERIES, QUERY_VECTORS,
                reranker, RUN_DIR / "queries", query_embedding_manifest=QUERY_EMBEDDINGS,
                config=RETRIEVAL_CONFIG, batch_size=CONFIG["embedding"]["batch_size"],
                max_new_queries=MAX_NEW_QUERIES)
            atomic_json(RUN_DIR / "reranker_runtime.json", {"git_commit": CODE_COMMIT,
                "model": reranker.identity, "gpu": torch.cuda.get_device_name(0),
                "seconds_this_call": time.monotonic() - started, "oom_backoffs_this_call": reranker.oom_backoffs})
        finally:
            reranker.close()
            del reranker
        print(json.dumps(PREDICTION_REPORT, ensure_ascii=False, indent=2))
        if PREDICTION_REPORT["state"] != "COMPLETE":
            raise RuntimeError("Query còn thiếu: chạy lại cell này để resume trước khi xuất submission.")
        '''),
        md("""
        ## 6. Kiểm source/ID/schema và xuất submission — CPU

        Kiểm đủ 1.200 query chính thức, official doc IDs, provenance và exact source
        text của mỗi chunk. `submission.zip` chỉ chứa một `results.json` ở root.
        Evidence/model/data manifests nằm ngoài ZIP. Không tính F2 khi chưa có nhãn
        và scorer BTC; đây là submission pilot để thử đường chạy trên hệ thống chấm.
        """),
        code('''
        from vietmedbridge.submission import export_submission

        EXPORT = export_submission(RECORDS, QUERIES, CATALOG, PREDICTION_REPORT,
            RUN_DIR / "submission", expected_count=CONFIG["expected_queries"], work_dir=WORK_DIR,
            evidence={"git_commit": CODE_COMMIT, "dense": CORPUS_EMBEDDINGS["encoder"],
                      "reranker": RERANKER_IDENTITY, "index": INDEX.manifest,
                      "retrieval_config": CONFIG["retrieval"]})
        print(json.dumps(EXPORT, ensure_ascii=False, indent=2))
        print("ZIP:", RUN_DIR / "submission/submission.zip")
        print("Corpus pilot:", len(CATALOG.documents), "documents; chưa đại diện toàn corpus.")
        # Xem vài query cùng parent output trước khi upload lên hệ thống BTC.
        for q, record in zip(QUERIES[:3], RECORDS[:3]):
            p = record["prediction"]
            print("\\nQuery", q["id"], q["query"])
            print("Documents:", p["relevant_docs"])
            print("First chunk:", p["relevant_chunks"][0]["chunk_text"][:500])
        '''),
        md("""
        ## 7. Tải ZIP

        Chạy cell này để tải file và tự upload lên hệ thống BTC. Khi nhận score,
        giữ cả submission_manifest.json để so sánh baseline với lần mở rộng 100k
        hoặc fine-tune sau này. Score thấp có thể do coverage pilot rất nhỏ; chưa
        đủ để kết luận chọn model sai.
        """),
        code('''
        from google.colab import files
        files.download(str(RUN_DIR / "submission/submission.zip"))
        '''),
    ])


if __name__ == "__main__":
    main()
