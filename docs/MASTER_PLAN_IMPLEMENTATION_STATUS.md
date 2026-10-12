# Trạng thái triển khai master plan Stage 3

Tài liệu chuẩn vẫn là [`R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md`](../R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md). Trang này phân biệt ba việc khác nhau: code đã có, lượt chạy đã được xác minh bằng artifact, và chất lượng cần nhãn/điểm BTC để kết luận. Có module không đồng nghĩa đã chứng minh tăng điểm.

## Đã có trong code và notebook

- **Nguồn, dữ liệu và tính tái lập:** snapshot/manifest có hash; crawl/import external; parse và source-preserving chunk; kiểm coverage official IDs; freeze candidate; ghép nhiều build theo lineage; resume bằng shard/checkpoint. Các luồng này nằm trong notebook 00–03 và `dataset`, `crawl`, `external_import`, `build`, `health`, `freeze`.
- **Search corpus:** child và document dense representations; BGE-M3 cùng Qwen3-Embedding-8B; BM25 tiếng Việt/Anh/Trung với tokenization y khoa và field boosts; RRF; query translation, PICO-lite, subquery và HyDE additive; rerank document/child bằng Qwen3-Reranker-8B; chọn source parent nguyên văn và kiểm LCS/near-duplicate. Model revisions và giới hạn tham số được ghim trong config/manifest.
- **Index quy mô lớn:** cache vector theo input identity, checksum và part checkpoint; disk catalog/BM25; FAISS IVF-SQ8 phân mảnh và kiểm ANN/exact; backend Qdrant server HNSW/on-disk; lưu evidence để truy vấn/khôi phục. Khả năng hoạt động và recall của từng backend phải được benchmark trên corpus thực tế trước khi chọn.
- **Submission và vận hành:** đủ 1.200 query IDs; kiểm schema, source span và ZIP trước tải; artifact manifest; checkpoint theo query/stage; log runtime và score do người dùng báo cáo. ZIP hợp lệ được sao lưu thành release bất biến có hash và có thể chọn lại local; notebook không upload submission thay người dùng.
- **Supervised/research workflow:** tạo draft có nguồn, review gate, train/dev/held-out độc lập, hard-negative review, QLoRA reranker, dev cutoff sweep, optional ablations, held-out frozen evaluation và budget từng model. Nếu thiếu nhãn thì 05 phải dừng ở trạng thái chờ; không tạo nhãn gold từ dự đoán.
- **Audit mới:** báo cáo recall ở các tầng còn quan sát được, ngôn ngữ, độ phức tạp, độ dài reference và lỗi theo tầng. Parent sweep 256/384/512/640/768 cùng merge 512→768 replay từ scored checkpoints trên CPU; dev chọn biến thể, held-out chỉ chấm biến thể đã chọn. Không gọi lại model và không tự promote policy.

## Đã chạy ở mức nào

- Sếp đã báo baseline BGE/BM25 trên khoảng 100k dữ liệu đạt **0.0081** trên dashboard. Đây là điểm submission thực tế được báo cáo, không phải kết quả của pipeline full mới.
- Các build/shard khoảng 700k đã được xử lý và freeze theo những output Sếp đưa trong cuộc trao đổi. Tôi chưa truy cập được Drive để xác minh lại toàn bộ manifests/coverage hay các lượt embedding Kaggle đã nhập về; không thể kết luận rằng mọi shard 700k đã có đủ BGE, Qwen, vector index và prediction artifacts.
- Lượt full-plan 04 đã tạo checkpoint/partial outputs trong các lần chạy, nhưng log hội thoại không đủ chứng minh bản retrieval full đã hoàn tất và đạt điểm nào. “READY_FOR_MANUAL_UPLOAD” chỉ nói ZIP có schema/source hợp lệ, không nói relevance tốt.
- Các bộ nhãn gần đây từng được mô tả là AI-assisted, `human_validated=false`, hoặc chờ review. Chúng không đủ làm bằng chứng độc lập để báo F2 dev/held-out hay fine-tune có ý nghĩa. Notebook 05 cần train/dev labels được review; đánh giá held-out chỉ mở khi có bộ held-out độc lập được review.

## Những gì vẫn cần dữ liệu hoặc một lượt chạy

1. **Xác nhận corpus chính xác:** sau khi nhóm thêm shard, 03 phải công bố batch/lineage với coverage audit; 04 cần dùng đúng active cumulative candidate. Phải kiểm tổng official IDs, duplicate/overlap, failures và quality counts bằng manifest mới nhất thay vì cộng các output worker thủ công.
2. **Hoàn tất embeddings/index trên candidate đó:** chỉ các model/vector artifacts có manifest `COMPLETE`, identity và checksums khớp mới được tính. Qwen vectors thiếu không được tính là đã hoàn tất chỉ vì notebook Kaggle báo partial checkpoint; nếu thiếu phải resume từ đúng job/checkpoint hoặc chọn cấu hình cached đã được xác minh.
3. **Nhãn độc lập:** review query/source spans và hard negatives; cần train + dev cho QLoRA/cutoff selection và held-out riêng cho đánh giá frozen. Nhãn dev/held-out không được sinh hoặc chọn bởi cùng model đang đo.
4. **Đo retrieval:** chạy audit/cutoff/parent sweep trên dev, các ablations theo plan, xem lỗi theo ngôn ngữ và query; chỉ sau đó đánh giá policy đã freeze trên held-out. Audit của checkpoint cũ chỉ có thể nói về candidate pool đã lưu, không dựng lại nhánh retrieval đã bị bỏ trước khi lưu.
5. **Đánh giá cạnh tranh:** benchmark thời gian, ANN recall, bộ nhớ và chất lượng trên candidate hiện hành; kiểm thử exact/FAISS/Qdrant khi chọn vector backend. Nộp ZIP thủ công để lấy BTC score thật; ghi dashboard result gắn đúng ZIP hash.
6. **Mở rộng 4 triệu URL:** chưa có cơ sở suy ra thời gian hoặc điểm từ run 700k. Cần benchmark throughput theo stage và ngoại suy theo phần dữ liệu thực sự còn mới; không nhân đơn giản bằng số URL nếu dedup/cache/retry khác nhau.

## Luồng làm việc từ đây

`00 audit/snapshot → 01 crawl khi cần → 02 import/parse/chunk, có checkpoint → 03 validate/freeze/coverage → 04 CPU compose & catalog → 04 GPU full inference hoặc cấu hình cached đã xác minh → xem diagnostics và ZIP/release → 05 chỉ khi đã có nhãn review độc lập → dev selection/ablation → held-out frozen audit → tải ZIP, nộp BTC thủ công, ghi điểm.`

Nếu nhóm chỉ muốn chạy end-to-end để lấy một submission và chưa có nhãn, chạy 04 cho candidate đang active; không cần chạy 05 trước. Nếu candidate hoặc retrieval policy đổi, dùng run mới để tránh trộn checkpoint. Khi có nhãn hợp lệ, 05 tạo bằng chứng chất lượng và adapter; 04 sau đó nạp adapter đã chọn vào namespace mới.

`docs/FULL_PLAN.md` ghi cấu hình/runtime cụ thể. Tài liệu này không thay master plan, không xác nhận dữ liệu ngoài máy, và không tuyên bố điểm tương lai.
