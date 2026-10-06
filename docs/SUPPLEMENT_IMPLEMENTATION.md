# Phạm vi triển khai supplement

Triển khai dựa trên `R2AI_STAGE3_DATA_PIPELINE_SUPPLEMENT.md` và hai master plan
Stage 3 hiện có. Mục tiêu của lần cập nhật này là xử lý data trên Colab đến
FROZEN_CANDIDATE, chuẩn bị evidence để team duyệt mở rộng. Không coi đây là
hoàn tất toàn bộ supplement, retrieval benchmark hoặc full-corpus run.

## Phần đã có code

- Inventory toàn file URL bằng SQL có spill; mẫu Stage A ~1k cân bằng domain
  tiers, format và language hints. URL canonical chỉ phục vụ inventory.
- Raw shards có byte budget, checksum, outcome/latency và requested-range coverage;
  checkpoint/retry immutable attempts, marker commit cuối cùng.
- Source immutable, quality rules có version, conservative error-page quarantine,
  HTML/XML/JATS/PDF extraction và cảnh báo table linearization.
- Section spans chính xác, paragraph/sentence boundaries, child/parent BGE token
  budgets; child identity độc lập parent budget và on-demand parent context.
- Source/retrieval hash tách riêng; dense/sparse text builder. Reducer alias toàn
  snapshot giữ đủ official IDs và child IDs.
- Whole-snapshot integrity gate; health JSON, HTML source audit, distribution
  statistics và baseline-relative drift warnings với threshold đã calibrate.
- 11 synthetic golden fixtures; workflow 100–500 golden nguồn thật có assertions
  do người review nhập, replay pinned raw và kiểm source/span regression.
- Freeze candidate và evidence records cho Stage A/B1/B2/C/full authorization;
  không tự ghi human approval, retrieval pass hoặc budget đã đo.
- Bốn notebook data Colab, code/tokenizer locks và hướng dẫn nâng API v1 → v2.
- Notebook 04: pretrained BGE-M3 dense embeddings có vector part checkpoints và
  OOM batch backoff; BM25 + FAISS exact cosine, RRF và pretrained BGE reranker.
  Giữ official aliases, parent source output và checkpoint mỗi query; validator
  bao phủ 1.200 query rồi xuất ZIP một JSON. Code lock retrieval tách khỏi data.
  Reader đã được kiểm với candidate thật; GPU inference cần chạy trên Colab.

## Những điều cần artifact thực tế từ Colab

- Chạy Stage A, xem failures/domain/source quality và đo storage/throughput.
- Team review các nguồn thật, điền expected snippets và chạy golden replay.
  Các file candidates đang PENDING_HUMAN_REVIEW, không phải golden đã được duyệt.
- Calibrate quality/drift thresholds và xử lý domain gây mất nội dung.
- Đo ổ Drive, memory, runtime, thời gian crawler và tỷ lệ dữ liệu đủ dùng.
  Cấu hình budget hiện null; pipeline không lấy con số máy local làm budget Colab.

## Các phần giữ cho bước retrieval

Baseline pilot đã có BGE embedding sharded, GPU/OOM batching, model/encoding-aware
vector reuse, BM25 và FAISS exact search. Full/selective policy, Lucene/ANN,
exact-vs-ANN recall, real qrels/span labels, canary relevance benchmark, scorer
tương thích, fine-tune và Chunk F2 ablation chưa triển khai.
Document relevance labels đơn thuần chưa đủ kiểm Chunk F2. Query-selective
embedding cần sparse fallback và benchmark recall trước khi chọn.

PROMOTED registry/rollback cho release index chưa có. Candidate/snapshot manifests
và raw attempts hiện hỗ trợ chọn lại dữ liệu cũ để debug/rebuild. Không có full-run
forecast, automatic per-shard drift kill-switch, bulk PMC/PubMed adapters hoặc OCR.

## Bổ sung retrieval v2 — 06/10/2026

Notebook 04 bổ sung document→child→parent cascade, conservative Qwen 4B query
translation, content-language sparse routing, weighted RRF, per-doc quota,
sliding MaxP và token LCS dedup. Hai rerank stages giữ toàn bộ pairs/scores với
checksum/completeness và resume; reuse embeddings v1 qua inputs/policy/part checks.
Scorer local và cutoff sweep đã có theo plan, nhưng chưa chứng minh tương thích
scorer BTC hoặc chất lượng F2 vì thiếu official labels/scorer. Chưa có semantic
translation benchmark, bilingual alias glossary hay Lucene language analyzers.

Full-scale indexing, selective encoding, ANN, canary relevance gates, train/dev
hard negatives/fine-tune, release PROMOTED và human QA vẫn chưa hoàn tất. Chạy
Colab canary trước khi hoàn tất bản pilot; source integrity không thay relevance.

## Điều chỉnh so với ví dụ trong supplement

Shard mặc định 512 records và tối đa 512 MiB JSONL trước nén. Giữ batch fetch
concurrency 4 và build từng document, tránh giữ 5k–20k full documents trong RAM.
Con số này là cấu hình pilot, chưa phải throughput tối ưu.

Freeze candidate diễn ra trước index experiments; promotion sau benchmark.
Marker xác nhận tập file đã commit, không chỉ dựa vào rename. Dedup diễn ra trên
toàn selected snapshot và giữ aliases, thay vì loại duplicate trong từng shard.

Golden nguồn thật chưa có sẵn nên cho phép một pilot giới hạn bằng synthetic
suite để tạo nguồn review; **không mở Stage B1 khi chưa có 100–500 nguồn reviewed
golden pass**. Đây là khác biệt bootstrap được ghi rõ với yêu cầu golden thật trước
Stage A trong plan. Chưa tuyên bố Stage A đạt Definition of Done.

`PLAN_ViBioMIR_Data_Pipeline_v2.md` của phương án từng được review không được dùng
để thay thế master plan hoặc làm nguồn triển khai của lần này.
