# VietMedBridge

Chuẩn bị dữ liệu cho truy hồi y sinh đa ngôn ngữ ViBioMIR. Phần tính toán và thu
thập dữ liệu chạy trên Google Colab; snapshot, checkpoint và Parquet đầu ra lưu
trong Google Drive.

## Quy trình Colab đã chốt

Luồng làm việc chính chỉ gồm **00 → 01 → 02 → 03**. Không chạy các notebook
01b–01g trong lượt crawl baseline quy mô lớn.

1. **00 — Tải và audit dataset:** [mở notebook](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/00_colab_dataset_audit.ipynb).
   Tải snapshot cố định, xác minh ID/URL, tạo inventory, Stage A sample và golden fixtures.
2. **01 — Crawl baseline:** [mở notebook](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/01_colab_crawl_sources.ipynb).
   HTTPx xử lý HTTP với robots guard, giới hạn tốc độ, redirect từng chặng và shard checkpoint.
   Stage A dùng lại `stage-a-v2` đã hoàn tất: 1.000 outcome, 917 capture HTTP trong kết quả
   đã kiểm tra. Notebook xác minh input, official snapshot, cặp ID/URL và hash raw trước
   khi đọc; không crawl lại hay đổi fingerprint của run cũ.
   Lượt đầu giữ Stage A 1.000 ID; để tăng phạm vi, dùng `MODE="range"` với range
   và `RUN_NAME` riêng, ổn định cho từng milestone. `MAX_NEW_SHARDS` giới hạn mỗi
   phiên; không để hai runtime ghi cùng shard.
3. **02 — Trích văn bản và chia đoạn:** [mở notebook](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_extract_and_chunk.ipynb).
   Mặc định đọc `stage-a-v2`, tạo build mới `stage-a-data-v2-restored` với tài liệu,
   section và parent/child chunks; chất lượng của 917 capture còn cần kiểm tra.
4. **03 — Kiểm tra, audit và freeze:** [mở notebook](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/03_colab_validate_and_freeze.ipynb).
   Kiểm coverage, integrity, golden và source audit; candidate chỉ được promote sau human review.

Đích hiện tại là xây baseline crawl đến khoảng 1 triệu URL qua các gate đã chốt:
Stage A 1.000 → Stage B1 10.000 → Stage B2 100.000 → Stage C 1.000.000.
Notebook 01 chỉ chạy baseline và resume; không retry lỗi hoặc chạy recovery trong
giai đoạn này. Các lỗi vẫn có outcome theo official ID trong ledger để không mất dấu.
Nhóm lỗi còn lại được xử lý sau khi đo chất lượng; riêng 02b kiểm thử cách cứu
trang cookie Lao Động trên Stage A mà không thay đổi lượt crawl baseline.

Các notebook trong luồng hiện tại: 00 audit dataset, 01 baseline crawl, 02 extract/chunk,
03 validate/freeze. Baseline dùng HTTPx, không cần cài Chromium hoặc chọn ACTION; chạy
CPU, không cần GPU. Scrapling/Crawl4AI và các công cụ recovery vẫn được giữ để kiểm
chứng sau. `stage-a-v3` được giữ làm kết quả thí nghiệm, không là input mặc định của build.

Sếp đã hoàn tất `stage-a-v2` thì mở notebook 02 bản mới và chạy tiếp; không cần chạy
lại 00 hay crawl lại 01. 02 và 03 cùng dùng `stage-a-data-v2-restored`. Bootstrap nâng
code lock một lần sang workflow HTTPx rồi ghim commit; nếu runtime đã import code cũ,
restart session một lần trước khi chạy Bootstrap. Báo cáo golden được chạy lại bằng
code hiện tại, không tự duyệt milestone.

### Thử khôi phục trang Lao Động trong Stage A

Trên branch `feature/laodong-extractor`, notebook [02b — recover Lao Động](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/feature/laodong-extractor/notebooks/02b_colab_recover_laodong.ipynb)
đọc raw `stage-a-v2` đã lưu trên Drive. Nó chỉ lấy lại các bản ghi Lao Động có
trang JavaScript đặt cookie `D1N`, kiểm tra ID bài và vùng nội dung, rồi tạo raw
run `stage-a-laodong-recovered-v1` đủ 1.000 outcome. Sau đó notebook build
`stage-a-data-v3-laodong` và in coverage mới. Bản raw và build cũ được giữ nguyên.
Chạy CPU, trong runtime Colab mới. Kết quả thực tế phụ thuộc vào raw đã lưu và
khả năng truy cập URL tại thời điểm chạy; không mặc định coi toàn bộ 123 ID Lao
Động là đã cứu được. Muốn chạy notebook 03 cho build mới, đặt `CODE_REVISION`,
`CRAWL_RUN` và `BUILD_RUN` theo hướng dẫn ở cuối notebook 02b.

## Notebook thí nghiệm Stage A — không thuộc luồng chính

Các notebook `01b–01g` phục vụ retry/chẩn đoán/recovery đã được chuyển khỏi thư mục
notebook đang dùng sang [`archive/notebooks/stage-a-1000-recovery/`](archive/notebooks/stage-a-1000-recovery/).
Luồng baseline vẫn là 00–03; 02b là thí nghiệm có chọn lọc cho nhóm Lao Động
đã parse thất bại trong Stage A.

Trong mỗi runtime Colab mới, chạy Bootstrap để mount cùng Drive và nạp code lock. Giữ cùng
`DATA_ROOT` trong mọi notebook. Đổi code/config khi fetch tiếp cần run mới; checkpoint
đã hoàn tất có thể được đọc để build lại sau khi xác minh provenance. Không để một
runtime đang chạy trộn package code giữa các stage.

Notebook 03 xuất golden candidates để team đánh expected snippets rồi replay. Corpus query
chưa có relevance labels; không coi self-retrieval là nhãn. Stage B2/C cần retrieval
regression có nhãn; full run cần budget thực đo và human audit theo gate.

## Dataset

Nguồn: [AIGuruTinix/ViBioMIR](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR).
Hai cấu hình Hub là query và corpus; cả hai dùng split tên train. Cấu hình query
hiện có id/query; corpus có id/url và chưa chứa văn bản tài liệu.

Revision pin kiểm tra ngày 2026-10-02: 0148f6f80ffafed5c005af6d506ccfd9d3fb47a7.
Metadata tại thời điểm kiểm tra báo 4.394.718 dòng corpus, trong khi dataset card
ghi 4.420.561 liên kết. Pipeline đếm trực tiếp Parquet và không suy luận rằng ID
liên tục từ 1. ID chính thức được giữ nguyên, kể cả khi URL hoặc nội dung trùng.
Split tên train chưa cung cấp reference labels để tính F2.

## Artifact dữ liệu

- raw/: Parquet và dataset card nguyên bản, revision và SHA-256.
- reports/: inventory, thống kê domain, mẫu phân tầng và golden reports.
- crawl/: raw snapshot nén theo shard, manifest và trạng thái lỗi của từng ID.
- processed/: documents, sections, children, parents, failures, input ledger và snapshot manifest.
- processed/<run>/reports/: canonical/representation aliases, health JSON, HTML audit và golden candidates.
- gates/: evidence của milestone do người review ghi nhận.

Mỗi đoạn có doc_id, source_text_sha256, start_char/end_char và chunk_id ổn định.
text là lát cắt source_text; retrieval_text được chuẩn hóa riêng. Offset tham
chiếu văn bản đã trích, tính theo ký tự Unicode Python. Raw HTML/PDF/XML và hash
được giữ để truy nguyên về nguồn HTTP. Mỗi official ID vẫn có bản ghi riêng.

Đọc các file được liệt kê trong build.json hoặc snapshot manifest; tránh glob
toàn bộ processed folder vì các attempt cũ được giữ để rollback. Xem thêm
[hợp đồng dữ liệu và giới hạn hiện tại](docs/DATA_PIPELINE.md).

## Môi trường local

Tạo môi trường từ thư mục repo:

    conda env create -f environment.yml
    conda activate r2ai-stage3
    conda env config vars set PYTHONNOUSERSITE=1
    python -m ipykernel install --user --name r2ai-stage3 --display-name "Python (r2ai-stage3)"

Sau khi đặt biến môi trường, activate lại env. Local phục vụ chỉnh sửa và kiểm tra
nhẹ; các notebook dành cho Colab. Thư viện dữ liệu và tokenizer không yêu cầu GPU.

Env r2ai-stage3 đã được tạo trên máy làm việc; package hiện là 0.2.0.

Kiểm tra mã:

    python -m pytest -q
    python scripts/check_notebooks.py

Kiểm tra riêng tokenizer BGE-M3, không tải model weights:

    python scripts/check_tokenizer.py --report artifacts/tokenizer-validation.json

Dùng --local-tokenizer <snapshot-directory> nếu tokenizer pinned đã có trong cache.

Sinh lại notebook sau khi sửa cell source:

    python scripts/write_notebooks.py

## Phạm vi hiện tại

Giai đoạn này triển khai phần data của plan gốc và supplement đã chốt. Chưa tạo embedding/index,
chưa huấn luyện reranker và chưa triển khai scorer F2 chính thức. Child 180,
overlap 40 và parent 512 token là các tham số khởi đầu cho thí nghiệm.
Section parser dùng heading có thật trong nguồn và fallback body khi không tìm thấy.
PDF scan/OCR và adapter đặc thù nguồn lớn cần bổ sung theo báo cáo sample. LOW quality
vẫn eligible; lỗi crawl/parse được lưu và không có văn bản thay thế cho nguồn lỗi.

FROZEN_CANDIDATE là đầu vào bất biến cho index/benchmark. PROMOTED cần evidence
retrieval, tài nguyên và human review ở giai đoạn sau. Xem
[phạm vi triển khai supplement](docs/SUPPLEMENT_IMPLEMENTATION.md) và
[kết quả kiểm tra thực tế](docs/VALIDATION.md).

Hai repo r2ai-stage-1 và r2ai-stage-2 ngoài VietMedBridge dùng làm nguồn tham khảo.
Các file kế hoạch .md trong repo giữ bối cảnh thiết kế. Data, cache và artifact
không được commit lên GitHub.

Dataset card công bố CC BY-NC 4.0 và hướng dẫn ghi nguồn
[MedKB](https://medkb.tinix.ai/); nội dung crawl giữ URL và provenance của nguồn tương ứng.
