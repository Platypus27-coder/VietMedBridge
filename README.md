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
   Ghi raw response/hash và kết quả cho từng ID, có robots guard, shard checkpoint và resume.
   Lượt đầu giữ Stage A 1.000 ID; để tăng phạm vi, dùng `MODE="range"` với range
   và `RUN_NAME` riêng, ổn định cho từng milestone. `MAX_NEW_SHARDS` giới hạn mỗi
   phiên; không để hai runtime ghi cùng shard.
3. **02 — Trích văn bản và chia đoạn:** [mở notebook](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_extract_and_chunk.ipynb).
   Chạy trên range đã hoàn tất, tạo tài liệu, section và parent/child chunks.
4. **03 — Kiểm tra, audit và freeze:** [mở notebook](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/03_colab_validate_and_freeze.ipynb).
   Kiểm coverage, integrity, golden và source audit; candidate chỉ được promote sau human review.

**Đích khoảng 1 triệu URL là hợp lý như một baseline lớn, nhưng cần đi qua gate theo giai đoạn:**
Stage A 1.000 → Stage B1 10.000 → Stage B2 100.000 → Stage C 1.000.000.
Mỗi milestone cần hoàn tất range, review 100–500 golden nguồn thật và các gate retrieval
áp dụng cho stage đó. Không nhảy thẳng từ Stage A lên 1 triệu; code sẽ chặn nếu thiếu
milestone/evidence. Trước full run lớn hơn, cần đo storage, thời hạn và budget theo gate.

Sau khi baseline mục tiêu hoàn tất, xử lý failures từ ledger theo domain/state trong **một
recovery workflow thống nhất**. Không retry rải rác trong các notebook thí nghiệm giữa lúc
đang crawl baseline. `01` vốn lưu outcome/checkpoint từng ID nên lỗi vẫn được giữ để recovery
sau; robots Disallow hoặc policy chưa xác minh vẫn phải giữ hold. Crawl/recovery chạy CPU;
GPU không cần cho các bước này.

## Notebook thí nghiệm Stage A — không thuộc luồng chính

Các notebook cũ đã được chuyển vào
[`notebooks/experiments/stage-a-1000/`](notebooks/experiments/stage-a-1000/).
`01b–01f` là retry/chẩn đoán/review từng phần; `01g` là recovery nâng cao chỉ dành cho
đúng mẫu Stage A 1.000 URL và các artifact đã ghim hash. Chúng là bằng chứng/kinh nghiệm
để xây workflow recovery tổng thể sau baseline, không phải chuỗi notebook Sếp cần chạy
cho corpus lớn.

Trong mỗi runtime Colab mới, chạy Bootstrap để mount cùng Drive và nạp code lock. Giữ cùng
`DATA_ROOT` trong mọi notebook. Đổi code/config cần run mới; không để một runtime đang chạy
trộn package code giữa các stage.

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
