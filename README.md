# VietMedBridge

Chuẩn bị dữ liệu cho truy hồi y sinh đa ngôn ngữ ViBioMIR. Phần tính toán và thu
thập dữ liệu chạy trên Google Colab; snapshot, checkpoint và Parquet đầu ra lưu
trong Google Drive.

## Chạy trên Google Colab

Mở lần lượt các notebook, chọn runtime CPU và chạy từ trên xuống:

1. [00 — Tải và audit dataset](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/00_colab_dataset_audit.ipynb):
   tải hai file Parquet ở revision cố định, kiểm tra schema/ID, thống kê domain và tạo mẫu crawl.
2. [01 — Crawl nguồn](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/01_colab_crawl_sources.ipynb):
   thử mẫu nhỏ, lưu bytes nguồn và hash, retry lỗi tạm thời, resume từng shard.
3. [02 — Trích văn bản và chia đoạn](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_extract_and_chunk.ipynb):
   trích HTML/XML/PDF, tải tokenizer BGE-M3, tạo child/parent và kiểm tra vị trí nguồn.

Package cho phép Python 3.11–3.13. Nếu cần giữ runtime ổn định, chọn Runtime →
Change runtime type → Runtime Version 2026.07 (Python 3.12), theo
[danh sách runtime của Google](https://research.google.com/colaboratory/runtime-version-faq.html).

Cell đầu mount Drive và cài thư viện từ repo. Giữ cùng DATA_ROOT trong cả ba
notebook; mặc định là MyDrive/VietMedBridge/data. code_lock.json lưu commit đã
chạy để các notebook sau dùng lại cùng mã nguồn. Tokenizer cũng được ghim revision.

Notebook 01 mặc định chạy smoke sample của các domain lớn. Để xử lý phạm vi lớn,
chọn MODE="range", đặt START_ROW/STOP_ROW và giữ range/cấu hình ổn định khi resume.
MAX_NEW_SHARDS giới hạn lượng công việc trong một phiên Colab. Đổi range hoặc cấu
hình cần tên run mới. Các runtime đồng thời phải dùng worker index khác nhau.
Với hàng triệu URL, cần kiểm tra lỗi theo domain và bổ sung bulk/API adapter cho
những nguồn lớn trước khi triển khai toàn corpus.

## Dataset

Nguồn: [AIGuruTinix/ViBioMIR](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR).
Hai cấu hình Hub là query và corpus; cả hai dùng split tên train. Cấu hình query
hiện có id/query; corpus có id/url và chưa chứa văn bản tài liệu.

Revision khởi đầu: ca87a68e42843d0a49b57d02e6e3d28ea15273c9.
Metadata tại thời điểm kiểm tra báo 4.394.718 dòng corpus, trong khi dataset card
ghi 4.420.561 liên kết. Pipeline đếm trực tiếp Parquet và không suy luận rằng ID
liên tục từ 1. ID chính thức được giữ nguyên, kể cả khi URL hoặc nội dung trùng.
Split tên train chưa cung cấp reference labels để tính F2.

## Artifact dữ liệu

- raw/: Parquet và dataset card nguyên bản, revision và SHA-256.
- reports/: audit, thống kê domain và mẫu link giữ nguyên ID/URL.
- crawl/: raw snapshot nén theo shard, manifest và trạng thái lỗi của từng ID.
- processed/: documents, children, parents, failures cùng manifest đầu ra.

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

Kiểm tra mã:

    python -m pytest -q
    python scripts/check_notebooks.py

Sinh lại notebook sau khi sửa cell source:

    python scripts/write_notebooks.py

## Phạm vi hiện tại

Giai đoạn này tạo nền xử lý dữ liệu có thể resume. Chưa tạo embedding/index,
chưa huấn luyện reranker và chưa triển khai scorer F2 chính thức. Child 180,
overlap 40 và parent 512 token là các tham số khởi đầu cho thí nghiệm.
PDF scan/OCR, adapter đặc thù nguồn lớn và section parsing cần được bổ sung theo
báo cáo sample. Lỗi crawl/parse được lưu; không tạo văn bản thay thế cho nguồn lỗi.

Hai repo r2ai-stage-1 và r2ai-stage-2 ngoài VietMedBridge dùng làm nguồn tham khảo.
Các file kế hoạch .md trong repo giữ bối cảnh thiết kế. Data, cache và artifact
không được commit lên GitHub.

Dataset card công bố CC BY-NC 4.0 và hướng dẫn ghi nguồn
[MedKB](https://medkb.tinix.ai/); nội dung crawl giữ URL và provenance của nguồn tương ứng.
