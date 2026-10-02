# VietMedBridge

Chuẩn bị dữ liệu cho truy hồi y sinh đa ngôn ngữ ViBioMIR. Phần tính toán và thu
thập dữ liệu chạy trên Google Colab; snapshot, checkpoint và Parquet đầu ra lưu
trong Google Drive.

## Chạy trên Google Colab

Mở lần lượt các notebook, chọn runtime CPU và chạy từ trên xuống:

1. [00 — Tải và audit dataset](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/00_colab_dataset_audit.ipynb):
   tải snapshot cố định, kiểm tra schema/ID, inventory URL, mẫu Stage A ~1.000 URL phân tầng và golden regression.
2. [01 — Crawl nguồn](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/01_colab_crawl_sources.ipynb):
   crawl Stage A, lưu bytes nguồn/hash và outcome cho từng ID, resume/retry từng shard.
3. [01b — Phục hồi URL lỗi Stage A](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/01b_colab_recover_failed_urls.ipynb):
   kiểm đủ ID/URL của mẫu, retry một lượt trong cùng checkpoint và xuất ledger lỗi theo domain.
4. [01c — Chẩn đoán robots và thử Scrapling](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/01c_colab_robots_and_scrapling_pilot.ipynb):
   đọc báo cáo sau retry, phân loại robots và thử phục hồi có checkpoint riêng trên các URL được phép; không đổi `stage-a-v2`.
5. [01d — Thử lại có robots guard và pacing](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/01d_colab_guarded_recovery.ipynb):
   chỉ xử lý những URL chưa thành article candidate từ experiment 01c đã khóa hash; kiểm robots trước từng request/redirect/retry và xuất báo cáo riêng.
6. [02 — Trích văn bản và chia đoạn](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_extract_and_chunk.ipynb):
   trích HTML/XML/JATS/PDF, đánh dấu chất lượng, tạo section và child/parent bằng tokenizer BGE-M3.
7. [03 — Validate, audit và freeze](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/03_colab_validate_and_freeze.ipynb):
   kiểm toàn snapshot, dedup xuyên shard có alias, xuất health report/HTML audit và freeze candidate.

Package cho phép Python 3.11–3.13; notebook kiểm tra phiên bản trước khi cài thư viện.

Trong **mỗi runtime Colab mới hoặc vừa restart**, chạy cell Bootstrap đầu notebook:
nó mount Drive, clone/cập nhật repo rồi cài package. code_lock.json lưu commit,
nhưng không giữ package đã cài trong runtime đã mất. Giữ cùng DATA_ROOT trong mọi
notebook; mặc định là MyDrive/VietMedBridge/data. code_lock.json lưu commit đã
chạy để các notebook sau dùng lại cùng mã nguồn. Tokenizer cũng được ghim revision.

Nếu Drive đang khóa code cũ API v1, đặt CODE_REVISION="main" trong bootstrap một
lần để cập nhật sang API v2; restart session nếu runtime đã import package cũ.
Dùng tên run mới stage-a-v2/stage-a-data-v2; các artifact cũ được giữ để rollback.

Notebook 01 mặc định chạy Stage A phân tầng; MODE="smoke" vẫn dùng được. Để xử lý phạm vi lớn,
chọn MODE="range", đặt START_ROW/STOP_ROW và giữ range/cấu hình ổn định khi resume.
MAX_NEW_SHARDS giới hạn lượng công việc trong một phiên Colab. Đổi range hoặc cấu
hình cần tên run mới. Các runtime đồng thời phải dùng worker index khác nhau.
Với hàng triệu URL, cần kiểm tra lỗi theo domain và bổ sung bulk/API adapter cho
những nguồn lớn trước khi triển khai toàn corpus.

Nếu Stage A có URL lỗi, chạy notebook 01b với `code_lock.json` của run hiện tại.
Notebook kiểm đủ official ID/URL của mẫu, retry một lượt, rồi ghi danh sách URL
chưa tải được và số URL của từng domain bị ảnh hưởng trong inventory toàn corpus
vào `reports/crawl_recovery/stage-a-v2/`. Retry không vượt robots.txt hoặc HTTP 403;
các nguồn này cần quyền truy cập hoặc API/bulk được phép. Giữ cùng code lock khi
chạy notebook 02–03 trên checkpoint Stage A này.

Notebook 01c dùng runtime CPU mới và `recovery_code_lock.json` riêng để tải mã
mới mà không làm đổi code hash của checkpoint Stage A. Nó đối chiếu
`failures_after_retry.csv` với official corpus, ghi từng robots probe/HTTP thử
nghiệm lên Drive. Browser hiện tạm dừng vì Scrapling 0.4.15 có thể tiếp tục
điều hướng khi callback cài route guard lỗi. Notebook 01d dùng code lock riêng,
chỉ thử lại các URL chưa thành ứng viên bằng HTTP có kiểm robots và pacing ở
từng bước; không cần chạy lại 01c. `article_candidate` chỉ là ứng viên cần người
kiểm nội dung, chưa được nhập vào canonical corpus. Muốn tiếp
tục notebook 02 trên raw `stage-a-v2`, dùng lại `code_lock.json` cũ trong một
runtime mới; thay package code giữa runtime đang import sẽ bị chặn.

Stage A dùng 11 fixture synthetic để bootstrap. Notebook 03 xuất 100 nguồn để
team gán expected snippets và replay thành golden nguồn thật. Scale-up cần bộ
100–500 nguồn này pass, human review và toàn bộ range đã có outcome. Stage B2/C
cần retrieval regression có nhãn; full run cần budget đo thực tế. Các gate không
tự coi sample audit, query hoặc self-retrieval là nhãn relevance.

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
