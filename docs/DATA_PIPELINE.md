# Hợp đồng xử lý dữ liệu

## Luồng chạy

Dataset pinned revision → snapshot Parquet + SHA-256 → audit + crawl sample →
HTTP raw shards → source extraction → BGE-token spans → Parquet + verified offsets.

Notebook chạy trên Google Colab. Google Drive giữ kết quả bền vững; temporary
files, DuckDB spill và file shard đang viết dùng /content/vmb_work. Chỉ publish
file hoàn chỉnh vào Drive; manifest hoàn thành được ghi sau cùng.

## ID và nhãn

query.parquet dùng id/query; links_corpus.parquet dùng id/url. Không đổi ID theo
vị trí dòng. start/stop của crawler là vị trí dòng Parquet, không phải doc_id.
Crawl subset phải khớp cả ID và URL với official corpus.

URL hoặc nội dung giống nhau không cho phép xóa official ID. Khi cần dedup để
tiết kiệm storage/index, phải giữ mapping tất cả official IDs. Hiện tại mỗi ID
được giữ riêng. Dataset query hiện không có relevant_docs/relevant_chunks; không
coi tên split train là bằng chứng có nhãn huấn luyện hoặc evaluation gold.

## Snapshot và resume

Raw Hub files được ghim revision; checksum LFS trên Hub được đối chiếu khi tải.
Audit dùng SQL với memory_limit=512MB và disk spill, tránh set/DataFrame toàn corpus.
Sample lấy vài ID đầu theo top domain, dùng kiểm tra crawler và parser.

Crawl run signature bao gồm hash file input, corpus nguồn, range, cấu hình,
version thư viện và hash code. Worker partition chỉ chia shard trong cùng range.
Resume từ manifest hoàn thành; file tạm không được xem là checkpoint.

Retry-failed tạo raw attempt mới, giữ lại nguồn thành công và raw attempt cũ.
Không chạy hai worker ghi cùng shard. Google Drive không được dùng làm SQLite
database hoặc nơi ghi từng file cho hàng triệu tài liệu.

## Source text và span

Raw snapshot chứa decoded HTTP entity bytes (sau content-encoding), body SHA-256,
URL yêu cầu, URL cuối, content type, fetch time và trạng thái.

HTML dùng trafilatura; XML/JATS lấy tiêu đề/abstract/body; PDF lấy text theo thứ tự
trang. source_text là chuỗi văn bản đã trích, chưa được normalize NFC cho retrieval.
source_text_sha256 hash UTF-8 của chuỗi này.

Mọi child/parent lưu start_char/end_char và phải thỏa:

    span.text == source_text[span.start_char:span.end_char]

Offset là ký tự Unicode Python. Không gọi đây là offset byte của raw HTML/PDF.
retrieval_text chuẩn hóa NFC/whitespace riêng, không dùng làm bằng chứng output.
Chunk ID gồm official ID, source hash, span offsets và chunk/tokenizer policy.
Parent là token window trong cùng tài liệu; section inference chưa triển khai.

BGE fast tokenizer dùng add_special_tokens=False, offset_mapping=True. Đây là
chính sách chia đoạn nội bộ; không khẳng định đã clone tokenizer/normalization,
LCS, duplicate merging hay macro aggregation của scorer chính thức.

## Trạng thái và coverage

Mọi ID được yêu cầu crawl đều có record, kể cả robots_blocked, HTTP lỗi, timeout
và body_too_large. Parser ghi failures cho format chưa hỗ trợ, challenge page,
PDF scan không có text, XML hỏng hoặc source không trích được.

Nguồn ngắn vẫn được giữ với quality flag. Language là hint có phương pháp và
mức tin cậy, chưa làm hard filter. Trang HTTP 200 vẫn có thể cần kiểm tra thêm
để phát hiện error template/bot page chưa có trong rule hiện tại.

build.json liệt kê chính xác các Parquet được chọn và counts cho documents,
children, parents, crawl_failed/parse_failed. Snapshot manifest giữ danh sách
đầu vào và checksum đầu ra. Số recorded/parsed là coverage của run đã chọn;
không đồng nghĩa coverage toàn corpus hoặc recall đối với gold.

## Bước tiếp theo dựa trên output Colab

1. Kiểm tra domain distribution và các format/API thực tế.
2. Phân tích tỷ lệ crawl/parse lỗi, sample text, encoding và chất lượng PDF.
3. Bổ sung bulk/API adapter cho nguồn lớn, OCR/section parsing theo nhu cầu.
4. Freeze snapshot corpus đạt coverage chấp nhận được.
5. Xây retrieval baseline, scorer tương thích và chunk/cutoff ablation khi có nhãn.
