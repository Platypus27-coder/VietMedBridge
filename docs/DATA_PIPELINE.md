# Hợp đồng dữ liệu API v2

Colab 00 → inventory/mẫu phân tầng/golden → Colab 01 crawl raw → Colab 02
extract/structure/chunk → Colab 03 whole-snapshot validation, global aliases,
health audit và FROZEN_CANDIDATE. Cấu hình khởi đầu nằm trong
`configs/data_pipeline.json`; code và tokenizer được khóa theo revision.

## Official ID và provenance

`query.parquet` dùng id/query; `links_corpus.parquet` dùng id/url. Giữ nguyên
ID/URL, gồm cả URL trùng và ID không liên tục. start/stop là vị trí dòng Parquet.
Subset phải khớp ID/URL chính thức; notebook build bắt buộc cung cấp official corpus.
Mỗi input có một outcome trong documents hoặc failures và một record trong ledger.

Raw shard giữ decoded HTTP entity bytes, body hash/size, original/final URL,
content type, fetch time, status và latency. Không gọi đây là bytes trước HTTP
content-encoding. Raw được nén JSONL để gói nhiều nguồn trong một shard.

`source_text` là chuỗi đã trích từ raw, giữ nguyên sau extraction. source hash
là SHA-256 UTF-8. Child/parent phải có cùng doc_id/source hash và thỏa:

    span.text == source_text[span.start_char:span.end_char]

Offset dùng ký tự Unicode Python, end-exclusive; không phải byte offset HTML/PDF.
Submission evidence dùng `text`; retrieval representation được tạo riêng.

## Extraction, structure và chất lượng

HTML dùng trafilatura rồi fallback BeautifulSoup nếu không có text. XML/JATS
giữ title/abstract/body và linearize table theo hàng/cell. PDF dùng pypdf theo
thứ tự trang. Không có OCR; PDF thiếu text, format không hỗ trợ và trang lỗi rõ
ràng đi vào failures cùng reason code. Exception lập trình bất thường làm shard
fail, không bị nuốt thành lỗi nguồn thông thường.

Section được neo vào heading trùng chính xác một dòng source_text. Parser
không đoán heading bằng semantic similarity. Không tìm được heading thì dùng
body section và NO_SECTIONS flag. Section spans được lưu Parquet; paragraph và
sentence spans được tính từ source/section để chọn ranh giới chunk. Sentence
splitter bảo vệ các viết tắt đơn giản như H. pylori và số thập phân; chưa phải
bộ phân đoạn biomedical đầy đủ. HTML/JATS table cần human audit; PDF layout/table
chưa được phục hồi chuyên biệt.

Quality rules ghi HIGH/MEDIUM/LOW, độ dài, số paragraph/section, tỷ lệ alphabetic,
dòng lặp và tín hiệu biomedical. Các ngưỡng này là heuristic ban đầu. LOW và
language unknown vẫn eligible. Language ghi phương pháp và confidence thực có;
source-declared language không được gán xác suất giả. Failure language chưa biết
không nằm trong thống kê language của extracted documents.

## Child, parent và representation

Baseline child 180, overlap 40, parent 512 token, ưu tiên section và ranh giới
paragraph/sentence. Tokenizer BGE-M3 fast được ghim revision, không tải weights.
Khi ưu tiên section, tokenizer chạy từng section chính xác rồi chuyển offset về
source toàn tài liệu; tránh token SentencePiece mang newline qua heading.
token_start/end tham chiếu chuỗi token nối từ các section của policy này.

Source slices được retokenize để kiểm ngân sách thực tế. Child ID gồm official
doc_id, source hash, offsets và child policy; parent_tokens không nằm trong child
policy. `parent_for_child` tạo context theo budget khác mà giữ child identity.
Parent chứa child và ưu tiên cùng section. Thay parent budget tạo parent ID mới.

Dense builder dùng title + heading + normalized child body, kiểm cả special
tokens ở budget 512. Chỉ prefix tùy chọn có thể được rút ngắn; không truncate body.
Hash representation tính trên đúng formatted text này. Sparse builder có cùng
cách ghép, chưa được dùng để xây BM25 trong giai đoạn data.

`content_hash` nhóm source đã NFC/whitespace-normalize. Global canonical aliases
giữ mỗi parsed official doc_id đúng một lần; representation aliases giữ mọi
child_id. Hai official IDs cùng nội dung vẫn tồn tại trong source/chunk Parquet.
Đây là mapping để dedup index/embedding sau này, chưa physical-dedup source storage.
Embedding reuse còn cần cùng model, tokenizer và encoding policy.

## Checkpoint và commit

Crawl signature khóa input hash, corpus hash, requested range, config, runtime
versions và code hash. Shard giới hạn cả số record và byte JSONL trước nén, gồm
base64/metadata. Vượt budget làm attempt fail; chọn shard nhỏ hơn trong run mới,
không tự truncate nguồn. Các thư mục bucket tránh một folder chứa toàn bộ shard.

Temporary files và DuckDB spill nằm ở /content/vmb_work. Publish từng file hoàn
chỉnh, kiểm checksum; marker .done.json ghi cuối cùng mới xác nhận shard commit.
Rename của một file không có nghĩa nhiều file đã commit cùng lúc. Temp names
riêng giúp worker ở shard khác nhau không va chạm khi ghi manifest chung.
Một build/report chỉ dùng một writer. Không chạy hai worker ghi cùng raw shard.

Resume chỉ dùng done pointers và file hashes. Retry failed tạo raw attempt mới,
giữ nguồn thành công cùng raw attempt cũ. Build manifest chỉ chọn attempt hiện
tại; không glob Parquet để đưa các attempt cũ vào snapshot. Đổi code/config/range
thì dùng run name mới. Legacy API v1 cần recrawl/rebuild trong run API v2 mới;
không gắn metadata coverage mới lên checkpoint cũ chưa có bằng chứng.

## Hard validation, health và freeze

Validator dùng DuckDB 512 MB với disk spill, kiểm uniqueness toàn snapshot,
ID/URL membership, input/outcome conservation, source hashes, exact slices,
foreign keys, parent containment, saved token budgets và manifest counts.
Duplicate detection đếm số dòng so với distinct; không chuyển sang set rồi làm
mất bằng chứng duplicate. Lỗi integrity ngăn DATA_VALIDATED/freeze.

Health JSON có coverage/failures, quality/language/domain/type, latency, raw và
processed bytes, global duplicate counts, phân phối mean/P05/P50/P95/P99 của
chars/paragraphs/sections/chunks/tokens. HTML audit giới hạn preview và highlight
child đầu tiên. Quantile dùng DuckDB approx_quantile để không giữ toàn bộ các
giá trị trong aggregate state; mean/count vẫn tính trên mọi record được chọn.
Baseline comparison ghi relative changes; warning threshold phải
được team calibrate từ pilot. Khác sample/domain có thể tạo drift tự nhiên.
Chưa có kill-switch drift tự động theo shard hoặc forecast full-corpus đã hiệu chỉnh.

Freeze kiểm lại file hashes, code, tokenizer và integrity. FROZEN_CANDIDATE có
snapshot/candidate manifest hash và các derived artifacts; có thể freeze partial
snapshot để chẩn đoán. Milestone phải hoàn tất **toàn requested range**, kể cả
outcome lỗi, không chỉ tất cả shard đang có. Zero extracted docs không được freeze.

11 synthetic fixtures cho phép bootstrap pilot. Trước scale beyond pilot, team
phải điền expected assertions cho 100–500 nguồn thật, replay raw đã ghim, xem
audit và ghi reviewer. Stage B2/C/full còn yêu cầu labeled retrieval evidence;
full yêu cầu resource estimates trong ngân sách rõ ràng. Không suy ra qrels từ
query, dedup hoặc self-retrieval. Golden extraction không phải relevance labels.

Candidate chưa PROMOTED. Promotion/index registry, official scorer, ANN recall,
GPU throughput, BM25/index storage và Chunk F2 là công việc retrieval tiếp theo.
