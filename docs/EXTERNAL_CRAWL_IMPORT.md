# Nhập bản crawl độc lập của team

Chạy **02 → 03 trên CPU** trước. Không chạy lại 01, không cần 05.
Giữ DATA_ROOT đã dùng: `/content/drive/MyDrive/VietMedBridge/data`.

## Nguồn trên Drive

Batch mới Sếp vừa đưa lên nằm trong `data/incoming/team-crawl-archives-2026-10-08/`
và có sáu archive: `vibiomir_shard_00001.tar` đến `vibiomir_shard_00006.tar`,
tổng khoảng 19,05 GB. Cùng shard `00000` đã xử lý trước đó, đây là batch khoảng
700k URL cần dùng cho pipeline. 02 được cấu hình trỏ thẳng vào folder này và
chạy tuần tự cả sáu archive; không chọn lại riêng shard `00001`.

Archive gốc `vibiomir_shard_00000.tar` có 3.187.025.920 bytes, SHA-256:
`cfb0dd6cfcb1cc6f8665717942b076a0e71011736d29e4a175064fc7b168929a`.
Cổng upload connector giới hạn 100 MiB; bản chia phần dùng 96 MiB mỗi phần:

```text
VietMedBridge/data/incoming/vibiomir_shard_00000.tar.parts/
  archive_manifest.json
  vibiomir_shard_00000.tar.part00000
  ...
  vibiomir_shard_00000.tar.part00031
```

02 tự tìm thư mục này, kiểm đủ các phần/kích thước, ghép trên ổ local Colab với
buffer 1 MiB và kiểm SHA-256 từng phần lẫn toàn archive. Không cần ghép bằng tay
hoặc lưu thêm bản tar 3,19 GB trên Drive. Cần khoảng 3,5 GB local disk cho archive
và metadata, cộng dung lượng shard tạm/output. Archive và bản chia phần giữ nguyên.

Sau đó đọc bốn metadata: dataset_manifest, frontier, crawl_manifest và
extraction_manifest. Không tải website hoặc giải nén lại toàn bộ raw payload.
Importer nhận frontier Parquet có số shard tương ứng trong tar, ví dụ
`data/crawl_shards/shard_00001.parquet`; mỗi archive cần đúng một frontier shard
và một `BUILD_RUN` riêng.
Archive đang có 100.000 unique URLs tương ứng 100.039 official IDs do aliases;
metadata ghi 90.645 URLs EXTRACT_SUCCESS. Đây là trạng thái extraction cũ, chưa
phải số documents vượt kiểm chất lượng mới hoặc số có relevance labels.

## 02: import/chunk có checkpoint

Mặc định `INPUT_KIND="external"`, shard 2.048 official IDs. `EXTERNAL_SOURCE`
nhận file tar hoặc folder batch; folder sẽ nhập hết file tar theo thứ tự.
`BUILD_RUN` trống sẽ tự sinh tên riêng ổn định cho từng archive. `active_data_batch.json`
checkpoint toàn danh sách; Run all lại cùng folder để resume archive/shard đang dở.

Kiểm SHA snapshot official, URL → ID/alias, liên kết raw SHA/snapshot và số ký tự.
Giữ nguyên source_text, chia bằng tokenizer BGE cố định, kiểm mọi span. Frontier URL
không có crawl row hoặc crawl thành công nhưng thiếu extraction row được giữ thành
failure rõ ràng; audit ghi từng official ID bị ảnh hưởng vào
`external_coverage_gaps.csv`. URL mismatch, raw binding mismatch và sai snapshot
vẫn chặn import. Không
giả định ID liên tục. Nguồn lỗi/challenge/redirect/trích rỗng vào failures; LOW
quality được giữ kèm flags. Mỗi input ID có ledger và đúng một outcome.

Parquet ghi theo buffer, không tạo một row group cho từng document. Sort chỉ trên
ID; text chỉ lấy cho shard đang xử lý. DuckDB dùng giới hạn 512 MiB và local spill.
Metadata được cache trên local disk, không scan lại metadata trên Drive mỗi shard.

Checkpoint chỉ xuất bản sau đủ sáu file và hash của shard. Ngắt Colab thì Run all
cùng nguồn/run/config/code để tiếp tục. Không để hai runtime ghi cùng build.
`PARTIAL_DATA_VALIDATED` chưa được freeze; phải hoàn tất 100.039 input outcomes.
Thay code/source/policy thì dùng build run mới để không trộn checkpoints.

## 03: kiểm snapshot, freeze và chuẩn bị input trên disk

Tự đọc mọi build hoàn tất từ `active_data_batch.json` của 02, kể cả sau khi Colab
ngắt. Kiểm whole-snapshot source hashes, official membership, parent/child/section
offsets và counts cho từng archive, rồi freeze và chuẩn bị index inputs từng build.
Candidate lineage tự nối từng batch với candidate cũ; không cần chạy 03 sáu lần.
Health giữ global content/representation aliases. Freeze là candidate integrity, không tự duyệt y khoa,
relevance, raw replay hoặc milestone quy mô.

`processed/team-100k-data-v1/index_inputs/units.json` liệt kê các Parquet phần nhỏ,
mỗi text representation giống hệt chỉ encode một lần; mọi official aliases vẫn
ở candidate. Chuẩn bị input không có nghĩa embeddings/index đã hoàn tất.
`active_data_candidate.json` nối đúng candidate tới bước GPU, tránh dùng nhầm 864
documents cũ.

## 04 hiện tại: full system trên candidate lớn

Xem [FULL_SYSTEM_RUNBOOK.md](FULL_SYSTEM_RUNBOOK.md) để chạy bốn model theo
master plan, phân công ba người, ghép data mới và dùng lại vectors 100k.
Bản baseline mô tả bên dưới là lượt trước đã đạt **0.0081** theo kết quả Sếp gửi.
Nó được giữ trong code/artifacts để so sánh, không còn là nhánh mặc định của 04.

## Lượt baseline trước: BGE + BM25 trên corpus lớn

Benchmark T4 đã hoàn tất trên candidate `ce6987985fb015ca`: BGE batch 4 khoảng
46,2 texts/s, Qwen embedding 8B khoảng 6,65 texts/s, Qwen reranker khoảng 4,19
pairs/s; không OOM. Child-only projections lần lượt 3,45h và 23,97h, không gồm
I/O, document dense, query LLM, sparse/index hay toàn cascade. Mẫu 64 không là
cam kết thời gian hoặc bằng chứng relevance.

04 ở phiên bản baseline trước tự chạy **large baseline** nếu candidate vượt pilot 2.000 docs/
50.000 children. Giữ DATA_ROOT, chọn GPU T4 trở lên và Run all; không ACTION,
không chạy lại 02–03. Workflow API mới tự nâng retrieval code lock một lần.
Runtime đã import code cũ thì restart session trước khi Bootstrap.

1. SQLite local lưu source và mappings official ID/child/parent/unit. BM25 dùng
   FTS5 contentless để không nhân bản toàn bộ token text, ba language partitions,
   PyVI/Jieba/CJK tokens và title/headings/body/medical alias fields.
2. BGE encode **tất cả 573.854 unique input texts**, lưu vector float32 theo 141
   parts của 03. Sort độ dài trong mỗi part rồi restore input order. Batch 32 tự
   giảm khi OOM; chưa đo tốc độ batch này trên GPU thật. Vectors riêng khoảng
   2,35 GB, cộng SQLite và local vector cache. Không tải weights Qwen.
3. Encode đủ 1.200 original queries, unload BGE, dense search toàn corpus theo
   blocks trên GPU. Giữ top 2.048 child representations/query, bảo toàn mọi official
   aliases rồi lấy tối đa 200 document candidates; đây không phải document-dense.
4. Fuse với BM25 bằng weighted RRF. Local child dense/lexical ranking, lấy frozen
   source parents từ 03 và LCS dedup. Không neural reranker/LLM translation hoặc
   parent 640 mới trong lượt baseline. Cutoffs chưa được tune bằng nhãn.
5. Validate mọi query ID, document ID, source hash, anchor, parent offsets/text và
   ZIP trước xuất `results.json` ở root của `submission.zip`; lưu evidence bên ngoài.

Run nằm ở `data/retrieval/team-100k-data-v1-bge-bm25-v1-<candidate-prefix>/`.
Vector checkpoint + query checkpoint lưu Drive, checksum trước reuse. Ngắt phiên
thì Run all lại cùng config. Search hiện có checkpoint top-k giữa lượt; ngắt phiên thì resume từ phần đã
commit, không encode lại vectors. Catalog CPU publish atomic; ngắt trước
khi hoàn tất catalog thì xây lại riêng catalog. Không hai runtime ghi cùng run.

Đây là baseline partial corpus để lấy điểm sớm, **không phải full master plan**,
không tự promote quality/relevance hoặc giả fine-tune. Full architecture đã được nối vào corpus lớn ở 04 mới; vectors BGE đã kiểm
được tận dụng. Xem runbook hiện hành ở trên.
05 supervised chỉ chạy khi có nhãn review độc lập phù hợp. Large baseline đã được Sếp chạy 100k và submit; full architecture mới
chưa được đo toàn corpus trên GPU thật. Điểm BTC vẫn phải lấy từ ZIP chạy thật, không từ test/benchmark.
