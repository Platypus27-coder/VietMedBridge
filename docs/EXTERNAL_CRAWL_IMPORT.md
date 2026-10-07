# Nhập bản crawl độc lập của team

Chạy **02 → 03 trên CPU** trước. Không chạy lại 01, không cần 05.
Giữ DATA_ROOT đã dùng: `/content/drive/MyDrive/VietMedBridge/data`.

## Nguồn trên Drive

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
Archive đang có 100.000 unique URLs tương ứng 100.039 official IDs do aliases;
metadata ghi 90.645 URLs EXTRACT_SUCCESS. Đây là trạng thái extraction cũ, chưa
phải số documents vượt kiểm chất lượng mới hoặc số có relevance labels.

## 02: import/chunk có checkpoint

Mặc định `INPUT_KIND="external"`, `BUILD_RUN="team-100k-data-v1"`,
`EXTERNAL_SOURCE=None`, shard 2.048 official IDs. Auto discovery chỉ tìm trong
`data/incoming`, `data/data_temp` và `VietMedBridge/data_temp`; nhiều nguồn thì cần
đặt một EXTERNAL_SOURCE cụ thể.

Kiểm SHA snapshot official, URL → ID/alias, liên kết raw SHA/snapshot và số ký tự.
Giữ nguyên source_text, chia bằng tokenizer BGE cố định, kiểm mọi span. Không
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

Tự đọc active_data_build.json từ 02. Kiểm whole-snapshot source hashes, official
membership, parent/child/section offsets và counts. Health giữ global content/
representation aliases. Freeze là candidate integrity, không tự duyệt y khoa,
relevance, raw replay hoặc milestone quy mô.

`processed/team-100k-data-v1/index_inputs/units.json` liệt kê các Parquet phần nhỏ,
mỗi text representation giống hệt chỉ encode một lần; mọi official aliases vẫn
ở candidate. Chuẩn bị input không có nghĩa embeddings/index đã hoàn tất.
`active_data_candidate.json` nối đúng candidate tới bước GPU, tránh dùng nhầm 864
documents cũ.

## 04: benchmark có giới hạn trước full scale

Khi candidate vượt pilot 2.000 documents/50.000 children, 04 tự chuyển sang đo
64 inputs trên tối đa tám input parts; model nạp lần lượt BGE → Qwen embedding
8B → Qwen reranker 8B. Đo download/load riêng, warmup, hai lượt inference, VRAM,
OOM backoffs và child-embedding time projection. Checkpoint từng model.

Đây là mẫu timing, không phải thời gian bảo đảm cho corpus hoặc điểm relevance.
Không đo query LLM, document dense, sparse index/ANN/fusion hay full query cascade.
Chưa xuất submission 100k: phần catalog/index disk-backed cần tích hợp tiếp theo
ngân sách thực đo. 04 giữ nguyên full pretrained submission cho corpus pilot.
05 supervised chỉ chạy khi có nhãn review độc lập phù hợp.
