# Sau intake 10 nguồn BTC mới

Mở `notebooks/03_colab_freeze_new_data.ipynb`, một phiên Colab **CPU riêng → Run all**.
Notebook nhận tám nguồn 00000–00007 + retained của campaign 19/10 và nguồn 00008
của campaign bổ sung. Không dùng 03 cũ để dò `data/incoming`/worker của corpus cũ.

Notebook này không sửa active candidate, code lock, runtime lock hoặc model cache
trong `VietMedBridge/data`. Code producer/runtime được lấy từ lock intake đã lưu.
Nó không crawl/import/chunk lại và không tự coi các nguồn lỗi là document hợp lệ.

Lần đầu mỗi build, payload documents/children/parents/sections/failures/ledger
được copy có SHA từ Drive vào scratch Colab; health, membership và source-span gates
chạy trên bản local này. Candidate và báo cáo được publish về đúng build. SQL một
luồng, tối đa 2 GiB và không quá một phần tư MemAvailable. Chỉ giữ một build staging;
sau khi receipt đầy đủ trên Drive, bản staging local được dọn. Không tải về PC.

Index inputs dùng producer's InputResume/SQLite spool, part 4.096 text. Mỗi part
được lưu Drive. Nguồn đã freeze/input xong nhận lại bằng JSON đã seal, không quét
payload. Nguồn mới freeze xong nhưng input dở dùng lại candidate đó. SQLite spool
được commit và lưu lên Drive **trước part input đầu tiên**. Runtime mới khôi phục
một file SQL checkpoint có kiểm SHA; không quét lại Parquet nguồn đã stage.
Trong cùng VM, source loader chỉ đọc part chưa commit trong spool. Nếu input part
đã đủ nhưng manifest chưa seal, không nạp lại nguồn hoặc tokenizer. Checkpoint cũ
chưa có SQL backup có thể cần stage riêng build dở một lần; không đọc lại payload
của build đã `complete.json`.
Không chạy hai phiên notebook này cùng lúc.

Đích tổng:

```
VietMedBridge/data_staging/btc-new-71be0e12-20261019-v1/freeze_batch_v1/
  contract.json
  builds/<build-run>/frozen.json
  builds/<build-run>/complete.json
  builds/<build-run>/input_spool.json
  builds/<build-run>/input_spool/representatives.sqlite
  freeze_batch.json
```

Handoff `NEW_DATA_INPUTS_READY` gồm locator tương đối project, candidate checksum,
input-manifest checksum và counts cho cả 10 build. Đây chưa phải vector: consumer
GPU phải kiểm SHA payload khi dùng, encode hash thiếu dưới đúng model/encoding
contract và cập nhật index. Input count tổng theo build có thể trùng hash giữa
build; không cộng như số unique text toàn corpus. Việc hợp nhất vào candidate mới
cần một bước riêng, không đổi candidate của luồng submit đang chạy.

Intake complete/freeze không xác nhận đủ toàn bộ ID BTC, không biến crawl/parse
failures thành dữ liệu tốt, không phải relevance evaluation hoặc human review.
