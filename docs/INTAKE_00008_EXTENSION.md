# Nguồn 00008 đến sau campaign data mới

Ngày 20/10 đã đối chiếu `full_intake_status.json` và chín `sources/*/complete.json`:
campaign `btc-new-71be0e12-20261019-v1` hoàn tất tám frontier 00000–00007 và retained.
Counts khớp: 492.380 document bổ sung, 264.985 lượt native ID dùng lại, 360.327 lượt
native ID lỗi. Build receipts ghi 4.714.098 children và 3.385.129 parents bổ sung.
Đây là kiểm tra receipt và đối chiếu số, chưa phải audit lại mọi payload hoặc y khoa.

Folder AI Guru còn có `vibiomir_retain_shard_00008.tar` (Drive ID
`1lqL46iCpfD7qnub-8ih1avKj6K3Dporv`, 2.503.065.600 byte), ngoài source list của
campaign cũ. Không sửa hoặc nối archive này vào plan cũ đã khóa.

Mở [02_colab_import_new_00008.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_import_new_00008.ipynb),
chọn **CPU → Run all**, một phiên riêng. Không chạy lại notebook chín nguồn hoặc
bốn worker cũ để thêm 00008. Giữ luồng submission hiện tại chạy riêng.

Notebook dùng lại code/runtime lock của parent, còn điều phối được nhúng ngoài
`src/vietmedbridge` để không đổi producer fingerprint. Chỉ kiểm chín receipt nhỏ
và size của mapping khi nhận handoff. Mapping được kiểm SHA khi thực sự dùng để
lọc ID; sources cũ không qua importer/chunking lần nữa.

Audit nguồn 00008 chạy trong Colab qua Drive API Range: đọc descriptor/frontier,
khóa snapshot, frontier SHA và metadata phiên bản archive. Không tải toàn bộ TAR
2,50 GB hoặc tải dữ liệu về PC. Mọi membership, raw/extraction binding, quality và
source-span chunk gate tiếp tục dùng importer đã khóa. Archive/version/snapshot
khác bị từ chối; không tự bỏ checkpoint để chạy lại nguồn khác.

Checkpoint/kết quả trên Drive:

```
VietMedBridge/data_staging/btc-new-71be0e12-00008-20261020-v1/
  source_audit.json
  intake_plan.json
  intake_code_lock.json
  intake_runtime_lock.json
  intake_status.json
  sources/new-00008/complete.json
  processed/new-00008/
```

Đổi VM: mở lại đúng notebook, cùng folder rồi Run all. Audit đã commit được nhận
lại bằng JSON; source hoàn tất chỉ nhận receipt. Source dở giữ selection và các
shard đã commit theo cơ chế intake hiện có. Không đổi campaign, plan hoặc xóa lock.

Chỉ sau khi chạy mới biết số ID/documents thực tế của archive 00008. Không cộng
naive lượt ID mới vào độ phủ cũ để tuyên bố đủ BTC: cần đối chiếu hợp tập native ID
với 1.220.598 ID chính thức. `INTAKE_COMPLETE` ở đây chỉ xác nhận import/quality/chunk
CPU cho 00008, chưa phải freeze, embedding, index hoặc submission. Active pointers,
snapshot và model cache của phiên submit hiện tại không thay đổi.
