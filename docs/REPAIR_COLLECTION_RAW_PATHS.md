# Sửa phân loại raw của lượt thu gom AI Guru

Native archive dùng `data/raw/html/...`; bộ lọc cũ chỉ nhận `raw/...` và ghi lý
do `SUCCESS_RAW` vào nhóm lỗi crawl. Đây là lỗi kiểm tra metadata, không phải
bằng chứng crawl thất bại. Bộ lọc mới chấp nhận cả hai prefix cho native; retained
portable giữ contract `raw/...`. Path traversal, đường dẫn tuyệt đối, NUL,
backslash, hash sai và snapshot không khớp vẫn bị chặn.

Sau khi đủ inventory archive, gọi `flow.repair_raw_paths()` bằng module mới.
Đổi một dòng điều kiện rồi chạy `flow.run()` không cập nhật được các source đã
đánh dấu COMPLETE. Notebook mới tự gọi bước sửa sau khi hoàn tất inventory.

Bước sửa chỉ đọc bản ghi crawl thành công từ metadata đã lưu trên Drive. Nó dùng
cache thu gom và các receipt metadata của intake/processed cũ, chọn file bằng
SHA256 khớp nguồn gốc rồi kiểm checksum khi dùng. Không mở TAR, gọi crawl, encode,
rebuild catalog hay sửa active submission candidate. Nếu thiếu metadata đã lưu,
báo `SAVED_METADATA_MISSING`, không tự tải lại archive.

Mỗi leaf được sửa có receipt SQL và part recovery riêng theo policy
`native-data-raw-and-portable-raw-v2`; không ghi đè part hay receipt cũ. Hash của
raw/extraction, snapshot, source length và quality gate vẫn được kiểm. Một HTTP
SUCCESS_RAW có metadata không hợp lệ được tách thành `RAW_METADATA_REVIEW`.

Xem `raw_path_repair.state`, `raw_path_classification.pending_leaves` và
`counts_provisional`. Chỉ dùng số liệu làm kết quả cuối khi repair COMPLETE,
pending bằng 0 và counts_provisional=false. `ALL_FOLDER_SOURCES_COLLECTED` vẫn
chỉ xác nhận inventory đã đọc đủ. Text recovery còn cần chunk/freeze chuẩn.

Checkpoint được lưu sau từng leaf và seal SQL cuối trước khi báo hoàn tất.
Đổi VM dùng lại SQL checkpoint; nguồn đã sửa không được replay lại. Receipt index
được lưu để lần tiếp tục không liệt kê lại các manifest cũ. Bản metadata đang được
đọc có thể cần đọc lại nếu leaf bị ngắt trước commit; phần đã commit vẫn được giữ.
