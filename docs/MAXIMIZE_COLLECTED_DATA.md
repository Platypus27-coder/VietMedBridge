# Tận dụng tiếp nguồn AI Guru đã thu gom

Mở `notebooks/02_colab_maximize_collected_data.ipynb`, chọn **Colab CPU → Run all**.
Chỉ một phiên ghi vào campaign này. Không cần đổi worker/account/GPU, không gọi
website, không cần Kaggle/HF token. Người dùng tự chạy notebook.

Đầu vào là báo cáo collection đã hoàn tất sửa raw-path: không còn archive error,
`counts_provisional=false`. Khóa báo cáo, producer code, runtime, tokenizer và
chunk policy; không lặng lẽ trộn thêm dataset mới vào campaign đang chạy.

1. Giữ 798.948 ID có nội dung cũ qua reference; không đọc lại corpus hay vector.
2. Ưu tiên 2.304 ID mới có text: kiểm provenance, alias BTC, source span, chạy
   chunk/freeze và chuẩn bị model-input chỉ cho delta. Cùng policy BGE 180/40,
   parent 512 và pinned tokenizer. Không đổi text đã được chấp nhận.
3. Lập hàng đợi từ metadata đã cache cho các URL chưa có text nhưng có raw.
   Mỗi source metadata làm một lần; nếu thiếu cache, đọc đúng member range cũ.
4. Đọc raw cần thử từ TAR; native/nested/portable giữ locator riêng, original
   snapshot/crawl record được bảo toàn. Phải khớp hash và kích thước raw.
   Bounded gzip/zstd decoding, tối đa 64 MiB; file lớn hơn đi vào review.
5. Parser/gate hiện có xử lý lại body. Trang challenge, lỗi, không có text,
   retained LOW/QUARANTINE không bị đổi thành bài viết thành công. Raw cùng
   hash + URL + trạng thái review đã thử không được parse lại.
6. Text cứu thêm tạo delta thứ hai. Chỉ freeze/index-input mới; embedding vẫn
   cần chạy GPU riêng. Không đổi active candidate, BM25, vector cache hoặc
   output của luồng submit đang chạy.
7. Xuất toàn bộ ID còn thiếu cho nhóm crawl theo URL/nguyên nhân. Nhóm HTTP lỗi
   hoặc không có trong archive không thể biến thành bài viết chỉ bằng xử lý CPU.

Checkpoint SQL nén và text/chunk parts ở
`VietMedBridge/data_staging/btc-maximize-71be0e12-v1/`. Runtime mới khôi phục một
SQL snapshot; source, task, raw hash, chunk shard và freeze đã commit được bỏ
qua. Raw TAR chưa có index cần đọc header lần đầu; cursor lưu mỗi 1.024 header
hoặc 32 text mới. Đoạn chưa commit có thể làm lại khi ngắt. Không tải cả archive
về đĩa, không giải nén raw thành hàng nghìn file. Scratch chỉ ở máy Colab.

`delta_candidates.json` chứa hai delta candidate (nếu có tài liệu đạt gate).
`remaining_for_crawl.csv` gồm mọi ID chưa có tài liệu hợp lệ, không mất alias.
`raw_recovery_outcomes.csv` là kết quả từng raw hash/URL/review state đã thử.
`status.json` tách rõ text ready, chunked, review, needs raw và needs crawl.

Số 97.758 là ID BTC có metadata raw thành công, không phải số bài chắc chắn cứu
được. Số lần held trong nhiều bản archive cũng không phải số URL khác nhau.
Không hứa tăng coverage hoặc điểm trước khi chạy thực tế và đánh giá retrieval.

## Kiểm tra lúc raw recovery còn chạy

`status.json` chỉ được ghi tại ranh giới stage, có thể cũ hơn SQL checkpoint.
Không dùng số file trong `recovered/` làm số URL đã phục hồi.
Trong **phiên Colab CPU thứ hai**, dán toàn bộ
`scripts/check_maximize_progress_cell.py` vào một cell rồi chạy. Phiên writer
hiện tại tiếp tục bình thường. Cell không cần cài package, tokenizer hoặc model.

Cell đọc đúng một SQL snapshot bất biến theo `checkpoint.json`, kiểm hash và
binding của snapshot đó trên scratch Colab, truy vấn read-only theo ID BTC,
rồi lưu báo cáo nhỏ `coverage_progress.json` trên Drive. Không mở TAR, Parquet,
corpus hoặc vector; không ghi vào SQL/status/candidate của writer. Thời điểm
`checkpoint_updated_at` là mốc số liệu, có thể lùi hơn luồng live đang chạy.
`TEXT_READY` chưa qua chunk/freeze; báo cáo này không khẳng định đã embedding.

## Chuyển bước khi Drive chặn đọc raw hoặc cần chốt sớm

Khi phiên raw recovery đã dừng, mở
`notebooks/02_colab_finalize_recovered_text.ipynb` trên **Colab CPU → Run all**.
Trong runtime maximize hiện có cũng có thể dán cell
`scripts/finalize_ready_text_cell.py` để giữ tokenizer/package cache.
Không chạy cùng một writer khác và không chạy lại cell raw recovery.

Cell chốt membership của phần raw đã có text thành một phase
`raw-ready-<hash>` riêng, chunk/freeze/index-input theo policy cũ và thêm vào
`delta_candidates.json`. Không mở raw TAR, không đổi active submission, không
tạo receipt `raw-pass` giả. Source/cursor raw cũ giữ nguyên. Phần chưa đủ chất
lượng hoặc chưa có text vẫn hiện trong `remaining_for_crawl.csv` và status.

Nếu ngắt giữa batch, cùng cell tiếp tục batch chưa freeze. Batch đã freeze bất
biến; raw cứu thêm sau này tạo batch mới, không thêm vào candidate cũ. Các URL
đã chunk được ghi nhận để raw resume không trích xuất lại hoặc xuất trùng batch.
Đầu ra này là input GPU đã freeze, chưa phải corpus vectors hoặc submission.
