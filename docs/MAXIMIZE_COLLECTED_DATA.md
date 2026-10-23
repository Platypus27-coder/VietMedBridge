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

## Xuất các recovery batch đã freeze sang Kaggle

Mở `notebooks/04_colab_export_recovered_data_kaggle.ipynb` trên **Colab CPU →
Run all**, giữ `KAGGLE_ACCOUNTS=2`, `KAGGLE_NOTEBOOK_OUTPUT=""`. Colab Secret
`KAGGLE_JSON` của tài khoản xuất cần được bật Notebook access.

Notebook nhận tất cả batch trong `delta_candidates.json`, kiểm binding của
candidate và manifest input, chốt bản sao manifest bất biến theo checksum.
Chỉ input text chưa chọn xong mới được đọc; vector cache được tra bằng metadata
SQL. Chỉ text còn thiếu của từng model được đưa vào dataset. Không đọc TAR hoặc
chạy lại chunk/freeze, không nạp weights trên CPU, không đổi active submission.
Số input rows cộng các batch chưa trừ trùng; cùng text hash thuộc cùng GPU và
được dùng lại trong worker. Input manifest và vector namespace/model giữ nguyên.

Khi hiện `NEW_DATASET_READY`, hai tài khoản Add Input cùng dataset riêng tư đó,
dùng `notebooks/kaggle/04_new_data_account_0.ipynb` và
`notebooks/kaggle/04_new_data_account_1.ipynb`, Internet ON, T4 x2, Save & Run All.
Outputs được nhập bằng notebook Colab xuất ở trên. Điền hai handle notebook vào
`KAGGLE_NOTEBOOK_OUTPUT`, Run all; importer khóa đúng recovery batch từ job cũ
kể cả sau này campaign có batch mới. Kết quả vào `data/model_cache` và mapping
trong `embedding_batches/<hash>/embeddings_v1`, tách biệt batch 10 nguồn.

Đây là embedding các recovery delta đã freeze; không chứng minh đã có nội dung
cho toàn bộ URL BTC hoặc đủ Qwen/BGE cho mọi candidate khác.

## Luồng đầy đủ — toàn bộ dữ liệu hợp lệ đã xử lý

Dùng **`notebooks/04_colab_export_full_data_kaggle.ipynb` trên Colab CPU →
Run all**. Giữ `KAGGLE_ACCOUNTS=2`, Outputs trống để xuất. Đây là notebook
chính cho yêu cầu full data; hai notebook xuất từng batch ở trên chỉ là công cụ
theo phạm vi riêng. Người dùng tự chạy notebook; không chạy hai CPU writer cùng
scope một lúc. Secret Kaggle và hai GPU notebook dùng như trên.

Nguồn gồm baseline đã freeze, đủ 10 nguồn intake đã freeze, và mọi recovery
candidate trong `delta_candidates.json`. Giữ cả legacy content đã được yêu cầu
tận dụng, không lọc bỏ tài liệu cũ hoặc cộng số document rows thành số ID BTC.
Manifest nguồn và input gốc được dùng lại; không crawl/extract/chunk/freeze hay
xây lại catalog. BGE baseline COMPLETE được kế thừa qua một manifest đã seal,
không mở 1.797 map/input payload chỉ để kiểm kê. Qwen đã có trong content cache
và các model/view COMPLETE khác cũng được dùng lại đúng producer/model identity.

Trước khi xuất GPU, luồng đối chiếu reference của từng ID từ SQL collection và
SQL maximize với candidate frozen, alias SHA, snapshot và input manifests.
Phải có `FULL_SCOPE_READY` với **ít nhất 802.028 ID BTC**. Nếu thiếu nguồn,
binding sai hoặc số ID thấp hơn yêu cầu thì dừng, không tự đổi sang batch 3.080.
Membership lưu theo từng part trên Drive, rồi chốt report nhỏ. Chạy tiếp bỏ qua
membership đã commit; scope hoàn tất không mở lại SQL/corpus. Nguồn SQL là hai
checkpoint metadata riêng; không mở lại TAR hoặc vector payload.

Input part chưa chọn được đọc một lần vào scratch Colab, kiểm hash rồi lookup
text cache. Draft/task/journal được lưu sau từng part, runtime mới không duyệt
lại part đã chọn. Hash trùng giữa các nguồn thuộc cùng GPU nên dùng lại vector.
Sau khi chọn toàn bộ, hàng đợi thiếu được giữ trên Drive. Dataset mỗi lượt chứa
phần được giao trong trần 16 GB vector mới/acc; `scheduled_tasks`,
`unscheduled_tasks`, `all_remaining_tasks_scheduled` phân biệt rõ một lượt và
toàn bộ scope. Nhập Outputs rồi chạy tiếp để xuất phần còn thiếu, không tạo lại
hàng đợi. Không hứa toàn bộ Qwen xong trong một lượt hay trong một đêm.

Scope ở `data_staging/btc-full-embeddings-71be0e12-v1/<hash>/scope.json`,
mapping và tiến độ trong `embeddings_v1/`. `full_embedding_status.json` ghi
coverage BGE/Qwen của toàn scope; chỉ khi mọi nguồn đủ cả hai model mới có
`FULL_DATA_EMBEDDINGS_COMPLETE`. Active candidate/luồng submit không bị sửa.
