# Drive và Hugging Face cho VietMedBridge

Repo đích: `platypus123/biobridgeir-checkpoints`, loại **Dataset**, **Private**.
Drive tiếp tục chứa bản nguồn hiện tại, cấu hình, nhãn duyệt và báo cáo nhỏ.
Hugging Face giữ snapshot của active candidate cùng các checkpoint vector.
Colab tải phần cần thiết về ổ tạm để xử lý; không dùng `hf://` như một đường
dẫn `Path` thay cho Drive. Không cần crawl/chunk/embedding lại để đổi nơi lưu.

## Chạy ngay trên Drive, chưa chờ HF

Mở `notebooks/04_colab_retrieval_baseline.ipynb` trên **Colab T4 GPU** và Run all.
Mặc định `RUN_CACHED_BASELINE_FROM_DRIVE=True`, `USE_HF_STORAGE=False`.
`DRIVE_DATA_ROOT` là thư mục data cũ; tự nhận một shortcut VietMedBridge nếu
đường dẫn MyDrive mặc định không có active candidate.
`MIN_EXPECTED_INPUT_RECORDS=700244` chặn chạy nhầm candidate 100k cũ. Đây là
số IDs đã nhập, gồm cả thất bại; không đồng nghĩa có đủ 700.244 bài hợp lệ.

Lượt này dùng active candidate và BGE đã hoàn tất, khôi phục catalog BM25 đã
lưu về ổ local, encode 1.200 query, tìm kiếm rồi xuất ZIP. Không cần HF_TOKEN
hoặc upload HF. Drive chỉ được đọc; checkpoint query/search và ZIP mới nằm ở
`/content/vmb_results`. Vectors được cache trên ổ local khi truy hồi, không
encode lại corpus. Thiếu BGE/catalog hoặc sai runtime thì dừng; không tự
dựng lại catalog, ghép corpus hoặc chuyển sang full inference.

Giữ runtime đến khi tải submission ZIP. Cell cuối cho tải bản checkpoint
inference riêng, không chứa corpus/catalog/model weights. Runtime bị thu hồi
sẽ mất checkpoint local nếu chưa tải bản này; dữ liệu gốc vẫn ở Drive.
Đây là lượt baseline để lấy điểm trước, chưa chạy Qwen/reranker/fine-tune.
Muốn trở lại full workflow hoặc dùng HF, tắt `RUN_CACHED_BASELINE_FROM_DRIVE`.

## 1. Chuyển dữ liệu đang có — Colab CPU

Mở [`tools/drive_hf_storage.ipynb`](../notebooks/tools/drive_hf_storage.ipynb).

1. Tạo hoặc kiểm tra dataset repo trên HF, đặt **Private**. Notebook tạo private
   repo nếu chưa có, nhưng không tự đổi visibility của repo đang Public.
2. Tạo HF token có quyền ghi đúng repo. Thêm vào **Colab Secrets → HF_TOKEN**,
   bật **Notebook access**. Token không được lưu trong notebook, plan hay locator.
3. Cell 1: giữ đúng `DATA_ROOT` của thư mục data hiện tại. Repo đích đã điền sẵn.
   `MAX_SNAPSHOT_GB=90` là ngân sách của lượt chuyển, không phải quota đã đo.
4. Giữ `UPLOAD_PROFILE="bge_submit"` để lấy submission trước: chỉ chuyển metadata,
   catalog BM25 và BGE. Qwen, source shards và input text Parquets giữ ở Drive.
   Run all trên **CPU**. Xem thống kê số file, GB, documents và coverage đã chọn.
   Vượt ngân sách thì dừng trước upload; kiểm tra tài khoản HF trước khi đổi số.
5. Chờ `HF_STORAGE_VERIFIED`, tải `hf-storage-locator.json` nhỏ về máy.

Theo [giới hạn HF hiện công bố](https://huggingface.co/docs/hub/storage-limits),
tài khoản Free có 100 GB private, PRO có 1 TB private đi kèm. Quota tính cả các
repo/snapshot đã lưu; mức 90 GB trong notebook không xác nhận tài khoản còn đủ
dung lượng. Dữ liệu vài triệu URL có thể vượt mức này; cần kiểm kê thực tế.

Sau khi có submission, đặt `UPLOAD_PROFILE="all"` để backup phần còn lại. HF dùng
lại các objects BGE/catalog cùng checksum. Snapshot `all` gồm **active candidate**: metadata chính thức, source shards
đã xử lý, index inputs, catalog BM25 đã hoàn tất và các vector BGE/Qwen được
mapping của candidate đó tham chiếu. Không copy cả thư mục data một cách mù:
incoming TAR, các build cũ, cache model weights, file tạm và run lỗi không nằm
trong snapshot. `INCLUDE_SOURCE_SHARDS=False` chỉ phù hợp khi muốn chuyển bộ
retrieval nhỏ hơn trước; mặc định True giữ source shards cho các bước sau.

Upload theo batch 20 immutable objects. Mỗi object có địa chỉ SHA-256; checksum
server được kiểm sau commit. Một snapshot chỉ được công bố sau khi đủ objects.
Ngắt mạng/phiên: mở lại cùng notebook, DATA_ROOT và repo; objects đã commit
được kiểm và bỏ qua kể cả không còn progress file local. Snapshot mới dùng lại
objects cũ cùng checksum. Chỉ một coordinator chuyển và công bố tại một thời điểm.

Drive chỉ được đọc; progress và locator ghi ở `/content/vmb_storage`. File nguồn
không bị xóa. Vì thế **upload xong không tự giải phóng quota Drive**. Cần giữ
bản nguồn cho đến khi đã khôi phục/đối chiếu bản HF; việc dọn Drive là bước riêng.
Không đổi float32 sang float16 hay thay nội dung/chunk policy trong bước chuyển.

## 2. Submit bằng BGE đã đủ — Colab GPU

Mở bản mới [`04_colab_retrieval_baseline.ipynb`](../notebooks/04_colab_retrieval_baseline.ipynb).
Thêm HF_TOKEN có quyền **đọc** repo; chọn **T4**. Đầu cell 1:

```python
RUN_CACHED_BASELINE_FROM_DRIVE = False
USE_HF_STORAGE = True
HF_STORAGE_REPO = "platypus123/biobridgeir-checkpoints"
HF_STORAGE_PROFILE = "bge_submit"
HF_STORAGE_REVISION = None
```

Có thể thay None bằng full revision trong locator để khóa đúng snapshot. Nếu
để None, notebook lấy latest một lần rồi dùng SHA commit bất biến cho cả lượt.
Run all. Profile này tải metadata, catalog và BGE; không tải source shards,
input text Parquets hoặc Qwen. Source text/chunks đã nằm trong SQLite catalog.
Metadata, snapshot, runtime/model, checksum và catalog policy được kiểm. Thiếu
BGE/catalog phù hợp thì dừng trước inference, không tự encode corpus thay thế.
Catalog tương thích được kiểm trước tải các vector lớn.

Cell 2 tự chọn BGE + BM25, một coordinator, không chạy chuẩn bị CPU hoặc chờ
Qwen. DATA_ROOT là `/content/vmb_data`; không mount hoặc ghi Google Drive.
Checkpoint query/search và ZIP nằm ở ổ local của Colab. Khi
`READY_FOR_MANUAL_UPLOAD` và 1.200 query, tải ZIP và nộp BTC.

**Phân biệt nơi lưu:** snapshot dữ liệu/vector đã bền vững trên HF. Checkpoint
inference của lượt submit này mới nằm trên ổ tạm, chưa tự đồng bộ lên HF;
runtime bị thu hồi sẽ mất tiến độ query/search local, nhưng corpus embeddings
đã lưu không cần tính lại. Giữ runtime đến lúc tải ZIP. Đừng hiểu
HF_STORAGE_VERIFIED là đã hoàn tất retrieval hay đã có điểm BTC.

## 3. Các bước sau

Sau khi đã upload snapshot `all`, profile `all` khôi phục toàn bộ snapshot khi đủ ổ local để dùng source shards,
index inputs và Qwen checkpoints. Phần Qwen còn thiếu vẫn còn thiếu; snapshot
không nâng coverage hoặc giả đã chạy full plan. Bản `bge_submit` chỉ xuất
baseline trên candidate hiện tại. Fine-tune, reranker và full query understanding
vẫn là các bước riêng như plan chính.

Đợt dữ liệu mới có thể tạo snapshot mới, không thay object cũ. Dataset repo
có Git history; xóa file ở bản mới không mặc nhiên giải phóng bytes trong lịch
sử. Không squash/xóa history tự động trong utility này.

Kiểm thử tự động dùng Hub API/GPU doubles: migration bị ngắt và resume, chặn
repo Public/ngân sách thiếu/checksum sai, restore theo profile và xuất ZIP đủ
1.200 query với source chunks hợp lệ từ SQLite. Chưa xác nhận upload thực tế,
quota của tài khoản hoặc hiệu năng GPU chỉ bằng các kiểm thử đó.
