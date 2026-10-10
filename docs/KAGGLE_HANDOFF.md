# Notebook 04: Colab CPU → Kaggle T4 x2 → Colab

Notebook hỗ trợ: [`04_kaggle_embedding_worker.ipynb`](../notebooks/04_kaggle_embedding_worker.ipynb).
Cùng một file nhận biết môi trường, không gắn username. Luồng 00–05 chính giữ nguyên.

## 1. Hoàn tất CPU trước

Phiên notebook **04_colab_retrieval_baseline** hiện tại tiếp tục chạy với
`PREPARE_ONLY=True`, `TEAM_WORKER_ID=None`, trên CPU. Chờ thông báo
`CPU_PREPARATION_COMPLETE`. Không mở thêm coordinator đang ghi cùng data.
Gói Kaggle chỉ được xuất sau checkpoint này, với candidate/input manifest đã khóa.
Không chạy lại crawl, chunk hoặc freeze để chuyển nền tảng.

## 2. Xuất input từ Drive trên Colab CPU

Mở notebook hỗ trợ trên Colab CPU. Trong cell 1:

```python
DATA_ROOT = Path("/content/drive/MyDrive/VietMedBridge/data")
RUNTIME_PLATFORM = "auto"
KAGGLE_NOTEBOOK_OUTPUT = ""
KAGGLE_ACCOUNTS = 2
KAGGLE_ACCOUNT_ID = 0
GPUS_PER_ACCOUNT = 2
SESSION_HOURS = 10.0
```

Notebook nhận diện runtime Colab/Kaggle bằng shell và biến môi trường, không
dựa vào sự tồn tại của `/kaggle/input`. Nếu dùng runtime đặc biệt hoặc bản cũ
báo `Add Input` ngay trên Colab, đặt `RUNTIME_PLATFORM="colab"` trong bản mới;
trên Kaggle giữ `"auto"` hoặc chọn `"kaggle"`. Dòng `Platform:` được in trước
khi notebook tìm dataset hoặc mount Drive.

Nếu tài khoản nhận thư mục chia sẻ qua shortcut thì `DATA_ROOT` phải trỏ đúng
thư mục data đó, cùng nơi đã chạy notebook 04. Đây là đường dẫn Drive; Kaggle
không dùng nó.

Trong tài khoản Kaggle **sẽ sở hữu dataset**, vào Settings → API → Legacy API
Credentials → Create Legacy API Key. Mở `kaggle.json`, sao chép toàn bộ nội dung
vào Colab Secrets với tên **KAGGLE_JSON**, bật Notebook access. Không đưa key
vào source code, Git hoặc gửi qua chat. Đây là cách xác thực legacy Kaggle
được [tài liệu Kagglehub](https://github.com/Kaggle/kagglehub#authenticate) hỗ trợ.

Run all. Notebook tự lấy username từ Secret, tạo private dataset tên riêng cho
job, kiểm tra lại `job.json` rồi in link dataset. [Kagglehub tạo dataset mới với
`is_private=True`](https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/datasets_helpers.py).
Gói chỉ chứa input embedding đã dedup, metadata và cache vector hợp lệ; không
upload HTML, toàn bộ parent/section corpus hoặc SQLite BM25. Gói được dựng ở
ổ tạm Colab, chỉ lưu pointer nhỏ trên Drive.

Một người xuất **một dataset chung**, sau đó chủ dataset cấp quyền đọc trên
Kaggle cho thành viên chạy tài khoản thứ hai. Cả hai Add Input đúng dataset đó;
không xuất riêng hai gói ở hai môi trường khác nhau. Username của người xuất
vẫn lấy tự động từ Secret, không cần dùng chung API key.

Gói mới khóa `KAGGLE_ACCOUNTS` và `GPUS_PER_ACCOUNT`; account ID được chọn riêng
ở mỗi phiên. Gói của bản notebook một tài khoản trước đây cần xuất bằng bản mới
để khóa cách chia hai tài khoản. Bước này dùng lại input đã chuẩn bị, không
chạy lại crawl/chunk/freeze. Nếu chỉ chạy một tài khoản, đặt `KAGGLE_ACCOUNTS=1`
ngay lúc xuất và ở phiên Kaggle.

## 3. Chạy hai tài khoản Kaggle song song

1. Tạo Kaggle notebook, giữ Private và import file notebook hỗ trợ.
2. **Add Input** private dataset vừa xuất. Chỉ thêm một input job VietMedBridge.
3. Bật **Internet**, chọn accelerator **GPU T4 x2** ở cả hai tài khoản.
   Giữ cùng `KAGGLE_ACCOUNTS=2`, `GPUS_PER_ACCOUNT=2`.
   Tài khoản thứ nhất đặt **`KAGGLE_ACCOUNT_ID=0`**;
   tài khoản thứ hai đặt **`KAGGLE_ACCOUNT_ID=1`**, ở đầu cell 1.
4. Chọn **Save Version → Save & Run All** để chạy phiên nền từ đầu.
5. Khi version hoàn tất, kiểm tra có Outputs `vmb_checkpoints/result-manifest.json`.

Hai bên có thể bắt đầu cùng lúc, không cần tài khoản 0 chạy trước. Tài khoản 0
chạy worker toàn nhóm 0–1; tài khoản 1 chạy worker 2–3. GPU vật lý trong mỗi
máy vẫn là 0–1. Part thứ `i` thuộc worker `i % 4`, nên toàn bộ parts được chia
đủ một lần và không giao trùng. Số worker/GPU được kiểm trước khi copy input
và nạp model; chọn sai accelerator sẽ dừng, không tự giảm thành một GPU.

Mỗi GPU nạp BGE rồi Qwen embedding theo thứ tự. Hai tài khoản xử lý các phần
khác nhau của cả hai model. Model/revision/precision/pooling giữ đúng cấu hình full system. Phiên
tự dừng nhận part mới trước 10 giờ tính từ cell đầu, hoặc khi output gần 18 GB;
part đang chạy được hoàn tất. Đây là khoảng dự phòng cho giới hạn thời gian và
20 GB output lưu của [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks),
không phải cơ chế kéo dài quota.

`CHECKPOINTED_PARTIAL` nghĩa là các part trong receipts đã hoàn tất, **không**
nghĩa toàn bộ corpus đã embedding. `WORKER_FAILED_WITH_CHECKPOINTS` cần xem
traceback; receipts đã hoàn tất vẫn nhập được. Số part BGE/Qwen có thể khác nhau.
Query translation, document embeddings, search, reranker và submission vẫn do
coordinator 04 thực hiện sau khi đủ corpus vectors.

File dưới `/kaggle/working` chỉ lấy về được sau khi Outputs của version đã lưu.
Không chờ runtime bị quota ngắt cứng: nếu version không lưu Outputs thì không
bảo đảm lấy được checkpoint phiên đó. Muốn chạy thêm phiên Kaggle, Add Input
Outputs đã lưu của phiên trước **của chính account ID đó** để resume; notebook
kiểm tra và mang chúng sang Outputs phiên mới. Outputs của account ID khác
được bỏ qua, không copy thành một bản thừa. Giữ cùng cách chia khi resume.
Giới hạn 18 GB tính cả các checkpoint mang theo, riêng cho từng phiên Kaggle.

## 4. Nhập kết quả về Drive

Sau khi Kaggle đã dừng, mở notebook hỗ trợ trên Colab CPU và điền:

```python
KAGGLE_NOTEBOOK_OUTPUT = "owner-a/notebook-a, owner-b/notebook-b"
```

Có thể dùng `username/notebook-slug/version` để lấy đúng version; một handle
đơn vẫn dùng được. Chọn một version mới nhất cho mỗi account ID. Secret của
người tải phải có quyền đọc cả hai Outputs (chủ notebook cần cấp quyền trên
Kaggle). Nếu không dùng chung quyền đọc, mỗi thành viên nhập Outputs của mình
bằng Secret riêng, **lần lượt** vào cùng DATA_ROOT. Run all: notebook dùng
[`kaggle kernels output`](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
tải vào ổ tạm Colab rồi kiểm toàn bộ receipts trước khi ghi vào Drive:
candidate, input manifest, model/runtime/source code, checksum file, shape/L2,
source text hash và vector row tương ứng. Nếu không khớp thì dừng trước khi
nhập. Ghi completion markers sau dữ liệu; nhập lại cùng kết quả là idempotent.

Mỗi lượt nhập in `coverage.bge` và `coverage.qwen`: số part đã có/tổng part và
số input text đã phủ, tính gộp trên Drive. `corpus_embeddings_complete=true`
chỉ khi cả hai nhánh có đủ part, không chỉ dựa vào hai notebook cùng báo chạy
xong. Khi phiên dừng do thời gian hoặc output budget, coverage có thể chưa đủ;
giữ checkpoint và chạy tiếp phần thiếu. Hai GPU mỗi tài khoản không đảm bảo
xong toàn bộ corpus trong một phiên. `corpus_complete=false` và scope vẫn
phân biệt bước embedding với coordinator/inference/submission còn lại.

Kết quả hợp lệ nằm thẳng trong `model_cache` và view `retrieval/full_resources`
của hệ thống, không cần convert hay chạy lại embedding đã nhập. Import lưu
runtime lock; bootstrap notebook 04/05 mới tự cài đúng phiên bản producer.
Không đổi encoder source code để hỗ trợ Kaggle.

## 5. Sau đó mới chia Colab workers

Dừng mọi worker Kaggle trước khi đổi cách chia phần. Mở notebook 04 chính bản
mới trên từng Colab GPU, cùng Drive, `PREPARE_ONLY=False`, cùng `TEAM_SIZE`,
worker IDs khác nhau. Các part đã nhập sẽ được xác minh và bỏ qua dù trước đó
Kaggle chia hai worker. Không có hai phiên đồng thời dùng cùng worker ID.
Khi workers hoàn tất, một coordinator `TEAM_WORKER_ID=None` tổng hợp và chạy
các bước còn lại.

Kiểm thử local dùng encoder giả để kiểm transfer/checkpoint/source bindings;
không chứng minh throughput, tương thích driver hoặc kết quả retrieval trên
Kaggle GPU. Phiên thật kiểm CUDA và producer runtime trước khi encode.
