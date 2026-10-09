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
KAGGLE_NOTEBOOK_OUTPUT = ""
SESSION_HOURS = 10.0
```

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

Sếp hoặc bạn Sếp đều có thể chạy bước này bằng Secret riêng và quyền đọc Drive.
Nếu người chạy Kaggle khác chủ dataset thì chủ dataset phải cấp quyền đọc trên
Kaggle trước khi Add Input. Không cần dùng chung API key.

## 3. Chạy một phiên Kaggle

1. Tạo Kaggle notebook, giữ Private và import file notebook hỗ trợ.
2. **Add Input** private dataset vừa xuất. Chỉ thêm một input job VietMedBridge.
3. Bật **Internet**, chọn accelerator **GPU T4 x2**. Không chỉnh worker ID.
4. Chọn **Save Version → Save & Run All** để chạy phiên nền từ đầu.
5. Khi version hoàn tất, kiểm tra có Outputs `vmb_checkpoints/result-manifest.json`.

Hai process chia input parts theo GPU, mỗi GPU nạp BGE rồi Qwen embedding theo
thứ tự. Model/revision/precision/pooling giữ đúng cấu hình full system. Phiên
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
Outputs đã lưu của phiên trước; notebook kiểm tra và mang chúng sang Outputs
phiên mới. Giới hạn 18 GB tính cả các checkpoint mang theo.

## 4. Nhập kết quả về Drive

Sau khi Kaggle đã dừng, mở notebook hỗ trợ trên Colab CPU và điền:

```python
KAGGLE_NOTEBOOK_OUTPUT = "username/notebook-slug"
```

Có thể dùng `username/notebook-slug/version` để lấy đúng version. Secret của
người tải phải có quyền đọc Outputs đó. Run all: notebook dùng
[`kaggle kernels output`](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
tải vào ổ tạm Colab rồi kiểm toàn bộ receipts trước khi ghi vào Drive:
candidate, input manifest, model/runtime/source code, checksum file, shape/L2,
source text hash và vector row tương ứng. Nếu không khớp thì dừng trước khi
nhập. Ghi completion markers sau dữ liệu; nhập lại cùng kết quả là idempotent.

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
