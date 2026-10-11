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

Runtime của máy xuất gói CPU không phải runtime chạy embedding GPU. Gói mới
dùng `configs/kaggle_runtime.json` (`torch==2.8.0+cu126`, transformers 4.57.1,
bitsandbytes 0.49.2), hoặc giữ runtime CUDA của vectors đã nhập trước đó. Không
lấy phiên bản torch CPU đang cài trên Colab để khóa GPU worker. PyTorch cung
cấp [wheel CUDA 12.6 cho bản 2.8.0](https://pytorch.org/get-started/previous-versions/).

Nếu dataset cũ đã khóa `torch=2.11.0+cpu` và **chưa có seed vectors**, dùng
notebook hỗ trợ mới trên Kaggle với **chính dataset đó**. Bootstrap kiểm seal,
model source và mọi checksum, tạo job CUDA mới ở ổ tạm Kaggle với liên kết
đến job gốc. Candidate, input files, số tài khoản/GPU và commit của encoder
giữ nguyên; không cần chạy lại 02/03/CPU catalog hoặc upload dataset lại.
Hai tài khoản nhận cùng job CUDA mới, nên Outputs vẫn nhập được cùng nhau.
Job có seed vectors không được tự đổi producer runtime. Dataset gốc không bị sửa.

Bootstrap cài runtime khóa qua pip rồi kiểm CUDA bằng **tiến trình Python mới**.
Launcher và GPU workers cũng chạy trong các tiến trình mới, nên torch CPU đã
import trong kernel notebook không chặn việc sửa runtime hoặc bị hot-reload.
Giữ phiên interactive cũng có thể chạy lại cell bootstrap rồi cell GPU sau
khi thay notebook. Muốn đóng tab mà vẫn chạy thì dùng **Save Version → Save &
Run All**. Bootstrap phải in torch CUDA, `CUDA available: True` và `GPUs: 2`;
mỗi GPU còn thực hiện phép tính nhỏ để kiểm CUDA kernel trước khi nạp weights.

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

### Xuất baseline trước khi Qwen hoàn tất

Nếu BGE đã đủ mọi input part của active candidate, có thể nộp một baseline
BGE + BM25 ngay. Dùng bản mới của **04_colab_retrieval_baseline.ipynb** trên
**Colab GPU**, giữ nguyên DATA_ROOT đã nhập cả hai Kaggle Outputs. Trong cell 2:

```python
PREPARE_ONLY = False
INFERENCE_MODE = "bge_bm25_cached"
TEAM_SIZE = 1
TEAM_WORKER_ID = None
MAX_NEW_QUERIES = None
```

Run all. Nhánh này đọc đúng native BGE maps đã nhập, kiểm input/model/runtime,
dùng lại catalog BM25 của bước CPU, encode 1.200 query, tìm kiếm trên toàn bộ
candidate rồi chọn source chunks và xuất ZIP. Không chạy lại 00–03, không
encode corpus và không đợi Qwen. Thiếu BGE thì báo WAITING, không tự tính bù.
Query embeddings, dense search và từng query có checkpoint riêng. Cần thêm
thời gian đọc vector/search/ranking; BGE đủ không đồng nghĩa ZIP đã có sẵn.

Khi `READY_FOR_MANUAL_UPLOAD` và `query_count=1200`, cell cuối tải ZIP để Sếp
nộp BTC. Báo cáo ghi rõ baseline, không giả đã chạy reranker/query LLM/fine-tune
và không dự đoán điểm. Các Qwen checkpoint giữ nguyên để tiếp tục full system
bằng `INFERENCE_MODE="full"` sau này. Không trộn Qwen mới đủ một phần vào
truy hồi toàn corpus.

File dưới `/kaggle/working` chỉ lấy về được sau khi Outputs của version đã lưu.
Không chờ runtime bị quota ngắt cứng: nếu version không lưu Outputs thì không
bảo đảm lấy được checkpoint phiên đó. Muốn chạy thêm phiên Kaggle, Add Input
Outputs đã lưu của phiên trước **của chính account ID đó** để resume; notebook
kiểm tra và mang chúng sang Outputs phiên mới. Outputs của account ID khác
được bỏ qua, không copy thành một bản thừa. Giữ cùng cách chia khi resume.
Giới hạn 18 GB tính cả các checkpoint mang theo, riêng cho từng phiên Kaggle.

### Tiếp tục từ Drive khi Outputs tích lũy lớn

Sau khi nhập checkpoint, bản notebook mới tự xuất một **continuation dataset**
nếu `corpus_embeddings_complete=false`. Dataset giữ nguyên job, model/runtime,
candidate và input checksums; thêm `continuation-plan.json` khóa commit của worker
đã hỗ trợ chạy tiếp. Commit mới chỉ thay đổi phần điều phối; cả hai file encoder
phải có đúng hash của job cũ trước khi được dùng.

Plan được lập từ map checkpoints đã kiểm trên Drive, gồm phần đã hoàn tất và
phần được giao trong lượt tới. Mỗi tài khoản được giao tối đa khoảng 16 GB vector
và metadata ước tính, để chừa khoảng trống trong ngân sách Outputs 18 GB.
`minimum_rounds_by_capacity` chỉ tính dung lượng; tốc độ GPU và thời hạn phiên có
thể làm số lượt thực tế lớn hơn. Không hứa đủ toàn bộ corpus trong một phiên.

Trên cả hai Kaggle account, dùng **notebook hỗ trợ bản mới**, bỏ input dataset cũ,
Add Input **chỉ continuation dataset mới nhất**, giữ account ID 0/1 và T4 x2.
Không gắn Outputs cũ trong chế độ này: vector cũ đã nằm trên Drive. Worker bỏ qua
những part đã hoàn tất, không nạp BGE khi nhánh đó đã đủ; Outputs chỉ chứa part
mới. Nhập chúng về cùng Drive rồi lập plan tiếp theo nếu còn thiếu.

Log phân biệt `ROUND_ASSIGNMENT_COMPLETE` (xong phần được giao trong lượt này),
`ACCOUNT_ASSIGNMENT_COMPLETE` (xong toàn bộ phần của account) và
`CHECKPOINTED_PARTIAL` (lượt còn dang dở). `account_coverage` in phần còn lại của
lượt và của toàn account; `worker_status` ghi `SESSION_TIME_LIMIT`,
`OUTPUT_SIZE_LIMIT` hoặc `WORKER_ERROR` khi có. Chỉ coverage cộng gộp trên Drive
của cả BGE và Qwen mới chốt đủ embedding corpus.

Ví dụ đã xác nhận trong log ngày 11/10: BGE 937/937, Qwen 151/937. Còn
3.218.991 văn bản Qwen, khoảng 52,7 GB vector float32, chưa tính metadata.
Hai Outputs có ngân sách 18 GB mỗi phiên không thể chứa hết phần này cùng lúc;
cần nhiều lượt có checkpoint. Không crawl/chunk hoặc tính lại BGE đã hoàn tất.

## 4. Nhập kết quả về Drive

Sau khi Kaggle đã dừng, mở notebook hỗ trợ trên Colab CPU và điền:

```python
KAGGLE_NOTEBOOK_OUTPUT = "owner-a/notebook-a, owner-b/notebook-b"
```

Có thể dùng `username/notebook-slug/version` để lấy đúng version; một handle
đơn lấy phiên mới nhất. `version` là số Version hiển thị trên Kaggle (ví dụ
`/2`), không phải `scriptVersionId` trong URL. Chọn một version chứa checkpoint
cho mỗi account ID; nếu vừa Save một bản mới không có Outputs, phải chọn lại
đúng bản đã chạy. Secret của
người tải phải có quyền đọc cả hai Outputs (chủ notebook cần cấp quyền trên
Kaggle). Nếu không dùng chung quyền đọc, mỗi thành viên nhập Outputs của mình
bằng Secret riêng, **lần lượt** vào cùng DATA_ROOT. Run all: notebook dùng
Kaggle API kiểm trạng thái và tải riêng `result-manifest.json` của **tất cả**
notebook trước. Log ghi rõ handle, trạng thái, số manifest, số saved parts và
account ID. Nếu thiếu, trùng manifest, khác job hoặc trùng account ID thì dừng
trước khi tải vector. Chỉ file được API trả trong lượt hiện tại mới được tính;
manifest cũ ở thư mục tạm không gây báo trùng hoặc che lỗi thiếu Outputs.
Sau preflight hợp lệ, tải vào ổ tạm Colab (tận dụng file đã tải nếu manifest
khớp) rồi kiểm toàn bộ receipts trước khi ghi vào Drive:
candidate, input manifest, model/runtime/source code, checksum file, shape/L2,
source text hash và vector row tương ứng. Nếu không khớp thì dừng trước khi
nhập. Ghi completion markers sau dữ liệu; nhập lại cùng kết quả là idempotent.
Outputs có thể có nhiều trang metadata; một trang không có file khớp regex
không có nghĩa là không có manifest. Trình nhập duyệt tới khi hết
`next_page_token`, kể cả Kaggle 2.0.2 có public API chỉ tải trang đầu. Với
phiên bản này, các trang sau được đọc qua SDK và file được tải streaming/atomic.
Không cần cài lại môi trường hay tính lại embedding để sửa phân trang.
Nếu Outputs đổi giữa preflight và tải vector, dừng trước khi nhập Drive và yêu
cầu cố định số version. Bước nhập không chạy model hoặc tính lại embedding.

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
