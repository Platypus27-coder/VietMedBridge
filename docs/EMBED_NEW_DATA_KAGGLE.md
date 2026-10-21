# Embedding batch mới trên Kaggle

Batch đầu vào là `NEW_DATA_INPUTS_READY` của `03_colab_freeze_new_data.ipynb`:
10 nguồn, 533.926 documents, 5.128.103 children, 1.225 input parts. Tổng
4.999.708 input rows là tổng distinct của từng build; có thể trùng giữa các build.
Đây không phải khẳng định đã đủ toàn bộ URL BTC hoặc corpus Qwen cũ đã hoàn tất.

## Chạy ngay

1. Mở `notebooks/04_colab_export_new_data_kaggle.ipynb` trên **Colab CPU**.
   Giữ `KAGGLE_ACCOUNTS=2` nếu dùng hai tài khoản, hoặc chọn 1 trước lần xuất đầu.
   `KAGGLE_NOTEBOOK_OUTPUT=""`. Secret `KAGGLE_JSON` thuộc người xuất, có Notebook access.
   Run all. Không chạy lại notebook 02/03.
2. Khi có `NEW_DATASET_READY: <link>`, cấp quyền đọc dataset private cho acc thứ hai.
   Hai acc Add Input cùng dataset đó. Import hai notebook đã đặt ID:
   `notebooks/kaggle/04_new_data_account_0.ipynb` và
   `notebooks/kaggle/04_new_data_account_1.ipynb`. Nếu chỉ dùng một acc, chạy bản 0.
   Chọn Internet ON, **T4 x2**, **Save Version → Save & Run All**.
3. Sau khi version lưu Outputs thành công, mở lại notebook Colab CPU, điền
   `KAGGLE_NOTEBOOK_OUTPUT="owner/notebook,owner2/notebook2"`, Run all.
   Có thể nhập một acc trước. Dùng chính job/team đã xuất; không thay team giữa lượt.
   Notebook nhập vector và xuất gói phần thiếu cho lượt sau.

Nếu mới nhập một acc, `WAITING_FOR_KAGGLE_OUTPUTS` yêu cầu nhập acc còn lại trước
khi mở lượt mới, tránh giao trùng phần mà đồng đội còn đang chạy. Run all khi
chưa nhập Outputs chỉ nhắc lại dataset hiện tại, không chọn lại corpus.

Mỗi lượt giới hạn kế hoạch 16 GB kết quả/acc, với deadline 10 giờ tính từ cell đầu
để kịp lưu Outputs. GPU có thể hết thời gian trước khi hết việc của lượt.
`ROUND_COMPLETE` chỉ nói lượt đó xong; `NEW_DATA_EMBEDDINGS_COMPLETE` trên Colab
mới xác nhận đủ cả hai model cho 1.225 input parts của batch mới.

## Cache và checkpoint

- Giữ BGE-M3 1024 chiều và Qwen3-Embedding-8B 4096 chiều, đúng model revision,
  pooling, corpus role, quantization và CUDA runtime đã dùng. Kiểm producer code
  trước khi nạp weights. Không lấy bản Torch CPU của máy xuất làm runtime GPU.
- Colab dùng lookup SQL đã lưu và chỉ đọc receipt vector mới. Không đọc payload
  vector để kiểm kê. Mỗi frozen input part được chọn riêng theo family; kết quả
  chọn lưu draft map và một missing-text Parquet. Journal commit sau payload.
  Đổi VM sẽ bỏ qua selection đã commit, không mở lại corpus đó để chọn lại.
- GPU chỉ nhận text thiếu. Mọi occurrence cùng text hash cùng model thuộc cùng
  worker, nên dùng lại block đã encode trong phiên. Mỗi task tối đa 512 text;
  chỉ task có journal hoàn tất mới tính coverage. Vector ghi `/kaggle/working`.
- Khi nhập, kiểm SHA, key coverage, encoder identity, vector shape/finite/L2.
  Sau đó nhập block vào native `data/model_cache`, ghép draft map bằng key và
  commit biên nhận. Lượt sau lọc cả vector mới có trong cache, kể cả text trùng
  giữa nguồn. Outputs đã nhập có biên nhận thì không tải hoặc kiểm payload lại.
- Input/hash/model-policy đổi thì dừng và giữ checkpoint. Không tự xóa artifact,
  reset build hoặc chạy lại import/chunk. Các file tạm nằm trên Colab/Kaggle,
  không tải dữ liệu về PC của người dùng.

Progress và gói xuất đặt ở:

`VietMedBridge/data_staging/btc-new-71be0e12-20261019-v1/freeze_batch_v1/embeddings_v1/kaggle-2x2/`

Đổi sang một acc dùng `kaggle-1x2`. Không chạy đồng thời hai phiên CPU xuất/nhập
cùng team hoặc hai GPU sessions trùng account ID của cùng gói. Hai acc chạy
GPU song song là được. Không Add Input gói 700k cũ vào các notebook mới này.

## Phạm vi

Luồng này tạo child embedding maps và content cache cho batch 10 nguồn mới.
Không đổi active submission candidate, không rebuild catalog/BM25, không chạy
query inference/rerank/submission và không tự hoàn thành Qwen của corpus cũ.
Ghép corpus, cập nhật index và chạy truy hồi là bước tiếp sau khi có đủ embedding.

Kiểm thử dùng producer cache thật, input Parquet và vector giả chuẩn hóa; không
tải model trên PC. Bao gồm cache cũ/trùng text, đổi runtime, partial round,
hai acc nhập lần lượt, identity mismatch, vector hỏng và notebook bootstrap.
Chưa xác nhận hiệu năng trên T4 thật bằng các kiểm thử này.
