# Vector index và vector DB

Notebook 04 nối ba backend vào cùng contract `scores, frozen_vector_positions`.
DiskCatalog tiếp tục giữ source text, official document/chunk aliases và BM25;
vector index không thay IDs hoặc dựng lại corpus.

## Mặc định trên Colab/Kaggle: FAISS ANN

`VECTOR_BACKEND="faiss_ivf_sq8"` dựng IVF + scalar quantization 8-bit từ
embedding đã có. Đây là chỉ mục vector portable, **không phải vector DB server**.
Index build/search chạy CPU; các model embedding/dịch/rerank vẫn cần GPU.
FAISS 1.12.0 đã nằm trong extra `retrieval`, không cài thêm PyTorch.

Training lấy tối đa 32.768 vector có sẵn, seed 1729, tối đa 1.024 cells
(giới hạn theo số sample). Mỗi shard tối đa 32 input parts, thường 131.072
vectors. Chỉ một shard nén và một source part nằm trong RAM; Linux search mmap,
Windows wheels không hỗ trợ mmap thì đọc từng shard. Corpus dưới 1.024 inputs
dùng FAISS Flat và ghi rõ backend trong manifest, không giả đó là ANN.

Templates và shards content-addressed nằm trong
`/content/vmb_results/vector_indexes/faiss_ivf_sq8` ở chế độ cached; full mode
dùng `DATA_ROOT/retrieval/vector_indexes/faiss_ivf_sq8`.
Mỗi shard có checksum và manifest công bố sau khi write thành công. Khi resume,
không đọc/encode/index lại source parts thuộc shard hợp lệ. Candidate mới giữ
nguyên part checksums/order trong một nhóm thì dùng lại shard đó; thay nội dung,
ranh giới part/group hoặc model thì tạo shard mới. Không hứa mọi thay đổi corpus
đều tái sử dụng 100% index. Việc bổ sung index không yêu cầu reembedding.

Trước inference, lấy 8 query positions trải đều trong batch thật; exact top-20
trên **toàn corpus** làm reference. ANN lấy pool cấu hình (4.096 ở strong),
chấm lại bằng vectors gốc và phải đạt recall trung bình >=0,95 trên mẫu đó.
Tự tăng nprobe 64 -> 128 -> 256 -> 512; không đạt thì dừng. Đây chỉ kiểm
sai số ANN, **không chứng minh relevance hoặc F2**. Bước reference quét corpus
một lần với ít query, có checkpoint; lượt đầy đủ không quét exact mọi query.
Refinement đọc tuần tự từng source part **một lần cho toàn bộ query pool**,
chỉ chấm các cặp query/vector được ANN chọn; từng part có checkpoint scores.
Không gather ngẫu nhiên toàn corpus lặp lại riêng cho từng query khi cache nhỏ.
Refinement vẫn cần source cache/IO; ANN không
loại bỏ mọi chi phí đọc vectors. Chi phí CPU training/build cũng cần đo thực.

Query search checkpoint từng batch; kết quả cuối, recall gate và scored cascade
gắn với index/model/source/policy. Đặt `VECTOR_BACKEND="exact"` khi chủ động
cần backend cũ để đối chiếu. Không tự fallback exact khi ANN quality gate fail.

Với 3.837.487 BGE inputs x 1.024 dimensions, riêng SQ8 codes khoảng 3,93 GB
thập phân cộng IDs/centroids/manifests. Đây là ước tính format, **chưa đo trên
700k**. Source vectors, SQLite catalog, models và temporary files chiếm thêm.
Không tự xóa chúng để lấy chỗ. Backup cuối 04 bao gồm index, document vectors,
query checkpoints; có thể nhiều GB. Runtime local mất thì phải restore backup,
không coi checkpoint local là backup cloud.

## Qdrant server

Trong cell 1: `VECTOR_BACKEND="qdrant"`, `QDRANT_URL="https://..."`;
Secret `QDRANT_API_KEY` nếu server yêu cầu xác thực. Ngoài Colab dùng
`VMB_QDRANT_URL`, `VMB_QDRANT_API_KEY` ở môi trường. Không đặt token vào URL,
source code hoặc chat. Remote API key cần HTTPS. Preflight/REST sử dụng httpx
đã có; không dùng Python Qdrant local mode cho corpus hàng triệu vectors.

Mỗi nhóm input parts tương ứng collection có tên hash từ encoder, vector
checksums và order. Vectors, payload và HNSW lưu disk server. Point IDs là
local row positions; manifest của candidate giữ global offsets. Các collections
cũ được chia sẻ khi thêm candidate có cùng groups; không mutate/delete chúng.
Payload chỉ chứa group/part bindings, không upload source text vào DB.

Upsert batch dùng `wait=true`; timeout sau acknowledgement được retry bằng
cùng ID nên không thêm bản sao. Resume hỏi **exact count trên server** cho
từng part, không chỉ tin local marker. Collection bị mất thì ingest lại
embedding có sẵn. Count sai/thừa hoặc vector policy khác thì dừng.
Chỉ công bố ready khi optimizer báo green/ok và collection lớn có HNSW index;
acknowledged nhưng đang optimize thì báo WAITING, rerun dùng lại các part đã
nhập. Query batch gửi ANN request, kiểm group/ID bounds, exact-refine và áp
cùng recall gate. Không có fallback âm thầm sang local/exact.

`deploy/qdrant.compose.yaml` ghim Qdrant 1.19.2, bind loopback và volume bền
vững. Trên máy chủ đã có Docker, đặt biến môi trường `VMB_QDRANT_API_KEY` rồi:

```bash
docker compose -f deploy/qdrant.compose.yaml up -d
```

Muốn Colab/Kaggle truy cập máy chủ phải có endpoint HTTPS/VPN/tunnel tương ứng;
localhost của notebook không phải localhost máy chủ. Compose này chưa cấp một
dịch vụ công khai, chưa có máy chủ được deploy tự động. Volume/server snapshot
phải được backup riêng; ZIP notebook chỉ chứa manifests/query artifacts khi
chọn Qdrant, không chứa vectors nằm trên server.

Các API/settings đối chiếu tài liệu chính thức:
[FAISS index formats](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
[FAISS on-disk indexes](https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM),
[Qdrant upsert](https://api.qdrant.tech/api-reference/points/upsert-points),
[Qdrant batch query](https://api.qdrant.tech/api-reference/search/query-batch-points),
[Qdrant collections](https://qdrant.tech/documentation/manage-data/collections/).

Tests chạy FAISS thật trên CPU, corruption/restart/append và pipeline xuất
1.200 query với model doubles. Qdrant được kiểm bằng REST transport mô phỏng,
chưa benchmark/live-test trên server hoặc corpus 700k. Chưa có số tăng tốc
hoặc điểm thi mới từ thay đổi này.
