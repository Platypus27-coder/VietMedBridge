# Tận dụng tiếp nguồn AI Guru đã thu gom

## Bản sửa 23/10: giảm ghi Drive và chốt theo từng lượt

Notebook hiện tại in `MAXIMIZE 20261023-small-journal-r1` khi bắt đầu. Dùng bản
mới này khi tiếp tục nhánh raw; dừng writer maximize cũ trước. Luồng export/GPU
802k chạy riêng. Không chạy lại collect/intake hoặc xóa checkpoint để bắt đầu lại.

Đã xác định overhead: 994 SQL snapshot trong 1.000 file mới nhất của campaign
chiếm gần59,4GiB, mỗi file khoảng60–62MiB. Code cũ sao lưu toàn bộ SQL sau mỗi
1.024TAR header hoặc32text mới. Bản này chỉ ghi thay đổi raw result, text reference
và cursor thành journal JSON nhỏ. Gộp SQL đầy đủ sau64journal hoặc ranh giới
source/stage. Old snapshots được giữ nguyên; không xóa dữ liệu để làm nhẹ.

Runtime mới khôi phục một baseSQL đã kiểm SHA rồi replay tối đa64journal đã seal.
Nếu bản local ghi trước khi head upload thành công, phần chưa chứng nhận bị bỏ
khi restore; không giả nhận tiến độ. Nếu mất kết nối sau khi gộp SQL nhưng trước
khi reset head, marker trong SQL chứng minh journal đã được gộp. Stage freeze vẫn
xuất SQL checkpoint hoàn chỉnh cho các consumer hiện có. File text/raw đã xử lý
không cần parse lại khi replay metadata.

Cell4 phục hồi raw theo ngân sách30phút, lưu cursor rồi chốt mọi text mới đã có
thành `raw-ready-<hash>` để đưa qua GPU riêng. Không đợi cả raw-pass xong.
Thời gian ghi/chunk/freeze và một lần đọc Drive đang dở có thể kéo dài hơn ngân
sách đọc; đây không phải cam kết cứu đủ96.982ID trong30phút. Parser, SHA/length,
source span, tokenizer, gate challenge/quality và model producer giữ nguyên.

`checkpoint.json` luôn chỉ SQL hoàn chỉnh; `raw_journal/head.json` giữ phần mới
chưa gộp. Report chỉ đọc SQL cũ có thể trễ so với journal. Khi cần xuất/chốt text,
dùng notebook/finalize mới có hỗ trợ journal, không dùng bản cũ để ghi đè progress.

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

## Luồng đầy đủ — xuất toàn bộ text, kế hoạch GPU riêng

Mở bản mới của `notebooks/04_colab_export_full_data_kaggle.ipynb` trên Colab
CPU, giữ KAGGLE_ACCOUNTS=2 và KAGGLE_NOTEBOOK_OUTPUT trống, Run all. Notebook
nhận baseline, 10 nguồn intake và mọi recovery candidate đã freeze, gồm cả
legacy content được yêu cầu giữ. Không crawl/extract/chunk/index lại. Bộ model,
runtime, text hash và namespace vector giữ nguyên; BGE COMPLETE được kế thừa.

FULL_SCOPE_READY phải có ít nhất 802.028 ID BTC / 13 nguồn đã chốt. Đó là số ID
có text trong phạm vi hiện tại, chưa phải toàn bộ BTC. Số document rows hoặc
model-input rows không được cộng thành số URL khác nhau. Kiểm tra cấu trúc/hash
không chứng nhận chất lượng ngữ nghĩa; text LOW và lỗi trích xuất đã biết giữ
nguyên trong lần đổi cách đóng gói này.

### Một thư viện chứa toàn bộ text thiếu

`FullQueueKaggle.export_inputs` dùng journal chọn text đã lưu, lấy tất cả task
chưa được import, không áp trần dung lượng output vào input. Copy mỗi file
missing Parquet đã có một lần theo source/model; task tham chiếu offset trong
file đó. Không tạo hàng chục nghìn file riêng cho từng task. Header thư viện
phải có input_export_complete=true, unscheduled_tasks=0. Có thể có dòng text
trùng giữa nguồn/model; worker ownership theo hash bảo đảm dùng lại vector.

Thư viện được khóa bởi input_library.json và full_package.json. Dataset có
prefix vmb-full-inputs-. Sau khi upload và đối chiếu job.json thành công, các
lượt sau dùng lại dataset đó, không đóng gói/upload lại text. Nếu runtime CPU
bị thay trước khi upload xong, chỉ file staging còn thiếu mới được khôi phục từ
saved missing inputs; checksum và job hash vẫn phải khớp. Key-only Parquet đã
lưu trên Drive dùng để lập kế hoạch, không mở lại corpus hoặc vector payload.

### Kế hoạch GPU nhỏ, thay theo lượt

#### Ba tài khoản, giữ thư viện gốc và checkpoint đã nhập

Chạy `notebooks/04_colab_prepare_3_accounts_kaggle.ipynb` trên Colab CPU.
Notebook đã đặt execution team 3 acc × 2 GPU. Team gốc của thư viện và journal
vẫn là 2×2: không đổi header text, không xuất lại 5,5 GB hoặc chọn lại corpus.
Chủ dự án đã xác nhận kế hoạch `ab7954e4a47f…` chưa chạy GPU. Chỉ đúng kế hoạch
đó, chưa có import receipt, mới được thay; bản cũ vẫn nằm trong `history`.
11.588 task đã nhập được loại bằng biên nhận cũ trước khi chia việc.

Kế hoạch v2 ghi rõ worker 0–5 cho từng task và phân tải theo số text còn thiếu;
mỗi task chỉ giao một GPU. Những task có cùng model và toàn bộ tập key được
ưu tiên cùng worker để dùng lại content cache. Task có tập key chồng lấn nhưng
khác nhau vẫn có thể encode cùng text ở các worker khác nhau; không tuyên bố
khử trùng tuyệt đối xuyên GPU. Không sửa text/model producer để đổi cách chia.
CPU chỉ đọc key-only index và lookup metadata cần cho kế hoạch mới, không
đọc lại corpus/text/vector payload. Retry upload dùng đúng kế hoạch đã lưu.

[Kaggle Docs](https://www.kaggle.com/docs/notebooks) ghi tối đa 20 GB Outputs
được lưu trong `/kaggle/working`. V2 dùng ngân sách dự kiến 18 GB/acc, gồm
vector float32 và allowance metadata, chừa thêm dung lượng dưới mức này.
Đây là ước tính, không phải cam kết mọi task sẽ xong trong một phiên GPU.

Giữ dataset ALL_TEXT_DATASET_READY. Cả ba acc thay dataset kế hoạch cũ bằng
GPU_PLAN_DATASET_READY mới, dùng lần lượt `04_full_data_3acc_account_0.ipynb`,
`04_full_data_3acc_account_1.ipynb`, `04_full_data_3acc_account_2.ipynb`. Hai file điều
phối nhỏ được khóa SHA trong kế hoạch; thư viện text/worker cũ không bị sửa.
T4 x2, Internet ON, Save Version / Save & Run All. Cấp quyền đọc hai dataset
cho cả hai thành viên còn lại. Sau khi ba Outputs được lưu, nhập bằng notebook
CPU mới; importer kiểm task thuộc đúng execution account, key, model, vector
SHA/shape và cache skip trước khi ghi receipt. Account 2 được nhập đúng vào
cùng cache native và journal, không tạo lại namespace vector.

Các đoạn dưới mô tả kế hoạch v1 hai acc, vẫn hỗ trợ nhập Outputs cũ.

`create_plan` giữ trần dự kiến 16 GB vector mới/acc/lượt, đây là lựa chọn cấu
hình của pipeline. Whitelist task, cached keys, job/library hash, team và dự
kiến dung lượng được seal riêng. Dataset kế hoạch có prefix vmb-full-plan-.
Colab in hai link ALL_TEXT_DATASET_READY và GPU_PLAN_DATASET_READY. Hai acc
Add Input cùng hai dataset, dùng 04_full_data_account_0.ipynb và
04_full_data_account_1.ipynb. Account ID đã đặt sẵn 0/1. Chủ dataset cấp quyền
đọc cả hai gói cho tài khoản còn lại.

Có thể dùng Kaggle CPU và chỉ chạy cell 1 để kiểm mọi file input, kể cả file
chưa được giao trong kế hoạch hiện tại. ALL_TEXT_INPUTS_PRESENT xác nhận toàn
thư viện, số task trong kế hoạch có thể nhỏ hơn. Cell này không cài/nạp weights
hoặc mở Drive. Khi chạy GPU dùng T4 x2, Internet ON, Save Version / Save & Run
All. Worker chỉ chạy whitelist của kế hoạch, kiểm SHA/text keys khi sử dụng.
Giới hạn phiên hoặc quota có thể khiến kế hoạch chưa hết việc; journal và
Outputs của task hoàn tất phải được lưu thành công.

Sau GPU điền hai owner/notebook vào KAGGLE_NOTEBOOK_OUTPUT rồi Run all notebook
CPU. Importer khóa đúng thư viện, kế hoạch, account, task, vector hash/dimension
và producer. Keys được GPU bỏ qua phải thật sự có trong lookup cùng producer
trên Drive. Nhập kết quả cả hai acc rồi mới tạo kế hoạch tiếp theo, loại task
đã xong và dùng lại vector trùng. Lượt sau chỉ thay dataset kế hoạch nhỏ, giữ
thư viện text. Không cần tải vector cũ vào Kaggle để resume.

### Chuyển từ gói 1,84 GB cũ

Chủ dự án đã xác nhận chưa chạy GPU gói cũ. Vì vậy bản notebook này đặt
MIGRATE_UNSTARTED_LEGACY=True để ghi lịch sử chuyển sang thư viện đầy đủ.
Job, dataset, active_round và receipt cũ không bị xóa; Outputs cũ vẫn nhập được.
Nếu một người khác đã chạy GPU gói cũ thì giữ False và nhập kết quả trước khi
chuyển. Active submission candidate luôn được giữ nguyên.

Scope và membership ở btc-full-embeddings-71be0e12-v1/<hash>/scope.json.
Thư viện, kế hoạch, import receipts và full_embedding_status.json ở
embeddings_v1/kaggle-2x2/. ROUND_COMPLETE chỉ hoàn thành kế hoạch của account;
FULL_DATA_EMBEDDINGS_COMPLETE mới có nghĩa mọi nguồn đủ BGE và Qwen. Query,
vector index, reranker và submission vẫn là giai đoạn riêng.

### Tránh IOPub timeout và giữ kết quả phiên bị lỗi

Hai notebook GPU ghi stdout/stderr đầy đủ vào
`/kaggle/working/vmb_full_queue/gpu-worker.log`. IOPub chỉ nhận các dòng tiến
độ có giới hạn, heartbeat và GPU_RUN_SUMMARY; không nhận JSON chứa toàn bộ
task/text hashes. Worker và manifest đã khóa trong dataset không bị sửa, nên
dataset text/kế hoạch hiện có vẫn dùng được mà không xuất lại.

Nếu worker lỗi nhưng đã tạo partial manifest đúng job/plan/account và checksum,
notebook báo process_state=FAILED rồi kết thúc để Kaggle có thể lưu Outputs.
Đây không phải xác nhận embedding hoàn tất. Thiếu manifest hoặc manifest sai
vẫn báo lỗi và giữ nguyên các file để kiểm tra. Giới hạn/quota/hạ tầng của
Kaggle vẫn có thể ngắt phiên trước khi Outputs được lưu.

Notebook CPU mới cho phép thăm dò Outputs của phiên ERROR/CANCELLED đã kết
thúc, ngoài COMPLETE. Nó phải tải mới và kiểm manifest từ Kaggle trước, rồi
importer kiểm đầy đủ task/vector/producer như thường lệ; RUNNING/QUEUED vẫn
bị chặn. Với phiên `kaisercoding/impact` bị IOPub timeout, nhập handle đó vào
KAGGLE_NOTEBOOK_OUTPUT để thử lấy phần đã lưu trước khi đổi kế hoạch. Nếu
Kaggle không lưu manifest/vector của phiên lỗi thì không thể xác nhận hay
khôi phục những phần đó chỉ từ log.

### Nhập Outputs không quét hàng trăm trang danh sách file

Luồng full-data biết sẵn đường dẫn
`vmb_full_queue/result/new-data-result.json`. Notebook CPU lấy trực tiếp file
này qua KaggleHub, khóa số saved version rồi lấy từng file được manifest tham
chiếu, cùng job/plan. Không yêu cầu archive. Bản tối ưu dùng
ListKernelSessionOutput page_size=200, version_label cố định, lấy URL trực tiếp
cho tối đa 200 file mỗi yêu cầu thay vì gọi download API riêng cho mỗi vector.
Chỉ có tối đa bốn lượt tải payload đồng thời; file đã hoàn tất giữ
completion marker trong cache Colab để lần thử sau dùng lại. Không tải log,
worker journals hoặc notebook HTML nếu importer không cần. JSON tiến độ/version
lưu ở `data_staging/kaggle_output_downloads` trên Drive.

Cursor từng trang hoàn tất lưu trong cache Colab; khi resume chỉ dùng lại cursor
nếu các file thuộc trang đã xử lý vẫn có completion marker. Không ghi URL có chữ
ký vào log hoặc Drive. Tiến độ KAGGLE_TRANSFER báo tổng file, số đã có trong cache,
số còn thiếu và ETA tải đo từ tốc độ thực tế. ETA này chưa gồm native validation
và nhập vector về Drive. File vector mới được kiểm SHA trước khi đánh dấu tải
xong, và native importer giữ các kiểm tra producer/task/dimension trước publish.

HTTP 429/5xx được thử lại có giới hạn, tôn trọng Retry-After. 401/403/404 không
bị lặp lại. Phải có manifest, job, plan và mọi vector mà manifest tham chiếu;
thiếu file thì dừng trước import. Hash và producer vẫn được native
importer kiểm như trước. Cache tạm không phải bản backup độc lập: chỉ checkpoint
đã nhập thành công về Drive mới dùng lại được sau khi mất runtime Colab.

Luồng tải cũng thử lại có giới hạn khi gặp ChunkedEncodingError, ConnectionError
hoặc Timeout; giữ cùng version và cache, không xóa file đang tải dở. Downloader
KaggleHub 1.0.3 dùng HTTP Range để tiếp tục một file khi server hỗ trợ và kiểm checksum
khi hoàn tất. Lỗi hết đĩa, sai checksum hoặc chứng chỉ không được coi là lỗi
mạng để thử lại. Với runtime đang dùng notebook cũ, chạy cell trong
`scripts/resume_kaggle_output_download_cell.py` để chuyển sang tải file có tên
(tuần tự trong cell vá), rồi chạy lại riêng cell 3; không
restart runtime hoặc chạy lại GPU để xử lý lỗi tải Outputs này.
