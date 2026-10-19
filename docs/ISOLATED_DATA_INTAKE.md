# Nhập data mới trong khi luồng submission vẫn chạy

Mở [02_colab_import_new_data.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_import_new_data.ipynb)
trong **một runtime Colab CPU riêng**, rồi **Run all**. Cho phép kết nối Drive
khi Colab hỏi. Notebook tự tìm project dùng chung, cài checkout riêng, khóa
code/runtime và chọn toàn bộ nguồn đã audit trong `configs/intake_ai_guru_full_20261019.json`.
Không cần tạo shortcut từng archive, sửa `INPUT_KIND`, chọn `WORKER_ID` hay bật GPU.

Để tăng tốc bằng **4 tài khoản Colab CPU**, dừng phiên intake tuần tự cũ trước,
giữ luồng submission chạy riêng. Mỗi tài khoản dùng chung đúng folder VietMedBridge
và mở một bản điền sẵn dưới đây, chọn **CPU → Run all**; không cần sửa code:

- Acc 0: [02_cpu_worker_0.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/parallel_cpu/02_cpu_worker_0.ipynb) — 00000, 00001, 00005.
- Acc 1: [02_cpu_worker_1.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/parallel_cpu/02_cpu_worker_1.ipynb) — 00002, 00003.
- Acc 2: [02_cpu_worker_2.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/parallel_cpu/02_cpu_worker_2.ipynb) — 00004, 00006.
- Acc 3: [02_cpu_worker_3.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/parallel_cpu/02_cpu_worker_3.ipynb) — 00007 và retained.

Cả bốn chạy ngay, không cần một acc chuẩn bị trước. WORKER_ID=0–3 và TEAM_SIZE=4
đã điền sẵn. Không mở hai phiên cùng ID. Các nguồn hoàn tất từ bản cũ được tham
chiếu bằng receipt, không copy hoặc chia chunk lại. Nguồn đang dở tiếp tục tại
root cũ và chỉ hoàn tất đúng nguồn đó; nguồn chưa bắt đầu có controller root riêng
trong `parallel_cpu4/jobs/<job>/`. Mỗi nguồn có một owner cố định; worker chỉ ghi
binding/result và báo cáo của mình. Reference chưa có được tải vào thư mục riêng
của worker, không cùng ghi một file `.partial` trên Drive.

Các frontier/retained đã audit có tập ID rời nhau. Với các nguồn mới chạy song
song, dedup dùng baseline cố định, không chờ output của worker khác. Nguồn dở
giữ nguyên dependency/selection ban đầu để signature của các shard đã commit
không đổi. Sau khi cả bốn báo `CPU_WORKER_COMPLETE`, **acc 0 chạy lại cell 4**
để tổng hợp. `WAITING_FOR_CPU_WORKERS` chỉ liệt kê nguồn chưa bàn giao, không
khởi động lại import. Reducer kiểm SHA các outcome/mapping cần dùng, ID duy nhất,
đủ 9 nguồn, khớp 1.117.692 ID đầu vào và các bộ đếm trước khi ghi
`FULL_INTAKE_COMPLETE` vào `full_intake_status.json`. Tệp thiếu/ID trùng không được
coi là hoàn tất. Tổng hợp thành công được checkpoint để lần sau chỉ đọc receipt.

Đổi runtime/tài khoản: mở lại đúng notebook worker ID đó, cùng folder project,
Run all. Không đổi TEAM_SIZE hoặc assignment giữa lượt. Bản tuần tự mới chặn
chạy khi đã có parallel plan để tránh vô tình xử lý lại các nguồn ở root cũ.
Tốc độ thực tế còn phụ thuộc I/O Drive và CPU mỗi phiên; không bảo đảm nhanh gấp 4.

Lượt đầy đủ gồm **tám frontier 00000–00007 và gói retained**, được xác minh lại
trong folder AI Guru ngày 19/10. Tất cả ID phải thuộc corpus BTC revision
`71be0e1239a19dfd7a717e2bc13e3bec21bd0334`. Bảy file vật lý cần dùng:

- `vibiomir_retain_shard_00001_with_resume_00000.tar` — chỉ inner 00000.
- `vibiomir_retain_shard_00002_with_resume_00001.tar` — inner 00001 và 00002.
- `vibiomir_retain_shard_00004_with_resume_00003.tar` — inner 00003 và 00004.
- `vibiomir_retain_shard_00005.tar`.
- `vibiomir_retain__shard_00006.tar` — hai dấu gạch dưới sau `retain`.
- `vibiomir_retain_shard_00007.tar`.
- `vibiomir_retained_71be0e12.tar` — bộ đọc portable riêng.

Các bản flat 00000/00003 và inner 00001 cũ trong gói outer 00001 là phiên bản
trùng, không nhập thêm lần nữa. Shard legacy 00000–00015 không được nhập nguyên
gói vào lượt corpus mới; gói retained cung cấp mapping chính thức của phần còn
hợp lệ. Đọc metadata qua các khoảng byte của Drive API; không tải/copy toàn
bộ archive hoặc crawl lại URL.

Ở chế độ tuần tự, notebook tự tiếp tục ba chặng: giữ nguyên lượt sáu nguồn 00000–00005 đã khóa;
chạy hai nguồn bổ sung trong `additional-frontiers/`; rồi chạy retained trong
`retained-recovery/`. Mỗi chặng sau dùng mapping đã commit của chặng trước để
bỏ qua tài liệu hợp lệ. Không sửa plan/receipt/code fingerprint của chặng cũ.

Đầu ra riêng nằm tại:

```
VietMedBridge/data_staging/btc-new-71be0e12-20261019-v1/
  intake_plan.json
  intake_code_lock.json
  intake_runtime_lock.json
  intake_contract.json
  intake_status.json
  full_intake_scope.json
  full_intake_status.json
  references/
  sources/<job>/metadata/
  sources/<job>/selection/source_selection.parquet
  sources/<job>/source_outcomes.parquet
  sources/<job>/complete.json
  processed/<job>/parts/
  processed/<job>/build.json
  processed/<job>/intake_provenance.json
  additional-frontiers/  # hai nguồn 00006–00007; cùng cấu trúc checkpoint
  retained-recovery/     # retained; cùng cấu trúc checkpoint
  parallel_cpu4/         # chế độ 4 worker: contract và kết quả tổng hợp
    jobs/<job>/         # nguồn chưa bắt đầu; cùng cấu trúc checkpoint
    bindings/<job>.json
    results/<job>.json
    workers/worker-<id>.json
    worker-<id>/references/
    collection.json
```

Luồng này không ghi vào `data/processed`, raw snapshot, active pointers, model
cache, khóa code/runtime, hoặc `submission_runs` của luồng nộp bài. Data mới
chưa được tự ghép vào candidate đang dùng. Hai runtime có thể cùng đọc Drive;
tốc độ vẫn phụ thuộc tài nguyên CPU và I/O của mỗi phiên.

Mọi nguồn được gắn với đúng file links và SHA-256 của snapshot BTC mới.
Mapping baseline được khóa theo candidate đã audit; không đọc lại active
candidate để tự đổi phạm vi giữa lượt. Chỉ bỏ qua URL khi **mọi native ID/URL
alias** của URL ấy đã có tài liệu hợp lệ trong baseline hoặc nguồn trước đó.
Một alias còn thiếu làm cả nhóm alias được đưa qua pipeline để giữ đúng mapping.
ID trùng nhưng URL khác là lỗi binding, không được tự ghép. Mapping baseline chỉ
được đọc để nhận biết tài liệu đã xử lý: không kéo các URL ngoài snapshot BTC mới
vào lượt nhập này. URL đã có trong baseline vẫn giữ được công việc trước đó.

Tám frontier bao phủ 800.000 URL (800.999 native IDs), có 455.245
`EXTRACT_SUCCESS` trước quality/chunk gates. Gói retained chứa 316.097 URL
(316.693 native IDs), không trùng tám frontier; 264.212 URL đã có trong candidate
baseline, 51.885 URL chưa có. Trong phần chưa có, 38.951 URL có extraction thành
công trước quality/chunk gates. Bộ nguồn bao phủ 1.117.692/1.220.598 native IDs
BTC mới; đây là độ phủ đầu vào đã có, không phải số tài liệu đạt chuẩn hay toàn
bộ corpus BTC. Số document thực tế được báo sau khi chạy.

Retained dùng `doc_url_map.parquet`, `raw_catalog.parquet` và `documents.parquet`.
Bộ đọc kiểm SHA metadata, ID/URL/alias với corpus mới, raw hash và snapshot của
lần crawl/extraction gốc; giữ nguyên snapshot cũ trong metadata capture và ghi
riêng snapshot membership mới. Không gán snapshot mới cho raw/extraction cũ.
Nhóm LOW/QUARANTINE từ một trong hai bảng raw/extraction được giữ để review,
không coi là tài liệu đạt chuẩn. Chỉ metadata chuẩn hóa của các URL cần bổ sung
được đưa qua importer/chunking, các URL lỗi vẫn giữ outcome theo từng native ID.
Metadata documents khoảng 1,12 GB được tải theo member range, checkpoint mỗi
8 MiB. Ngắt tải: tiếp tục offset đã commit, kiểm checksum toàn member trước khi
dùng; không tải lại từ byte 0 hoặc tải raw HTML của gói 12,58 GB.

Trước khi lọc phần trùng, importer kiểm tra metadata gốc để không che mất record
ngoài frontier, extraction sai raw hash, hoặc snapshot sai. Phần mới dùng nguyên
quy trình kiểm tra văn bản, structure, parent/child chunk, token budgets và
kiểm tra offset nguồn của pipeline hiện có. Raw HTML chỉ có locator/provenance;
lượt này không chứng nhận đã replay toàn bộ payload hoặc chất lượng y khoa.

Các URL lỗi/pending vẫn có kết quả theo từng native ID trong
`source_outcomes.parquet`. Ba trạng thái cuối:

- `ADDED_VALID_DOCUMENT`: đã tạo document/chunk và qua kiểm tra integrity.
- `REUSED_EXISTING_DOCUMENT`: đã có document hợp lệ trong baseline/nguồn trước.
- `PIPELINE_FAILURE`: giữ nguyên reason code để xử lý sau; không biến thành document giả.

`full_intake_status.json` tổng hợp ba chặng; `intake_status.json` là từng chặng.
Các counts là **lượt ID nguồn**, không phải số URL duy nhất:
cùng URL có thể có nhiều native ID. Không cộng các lượt này
để báo độ phủ URL duy nhất. `EXTRACT_SUCCESS` của crawler cũng chưa phải số
document cuối cùng qua quality/chunk gates.

Nếu runtime bị ngắt, mở lại cùng notebook và cùng project rồi Run all. Không
đổi campaign hoặc xóa checkpoint. Notebook đọc receipt nhỏ trước, không nạp
mapping/tokenizer nếu chưa cần. Các log tương ứng:

- `SKIP_COMPLETED_SOURCE`: nguồn đã xong, chỉ đọc JSON và trạng thái file;
  không băm/đọc lại nội dung Parquet, không khôi phục catalog hay embedding.
- `REUSE_SELECTION`: nguồn dở đã chọn xong ID; dùng lại lựa chọn đó, không đọc
  lại archive hoặc mapping baseline/nguồn trước để chọn lại URL.
- `RESUME_PARTS`: shard 2.048 ID đã commit được kiểm checksum và dùng lại;
  shard đang ghi dở chạy lại. Chỉ nguồn đang dở cần đọc metadata/kiểm shard cũ.
- `REUSE_VALIDATED_BUILD`: nguồn đã qua integrity nhưng bị ngắt trước receipt;
  tiếp tục bàn giao kết quả, không chia chunk hoặc kiểm toàn build lần nữa.
  Chỉ kiểm SHA của document/failure và mapping cần dùng để hoàn tất bàn giao.
- `SOURCE_CHECKPOINT_SAVED`: kết quả và receipt của nguồn đã được ghi xong lên Drive.

Mapping được copy về local và kiểm SHA khi thực sự cần cho một lựa chọn mới.
Receipt chỉ xác nhận lần xử lý đã commit; đọc trạng thái không thay thế một lần
audit lại toàn payload. Khi dùng mapping để lọc data mới vẫn phải kiểm SHA; bước
freeze sau này vẫn kiểm các artifact đầu vào. Tệp thiếu/đổi kích thước được báo
lỗi, không tự xóa checkpoint rồi xử lý lại nguồn đã xong.

Phần điều phối và retained adapter được nhúng vào notebook từ ba script
`intake_resume.py`, `retained_intake.py`, `full_intake.py`,
nằm ngoài package tạo dữ liệu: tối ưu resume không làm đổi code fingerprint của
importer/chunking. Giữ nguyên campaign và code/runtime lock đã lưu, kể cả khi
notebook mới mở với checkout cũ đã khóa. Đường dẫn/mtime metadata local cố định
để đổi tài khoản Colab không làm đổi identity của importer. Không cho hai phiên
cùng ghi campaign; không cần khởi động lại phiên đang chạy ổn chỉ để cập nhật notebook.

Sau lỗi `OutOfMemoryException` khi ghi metadata Parquet, dùng bản notebook đã
cập nhật nhưng giữ nguyên campaign và các code/runtime lock. Adapter SQL của
phần điều phối ghi từng nhóm 2.048 dòng thay cho row group mặc định lớn; dùng
một luồng, spill vào `WORK_DIR`, giới hạn tối đa 2 GiB và 1/4 RAM khả dụng lúc
bắt đầu. Không đổi file producer, chính sách chunk hoặc runtime đã khóa. Nguồn
đã có `SOURCE_CHECKPOINT_SAVED` được bỏ qua. Nếu chưa có `selection.json`, chỉ
chọn/ghi lại metadata của nguồn đang dở; metadata tải thành công vẫn được dùng
lại sau kiểm tra binding/SHA. File xuất chỉ được công bố sau khi ghi xong.

Chỉ sau `FULL_INTAKE_COMPLETE` và **9/9 nguồn**, gửi `full_intake_status.json`
để đối chiếu số thực tế. `INTAKE_COMPLETE` của sáu nguồn cũ chỉ là một chặng. Đây mới
là đầu ra CPU; chưa có embedding, index hay ZIP cho phần bổ sung. Lượt tiếp theo
cần kiểm tra/freeze additions, chuẩn bị index inputs, embedding text thiếu rồi
ghép candidate cho **phiên submission kế tiếp**. Giữ nguyên phiên nộp đang chạy.

Plan của campaign này được khóa: data về sau không tự lọt vào giữa lượt. Khi
có archive/version mới, audit và tạo plan/campaign mới, dùng cùng engine; không
ghi đè plan cũ hoặc sửa code đang chạy để nhận dữ liệu khác.
