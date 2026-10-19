# Nhập data mới trong khi luồng submission vẫn chạy

Mở [02_colab_import_new_data.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_import_new_data.ipynb)
trong **một runtime Colab CPU riêng**, rồi **Run all**. Cho phép kết nối Drive
khi Colab hỏi. Notebook tự tìm project dùng chung, cài checkout riêng, khóa
code/runtime và chọn các nguồn trong `configs/intake_ai_guru_20261019.json`.
Không cần tạo shortcut từng archive, sửa `INPUT_KIND`, chọn `WORKER_ID` hay bật GPU.

Lượt này **chỉ dùng sáu frontier mới 00000–00005** đã audit ngày 19/10, thuộc
dataset BTC revision `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`.
Shard legacy 00013–00015 không nằm trong lượt nhập này. Notebook chặn plan có
nguồn khác snapshot BTC mới trước khi tải mapping hoặc chạy importer.
Các phiên bản TAR chứa TAR được mở
theo đúng inner member. Notebook chỉ tải bốn metadata cần thiết qua các khoảng
byte của Drive API; không tải/copy toàn bộ archive hoặc crawl lại URL.

Đầu ra riêng nằm tại:

```
VietMedBridge/data_staging/btc-new-71be0e12-20261019-v1/
  intake_plan.json
  intake_code_lock.json
  intake_runtime_lock.json
  intake_contract.json
  intake_status.json
  references/
  sources/<job>/metadata/
  sources/<job>/selection/source_selection.parquet
  sources/<job>/source_outcomes.parquet
  sources/<job>/complete.json
  processed/<job>/parts/
  processed/<job>/build.json
  processed/<job>/intake_provenance.json
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

Sáu frontier bao phủ 600.000 URL (600.703 native IDs). Audit trước import ghi nhận
174 URL đã có trong baseline và 599.826 URL chưa có; không phải tất cả đều crawl
thành công. Có 321.499 `EXTRACT_SUCCESS` theo báo cáo crawler trước các bước kiểm
tra/chunking. Đây là một phần của 1.220.598 bản ghi BTC mới, chưa phải
độ phủ toàn bộ dataset. Số document thực tế được báo sau khi chạy.

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

`intake_status.json` tổng hợp **lượt ID nguồn**, không phải số URL duy nhất:
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

Phần điều phối tiếp tục được nhúng vào notebook từ `scripts/intake_resume.py`,
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

Sau `INTAKE_COMPLETE`, gửi `intake_status.json` để đối chiếu số thực tế. Đây mới
là đầu ra CPU; chưa có embedding, index hay ZIP cho phần bổ sung. Lượt tiếp theo
cần kiểm tra/freeze additions, chuẩn bị index inputs, embedding text thiếu rồi
ghép candidate cho **phiên submission kế tiếp**. Giữ nguyên phiên nộp đang chạy.

Plan của campaign này được khóa: data về sau không tự lọt vào giữa lượt. Khi
có archive/version mới, audit và tạo plan/campaign mới, dùng cùng engine; không
ghi đè plan cũ hoặc sửa code đang chạy để nhận dữ liệu khác.
