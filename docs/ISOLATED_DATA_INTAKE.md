# Nhập data mới trong khi luồng submission vẫn chạy

Mở [02_colab_import_new_data.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/02_colab_import_new_data.ipynb)
trong **một runtime Colab CPU riêng**, rồi **Run all**. Cho phép kết nối Drive
khi Colab hỏi. Notebook tự tìm project dùng chung, cài checkout riêng, khóa
code/runtime và chọn các nguồn trong `configs/intake_ai_guru_20261019.json`.
Không cần tạo shortcut từng archive, sửa `INPUT_KIND`, chọn `WORKER_ID` hay bật GPU.

Lượt này dùng chín frontier đã audit ngày 19/10: sáu frontier mới 00000–00005,
shard cũ 00014/00015, và bản phục hồi 00013. Các phiên bản TAR chứa TAR được mở
theo đúng inner member. Notebook chỉ tải bốn metadata cần thiết qua các khoảng
byte của Drive API; không tải/copy toàn bộ archive hoặc crawl lại URL.

Đầu ra riêng nằm tại:

```
VietMedBridge/data_staging/ai-guru-20261019-v1/
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

Mỗi nguồn được gắn với đúng file links và SHA-256 của snapshot tương ứng.
Mapping baseline được khóa theo candidate đã audit; không đọc lại active
candidate để tự đổi phạm vi giữa lượt. Chỉ bỏ qua URL khi **mọi native ID/URL
alias** của URL ấy đã có tài liệu hợp lệ trong baseline hoặc nguồn trước đó.
Một alias còn thiếu làm cả nhóm alias được đưa qua pipeline để giữ đúng mapping.
ID trùng nhưng URL khác là lỗi binding, không được tự ghép.

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
cùng URL có thể xuất hiện ở nhiều frontier/snapshot. Không cộng các lượt này
để báo độ phủ URL duy nhất. `EXTRACT_SUCCESS` của crawler cũng chưa phải số
document cuối cùng qua quality/chunk gates.

Nếu runtime bị ngắt, mở lại cùng notebook và cùng project rồi Run all. Không
đổi campaign hoặc xóa checkpoint. Source hoàn tất có receipt và được bỏ qua
sau kiểm tra mapping/outcomes nhỏ. Trong source đang dở, các shard 2.048 ID đã
commit được kiểm checksum và dùng lại; shard ghi dở chạy lại. Metadata phục vụ
source đang dở có thể phải đọc lại từ cache Drive; không chia chunk lại shard
đã hoàn tất. Đường dẫn/mtime của metadata local được cố định để đổi tài khoản
Colab không làm đổi identity của importer. Không cho hai phiên cùng ghi campaign.

Sau `INTAKE_COMPLETE`, gửi `intake_status.json` để đối chiếu số thực tế. Đây mới
là đầu ra CPU; chưa có embedding, index hay ZIP cho phần bổ sung. Lượt tiếp theo
cần kiểm tra/freeze additions, chuẩn bị index inputs, embedding text thiếu rồi
ghép candidate cho **phiên submission kế tiếp**. Giữ nguyên phiên nộp đang chạy.

Plan của campaign này được khóa: data về sau không tự lọt vào giữa lượt. Khi
có archive/version mới, audit và tạo plan/campaign mới, dùng cùng engine; không
ghi đè plan cũ hoặc sửa code đang chạy để nhận dữ liệu khác.
