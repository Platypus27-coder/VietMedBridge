# Chia corpus cho ba người chạy độc lập

Hai notebook dành riêng cho việc này là
[`01c_colab_team_worker.ipynb`](../notebooks/optional/01c_colab_team_worker.ipynb) và
[`01d_colab_merge_team_crawls.ipynb`](../notebooks/optional/01d_colab_merge_team_crawls.ipynb).
Chúng **chưa crawl URL nào** khi chỉ mở notebook hoặc chạy ô xem kế hoạch.

## Chia phần

Mỗi người có thể dùng Colab và Google Drive riêng. Trước khi bắt đầu, cả ba cần
cùng một bản `links_corpus.parquet`, cấu hình crawler, và commit Git. Chạy notebook
00 trên Drive của từng người hoặc sao chép cùng snapshot đã xác minh. Ghim cùng
một full commit SHA vào `CODE_REVISION` trong ô Bootstrap của notebook 01c.
Bootstrap dùng `optional_data_code_lock.json` riêng; giữ cùng SHA khi resume,
không nâng code giữa một lượt crawl. Notebook 00–04 vẫn là luồng chính.
Mỗi Drive cũng cần các milestone đã được duyệt trong `data/gates/`, báo cáo
golden và tokenizer lock của cùng snapshot/code; nếu thiếu, ô crawl sẽ dừng ở
cổng quy mô trước khi tải URL.

Trong notebook 01c, chỉ đặt `TEAM_MEMBER` thành 1, 2 hoặc 3. Giữ nguyên
`RUN_PREFIX` giống nhau cho cả ba. Chạy ô Bootstrap và ô kế hoạch trước;
đối chiếu `Plan hash`, `Official SHA-256` và `Code commit` giữa ba người.

Với snapshot 4.394.718 dòng và shard size 512 hiện tại, kế hoạch là:

| Người | `TEAM_MEMBER` | Dòng Parquet `[start, stop)` | Số URL |
| --- | ---: | ---: | ---: |
| 1 | 1 | `[0, 1465344)` | 1.465.344 |
| 2 | 2 | `[1465344, 2930176)` | 1.464.832 |
| 3 | 3 | `[2930176, 4394718)` | 1.464.542 |

Đây là **vị trí dòng trong Parquet**, không phải `doc_id`. Các khoảng không
giao nhau và bao phủ toàn bộ file. Notebook tính lại giới hạn từ file thật và
từ chối file khác SHA; bảng trên chỉ giúp đối chiếu.

Ô crawl kiểm cổng quy mô của repo **trước khi gửi request**. Full corpus chỉ
được chạy sau các milestone Stage A/B1/B2/C, kiểm thử truy hồi và kiểm ngân sách
theo `src/vietmedbridge/gates.py`. Không chỉnh hoặc bỏ qua gate để chạy sớm.
Khi được phép chạy, mỗi người giữ nguyên `TEAM_MEMBER`, `RUN_PREFIX`, commit và
cấu hình qua các phiên Colab. Checkpoint đã hoàn thành sẽ được tái sử dụng.

## Bàn giao và ghép

Mỗi người chuyển **toàn bộ thư mục run** có `run.json` và `parts/` cho người
ghép, không chỉ file thống kê. Đặt ba thư mục ở nơi cùng một Colab runtime đọc
được, ví dụ `data/team_uploads/<run_name>`. Chạy notebook 01d, sửa
`TRANSFER_ROOT` nếu cần. Ba đường dẫn phải theo thứ tự người 1, 2, 3.

Bước gộp kiểm tra SHA của official Parquet, khoảng dòng, cấu hình và code của
từng run; kiểm mỗi shard có đúng cặp ID/URL chính thức, decoded body hash/length;
kiểm hash raw và độ phủ
toàn corpus. Nếu thiếu, hỏng, trùng hoặc khác snapshot thì không công bố run
ghép. Raw gốc của ba người không bị sửa. Gộp có checkpoint trong thư mục
`.staging`, nên có thể chạy lại sau khi Colab ngắt. Cần thêm dung lượng để lưu
một bản sao raw ở nơi ghép.

Sau khi gộp thành công, chỉ một người chạy notebook 02 với
`CRAWL_RUN = "full-team-v1-merged"` và một `BUILD_RUN` mới, sau đó notebook 03
dùng đúng hai tên đó. Bước gộp chỉ xác nhận toàn bộ URL có **outcome crawl**,
bao gồm cả URL lỗi; nó không biến URL lỗi thành document dùng được.
