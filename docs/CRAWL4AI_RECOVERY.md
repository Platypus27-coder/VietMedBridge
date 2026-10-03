# Thu tiếp Stage A bằng Crawl4AI

Notebook `01g_colab_recover_remaining_stage_a.ipynb` chạy trên Colab CPU mới,
cùng `DATA_ROOT` của các bước trước. Bootstrap clone repo, cài package và khóa
commit riêng trong `remaining_recovery_code_lock.json`. Không đổi code lock của
raw Stage A. Notebook tự làm bước replay 01f; không cần chạy lại 01e/01f trước.

## Chọn đủ dữ liệu và giữ nguồn

`load_remaining_plan` đọc sample Parquet chính thức, `run.json`, các raw shard
đã hoàn tất và `failures_after_retry.csv`. Nó kiểm hash, range đủ, ID duy nhất,
URL khớp sample, failure IDs khớp raw outcomes và số đếm khớp triage. CSV robots
83 ID và attempts 01c được ghim hash theo thí nghiệm đã nhận. Bốn candidate
01c phải còn raw asset đúng hash. Replay browser kiểm CSV, marker và từng asset
trước khi trích text.

Với kết quả hiện tại, có 917 HTTP captures gốc, 11 browser captures đã nhận,
4 candidate cũ cần review và 68 ID cần thử tiếp. Ledger đầu ra ghi đủ 1.000 ID
dù một phiên chỉ xử lý một phần hoặc nguồn vẫn lỗi. HTTP capture/candidate
không tự được tính là bài hợp lệ hoặc tự nhập vào canonical corpus.

## Phần kỹ thuật được dùng từ repo

Nguồn đọc là Crawl4AI 0.9.4, commit
`e5d2e786d1a101225f3f6a3e6fd344d76eeb13af` của
[unclecode/crawl4ai](https://github.com/unclecode/crawl4ai/tree/e5d2e786d1a101225f3f6a3e6fd344d76eeb13af).

- `async_configs.py`: `check_robots_txt=False` là mặc định. Adapter chủ động
  dùng resolver VietMedBridge cho cả crawler product và browser user agent.
- `utils.py`/`RobotsParser`: helper upstream cho phép sau non-200/network error.
  Adapter không dùng helper này để quyết định truy cập nguồn.
- `async_webcrawler.py`: upstream robots check ở URL đầu vào, không kiểm từng
  redirect. Guard CDP riêng kiểm mỗi request, scope, robots và pacing.
- `async_crawler_strategy.py`: hook `before_goto` lỗi sẽ dừng điều hướng;
  `before_return_html` cho phép thu riêng response bytes và DOM. Nếu hook guard
  thiếu/lỗi, output không đủ điều kiện thành capture.
- `browser_manager.py`: dùng context mới cho mỗi ID, tắt persistent profile,
  giữ kiểm chứng TLS. Không dùng LLM extraction, trả phí API hoặc tải model weights.

HTTP sử dụng Scrapling Chrome TLS, cả khi đọc robots. Redirect được theo bằng
từng hop có kiểm tra, retry mạng/5xx có giới hạn và giữ `Retry-After`. HTTP 403
hoặc HTML 200 thiếu bài có thể chuyển browser khi robots cho phép. Robots 200
trả HTML challenge được ghi `ROBOTS_INVALID_CONTENT`, không coi là allow-all.
Explicit Disallow đã xác nhận, 401/403 robots, rate limit và robots chưa đọc được
giữ trạng thái hold; đổi client không đồng nghĩa đã được phép tải bài.

## Checkpoint và đầu ra

Mỗi ID hoàn tất có marker cùng hash của robots evidence, HTTP responses,
browser response/DOM và text/structure nếu trích được. Các variant dùng path
riêng, giữ được HTTP shell và DOM chứa bài trong cùng attempt. Nếu runtime
ngắt trước marker, phiên sau tạo attempt mới; bytes thay đổi không ghi đè vào
attempt cũ. Marker đã hoàn tất được kiểm hash và dùng lại.

`MAX_NEW_IDS` giới hạn công việc mỗi phiên; cùng `RECOVERY_LABEL` sẽ resume.
Đổi label tạo thí nghiệm retry mới. Không chạy hai runtime cùng experiment.
Kết quả gồm `coverage_1000.csv`, `recovery_results.csv`, source inputs, manifest,
summary, raw assets, extraction candidates và `export_hashes.json`. Cuối
notebook xuất `remaining-stage-a-<experiment>.zip` lên Drive và tải xuống để
đối chiếu các URL còn thiếu với nội dung thực tế.

## Bằng chứng và giới hạn

Kiểm thử chạy adapter Crawl4AI thật trên Edge headless với response mô phỏng:
redirect bị robots cấm được chặn trước khi gửi, redirect được phép tới HTTP 403
giữ đúng response bytes, DOM và status. Unit/integration tests còn kiểm guard
setup lỗi, input ledger trên Drive, retry, checksum corruption và interruption
trước marker. Xem `VALIDATION.md` cho kết quả toàn suite.

Các kiểm thử đó chứng minh cơ chế adapter. Chưa có kết quả chạy 68 URL còn lại
trên Colab. Tool không phục hồi được URL đã chết, nội dung yêu cầu quyền truy
cập hoặc nguồn vẫn chặn truy cập bằng mọi transport được thử. Các trường hợp
đó vẫn nằm trong ledger để xử lý theo nguồn, không tạo văn bản thay thế.
