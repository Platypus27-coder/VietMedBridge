# Thu tiếp Stage A bằng Crawl4AI

Notebook pilot `notebooks/experiments/stage-a-1000/01g_colab_recover_remaining_stage_a.ipynb` chạy trên Colab CPU mới,
cùng `DATA_ROOT` của các bước trước. Bootstrap clone repo, cài package và khóa
commit riêng trong `advanced_recovery_code_lock.json`. Không đổi code lock của
raw Stage A. Notebook tự làm bước replay 01f; không cần chạy lại 01e/01f trước.
Đây là công cụ thí nghiệm cho đúng mẫu Stage A 1.000 ID, chưa phải recovery
workflow tổng quát cho lượt crawl corpus lớn.

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
- `browser_manager.py`: quản lý browser/context; bản nâng cấp giữ session theo
  domain trong pool nhỏ và giữ kiểm chứng TLS. Không dùng LLM extraction,
  trả phí API hoặc tải model weights.

HTTP sử dụng Scrapling Chrome TLS, cả khi đọc robots. Redirect được theo bằng
từng hop có kiểm tra, retry mạng/5xx có giới hạn và giữ `Retry-After`. HTTP 403
hoặc HTML 200 thiếu bài có thể chuyển browser khi robots cho phép. Robots 200
trả HTML challenge được ghi `ROBOTS_INVALID_CONTENT`, không coi là allow-all.
Explicit Disallow đã xác nhận, 401/403 robots, rate limit và robots chưa đọc được
giữ trạng thái hold; đổi client không đồng nghĩa đã được phép tải bài.

## Nhánh recovery nâng cấp

Bản `advanced-v1` mở rộng các chức năng bên trên theo dữ liệu thất bại thực tế:

- Scrapling `FetcherSession` giữ HTTP cookies/connection state; Crawl4AI
  `RateLimiter` điều chỉnh tốc độ sau lỗi. `Retry-After` dài và HTTP 429 không
  bị bỏ qua bằng cách chuyển sang engine khác.
- Robots HTTP 403/HTML challenge/408/5xx/lỗi kết nối có nhánh browser riêng:
  Scrapling stealth trước, Crawl4AI stealth sau. Chỉ raw response HTTP 200 có
  nội dung robots nhận diện được mới đưa vào resolver. DOM chứa `<pre>`, trang
  lỗi hoặc `success=True` không đủ. Đọc file robots có giới hạn redirect và
  request, không phụ thuộc vào đọc lại chính robots và không mở bài chưa xét.
- Article HTTP 403/404/408/5xx/lỗi kết nối hoặc HTML thiếu bài được thử qua hai
  browser. Engine từng lấy được robots/bài trên domain đó được ưu tiên, để dùng
  lại cookies. 404 vẫn cần bằng chứng bài ở response cuối 200 mới tạo candidate.
- Crawl4AI bật `enable_stealth`; Scrapling dùng `AsyncStealthySession`/Patchright.
  Cài guard trực tiếp trước navigation vì callback `page_setup` của Scrapling
  có thể nuốt exception. Solver Cloudflare của bản Scrapling pinned được gọi
  khi phát hiện challenge và bọc timeout, không xem solver return là thành công.
- Chờ selector bài đạt ngưỡng nội dung, cuộn tối đa ba bước và ghi trạng thái
  ổn định. Profile theo domain có thể đổi selector, thời gian chờ, số bước cuộn,
  resource hosts và `block_ads`. Engine nào cũng phải qua extraction trước khi
  được coi là candidate; vẫn cần review.
- Cho phép tài nguyên tĩnh HTTPS công khai có kiểm robots; giữ riêng phạm vi
  document và resource, theo redirect HTTP công khai có kiểm tra. Dùng danh sách
  ad domains của Scrapling để chặn tracking không phục vụ nội dung. Request
  không thuộc phạm vi và private IP bị chặn. Challenge endpoints có giới hạn
  riêng cho frame/POST. Không lấy text từ iframe thành bài chính.
- Duy trì pool LRU hai browser theo engine/domain. Các robots bootstrap từ
  nhiều CDN được xếp tuần tự để không mở hàng chục browser. Profile chỉ nằm ở
  WORK_DIR, không nhập profile trình duyệt cá nhân và không xuất cookies vào ZIP.
- Bật thu response CDP rõ ràng vì Patchright có thể không phát Playwright
  response event. Lưu bytes/status của document chính, DOM riêng, screenshot
  khi lỗi và request log. Hủy trang sẽ hủy các robots request đang chờ, đợi
  request đang chạy kết thúc trước khi đóng HTTP session.

Các thư viện browser được pin: Scrapling 0.4.15, Crawl4AI tại commit trên,
Playwright/Patchright 1.63.0, playwright-stealth 2.0.3 và curl_cffi 0.16.3.
Cài toàn bộ dependencies trước khi import các parser để tránh nâng native
library giữa một kernel đang sử dụng chúng.

## Checkpoint và đầu ra

Mỗi ID hoàn tất có marker cùng hash của robots evidence, HTTP responses,
browser response/DOM và text/structure nếu trích được. Các variant dùng path
riêng, giữ được HTTP shell và DOM chứa bài trong cùng attempt. Nếu runtime
ngắt trước marker, phiên sau tạo attempt mới; bytes thay đổi không ghi đè vào
attempt cũ. Marker đã hoàn tất được kiểm hash và dùng lại.

`MAX_NEW_IDS` giới hạn công việc mỗi phiên; cùng `RECOVERY_LABEL` sẽ resume.
Đổi label tạo thí nghiệm retry mới. Không chạy hai runtime cùng experiment.
Kết quả gồm `coverage_1000.csv`, `coverage_by_domain.csv`, `unresolved_urls.csv`,
`recovery_results.csv`, source inputs, manifest,
summary, raw assets, extraction candidates và `export_hashes.json`. Cuối
notebook xuất `remaining-stage-a-<experiment>.zip` lên Drive và tải xuống để
đối chiếu các URL còn thiếu với nội dung thực tế.

## Bằng chứng và giới hạn

Kiểm thử chạy adapter Crawl4AI thật trên Edge headless với response mô phỏng:
redirect bị robots cấm được chặn trước khi gửi, redirect được phép tới HTTP 403
giữ đúng response bytes, DOM và status. Unit/integration tests còn kiểm guard
setup lỗi, input ledger trên Drive, retry, checksum corruption và interruption
trước marker. Xem `VALIDATION.md` cho kết quả toàn suite.

Hai engine nâng cấp đã được thử trên URL Long Châu ID 206172 ngoài thực tế,
đều HTTP 200, không guard error và trích cùng văn bản 5.934 ký tự, cùng SHA-256
`bc7b09db548d689f6cb2932700ea2a2d214fb363132aa77d572d722bbda5fc27`.
Đây là một URL đã biết, không được tính thành một ID mới trong nhóm 68.

Các kiểm thử đó chứng minh cơ chế adapter. Chưa có kết quả chạy 68 URL còn lại
trên Colab. Tool không phục hồi được URL đã chết, nội dung yêu cầu quyền truy
cập hoặc nguồn vẫn chặn truy cập bằng mọi transport được thử. Các trường hợp
đó vẫn nằm trong ledger để xử lý theo nguồn, không tạo văn bản thay thế.
