# Kiểm chứng data pipeline v2

Kiểm tra local trong Conda env `r2ai-stage3`, Python 3.11, ngày 02/10/2026.
Package editable đã cập nhật 0.2.0; PYTHONNOUSERSITE=1.

## Kết quả đã chạy

- **13 pytest tests passed**. Bao gồm ID không liên tục, duplicated URL giữ nhiều
  official IDs, official ID/URL mismatch, deterministic stratified sampling có
  rare formats/long-tail, source Unicode spans, parent containment, child identity
  độc lập parent size, global dedup xuyên shard và representation hash theo title.
- Kiểm tra resume/retry failed, checksum corruption, robots redirect/disallow,
  hard span failure không ghi done marker, parser bug không bị nuốt thành lỗi
  nguồn, duplicate input xuyên shard, byte budget, incomplete requested range
  không được duyệt, golden thiếu human assertions bị từ chối và snippet regression.
- Health baseline-relative warning, code/tokenizer/chunk-policy evidence gates,
  post-health artifact corruption bị bắt lại khi freeze. Không tự tạo qrels/pass.
- Bốn notebook qua nbformat và syntax Python, kể cả top-level await:
  00 có 11 cells; 01 có 9; 02 có 9; 03 có 10. Không lưu execution outputs.
- `pip check`: No broken requirements found.

## Tokenizer và mock workflow thực

Script `scripts/check_tokenizer.py` dùng BGE-M3 fast tokenizer đã cache ở revision
`5617a9f61b028005a4858fdac845db406aefb181`, local_files_only=True, không tải weights.

11 synthetic fixture cases đều pass: VI/EN/ZH, biomedical symbols/dosage, combining
Unicode, HTML table, XML/JATS section/table cells, source ngắn, PDF text/scan-like,
403/login/JS quarantine. Kiểm tra bằng tokenizer thật đã phát hiện và sửa token
SentencePiece gộp newline vào heading qua ranh giới section.

Bốn source dài Việt/Anh/Trung/mixed Unicode được kiểm với parent 256/512/640.
Child IDs không đổi khi chỉ đổi parent size; exact slices và token budgets đều
pass. Cấu hình parent 640 tạo tổng **104 child spans và 88 parent spans**; parent
on-demand 256 vẫn chứa đúng child nguồn.

Workflow offline HTTP mock với tokenizer thật đi qua crawl → extract/chunk →
Parquet → whole-snapshot validation → global aliases → health → freeze:
4 input IDs, 3 parsed documents, 1 HTTP 404; 64 children và 60 parents. Aliases
giữ 3 official document IDs thuộc 2 nhóm normalized content. Snapshot đạt
FROZEN_CANDIDATE; không có human milestone/retrieval approval giả.

## Chạy lại

    conda activate r2ai-stage3
    python -m pytest -q
    python scripts/check_notebooks.py
    python scripts/check_tokenizer.py --local-tokenizer <pinned-cache-directory> --report artifacts/tokenizer-validation.json
    python -m pip check

Token validation report được lưu trong artifacts/ và không commit. Không truyền
--local-tokenizer thì script tải tokenizer pinned từ Hub, vẫn không tải weights.

## Giới hạn bằng chứng

Crawl tests trong mục kiểm chứng ban đầu dùng HTTP mock, documents giả và
fixture synthetic. Kết quả Stage A và recovery thực từ Colab được ghi riêng
trong `reports/`; chưa review 100–500 nguồn thật hoặc crawl toàn corpus.
Metadata dataset/query/corpus đã kiểm ở revision pinned trong lần setup trước;
không coi metadata Hub là bằng chứng đã xử lý các URL.

Chất lượng extraction theo domain, coverage thực, storage/GPU forecasts, OCR,
official scorer compatibility, ANN recall và F2 cần artifact/labels/benchmark sau
Colab. Các gates hiện giữ những evidence chưa có ở trạng thái pending.

## Bổ sung ngày 03/10/2026: browser recovery và extraction

- **53 pytest tests passed**, gồm hai CDP tests chạy trên Edge với trang mô
  phỏng: chặn redirect trước request và lưu body HTTP 403 có kiểm robots.
- Adapter Long Châu được kiểm cho article boundaries, heading/paragraph order,
  nested lists, table/caption, inline dosage và combining Unicode. Canonical
  URL lệch/layout đổi/error page không được fallback thành văn bản giao diện.
- Mock integration đi qua crawl → domain adapter → sections → child/parent →
  Parquet validation, giữ official ID, raw hash và exact source spans.
- Recovery replay kiểm pinned CSV, marker/asset hashes, ID/URL, phát hiện
  capture sửa đổi, ghi provenance `response_bytes`/`rendered_dom` rõ ràng và
  resume output theo ID. Gói review chưa nhập vào canonical corpus.
- **9 notebooks** qua nbformat, empty-output checks và Python syntax.
- ZIP Colab đủ 11 ID có 22 method markers/56 assets đúng hash. Browser
  lấy được bài thật cho 11/11; 10 HTTP mới là Cloudflare challenge.
- Replay offline từ 11 browser response bodies giữ toàn bộ paragraph và
  heading trong article body; structure/hash checks và resume cả 11 đều pass.
  Không có table trong 11 body này; bảng/chú thích được kiểm bằng synthetic
  fixtures, vẫn cần source audit khi gặp bảng thật.

Xem `reports/stage-a-v2-colab-probe-4c49d3358aa818b7.md`. Coverage của các
domain còn lại, human review, reviewed ingestion và full-corpus budget chưa
được chứng minh bằng kết quả 11 URL này.

## Bổ sung ngày 03/10/2026: thu tiếp bằng Crawl4AI

- **68 pytest tests passed** trong runtime riêng dùng Python 3.11 và Crawl4AI
  0.9.4 từ commit `e5d2e786d1a101225f3f6a3e6fd344d76eeb13af`. Bao gồm
  bốn browser integration tests trên Edge headless: hai Scrapling tests cũ và
  hai Crawl4AI tests mới dùng response mô phỏng. Không sửa Conda env gốc để
  thêm dependency Crawl4AI.
- Crawl4AI redirect guard chặn đường dẫn bị robots cấm trước request; khi
  redirect được phép tới HTTP 403, status, actual response bytes và DOM đều
  được giữ. Setup/hook thiếu không biến `success=True` thành article capture.
- Loader của 01g được kiểm bằng workflow crawl thật với HTTP mock, Parquet,
  raw shard, triage, legacy assets và browser replay. Nó chọn đúng ID chưa
  tải, dùng lại capture và từ chối URL lệch mapping dù CSV hash đã cập nhật.
- Chrome robots transport kiểm retry 5xx/Retry-After và decoded bodies;
  robots 200 trả HTML challenge bị giữ. Robots 403/429/5xx hoặc Disallow đã
  xác nhận không được chuyển sang tải bài/browser.
- Recovery kiểm ledger đủ official IDs, resume không fetch lại ID hoàn tất,
  phát hiện checksum corruption, giữ riêng HTTP shell/raw browser/DOM và
  tiếp tục được khi ngắt sau asset nhưng trước marker, dù response mới đổi.
- **10 notebooks** qua nbformat, empty-output checks và Python syntax;
  notebook 01g có 10 cells, bootstrap/Drive/code lock và ZIP export riêng.

Chưa chạy 68 URL còn lại trong môi trường mạng Colab của người dùng. Các
kiểm thử mới chứng minh input/recovery/checkpoint/guard, chưa chứng minh
1.000 bài đã tải đủ hoặc toàn corpus có thể crawl. 917 HTTP captures,
11 browser captures và 4 legacy candidates vẫn cần extraction/content review
trước khi được tính vào valid-article coverage hoặc nhập canonical corpus.
