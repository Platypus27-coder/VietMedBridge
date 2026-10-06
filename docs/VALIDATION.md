# Kiểm chứng data pipeline v2

Kiểm chứng retrieval baseline ngày 05/10/2026 nằm trong
[RETRIEVAL_BASELINE.md](RETRIEVAL_BASELINE.md): 21 tests liên quan pass, candidate
thật được đọc bằng tokenizer pinned và notebook 04 được smoke test bằng model
giả. Artifact inference GPU v1 thực của Sếp đã được audit riêng; chưa có official
score. Không dùng kết quả CPU giả lập để chứng minh chất lượng GPU.

## End-to-end competition pilot — 06/10/2026

- **137 tests passed, 9 skipped** trong regression CPU. Skips thuộc browser/
  Crawl4AI tùy chọn; không tải model weights hoặc chạy inference GPU local.
- Test mới đi qua 1.200 query IDs không liên tục: embedding, FAISS/BM25 thực,
  translation, document/child rerank, parent output và ZIP. Encoder/translator/
  reranker dùng test doubles; predictions này không dùng submit BTC.
- Ngắt sau 17 queries rồi resume đủ 1.200: không tính lại translations hoặc
  scored stages hoàn tất. Export từ run chưa đủ query bị từ chối.
- Binding yêu cầu đúng master plan, hash frozen catalog/query set/config;
  plan khác hoặc thay nội dung plan trong cùng run bị từ chối. Notebook mặc
  định MAX_NEW_*=None, giữ tất cả official queries và kiểm ZIP hash trước download.
- ZIP chỉ có `results.json` ở root, CRC/schema/IDs/exact source pass. Export
  lặp lại cùng predictions giữ ZIP hash, feedback đã ghi không bị reset; feedback
  của ZIP khác bị từ chối. Thiếu hoặc hỏng optional labels không chặn ZIP hợp lệ.
- Cả 7 notebook active/optional qua nbformat, Python syntax và empty-output
  checks; notebook 04 hiện có 17 cells và không phát sinh notebook phụ.
- Đây là kiểm kỹ thuật end-to-end trên CPU. Chưa có lượt Qwen/BGE cascade mới
  chạy thật trên Colab, chưa upload ZIP mới hoặc nhận điểm BTC. Run GPU v1 cũ
  giữ evidence riêng; không được gán thành kết quả của workflow mới.

## Retrieval cascade v2 — 06/10/2026

- Bộ regression CPU: **132 passed, 9 skipped**; skips thuộc browser/Crawl4AI
  integration tùy chọn. Sau khi siết cache fingerprint/label schema, nhóm
  retrieval/cascade chạy lại **23 passed**, gồm mở lại index sau JSON roundtrip
  với IDs khác độ dài để kiểm resume qua runtime mới.
- Kiểm translation schema/numbers/acronyms/constraint cues, partial translation
  canary resume cùng signature, document/child stage disconnect/reuse, giữ aliases,
  quotas, quarantine ledger, source-preserving export và checksum tampering.
- Posting BM25/FAISS là implementation thực trên CPU; reranker/translator là
  test doubles. Test LCS so với dynamic programming, same-doc-only relevance,
  dedup trước precision, macro F2 theo từng query, dev-only cutoff sweep và recall.
- Dữ liệu thật: đọc lại frozen candidate 864 documents/9.076 children/8.760 units
  cùng matrices BGE-M3 từ Colab (8.760×1.024 và 1.200×1.024). Canary 3 queries
  kiểm **1.040 reranker pairs**, parent spans/IDs/ZIP và resume zero model calls.
  Không inference GPU mới, không đánh giá F2/relevance bằng các điểm giả lập.
- Kiểm MaxP bằng query thật dài nhất với **toàn 9.076 children**: cặp tokenized
  dài nhất **512**, không vượt passage budget. Tokenizer BGE pinned thực dùng
  local cache; không tải model weights.
- Source holds: 9 documents (4 encoded payload + 5 article→homepage redirects)
  được ghi ledger và giữ khỏi retrieval. Frozen sources/IDs không bị xóa. Còn
  855 eligible docs; content-language routing cho 429 VI/426 ZH, **0 EN** trong
  corpus này. Đây là coverage diagnostic, không phải chất lượng toàn corpus.
- 7 notebook active/optional qua nbformat, Python syntax và empty-output checks;
  chỉ regenerate notebook 04. Evidence CPU thật + test doubles nằm local ignored
  `artifacts/retrieval-cascade-check/summary.json`.

Chưa đo Qwen translation hoặc cascade reranker thực trên Colab, chưa có BTC scorer/
labels. Không tuyên bố kiến trúc mới có F2 tốt hơn hoặc đã đạt best configuration.

### Rà theo plan, validator v2.1

Tái hiện regression fail khi original có `metformin` trong `entities` nhưng bản
dịch thay thuốc khác vẫn pass. Sau sửa kiểm Latin entities/word boundaries và
intolerance cue, **24 retrieval/cascade tests pass**. Notebook schema/syntax/
empty-output checks cũng pass. Namespace/lock v2.1 không reuse translation/
score checkpoints từ validator v2; cache vectors v1 giữ nguyên. Đây là validation
CPU, không phải semantic translation benchmark/GPU inference mới.

## Review và merge branch Lao Động ngày 05/10/2026

Đã review đến commit `48a5479`: article adapter và cookie recovery Lao Động,
language/homepage/encoded-payload guards, site cleanup và independent team crawl.
Trong review đã tái hiện và sửa việc loại nhầm section `Related work`/recommended
treatment, kể cả bộ lọc class của Trafilatura, và cắt đuôi bài khi một cụm UI
tiếng Trung xuất hiện giữa câu. Tail marker nay phải là heading riêng một dòng.

Report cells đọc file trong manifest đã kiểm hash, không glob các attempt cũ.
Team merge kiểm thêm decoded body hash/length trước khi công bố run. Notebook
tùy chọn dùng code lock/runtime riêng và tên build mới; main workflow giữ 00–04.

- **117 tests pass, 9 tests tùy chọn skip** do thiếu browser/crawl4ai runtime.
  Tests cover crawl/robots/recovery, giữ source evidence, team merge/resume,
  response corruption và stale Parquet không được nhập vào report.
- 7 notebooks active và 3 notebooks quality/recovery mới archive qua nbformat,
  empty-output và syntax checks. Không chạy crawler tới website thật.
- Reader mới vẫn xác minh 864 documents, 9.076 children, 6.906 parents và 8.760
  representations trong candidate v3 hiện có. 04 smoke test và resume dùng real
  data/queries/index, inference test doubles; chưa chạy GPU weights/official score.
- Replay saved FamilyDoctor raw ID 2474168 giữ đủ nội dung clinical trong
  `#viewContent`. Kết quả này chưa phải human audit cho toàn corpus.

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

## Bổ sung ngày 03/10/2026: hai engine recovery và robots bootstrap

Full regression đã chạy **84 tests pass**; sau khi bổ sung kiểm tra concurrency,
nhóm advanced recovery chạy lại **16/16 pass**. Bốn integration cases mới chạy
Crawl4AI stealth và Scrapling/Patchright thật trên Edge headless, dùng response
mô phỏng: JS từ CDN tải nội dung muộn, cookies tồn tại giữa hai URL cùng domain,
và robots response bytes được giữ nguyên. Các kiểm thử còn chứng minh:

Cell cài đặt notebook ghim Crawl4AI 0.9.4 trên PyPI, Scrapling 0.4.15 và các
phiên bản browser đã kiểm thử; pip không còn bị chạy ở chế độ im lặng để lỗi
dependency nếu có sẽ hiện ngay trong output Colab.

- HTTP robots 403 có thể chuyển sang raw robots 200 bằng browser, nhưng
  Disallow thực vẫn chặn bài; DOM/HTML error/plain-text error không thành allow-all.
- Auth/rate limit/Retry-After dài không bị bỏ qua để chạy browser tiếp.
- HTTP connection error và HTTP 403 đi qua hai engine, giữ assets của cả hai,
  không gán raw bytes cũ cho engine mới, resume không fetch lại ID hoàn tất.
- Robots bootstrap không tự điều hướng vào bài, chặn private IP, và các bootstrap
  từ nhiều CDN được xếp tuần tự/cache để giới hạn số browser.
- Hủy browser không phát thêm các robots request đang xếp hàng; HTTP request
  đang chạy được đợi hoàn tất trước khi đóng session.

Thử ngoài thực tế tại URL Long Châu official ID **206172**, ngày 03/10/2026
19:31–19:32 Asia/Saigon: cả hai engine HTTP 200, không guard error, trích đúng
**5.934 ký tự** với cùng source-text SHA-256
`bc7b09db548d689f6cb2932700ea2a2d214fb363132aa77d572d722bbda5fc27`.
Raw response hashes khác nhau do bytes trả về giữa các lần truy cập khác nhau:

- Crawl4AI: `8e3dd415d6041ec033a6ffbd1ec609ce8f727e8e70b2415b296c93faf2414835`.
- Scrapling: `e1dab819e62a27efd126505af8dc784e66da4aacdd323ac73f83418658690b04`.

Raw/DOM/log lưu local trong `data/advanced_live_final/`, không commit. Lượt này
kiểm adapter mới trên một URL đã biết; không cộng thêm vào coverage nhóm 68.
Notebook 01g được sinh lại và cả 10 notebook qua schema/syntax/empty-output checks.
Kết quả 68 URL còn thiếu vẫn cần chạy trên Colab, cùng Drive của run Stage A.
