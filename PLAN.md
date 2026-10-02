# PLAN — ViBioMIR Full-Corpus Data Pipeline
## Verified, Recoverable, Quality-Gated Crawling → Canonical Corpus → Child/Parent Chunking → Hybrid Indexing

> **Status:** Kế hoạch triển khai có điều kiện — hợp nhất kiến trúc R2AI Stage 3, `PLAN v2`, Data Pipeline Supplement, pilot 1.000 URL và đánh giá Scrapling. Các gate dữ liệu nguồn/độ phủ và A/B Colab vẫn cần bằng chứng trước khi scale.
>
> **Mục tiêu:** xây một data pipeline có thể xử lý vài triệu URL mà **không phải tin mù vào output**, có khả năng tự kiểm tra liên tục, tự dừng khi invariant bị phá, retry/recover các lỗi kỹ thuật có thể phục hồi, bảo toàn `doc_id` của BTC và tạo corpus/chunk đủ chất lượng cho document + chunk retrieval của ViBioMIR.
>
> **Nguyên tắc tối cao:** `CORRECTNESS → COVERAGE → QUALITY → RETRIEVABILITY → SCALE → SPEED`.

---

# 0. Executive Summary

Pipeline cuối cùng được chốt theo luồng:

```text
links_corpus.parquet + query.parquet
            │
            ▼
      DATASET SNAPSHOT
            │
            ▼
       URL INVENTORY
            │
            ▼
      ROBOTS RESOLVER
            │
            ▼
   MULTI-TIER CRAWL/FETCH
            │
     ┌──────┴─────────┐
     ▼                ▼
 RAW ARCHIVE      CRAWL MANIFEST
     │
     ▼
 CONTENT ROUTER
 HTML / PDF / TEXT
     │
     ▼
 MAIN CONTENT EXTRACTION
     │
     ▼
 IMMUTABLE SOURCE_TEXT
     │
     ├── exact provenance
     ├── extractor version
     └── source hash
     │
     ▼
 CONSERVATIVE NORMALIZATION
     │
     ▼
 LANGUAGE + QUALITY + STRUCTURE
     │
     ▼
 EXACT CONTENT DEDUP
 (preserve every BTC doc_id)
     │
     ▼
 STRUCTURE PARSING
 section → paragraph → sentence
     │
     ├───────────────┐
     ▼               ▼
 CHILD CHUNKS     PARENT SPANS
 retrieval       submission context
     │               │
     └──────┬────────┘
            ▼
    INTEGRITY + QUALITY GATES
            │
            ▼
       ATOMIC COMMIT
            │
            ▼
         FREEZE
            │
            ▼
 RETRIEVAL/RESOURCE BENCHMARK
            │
            ▼
      RESOURCE GATE
       /         \
 FULL EMBED   DOC-FIRST SELECTIVE
       \         /
            ▼
       BM25 + FAISS
            │
            ▼
     RETRIEVAL REGRESSION
            │
            ▼
        PROMOTED CORPUS
```

Không có stage nào được phép “chạy xong rồi mới kiểm”. Mỗi shard đi theo:

```text
PROCESS
   ↓
VALIDATE
   ↓
QUALITY CHECK
   ↓
PASS ?
├── YES → ATOMIC COMMIT → NEXT SHARD
└── NO  → STOP/PAUSE → REPORT → FIX → RERUN SHARD ONLY
```

---

# 1. Current Pilot Baseline — 1.000 URL

Kết quả crawl **ban đầu**, trước lượt retry có kiểm soát:

| Trạng thái | URL | Tỷ lệ trên 1.000 |
|---|---:|---:|
| Crawl thành công | 912 | 91.2% |
| `robots_blocked` | 54 | 5.4% |
| `robots_unavailable` | 13 | 1.3% |
| HTTP 403 | 15 | 1.5% |
| Timeout | 5 | 0.5% |
| HTTP 404 | 1 | 0.1% |

Tổng URL chưa crawl được:

```text
88 / 1000 = 8.8%
```

Không được suy từ bảng trên rằng toàn bộ `robots_unavailable` là policy block:

```text
54 / 1000 = 5.4%  robots.txt đã xác nhận Disallow
13 / 1000 = 1.3%  chưa biết lý do robots.txt không khả dụng
21 / 1000 = 2.1%  HTTP/network outcome của URL đích
```

`robots_unavailable` có thể là 4xx, 5xx, timeout hoặc lỗi mạng; HTTP 403
cũng có thể là chính sách truy cập, không mặc nhiên là lỗi kỹ thuật. Vì vậy
**chưa thể tính tỷ lệ technical failure chính xác** từ CSV ban đầu.

Sau một lượt retry, Colab báo **917 thành công và 83 chưa lấy được nội dung**.
Phân bố lỗi sau retry phải đọc từ `failures_after_retry.csv`; không dùng số
88 ban đầu làm quyết định recovery hiện tại. Mẫu 1.000 URL phân tầng cũng
không được ngoại suy trực tiếp thành tỷ lệ lỗi của 4,39 triệu URL.

## 1.1. Quyết định từ pilot

1. Không thay toàn bộ crawler chỉ vì 88 URL fail.
2. `robots_blocked` không được coi là technical failure.
3. `robots_unavailable` phải được phân loại chi tiết theo RFC 9309, không dùng một bucket chung.
4. HTTP 403 là nhóm ưu tiên A/B test với Scrapling recovery tier.
5. Timeout dùng retry/backoff/session tuning trước khi browser fallback.
6. HTTP 404 đánh dấu dead source; không dùng browser để cố cứu.
7. Tất cả retry/recovery phải đo bằng **valid article recovery**, không chỉ HTTP 200.

---

# 2. Competition-Specific Hard Constraints

Data pipeline phải phục vụ trực tiếp submission format của ViBioMIR.

## 2.1. BTC `doc_id` là immutable primary key

Input chính thức:

```text
links_corpus.parquet
├── id: integer
└── url: string
```

Quy ước nội bộ:

```text
btc_doc_id := links_corpus.id
```

Không được:

```text
reset_index()
DataFrame row number làm doc_id
hash URL thay BTC doc_id
canonical_content_id thay BTC doc_id khi submission
```

Mọi object downstream phải trace được về BTC ID:

```text
BTC doc_id
  ↓
crawl record
  ↓
raw asset
  ↓
processed document
  ↓
section
  ↓
child chunk
  ↓
parent span
  ↓
embedding/vector ID
  ↓
retrieval result
  ↓
BTC doc_id
```

## 2.2. Submission chunk phải là source-derived text

`chunk_text` không được là:

- translation;
- HyDE passage;
- LLM rewrite;
- Markdown do renderer thêm ký hiệu;
- text tổng hợp từ nhiều vị trí không contiguous;
- `title + section + body` nếu title/section không nằm trong source span đó.

Contract bắt buộc:

```python
chunk_text == document.source_text[start_char:end_char]
```

Nếu contract này fail dù **1 chunk**, shard đó không được promote.

## 2.3. External corpus không mở rộng submission document universe

External biomedical resources chỉ dùng cho:

- query understanding;
- aliases/terminology;
- training;
- augmentation;
- reranker hard negatives;
- evaluation phụ trợ.

Document candidate cuối để submit phải thuộc `links_corpus.parquet`.

---

# 3. Design Principles

## 3.1. Raw-first, replayable pipeline

Sai:

```text
URL → extract → discard response
```

Đúng:

```text
URL
 ↓
RAW RESPONSE ARCHIVE
 ↓
extractor v1/v2/v3
```

Lợi ích:

- thay extractor không crawl lại;
- debug 403/challenge/error pages;
- chống link rot;
- audit được source;
- test extractor mới offline;
- reproducibility.

## 3.2. Freeze trước embedding

```text
crawl
→ extract
→ source_text
→ clean/structure
→ validate
→ dedup mapping
→ child/parent
→ FREEZE
→ embed/index
```

Không embed trực tiếp trong crawler.

## 3.3. QA nằm bên trong pipeline

QA không phải task cuối cùng.

```text
shard build
→ invariant checks
→ quality checks
→ drift checks
→ report
→ commit
```

## 3.4. Không scale bằng niềm tin

Mỗi lần tăng quy mô phải có gate:

```text
Golden
→ 1k
→ 10k
→ 100k
→ 1M
→ Full
```

Không nhảy từ 1k thẳng vài triệu nếu Stage B chưa chứng minh stability.

---

# 4. Source Snapshot & Dataset Identity

Input:

```text
data/source/links_corpus.parquet
data/source/query.parquet
```

Tạo:

```text
data/manifests/dataset_manifest.json
```

Schema:

```json
{
  "dataset_name": "ViBioMIR",
  "snapshot_time_utc": "...",
  "links_path": "...",
  "links_rows": 0,
  "links_schema": {},
  "links_sha256": "...",
  "query_path": "...",
  "query_rows": 1200,
  "query_schema": {},
  "query_sha256": "...",
  "pipeline_git_commit": "..."
}
```

Hard gates:

```text
links.id not null
links.id unique
query.id not null
query.id unique
expected schema exact
checksum unchanged during a corpus build
```

Nếu source checksum đổi giữa run:

```text
IMMEDIATE STOP
```

---

# 5. URL Inventory & Domain EDA

Không crawl full ngay.

Tạo:

```text
data/inventory/url_inventory/
data/inventory/domain_stats.parquet
```

Fields:

```text
doc_id
original_url
canonical_url
scheme
host
domain
path
extension
query_string_present
url_valid
likely_content_type
```

Không dùng URL canonicalization để đổi BTC identity. `canonical_url` chỉ là operational metadata.

Required reports:

```text
URLs/domain
HTTP vs HTTPS
extension distribution
exact duplicate URL count
top 20 domains
top 100 domains
long-tail domains
PDF-like URLs
language hints from TLD/path
```

Output phải giúp trả lời:

1. Bao nhiêu domain phải crawl?
2. Domain nào chiếm phần lớn corpus?
3. Domain nào đáng viết extractor riêng nếu extraction lỗi?
4. Bao nhiêu PDF / HTML / unknown?
5. Duplicate URLs có đáng kể không?

---

# 6. Robots Resolver — P0

Đây là thay đổi bắt buộc từ pilot.

## 6.1. Không dùng một status `robots_unavailable`

Phải tách:

```text
ROBOTS_OK_ALLOWED
ROBOTS_OK_DISALLOWED
ROBOTS_404
ROBOTS_401
ROBOTS_403
ROBOTS_429
ROBOTS_4XX_OTHER
ROBOTS_5XX
ROBOTS_TIMEOUT
ROBOTS_DNS_ERROR
ROBOTS_SSL_ERROR
ROBOTS_CONNECTION_ERROR
ROBOTS_REDIRECT_LIMIT
ROBOTS_PARSE_PARTIAL
ROBOTS_PARSE_FAILED
```

## 6.2. RFC 9309 semantics

Implementation baseline:

### Robots fetch thành công

```text
2xx + parseable robots
→ follow rules
```

Nếu path `Disallow`:

```text
ROBOTS_OK_DISALLOWED
→ POLICY_BLOCKED
→ không fetch page
```

### 4xx robots resource unavailable

Theo RFC 9309, 400–499 khi fetch `/robots.txt` có thể biểu thị resource
unavailable và crawler **MAY access** tài nguyên server. `MAY` không bắt buộc
project phải truy cập. HTTP 429 còn báo hiệu giới hạn tốc độ, có thể kèm
`Retry-After`; không được diễn giải thành tín hiệu mở quyền crawl.

Policy của project:

```text
ROBOTS_404 / ROBOTS_410
→ ALLOW_BY_RFC_UNAVAILABLE
→ crawl page bằng polite rate limits; audit điều khoản nguồn trước khi scale domain

ROBOTS_401 / ROBOTS_403 / ROBOTS_4XX_OTHER
→ HOLD_FOR_SOURCE_REVIEW
→ không tự động fetch page

ROBOTS_429
→ RATE_LIMITED
→ tôn trọng Retry-After/backoff rồi kiểm tra robots lại
```

Nhưng giữ audit reason rõ ràng.

### 5xx / network unreachable

```text
ROBOTS_5XX / timeout / DNS / connection
→ UNREACHABLE
→ assume complete disallow for current crawl window
```

Không bypass bằng browser.

### Redirect

Follow tối thiểu logic tương thích RFC; giới hạn rõ ràng, log chain.

## 6.3. Robots cache

Cache theo authority/domain:

```text
domain → robots_state
```

Không fetch `/robots.txt` cho mỗi document.

Schema:

```text
domain
robots_url
fetch_status
http_status
fetched_at
expires_at
content_hash
parser_version
rules_summary
crawl_delay
request_rate
```

Cache TTL mặc định không vượt 24h trong điều kiện bình thường; unreachable có thể
dùng cached copy theo policy. Không cache một lỗi tạm thời thành `allow-all`.

## 6.4. Robots guardrail

Không dùng Scrapling/Stealthy/browser để bypass một path đã được xác định `Disallow`.

Scrapling có thể được dùng cho:

- allowed pages bị 403;
- JS rendering;
- retry/rate limit;
- browser compatibility.

Không dùng để biến:

```text
ROBOTS_OK_DISALLOWED
```

thành fetch attempt.

---

# 7. Multi-Tier Crawl Architecture

Crawler phải ưu tiên cheap path trước.

```text
URL
 ↓
ROBOTS RESOLVER
 ↓
ALLOWED?
 ├── NO → POLICY_BLOCKED
 └── YES
       ↓
TIER 1 FAST HTTP
       ↓
VALID?
├── YES → RAW COMMIT
└── NO
     ↓
CLASSIFY FAILURE
     ↓
 ┌──────────────┬────────────┬───────────────┐
 ▼              ▼            ▼               ▼
403/BLOCK      429        TIMEOUT         JS_EMPTY
 ▼              ▼            ▼               ▼
TIER 2       THROTTLE      RETRY         TIER 3
browser-like   + retry     BACKOFF       browser render
HTTP             │            │               │
 └───────────────┴────────────┴───────┬───────┘
                                      ↓
                                VALID ARTICLE?
                              ┌───────┴────────┐
                              YES              NO
                               ↓                ↓
                          RAW COMMIT       QUARANTINE
```

## 7.1. Source access and content coverage

`links_corpus.parquet` chỉ cung cấp `id/url`; ghi đủ outcome cho mọi ID
**không đồng nghĩa** đã có nội dung để index. Báo cáo bắt buộc tách:

```text
id_accounting_rate = IDs có terminal outcome / official IDs được yêu cầu
valid_article_coverage = IDs có source-derived article hợp lệ / official IDs được yêu cầu
```

Không promote theo `id_accounting_rate` một mình. Trước 10k, team cần chốt
ngưỡng `valid_article_coverage` cho từng milestone dựa trên số liệu theo domain.

Với domain thực sự Disallow, 403 kéo dài hoặc URL chết, mở access ledger:

1. Xin BTC snapshot nội dung/WARC gắn `doc_id` hoặc mirror/API và điều kiện dùng.
2. Kiểm tra API, feed, bulk export hoặc bản lưu của **cùng tài liệu** được phép dùng.
3. Nếu chưa có nguồn hợp lệ, giữ `doc_id` ở trạng thái unresolved và báo rõ
   coverage gap; không thay bằng bài khác hay text do mô hình tạo.

Mỗi recovery lưu `original_url`, `final_url`, `source_method`, thời điểm,
raw/content hash và bằng chứng mapping về BTC ID. Browser-rendered DOM phải
được gắn `capture_kind=rendered_dom`, không gọi là HTTP response bytes gốc.

---

# 8. Scrapling Integration Strategy

Scrapling là **candidate crawl/fetch framework**, không phải canonical extractor và không phải permission bypass.

## 8.1. Features relevant to ViBioMIR

Theo official repository/docs hiện tại, Scrapling cung cấp:

- concurrent Spider API;
- per-domain throttling;
- AutoThrottle;
- blocked request detection/retry;
- multiple HTTP/browser sessions;
- pause/resume checkpointing;
- robots compliance option + per-domain cache;
- dynamic/browser fetching;
- response replay/development mode.

## 8.2. Vai trò đề xuất

### Option A — giữ crawler hiện tại làm Tier 1

```text
Current fast crawler
       ↓ fail
Scrapling recovery tier
```

Đây là lựa chọn ít rủi ro nhất trong ngắn hạn.

### Option B — Scrapling Spider làm orchestration layer

```text
Scrapling Spider
├── fast session
├── browser-like HTTP session
└── dynamic browser session
```

Sau pilot A/B nếu Scrapling cho stability/resume/throttling tốt hơn, có thể promote Option B.

## 8.3. Không bật browser cho full corpus

Browser chỉ fallback:

```text
normal HTTP
→ failure classification
→ browser only when justified
```

Không:

```text
4M URLs → StealthyFetcher/browser
```

## 8.4. Scrapling recovery experiment

Từ pilot hiện tại lấy:

```text
15 HTTP 403
5 timeout
13 robots_unavailable (sau khi phân loại RFC)
```

### 403 matrix

Test:

```text
A. current HTTP
B. Scrapling fast/browser-like HTTP
C. Scrapling DynamicFetcher/browser session
```

Measure:

```text
HTTP success
valid article recovery
latency
CPU
RAM
browser startup cost
response bytes
block/challenge detection
```

**Metric chính:**

```text
VALID_ARTICLE_RECOVERY_RATE
```

Không dùng `HTTP_200_RATE` làm metric duy nhất.

### Timeout matrix

Test:

```text
retry=1/2/3
connect/read timeout
exponential backoff
jitter
connection pooling
per-domain concurrency
```

Browser fallback chỉ thử nếu evidence cho thấy HTTP endpoint phụ thuộc browser/session.

## 8.5. Promotion criteria cho Scrapling

Promote Scrapling recovery nếu:

```text
403 recovery tăng đáng kể
AND recovered body là article thật
AND resource cost chấp nhận được
AND resume/checkpoint reliable
AND không làm giảm polite/compliance guarantees
```

---

# 9. Crawl Failure Taxonomy

Không dùng `crawl_failed=true` chung chung.

Required status:

```text
PENDING
SUCCESS

POLICY_ROBOTS_DISALLOWED
ROBOTS_4XX_UNAVAILABLE_ALLOWED
ROBOTS_UNREACHABLE

HTTP_401
HTTP_403
HTTP_404
HTTP_429
HTTP_5XX

TIMEOUT_CONNECT
TIMEOUT_READ
DNS_ERROR
SSL_ERROR
CONNECTION_ERROR
REDIRECT_LOOP

BLOCK_PAGE
CAPTCHA_OR_CHALLENGE
JS_EMPTY
LOGIN_REQUIRED
PAYWALL_OR_AUTH_REQUIRED
EMPTY_RESPONSE
UNSUPPORTED_TYPE

RETRY_EXHAUSTED
QUARANTINED
```

Mỗi failure phải có:

```text
reason_code
attempt_count
last_method
last_http_status
last_error
first_attempt_at
last_attempt_at
```

---

# 10. Crawl Manifest

Schema tối thiểu:

```text
doc_id: int64
original_url: string
final_url: string
domain: string

robots_state: string
robots_reason: string

crawl_status: string
http_status: int?
content_type: string?

fetch_tier: string
fetcher_name: string
session_type: string

raw_archive: string?
raw_member: string?
raw_sha256: string?
raw_size_bytes: int?
compressed_size_bytes: int?

attempts: int
elapsed_ms: int
started_at
finished_at

error_code: string?
error_message: string?
```

Hard check:

```text
exactly one current terminal crawl state per doc_id
```

History/retries có thể lưu riêng trong `crawl_attempts`.

---

# 11. Raw Storage Architecture

Không tạo hàng triệu file trong một directory.

Ưu tiên một trong:

```text
raw/archive_000000.tar.zst
raw/archive_000001.tar.zst
...
```

hoặc shard directory đủ nhỏ:

```text
raw/shard_0000/...
```

Khuyến nghị archive-based khi filesystem/inode là concern.

## 11.1. Raw index

```text
doc_id
archive_id
member_name / byte_offset
mime_type
sha256
compressed_size
uncompressed_size
```

## 11.2. Atomic raw commit

```text
write temp
→ checksum
→ fsync/close
→ atomic rename
→ manifest status DONE
```

Không publish half-written raw artifact.

---

# 12. Content Router

Phân loại response:

```text
HTML
PDF
plain text
JSON/XML if meaningful
unsupported binary
```

Dựa trên:

- HTTP `Content-Type`;
- magic bytes;
- extension only as weak hint.

Không tin extension tuyệt đối.

---

# 13. Extraction Pipeline

## 13.1. HTML

Baseline:

```text
raw HTML
↓
Trafilatura main-content extraction
↓
fallback parser/custom domain extractor
```

Scrapling parser có thể hỗ trợ DOM access nhưng không tự động thay Trafilatura làm canonical main-content extractor.

## 13.2. PDF

Baseline:

```text
PyMuPDF
```

Phải log:

```text
page_count
extracted_char_count
chars_per_page
possible_scan_pdf
```

OCR không bật full mặc định.

Detection P0:

```text
PDF nhiều page + gần như không text
→ POSSIBLE_SCAN_PDF
```

OCR implementation P1/P2 tùy tỷ lệ scan thật.

## 13.3. JavaScript

Nếu HTTP raw:

- body rất nhỏ;
- app shell;
- content marker không có;
- DOM after browser pilot có article;

thì status:

```text
JS_EMPTY
```

và browser recovery tier có thể chạy.

## 13.4. Domain-specific extractor

Implementation nói chung P1.

Nhưng **top-domain extraction audit là P0**.

Nếu domain chiếm tỷ trọng lớn và generic extractor fail đáng kể, domain extractor được nâng lên P0.

---

# 14. Immutable `source_text`

Đây là core contract của corpus.

Sau extraction, tạo:

```text
source_text
```

Đây là source of truth cho:

- offsets;
- child text;
- parent text;
- final `chunk_text` submission.

Lưu:

```text
source_text_sha256
extractor_name
extractor_version
extraction_config_hash
raw_sha256
```

Không mutate `source_text` in-place sau khi document được commit.

Nếu extractor thay đổi:

```text
corpus_version mới
```

---

# 15. Three Text Concepts

Không được lẫn ba khái niệm.

## 15.1. `source_text`

Exact canonical extracted text.

Dùng cho:

```text
submission spans
source offsets
provenance
```

## 15.2. `normalized_text`

Có thể normalize:

```text
Unicode NFC
HTML entity decode trước source finalization nếu contract đã chốt
whitespace rules
```

Nếu normalized text khác source, phải có mapping hoặc không dùng normalized offsets làm submission offsets.

## 15.3. `retrieval_text`

Có thể là:

```text
title
section
chunk body
```

hoặc representation riêng sparse/dense.

Không submit `retrieval_text`.

---

# 16. Conservative Cleaning Rules

Bảo toàn biomedical tokens:

```text
H. pylori
HbA1c
BRCA1
HER2
SARS-CoV-2
SpO₂
0.5 mg
5 mg/kg
ICD-10
TNM
p < 0.05
95% CI
```

Không:

```text
remove all punctuation
remove digits
strip Vietnamese accents
translate corpus
aggressive stemming in source_text
```

Cleaner regression suite phải chứa các biomedical patterns này.

---

# 17. Language Detection

Fields:

```text
language
language_confidence
```

Values:

```text
vi
en
zh
mixed
other
unknown
```

Dùng cho:

- EDA;
- BM25 analyzer routing;
- quality monitoring;
- debugging;
- cross-lingual coverage report.

Không dùng để loại document tự động nếu confidence thấp.

---

# 18. Data Quality Gate — Extraction

Integrity đúng không đồng nghĩa data tốt.

Mỗi document tính:

```text
clean_char_count
paragraph_count
sentence_count
alphabetic_or_cjk_ratio
duplicate_line_ratio
link_text_ratio
boilerplate_signal
title_present
heading_count
reference_signal
```

## 18.1. Error page detector

Detect examples:

```text
Access Denied
403 Forbidden
Page Not Found
Enable JavaScript
Checking your browser
Sign in
Cloudflare challenge
```

Không để challenge page thành biomedical document.

## 18.2. Quality tiers

```text
HIGH
MEDIUM
LOW
QUARANTINE
```

Nhưng luôn lưu reason codes:

```text
TOO_SHORT
HIGH_BOILERPLATE
ERROR_PAGE
POSSIBLE_TRUNCATION
MISSING_TITLE
SCAN_PDF
JS_REQUIRED
ACCESS_DENIED
```

Không dùng `LOW` đồng nghĩa irrelevant.

---

# 19. Structure Parsing

Ưu tiên hierarchy:

```text
Document
 ↓
Section
 ↓
Paragraph
 ↓
Sentence
```

Schema section:

```text
section_id
doc_id
heading
start_char
end_char
section_index
```

Structure parser phải preserve source order.

Warnings:

```text
section overlap
invalid offsets
empty section
extreme heading density
reference section dominating
```

---

# 20. Child/Parent Chunking Architecture

Giữ kiến trúc R2AI Stage 3 ban đầu nhưng formalize.

## 20.1. Child chunks — retrieval units

Starting hypothesis:

```text
~180–256 BGE-M3 tokens
```

Không coi đây là tối ưu cố định.

Benchmark candidates:

```text
128
180
256
384
512
```

Child dùng cho:

- BM25;
- dense embedding;
- candidate retrieval;
- reranking.

## 20.2. Parent spans — submission context

Starting hypothesis:

```text
~384–640 tokens
```

Baseline:

```text
512
```

Parent được tạo/lookup từ source span quanh child, ưu tiên sentence/paragraph/section boundaries.

Parent không nhất thiết materialize mọi possible window trước.

## 20.3. Why two levels

```text
small child
→ stronger localized semantic signal

larger parent
→ better coverage of scorer reference information
```

Retrieval-optimal length không nhất thiết submission-optimal length.

---

# 21. Chunk Contracts

Child schema:

```text
child_id
canonical_chunk_id?
doc_id
canonical_content_id
section_id
chunk_index
source_start_char
source_end_char
text
token_count_retrieval
language
quality_tier
```

Parent schema:

```text
parent_id
doc_id
section_id
source_start_char
source_end_char
text
contains_child_ids[]
```

Hard invariants:

```python
child.text == source_text[child.start:child.end]
parent.text == source_text[parent.start:parent.end]
parent.start <= child.start
child.end <= parent.end
0 <= start < end <= len(source_text)
```

---

# 22. Chunk Quality Gate

Hard:

```text
exact span match = 100%
child IDs unique
parent IDs unique
foreign keys valid
bounds valid
containment valid
```

Soft metrics:

```text
token distribution
sentence count
starts mid-sentence
ends mid-sentence
cross-section boundary
near duplicate rate
very short chunk rate
very long chunk rate
```

Fragment warnings:

```text
pronoun-heavy beginning
numeric result without context
chunk consisting mostly references
navigation/boilerplate residual
```

Không hard reject semantic fragments chỉ bằng heuristic; dùng để QA/tuning.

---

# 23. Dedup Semantics

## 23.1. Content identity

```text
content_hash = SHA256(normalized canonical content)
```

Documents có same content có thể share:

```text
canonical_content_id
```

Nhưng giữ toàn bộ BTC IDs.

## 23.2. Retrieval representation identity

Nếu dense text bao gồm:

```text
title + section + child body
```

thì embedding reuse chỉ an toàn khi:

```text
retrieval_representation_hash identical
```

Không mặc định:

```text
same body_hash → same embedding
```

nếu title/section representation khác.

## 23.3. Alias mapping

```text
canonical_content_id
→ [btc_doc_id_1, btc_doc_id_2, ...]
```

Hard check:

```text
number of official BTC doc IDs represented after dedup
== number represented before dedup
```

Dedup tiết kiệm computation, không được xóa identity.

---

# 24. Sharding Strategy

Không tạo giant `documents.parquet` duy nhất.

```text
data/processed/documents/
  part-000000.parquet
  part-000001.parquet
...

data/chunks/child/
  part-000000.parquet
...
```

Shard target ban đầu:

```text
5k–20k docs/shard
```

Baseline:

```text
10k docs/shard
```

Tune theo raw/document size distribution.

Shard ID deterministic theo doc range hoặc stable partition assignment.

---

# 25. Atomic Commit Protocol

Shard lifecycle:

```text
PENDING
→ RUNNING
→ VALIDATING
→ DONE
```

Failure:

```text
FAILED
```

Write flow:

```text
part-0001.tmp
→ validate
→ checksum
→ atomic rename
→ part-0001.parquet
→ mark DONE
```

Không downstream-read `.tmp`.

---

# 26. Continuous Verification

Sau mỗi shard tự chạy:

## 26.1. Integrity

```text
invalid doc_id = 0
orphan document/chunk = 0
duplicate IDs = 0
source span mismatch = 0
child-parent containment mismatch = 0
checksum mismatch = 0
```

## 26.2. Distribution

```text
crawl status distribution
extract success
quality tiers
language distribution
chunks/doc
tokens/chunk
text length
near-duplicate rate
```

## 26.3. Domain segmentation

Luôn breakdown theo:

```text
domain
language
content_type
```

Global metrics không đủ.

---

# 27. Hard Fail vs Soft Warning

## 27.1. Immediate HARD FAIL

```text
dataset checksum changed
invalid BTC doc_id
lost BTC doc_id mapping
duplicate child_id
orphan chunk
exact source span mismatch
child outside parent
invalid bounds
corrupt shard
embedding ID mapping mismatch
index IDs outside frozen corpus
```

Action:

```text
STOP current shard
DO NOT COMMIT
```

Nếu systemic contract/config error:

```text
STOP pipeline
```

## 27.2. PAUSE / Soft anomaly

Examples:

```text
extract success drops sharply
quality LOW spike
chunks/doc drift
language distribution anomaly
domain failure spike
too-short chunks spike
```

Action:

```text
finish/rollback current shard safely
emit report
pause before next shard if threshold crossed
```

Threshold calibrate từ Stage A/B, không hard-code cảm tính ngay.

---

# 28. Golden Canary Dataset

Trước scale cần khoảng:

```text
100–500 manually inspected docs
```

Bao gồm:

```text
VI / EN / ZH
HTML / PDF
long / short
special Unicode
biomedical symbols
tables
duplicates
messy HTML
JS pages
error pages
```

Expected assertions:

```text
title snippets
required source snippets
language expectation
structure hints
chunk count reasonable range
biomedical token preservation
```

Mỗi thay đổi extractor/cleaner/chunker phải chạy golden regression.

---

# 29. Human QA Strategy

Automation không thay hoàn toàn human audit.

## 29.1. Milestone audit

Human inspect tại:

```text
Golden
1k
10k
100k
trước Full
```

Không cần stop thủ công từng giờ trong full run.

## 29.2. Stratified samples

Sample theo:

```text
top domains
VI/EN/ZH
PDF/HTML
quality tier
```

## 29.3. Adversarial samples

Luôn inspect:

```text
longest docs
shortest docs
highest chunk count
lowest text quality
highest boilerplate
weird language
scan PDF
JS recovered pages
403 recovered pages
```

---

# 30. HTML QA Report

Tự tạo report sau stage/shard sample:

```text
reports/qa/shard_XXXX.html
```

Mỗi example hiển thị:

```text
doc_id
URL
crawl/fetch tier
raw preview
source_text preview
quality reasons
sections
child boundaries
parent boundaries
```

Mục tiêu: mở 2–5 phút là nhìn thấy lỗi extraction/chunking rõ ràng.

---

# 31. Corpus Health Report

Artifact bắt buộc sau mỗi milestone.

Ví dụ:

```text
CORPUS HEALTH

Coverage
--------
total URLs
policy robots blocked
robots 404/410 allowed after source review; 401/403/429 held
technical fetch success
technical failure
valid article rate

Robots
------
disallowed
4xx unavailable
5xx unreachable
timeout/DNS

Recovery
--------
403 recovered by fast/browser-like
403 recovered by browser
JS recovered
retry-exhausted

Extraction
----------
extract success
error pages
possible truncation
scan PDF

Quality
-------
HIGH / MEDIUM / LOW / QUARANTINE

Languages
---------
VI / EN / ZH / mixed / unknown

Chunking
--------
mean/median/P95 tokens
chunks/doc
near duplicate rate
source-span failures = 0

Retrieval
---------
BM25 Recall@K
Dense Recall@K
Hybrid Recall@K
```

---

# 32. Stage A — 1k Pilot v2

Pilot hiện tại đã có 1k crawl result, nhưng cần **re-run targeted diagnostics** thay vì crawl lại toàn bộ ngay.

## 32.1. Reclassify nhóm robots_unavailable còn lại

Ban đầu có 13 ID nhóm này; sau retry phải lấy danh sách hiện tại từ
`failures_after_retry.csv`. CSV cũ không chứa HTTP status của `/robots.txt`,
nên phải probe lại và giữ thời điểm cùng bằng chứng mới. For each:

```text
exact robots HTTP status
redirect chain
network failure type
cached robots availability
```

Convert sang RFC-aware state.

## 32.2. Recovery test nhóm 403 còn lại

Ban đầu có 15 ID; chỉ chọn các ID còn lỗi sau retry và robots cho phép.

Run matrix:

```text
current fetch
Scrapling fast/browser-like HTTP
Scrapling browser session
```

Validate article content.

## 32.3. Retry nhóm timeout còn lại

Ban đầu có 5 ID. Run controlled backoff matrix trên các ID còn lỗi.

## 32.4. 54 robots-blocked

Audit parsing correctness trên sample:

```text
right user-agent?
right allow/disallow longest-match semantics?
right path?
```

Nếu parser đúng, giữ policy blocked.

Không dùng recovery bypass.

## 32.5. Stage A DoD

```text
✓ robots statuses fully classified
✓ 403 recovery experiment complete
✓ timeout recovery experiment complete
✓ top-domain extraction manual audit
✓ raw storage measurement
✓ extraction quality sample
✓ language distribution
✓ failure report by domain
✓ valid-article coverage được đo tách khỏi ID accounting
✓ đường lấy nội dung cho domain bị chặn/URL chết đã được BTC hoặc nguồn xác nhận,
  hoặc coverage gap được ghi rõ và team chấp nhận
✓ crawler configuration selected for Stage B1
```

---

# 33. Stage B1 — 10k

Goals:

```text
longer crawl stability
robots cache behavior
Scrapling fallback cost
raw storage growth
extract quality
chunk integrity
shard resume
```

Must produce:

```text
crawl_report.json
robots_report.json
recovery_report.json
extraction_report.json
chunk_report.json
corpus_health.html
```

Hard requirement:

```text
exact source span failures = 0
```

---

# 34. Stage B2 — 100k

Goals:

```text
domain long-tail behavior
resource estimates
real chunk distribution
retrieval benchmark
index prototype
failure recovery at scale
```

Only here do we decide child size candidate winner.

Benchmark child sizes:

```text
128 / 180 / 256 / 384 / 512
```

Parent sizes:

```text
384 / 512 / 640 / 768
```

Do not assume 180/512 wins.

---

# 35. Stage C — 100k → 1M

Run only after B2 passes.

Measure:

```text
memory leaks
file descriptor growth
raw archive behavior
crawler checkpoint/resume
manifest scale
Parquet performance
Lucene growth
FAISS build memory
embedding throughput variance
```

Stop if:

```text
technical failure rate spikes
major domain extraction breaks
storage estimate exceeds budget
GPU-hours exceed deadline
ANN fidelity unacceptable
```

---

# 36. Extraction Quality Promotion Gate

Before corpus freeze:

Need evidence that main-text extraction is usable.

Per domain/language/type report:

```text
article_present_rate
error_page_rate
boilerplate_warning_rate
possible_truncation_rate
manual_good_rate
```

Top domains get manual sample.

A corpus is not promoted because HTTP success is high.

Metric of interest:

```text
VALID_ARTICLE_RATE
```

---

# 37. Retrieval Quality Gate

Self-retrieval catches plumbing bugs only.

Need:

```text
Recall@10
Recall@50
Recall@100
Recall@1000
nDCG@10
nDCG@100
```

Nếu có qrels chính thức, dùng.

Nếu không có qrels:

```text
50–100 query mini-dev
candidate pool = union(BM25 top20, dense top20, hybrid top20)
manual relevance labels
```

Track regression by build version.

Rule:

```text
cleaner/extractor/chunker mới nhìn đẹp hơn
BUT retrieval quality giảm đáng kể
→ do not promote
```

---

# 38. Sparse Index

Default:

```text
Pyserini / Lucene BM25
```

Input generated from child chunks.

Do not store unnecessary fields in Lucene if raw/source text is in Parquet.

Benchmark:

```text
shared multilingual index
vs
BM25_vi / BM25_en / BM25_zh / BM25_mixed
```

Chinese analyzer/tokenization must be tested separately.

---

# 39. Dense Embedding

Baseline candidate:

```text
BAAI/bge-m3
```

Before competition freeze, model eligibility must be verified independently against competition model rules.

Manifest:

```text
model_name
model_revision
tokenizer_name
tokenizer_revision
embedding_dim
max_length
precision
normalize_embeddings
chunk_version
retrieval_representation_version
```

Token count measured on actual dense representation:

```text
title + section + child body
```

not only child body.

---

# 40. Embedding Shards

Do not store tens of millions vectors in one monolithic array.

```text
embeddings/shard_00000/
  vectors.npy
  ids.parquet
```

Target:

```text
250k–1M vectors/shard
```

depending on hardware/storage.

Each shard:

```text
vector_count
dtype
dimension
checksum
status
model/tokenizer/config hash
```

---

# 41. Resource Gate

Measure rather than assume.

Required:

```text
valid canonical docs
mean child chunks/doc
child length buckets
tokens/s
chunks/s
VRAM peak
raw GB/doc
processed GB/doc
vector GB
Lucene GB
FAISS GB
```

Compute:

```text
estimated_chunks
estimated_GPU_hours
estimated_storage
estimated_deadline_margin
```

---

# 42. Full vs Selective Embedding

## 42.1. Full

Use when:

```text
GPU-hours within budget
storage within budget
deadline safe
retrieval quality justifies chunk dense full
```

## 42.2. Preferred fallback: doc-first query-conditioned selective

If full chunk embedding too expensive:

```text
ALL docs
 ↓
BM25 full
+
cheap/document-level multilingual dense
 ↓
run all 1200 queries
 ↓
union candidate docs
 ↓
chunk-level embed those candidate docs
```

This is preferred over dropping LOW-quality docs blindly.

## 42.3. Quality-conditioned fallback

Can be secondary:

```text
HIGH/MEDIUM → chunk dense
LOW → sparse/document-level dense
```

but must prove it does not destroy cross-lingual recall.

---

# 43. Dense Index

Reference exact index on subset:

```text
IndexFlatIP
```

Full-scale default candidate:

```text
FAISS IVF-PQ
```

Benchmark:

```text
nlist
M
nbits
nprobe
OPQ optional
```

Measure ANN fidelity:

```text
ANN Recall@10
ANN Recall@100
```

against Flat on sampled subset.

---

# 44. Index Integrity

Hard checks:

```text
index vector count == embedding manifest count
all vector IDs resolve to child IDs
all child IDs resolve to frozen corpus
no duplicate vector IDs
same model/config across shards
```

Self-retrieval smoke:

```text
sample child text → retrieve → own child/doc rank high
```

This catches ID misalignment/index corruption, not semantic quality.

---

# 45. Manifests

Hierarchy:

```text
dataset_manifest
    ↓
crawl_manifest
    ↓
corpus_manifest
    ↓
chunk_manifest
    ↓
embedding_manifest
    ↓
index_manifest
```

Each downstream artifact stores parent manifest hash.

Example corpus manifest:

```json
{
  "corpus_version": "...",
  "dataset_manifest_sha256": "...",
  "crawl_manifest_sha256": "...",
  "extractor_version": "...",
  "cleaner_version": "...",
  "document_count": 0,
  "canonical_content_count": 0,
  "btc_doc_id_count": 0,
  "source_text_hash_aggregate": "..."
}
```

---

# 46. Versioning & Promotion

Không overwrite artifact đã dùng benchmark.

```text
corpus/v001
corpus/v002
chunks/v001
chunks/v002
indices/v001
```

States:

```text
BUILDING
VALIDATED
QUALITY_REVIEWED
RETRIEVAL_TESTED
PROMOTED
REJECTED
```

Only `PROMOTED` can feed competition submission pipeline.

---

# 47. Rollback

Promotion pointer:

```text
current_corpus -> v007
current_index  -> v012
```

Nếu v008 regression:

```text
reject v008
keep v007 active
```

No destructive overwrite.

---

# 48. Recovery Protocol

## Crawl crash

Resume pending docs from manifest/checkpoint.

## Shard processing crash

Delete/rebuild current `.tmp`; committed shards untouched.

## Extractor bug

Use raw archive to re-extract affected shards/domains without recrawl.

## Chunker bug

Rebuild chunks from immutable documents only.

## Embedding bug

Rebuild embedding shards from frozen child chunks; no recrawl/re-extract.

## Index bug

Rebuild index from embedding shards.

This layering is intentional to minimize rework.

---

# 49. Config — Proposed Master Baseline

```yaml
dataset:
  links_path: data/source/links_corpus.parquet
  query_path: data/source/query.parquet

robots:
  obey: true
  cache_enabled: true
  normal_ttl_hours: 24
  follow_redirects: true
  max_redirects: 5
  rfc9309_404_410_policy: allow_with_audit_before_scale
  rfc9309_401_403_other_4xx_policy: hold_for_review
  robots_429_policy: retry_after_backoff
  unreachable_policy: disallow

crawler:
  engine: candidate
  global_concurrency: 4  # pilot hiện tại; chỉ tăng sau đo tải/domain
  per_domain_concurrency: 2
  connect_timeout_seconds: 10
  read_timeout_seconds: 30
  max_retries: 3
  user_agent: "ViBioMIR-Research-Retriever/1.0"
  autothrottle: true

recovery:
  enable_scrapling: true
  http_403_browser_like: true
  browser_fallback: true
  browser_only_after_fast_failure: true
  retry_404: false
  bypass_robots_disallow: false

raw:
  archive_format: gzip_jsonl_shards  # hiện đã triển khai; benchmark trước khi đổi
  atomic_commit: true

extraction:
  html_primary: trafilatura
  html_fallback: structured_dom
  pdf_primary: pymupdf
  detect_scan_pdf: true
  ocr_default: false

cleaning:
  unicode_normalization: NFC
  preserve_numbers: true
  preserve_punctuation: true
  preserve_accents: true

quality:
  enabled: true
  error_page_detector: true
  store_reason_codes: true
  drift_detection: true

chunking:
  child_target_tokens: 180
  child_candidates: [128, 180, 256, 384, 512]
  parent_target_tokens: 512
  parent_candidates: [384, 512, 640, 768]
  section_aware: true
  sentence_boundary: true
  prepend_title_for_retrieval: true
  prepend_section_for_retrieval: true

sharding:
  documents_per_shard: 10000
  atomic_commit: true

embedding:
  model: BAAI/bge-m3
  precision: fp16
  normalize: true
  policy: undecided

sparse_index:
  backend: pyserini-lucene

dense_index:
  backend: faiss
  type: IVF-PQ

evaluation:
  recall_k: [10, 50, 100, 1000]
  ndcg_k: [10, 100]
```

All numerical values above are starting configurations, not proven optima.

---

# 50. Repository Structure

```text
VietMedBridge/
├── configs/
│   └── config.yaml
│
├── data/
│   ├── source/
│   ├── inventory/
│   ├── raw/
│   ├── processed/
│   │   ├── documents/
│   │   ├── sections/
│   │   └── aliases/
│   ├── chunks/
│   │   ├── child/
│   │   └── parent/
│   ├── dev/
│   └── manifests/
│
├── embeddings/
│   └── shards/
│
├── indices/
│   ├── sparse/
│   └── dense/
│
├── reports/
│   ├── pilot/
│   ├── stage_a/
│   ├── stage_b1/
│   ├── stage_b2/
│   ├── stage_c/
│   └── qa/
│
├── src/
│   ├── dataset/
│   ├── inventory/
│   ├── robots/
│   ├── crawler/
│   │   ├── fast.py
│   │   ├── scrapling_recovery.py
│   │   └── failure_classifier.py
│   ├── storage/
│   ├── extraction/
│   ├── preprocessing/
│   ├── quality/
│   ├── structure/
│   ├── chunking/
│   ├── validation/
│   ├── embedding/
│   ├── indexing/
│   └── evaluation/
│
├── tests/
│   ├── unit/
│   ├── golden/
│   ├── integration/
│   └── regression/
│
├── scripts/
│   ├── 01_snapshot.py
│   ├── 02_inventory.py
│   ├── 03_reclassify_robots.py
│   ├── 04_scrapling_recovery_pilot.py
│   ├── 05_crawl.py
│   ├── 06_extract.py
│   ├── 07_build_documents.py
│   ├── 08_chunk.py
│   ├── 09_validate_corpus.py
│   ├── 10_freeze.py
│   ├── 11_benchmark_embed.py
│   ├── 12_build_bm25.py
│   ├── 13_build_dense.py
│   ├── 14_eval.py
│   └── 15_health_report.py
│
├── PLAN.md
└── README.md
```

---

# 51. Test Suite — Mandatory

## 51.1. Robots

```text
200 robots allow
200 robots disallow
404 robots
403 robots
500 robots
robots timeout
redirect chain
partial parse
user-agent matching
allow/disallow specificity
```

## 51.2. Crawler

```text
403 classification
429 Retry-After
read timeout
connect timeout
HTML valid
JS empty
challenge page
404 no retry
```

## 51.3. Extraction

```text
normal article
navigation-heavy page
error page
PDF text
scan PDF detection
JS recovered rendered HTML
```

## 51.4. Cleaner

Biomedical preservation cases.

## 51.5. Chunking

```text
sentence boundary
section boundary
long paragraph
short paragraph
exact offset
child-parent containment
```

## 51.6. Dedup

```text
same body same title
same body different title
all BTC aliases retained
retrieval hash logic
```

## 51.7. Shard

```text
crash mid-write
atomic commit
resume
corrupt temp
checksum mismatch
```

## 51.8. Index

```text
vector count
ID mapping
self retrieval
ANN fidelity subset
```

---

# 52. Observability Metrics

## Crawl

```text
requests/s
success rate
403 rate
429 rate
timeout rate
robots states
recovery rate
browser fallback rate
latency P50/P95
bytes/s
```

## Extract

```text
extract success
chars/doc
paragraphs/doc
error page rate
scan PDF rate
quality tiers
```

## Chunk

```text
chunks/doc
tokens/chunk P50/P95/P99
source-span failure
parent containment failure
near duplicate rate
```

## Resource

```text
CPU
RAM
open files
disk write throughput
raw storage growth
processed storage growth
```

## Embed

```text
chunks/s
tokens/s
VRAM
GPU utilization
batch size
length buckets
```

---

# 53. Kill Switch Policy

Immediate STOP if:

```text
source span mismatch > 0
invalid/lost doc_id > 0
corpus checksum changed
systematic crawler policy bug
raw archive corruption
manifest corruption
```

Pause after shard if:

```text
403 rate spikes unusually
technical failure rate sharply increases
extract success drops
quality LOW/QUARANTINE spikes
chunks/doc distribution jumps
storage forecast crosses budget
```

Browser recovery cost spike can disable browser tier while fast crawl continues only if doing so does not invalidate corpus consistency.

---

# 54. What We Do NOT Do

Không:

```text
bypass robots Disallow
bypass login/paywall/authentication restrictions
browser crawl every URL by default
trust HTTP 200 as valid article
throw away raw responses
reassign BTC doc_id
submit generated/rephrased chunk text
embed before corpus freeze
promote artifact without validation
use a single giant file for everything
use aggregate success rate without per-domain breakdown
```

---

# 55. Immediate Implementation Roadmap

## P0.1 — Fix robots resolver

1. Implement RFC-aware status taxonomy.
2. Re-probe và phân loại những `robots_unavailable` còn lại sau retry.
3. Unit tests for 4xx/5xx/timeout/redirect.
4. Per-domain cache.

## P0.2 — Run Scrapling recovery experiment

1. Install/pin Scrapling version.
2. Build recovery adapter, not full migration.
3. Test nhóm 403 còn lại sau retry, robots đã cho phép.
4. Test timeouts where justified.
5. Measure valid article recovery and cost.
6. Decide fast crawler vs Scrapling orchestration.

## P0.3 — Audit raw + manifest contracts hiện có

1. Stable crawl manifest.
2. Đo gzip JSONL shards hiện có; chỉ thử tar.zst khi có lợi ích đo được.
3. SHA256.
4. Atomic commit.
5. Resume.

## P0.4 — Audit source-text contract hiện có

1. Extractor interface.
2. Immutable `source_text`.
3. Source hash.
4. Error-page detector.
5. Golden extraction suite.

## P0.5 — Audit child/parent + validators hiện có

1. Section/paragraph/sentence parser.
2. Child builder.
3. Parent resolver.
4. Exact-span checks.
5. Containment checks.

## P0.6 — Audit quality & health reports hiện có

1. Quality reason codes.
2. Per-domain/language/type reports.
3. Drift detector.
4. HTML sample report.

## P0.7 — Stage B1 10k

Only after all above pass **và** source-access/valid-article-coverage gate ở
§7.1 đã được chốt. Các mục P0.3–P0.6 đã có implementation cơ bản trong
`src/vietmedbridge`; cải tiến theo gap audit thay vì viết lại pipeline.

---

# 56. Decision Table for Current Pilot Failures

| Initial pilot failure (trước retry) | Immediate action | Scrapling? | Final state if unresolved |
|---|---|---|---|
| 54 robots blocked | verify parser, respect Disallow | No bypass | `POLICY_ROBOTS_DISALLOWED` |
| 13 robots unavailable | probe lại, ghi status/exception | Not primary | chỉ 404/410 có thể allow; 429 backoff; còn lại review/pause |
| 15 HTTP 403 | recovery A/B test | Yes, candidate | `HTTP_403/RETRY_EXHAUSTED` |
| 5 timeout | backoff/retry/tune | Maybe | `TIMEOUT_* / RETRY_EXHAUSTED` |
| 1 HTTP 404 | no expensive retry | No | `HTTP_404` |

---

# 57. Promotion Checklist Before 10k

```text
[ ] dataset snapshot frozen
[ ] robots resolver RFC tests pass
[ ] mọi robots-unavailable còn lại sau retry được re-probe và phân loại
[ ] 54 robots-blocked sample audit confirms parser correctness
[ ] 403 Scrapling experiment completed
[ ] timeout recovery policy fixed
[ ] raw archive replay works
[ ] crawl manifest complete
[ ] source_text immutable contract implemented
[ ] exact source offsets implemented
[ ] golden extractor tests pass
[ ] biomedical cleaner tests pass
[ ] child/parent chunk tests pass
[ ] shard atomic commit works
[ ] health report generated automatically
[ ] valid_article_coverage và domain coverage đo được, ngưỡng được team chốt
[ ] domain bị chặn/URL chết có access path từ BTC/nguồn hoặc gap được ghi rõ
```

Nếu một P0 item chưa pass, không scale 10k.

---

# 58. Promotion Checklist Before 100k

```text
[ ] Stage B1 10k completed without hard invariant failures
[ ] recovery metrics stable
[ ] top-domain extraction audit acceptable
[ ] per-language extraction coverage understood
[ ] chunk distributions stable
[ ] QA sample reviewed
[ ] raw/processed storage extrapolation available
[ ] resume tested by intentional interruption
[ ] mini-dev evaluation available or explicit limitation documented
```

---

# 59. Promotion Checklist Before Full Corpus

Must answer quantitatively:

```text
1. How many URLs are policy-blocked?
2. How many are technically reachable?
3. What is valid-article recovery after fallback?
4. Which domains remain problematic?
5. Raw storage forecast?
6. Valid canonical documents?
7. Exact dedup rate?
8. Child chunks/doc distribution?
9. Best child/parent config on dev?
10. Full embedding GPU-hours?
11. Dense index size?
12. Sparse index size?
13. ANN Recall@100?
14. Full crawl deadline margin?
15. Zero source-span errors?
16. Zero lost BTC doc_ids?
17. Retrieval quality regression passes?
18. Valid-article coverage đạt ngưỡng đã chốt theo domain?
19. Với domain bị chặn nhiều, đã có snapshot/API/quyền từ BTC hoặc nguồn?
```

If these are not known, full run is not approved.

---

# 60. Final Definition of Done

Data pipeline is considered complete only when:

## Correctness

```text
all BTC IDs accounted for
all child/parent spans exact
all mappings valid
no orphan artifacts
```

## Coverage

```text
policy-blocked separated from technical failures
recoverable failures have controlled fallback
per-domain coverage is measured
valid_article_coverage measured separately from ID accounting
coverage threshold agreed before scale-up
blocked/dead high-impact domains have BTC/source access path or explicit gap
```

## Quality

```text
valid article extraction demonstrated
quality reason codes available
human + automated QA pass
```

## Reproducibility

```text
raw archived
manifests/hashes pinned
pipeline versions pinned
artifacts immutable/versioned
```

## Retrieval readiness

```text
BM25 index valid
dense index valid
candidate recall measured or limitation explicit
regression suite passes
```

## Operations

```text
shard resume works
atomic commit works
kill switch works
health report automatic
rollback available
```

---

# 61. Final Architecture Decision

Current recommended build is:

```text
FAST / POLITE HTTP FIRST
        ↓
RFC-AWARE ROBOTS RESOLUTION
        ↓
FAILURE CLASSIFICATION
        ↓
SCRAPLING AS CONTROLLED RECOVERY TIER
        ↓
RAW ARCHIVE
        ↓
TRAFILATURA / PyMuPDF / DOMAIN FALLBACK
        ↓
IMMUTABLE SOURCE_TEXT
        ↓
DATA QUALITY GATES
        ↓
CHILD RETRIEVAL CHUNKS
        +
PARENT SUBMISSION SPANS
        ↓
CONTINUOUS INTEGRITY CHECKS
        ↓
FREEZE
        ↓
BM25 + MULTILINGUAL DENSE
```

Scrapling **không thay thế toàn bộ data pipeline**. Nó giải quyết một phần rất cụ thể nhưng có giá trị cao:

```text
FETCH RELIABILITY / RATE ADAPTATION / BROWSER FALLBACK / CHECKPOINTING
```

Canonical extraction, provenance, dedup, quality control, chunking và retrieval indexing vẫn thuộc pipeline của VietMedBridge.

---

# 62. External Technical References Used for This Plan

1. **RFC 9309 — Robots Exclusion Protocol**
   - defines successful access, redirects, 4xx “Unavailable”, 5xx/network “Unreachable”, parsing behavior and caching guidance.

2. **D4Vinci/Scrapling official repository and documentation**
   - Spider concurrency and per-domain controls;
   - AutoThrottle;
   - robots compliance/cache;
   - blocked-request detection/retry;
   - HTTP/browser multi-session fetching;
   - pause/resume and development-mode response replay.

3. **R2AI Stage 1/Stage 2 solution analyses**
   - Stage 1: candidate recall, canonical corpus, normalization and data quality;
   - Stage 2: provenance, replay, validation, artifact promotion, immutable versions and rollback.

---

# 63. One-Line Rule for the Team

> **Không được scale vì pipeline “chạy được”; chỉ scale khi pipeline chứng minh rằng output của stage nhỏ là đúng, có chất lượng, truy nguyên được và có thể phục hồi.**
