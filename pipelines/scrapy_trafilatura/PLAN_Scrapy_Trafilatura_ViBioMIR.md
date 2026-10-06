# PLAN — ViBioMIR Scrapy + Trafilatura Data Ingestion Pipeline

**Deployment bổ sung 2026-10-04: Kaggle theo yêu cầu người dùng.** Giữ nguyên pilot Windows 10k và bộ re-extraction v2 đã kiểm chứng. Bộ chạy mới dùng `notebooks/02_kaggle_crawl_shard.ipynb`: một frontier cố định mỗi phiên CPU/Internet ON, chạy tuần tự; 4.392.746 URL duy nhất được chia thành 43 shard × 100.000 và một shard 92.746. Mỗi shard đọc trực tiếp Parquet, không sao chép SQLite inventory toàn corpus. Phân hoạch là các khoảng liên tiếp trên inventory đã sắp theo hash URL, đóng băng trong `campaign_manifest.json`; không đổi biên shard khi resume.

Kaggle công bố 12 giờ CPU/GPU và 20 GB output được lưu: [tài liệu notebook](https://www.kaggle.com/docs/notebooks). Cấu hình mặc định: crawl tối đa 6 giờ, extraction 3 giờ, 16 request đồng thời/2 mỗi domain, 2 extraction worker, raw nén tối đa 4 GiB và event extraction tối đa 1 GiB. Giới hạn thời gian dừng việc cấp task rồi drain; trạng thái PARTIAL giữ URL chưa hoàn tất. Giới hạn storage tính tích lũy trong shard: nếu chạm ngưỡng, resume với cùng ngưỡng sẽ dừng tiếp; cần kiểm tra kích thước trước khi tăng trong giới hạn phiên. Không dùng GPU cho crawler.

Cuối phiên, stream raw/events/checkpoints/frontier/extracted vào một tar có SHA256 mỗi member và toàn archive; kiểm tra lại toàn bộ trước khi dọn cache tạm của riêng phiên. Lưu `.tar`, `.tar.sha256` và report bằng Save & Run All có output. Phiên tiếp theo mount bundle cũ để tiếp tục cùng shard; queue/index được dựng lại từ manifest, bỏ qua mọi URL đã có kết quả, giữ domain thất bại để xử lý sau theo quyết định trước. Chỉ chuyển shard sau `COMPLETED`; PARTIAL không đồng nghĩa hoàn thành. Nếu môi trường bị kill trước export, chỉ output đã lưu ở phiên trước bảo đảm resume được.

Source snapshot và native ID mapping được giữ trong Dataset input. Report mỗi shard không thay cho outcomes toàn corpus; global dedup/mapping và merge sau khi thu đủ shard là bước riêng. README ghi thao tác Kaggle; `IMPLEMENTATION_STATUS.md` ghi kết quả test và trạng thái chạy thực tế.

Rà soát kỹ thuật: **2026-10-02**, cập nhật triển khai **2026-10-03**. Baseline: **Scrapy 2.13.4, Trafilatura 2.1.0, Python 3.11** theo yêu cầu triển khai bổ sung của người dùng. Môi trường Windows hiện dùng Python 3.11.15; exact dependencies nằm trong `requirements-lock.txt`. Đây là các phiên bản được chọn để đối chiếu API, không phải tuyên bố về phiên bản mới nhất hay kết quả benchmark. Xem §28 và §37 về lockfile, nguồn kiểm chứng và các nhận xét cần đính chính.

**Schema nguồn đã xác minh:** snapshot HF `AIGuruTinix/ViBioMIR` revision `0148f6f80ffafed5c005af6d506ccfd9d3fb47a7` có corpus **4.394.718 dòng**, cột `id: int64`, `url: string`; query **1.200 dòng**, chỉ có `id: int64`, `query: string`. File query này **không có gold doc IDs/qrels**. Các mục gold bên dưới là contract cho nhãn nếu có; không được join query `id` với corpus `id` để tự tạo gold. Người dùng xác nhận BTC không cung cấp qrels và đồng ý tiếp tục: config giữ `gold_doc_ids_column: null`, report giữ `available: false`/coverage null, LOW giữ trong benchmark. Người dùng chấp nhận nội dung Stage A, hoãn domain không truy cập được và chọn Windows hiện tại cho pilot 10k. Quyết định lưu riêng trong `review_decision.json`; không tự điền đánh giá từng doc trong CSV. Xem `IMPLEMENTATION_STATUS.md` và báo cáo pilot để biết kết quả thực thi.

## 0. Mục tiêu và phạm vi

Xây dựng pipeline dữ liệu đầu vào cho ViBioMIR từ corpus dạng `doc_id + url` thành corpus tài liệu sạch, có thể chuyển trực tiếp sang bước chunking/embedding.

```text
links_corpus.parquet
        ↓
URL inventory + duplicate mapping
        ↓
Scrapy downloader
        ↓
Raw response cache + Crawl manifest
        ↓
Trafilatura extraction workers
        ↓
Basic cleaning + quality checks
        ↓
Exact content dedup mapping
        ↓
documents.parquet
        ↓
HANDOFF → chunking / embedding
```

Phạm vi PLAN này chỉ bao gồm: đọc URL từ ViBioMIR; chuẩn hóa URL tối thiểu; gom duplicate URL; tải dữ liệu bằng Scrapy; retry/throttle/resume; lưu raw response; lưu crawl manifest; extract HTML bằng Trafilatura; extract PDF bằng PyMuPDF; cleaning bảo thủ; basic language detection; basic quality tier; exact content dedup; xuất `documents.parquet`.

Không triển khai trong P0: chunking production, embedding, FAISS, BM25, reranker, LLM query expansion, OCR production, Playwright production, near-duplicate MinHash/SimHash.

---

# 1. Nguyên tắc kiến trúc bắt buộc

## 1.1. Không crawl theo `doc_id` trực tiếp

ViBioMIR có thể có nhiều `doc_id` trỏ tới cùng một URL. Không tải cùng URL nhiều lần.

```text
doc_1 ─┐
       ├── crawl_url_id = U001 → https://example.com/a
doc_2 ─┘
```

Crawler chỉ tải `U001` một lần. Sau extraction vẫn giữ mapping từ nội dung về toàn bộ BTC `doc_id`.

**Không được làm mất bất kỳ `doc_id` nào.**

## 1.2. Scrapy chỉ chịu trách nhiệm network ingestion

Scrapy làm: scheduling, HTTP download, redirect, retry, throttle, robots policy, response metadata, raw cache, crawl status, resume/checkpoint.

Không gọi Trafilatura trực tiếp trong production callback cho hàng triệu URL. Scrapy chủ yếu I/O-bound, Trafilatura CPU-bound; trộn hai việc sẽ khó benchmark, dễ block crawler và khiến đổi extractor phải crawl lại.

Production split:

```text
STAGE 1 — SCRAPY
URL → raw bytes + manifest

STAGE 2 — TRAFILATURA
raw HTML bytes → main content Markdown + plain text + title metadata
```

## 1.3. Raw response là source of truth cho extraction

Mỗi response hợp lệ phải được cache trước khi extraction.

```text
raw/
├── html/
├── pdf/
└── text/
```

Nếu thay extractor, chỉ chạy lại Stage 2, không request Internet lại.

Cache `response.body` dạng **bytes**, sau xử lý HTTP Content-Encoding của Scrapy nhưng trước charset decoding; không dùng `response.text` làm raw. Giữ header gốc và checksum để kiểm tra encoding VI/ZH. `raw_path` có thể trỏ tới file lẻ hoặc container kèm member/index (§12).

## 1.4. `doc_id` BTC là immutable

Chuỗi mapping phải giữ được:

```text
doc_id
→ crawl_url_id
→ raw response
→ extracted content
→ canonical_content_id
→ final document mapping
```

Không tự renumber.

## 1.5. Mọi stage phải resume được

Các stage `inventory`, `crawl`, `extract`, `compact` phải idempotent, checkpointed và có thể chạy tiếp sau crash.

---

# 2. Kiến trúc tổng thể

```text
                    links_corpus.parquet
                            │
                            ▼
                 [A] INVENTORY BUILDER
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
 unique_urls.parquet                 url_doc_map.parquet
           │
           ▼
                  [B] CRAWL SHARD BUILDER
                            │
                            ▼
                     crawl_shards/
                            │
                            ▼
                    [C] SCRAPY SPIDER
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
         RAW CACHE                  CRAWL MANIFEST
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 [D] EXTRACTION WORKERS
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
            Trafilatura   PyMuPDF    text decode
                 │          │          │
                 └──────────┼──────────┘
                            ▼
                    extracted shards
                            │
                            ▼
                [E] CLEAN + QUALITY
                            │
                            ▼
                 [F] EXACT CONTENT DEDUP
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
       documents.parquet        content_doc_map.parquet
              │
              ▼
              HANDOFF → CHUNKING
```

---

# 3. Repo structure

```text
vibiomir-retrieval/
├── configs/
│   ├── config.yaml
│   └── deployment.yaml
├── data/
│   ├── source/
│   │   ├── links_corpus.parquet
│   │   ├── query.parquet
│   │   └── dataset_manifest.json
│   ├── inventory/
│   │   ├── unique_urls.parquet
│   │   ├── url_doc_map.parquet
│   │   ├── domain_stats.parquet
│   │   ├── gold_doc_map.parquet
│   │   └── invalid_urls.parquet
│   ├── crawl_shards/
│   ├── raw/
│   │   ├── html/
│   │   ├── pdf/
│   │   └── text/
│   ├── manifests/
│   │   ├── crawl/
│   │   └── extraction/
│   ├── processed/
│   │   ├── extracted/
│   │   └── documents/
│   └── mappings/
│       ├── content_doc_map.parquet
│       └── doc_outcomes.parquet
├── crawl_jobs/
├── reports/
│   ├── stage_a/
│   └── stage_b/
├── src/
│   ├── inventory/
│   │   ├── url_normalizer.py
│   │   ├── build_inventory.py
│   │   └── build_shards.py
│   ├── crawler/
│   │   ├── spiders/corpus_spider.py
│   │   ├── items.py
│   │   ├── pipelines.py
│   │   ├── middlewares.py
│   │   ├── raw_store.py
│   │   └── settings.py
│   ├── extraction/
│   │   ├── router.py
│   │   ├── html.py
│   │   ├── pdf.py
│   │   ├── text.py
│   │   └── worker.py
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   ├── language.py
│   │   ├── quality.py
│   │   └── dedup.py
│   ├── storage/
│   │   ├── jsonl_writer.py
│   │   ├── parquet_writer.py
│   │   └── manifests.py
│   └── reports/build_report.py
├── scripts/
│   ├── 01_build_inventory.py
│   ├── 02_build_crawl_shards.py
│   ├── 03_run_crawl.py
│   ├── 04_extract_raw.py
│   ├── 05_build_documents.py
│   └── 06_report.py
├── tests/
├── scrapy.cfg
├── pyproject.toml
└── README.md
```

---

# 4. Configuration baseline

Dùng một file `configs/config.yaml` ở giai đoạn đầu.

```yaml
dataset:
  links_path: data/source/links_corpus.parquet
  query_path: data/source/query.parquet
  manifest_path: data/source/dataset_manifest.json
  sample_seed: 42

inventory:
  remove_fragments: true
  remove_tracking_params: false

sharding:
  urls_per_shard: 100000
  execution: sequential

crawler:
  concurrent_requests: 64
  concurrent_requests_per_domain: 2
  download_timeout: 30
  retry_times: 3
  retry_http_codes: [408, 500, 502, 503, 504, 522, 524]
  rate_limit:
    respect_retry_after: true
    default_cooldown_seconds: 60
    max_batch_retries: 3
  autothrottle:
    enabled: true
    start_delay: 0.5
    max_delay: 60
    target_concurrency: 1.5
  robots_obey: true
  user_agent: "ViBioMIR-Research-Retriever/1.0"
  max_response_bytes: 52428800       # DOWNLOAD_MAXSIZE: 50 MiB
  warn_response_bytes: 10485760      # DOWNLOAD_WARNSIZE: 10 MiB
  reactor_threadpool_maxsize: 32     # Thử nghiệm 10/32/64 ở Stage B
  dnscache_enabled: true
  dnscache_size: 100000
  download_slots: {}                # Override hostname sau Stage A

raw_store:
  layout: loose_files               # Chốt loose_files/packed_shards ở Stage B
  compression: zstd
  compression_level: 1

extraction:
  max_tasks_per_child: 500
  max_inflight_per_worker: 2
  html:
    fast_first: false
    output_format: markdown
    include_formatting: true
    include_comments: false
    include_links: false
    include_tables: true
    min_chars_fast_result: 500
    max_input_bytes: 10485760
    timeout_seconds: 30
  pdf:
    parser: pymupdf
    max_input_bytes: 52428800
    timeout_seconds: 60
  text:
    max_input_bytes: 10485760
    timeout_seconds: 30

quality:
  min_chars: 200
  low_handoff_policy: pending_gold_audit
  handoff_mode: benchmark

language:
  backend: heuristic                 # Benchmark với fastText lid.176.ftz
  low_confidence_label: unknown

dedup:
  widespread_duplicate_min_urls: 100
  widespread_duplicate_min_domains: 5

coverage:
  blocked_review_threshold: 0.10
  stage_a_gold_url_budget: 200

storage:
  jsonl_records_per_part: 50000
  parquet_rows_per_part: 100000
```

Đây là pilot config, không phải full-scale config bất biến.

`429` được xử lý bằng cooldown + retry batch riêng (§9–10), nên không nằm trong danh sách retry tức thì của RetryMiddleware. Các giới hạn extraction áp dụng trên **bytes đã giải nén**, độc lập với download cap. `output_format: markdown` là format handoff; adapter Python lấy `Document` rồi serialize cấu trúc (§16).

---

# 5. Inventory Builder

## 5.1. Input

`links_corpus.parquet` với fields `id`, `url`. Rename nội bộ `id → doc_id` nhưng không sửa giá trị.

### Dataset snapshot và gold mapping

Trước inventory, lưu bản bất biến của **cả** `links_corpus.parquet` và `query.parquet`; tính SHA256 từ bytes của từng file. `dataset_manifest.json` ghi `snapshot_id`, nguồn/revision BTC, thời điểm lấy, path, SHA256, byte size, row count, Parquet schema và adapter version. Mọi run/report tham chiếu `snapshot_id` và config hash; không ghi đè snapshot đang dùng để resume.

Đọc schema thực tế của `query.parquet` trước khi viết adapter. Chuẩn hóa các trường được BTC định nghĩa thành `query_id`, `gold_doc_id` trong `gold_doc_map.parquet` (một row/quan hệ); không đoán tên cột hay tự đổi relevance label. Giữ nguyên kiểu/giá trị ID khi join. Nếu có nhiều split, ghi split và chỉ dùng nhãn được phép truy cập.

Kiểm tra mọi gold ID có trong corpus; liệt kê `gold_doc_ids_missing_from_inventory`, URL invalid và duplicate mappings. Dùng gold để **kiểm tra coverage**, không dùng nhãn relevance để tùy chỉnh nội dung hay quy tắc riêng cho từng gold doc. Schema nguồn đã xác minh khi triển khai: query chỉ có `id/query`, nên bước này chờ qrels riêng từ BTC; không coi bảng gold rỗng là coverage đạt.

## 5.2. URL normalization v1

Chỉ làm các biến đổi không ảnh hưởng semantics request:

- strip whitespace;
- lowercase scheme;
- lowercase hostname;
- remove fragment `#...`;
- normalize empty path nếu cần.

Không tự động xóa `?id=`, `?page=`, `?lang=`, `?article=` hay tracking parameters ở v1. Query parameters phải được bảo toàn trừ khi Stage A/B chứng minh an toàn.

## 5.3. URL validation

Status:

```text
VALID_HTTP_URL
INVALID_URL
UNSUPPORTED_SCHEME
EMPTY_URL
```

Chỉ support `http`, `https`. Không xóa row lỗi; ghi vào `invalid_urls.parquet`.

## 5.4. Unique crawl URL

Group theo `normalized_url`, sinh:

```text
crawl_url_id = SHA256(normalized_url)
```

Tạo:

`unique_urls.parquet`:

```text
crawl_url_id
original_url_example
fetch_url
domain
doc_count
```

`url_doc_map.parquet`:

```text
crawl_url_id
doc_id
original_url
```

Không dùng `dont_filter=True` để xử lý duplicate corpus; dedupe trước crawler hiệu quả hơn và tránh tải cùng nội dung nhiều lần.

## 5.5. Domain stats

Tạo `domain_stats.parquet`:

```text
domain
unique_url_count
doc_id_count
percentage
gold_doc_id_count
stage_a_sampled_urls
measured_terminal_urls_per_second
measured_mean_request_seconds
blocked_rate
estimated_download_hours
```

Dùng để đánh giá top domains, block rate và nhu cầu domain-specific extractor.

Các cột đo lường để null trước pilot, không điền 0 như một kết quả đã đo. `domain` là hostname; khi tổng hợp họ domain/subdomain phải ghi rõ group key. Ước lượng theo từng download slot và retry overhead theo §26.

---

# 6. Crawl shard builder

Không đưa toàn bộ vài triệu URL vào một pilot job.

Stage B baseline:

```text
~100k unique URLs/shard
```

Range dự kiến:

```text
50k–200k/shard
```

Single-machine baseline dùng:

```text
int(crawl_url_id, 16) % N_SHARDS
```

để phân bố domain tương đối đều.

Không dùng Python built-in `hash()` vì kết quả có thể thay đổi giữa process. Freeze `N_SHARDS` và thứ tự URL trong snapshot. Hash-sharding khiến domain lớn xuất hiện trong gần như mọi shard: baseline **một process, chạy các shard tuần tự**. Nó không làm giảm tổng thời gian xử lý domain đó; ETA tuần tự phải cộng thời gian từng shard (§26).

Nếu chạy nhiều Scrapy processes, phải nhớ `CONCURRENT_REQUESTS_PER_DOMAIN` là giới hạn theo process. Không scale processes trước khi có cơ chế phối hợp domain rate.

---

# 7. Scrapy Spider contract

Spider `CorpusSpider` nhận:

```text
--shard data/crawl_shards/shard_00012.parquet
```

Spider chỉ tạo Request từ frontier cố định của BTC.

Không:

```python
response.follow(...)
```

Không discover URL mới.

Với Scrapy 2.13.4, implement `async def start(self)` và đọc Parquet theo batch; chỉ thêm `start_requests()` nếu thật sự hỗ trợ bản <2.13. Tạo Request với callback/errback rõ ràng và `dont_filter=False`. [Scrapy 2.13 — Spider.start](https://docs.scrapy.org/en/2.13/topics/spiders.html#scrapy.spiders.Spider.start)

Request phụ trợ `robots.txt` và HTTP redirects của middleware được accounting riêng; không xem đó là frontier mới hay corpus documents.

Request chỉ mang metadata nhỏ, serializable:

```text
crawl_url_id
fetch_url
domain
shard_id
run_id
attempt_no
```

Không truyền DataFrame, DB connection, open file, large `doc_id` list hoặc object không pickle được trong `meta/cb_kwargs`.

---

# 8. JOBDIR và resume

Mỗi shard có JOBDIR riêng:

```text
crawl_jobs/shard_00012/
```

Run/resume cùng command:

```text
scrapy crawl corpus \
  -a shard=data/crawl_shards/shard_00012.parquet \
  -s JOBDIR=crawl_jobs/shard_00012
```

Không share JOBDIR giữa jobs. Không upgrade/downgrade Scrapy giữa lúc cần resume cùng JOBDIR.

Không cấu hình `JOBDIR_SYNC_EVERY`: không có setting này trong baseline đã đối chiếu. Dùng `JOBDIR`, bật `SCHEDULER_DEBUG` ở pilot để phát hiện Request không serialize được. Shutdown sạch là luồng resume bình thường; crash có thể làm mất in-flight requests, nên reconcile frontier với manifest khi khởi động lại. [Scrapy — Jobs](https://docs.scrapy.org/en/2.13/topics/jobs.html)

JOBDIR chỉ là resume layer; source of truth lâu dài là raw cache + manifest đã compact. Retry batch dùng frontier/JOBDIR mới để request lỗi không bị dupefilter cũ chặn; không xóa JOBDIR đang chạy.

---

# 9. Crawl settings

Các settings cần cấu hình:

```text
CONCURRENT_REQUESTS
CONCURRENT_REQUESTS_PER_DOMAIN
DOWNLOAD_TIMEOUT
DOWNLOAD_MAXSIZE
DOWNLOAD_WARNSIZE
DOWNLOAD_SLOTS
CONCURRENT_REQUESTS_PER_IP
REACTOR_THREADPOOL_MAXSIZE
DNSCACHE_ENABLED
DNSCACHE_SIZE
RETRY_ENABLED
RETRY_TIMES
RETRY_HTTP_CODES
AUTOTHROTTLE_ENABLED
AUTOTHROTTLE_START_DELAY
AUTOTHROTTLE_MAX_DELAY
AUTOTHROTTLE_TARGET_CONCURRENCY
ROBOTSTXT_OBEY
USER_AGENT
```

Pilot:

```text
global concurrency: 64
per-domain concurrency: 2
AutoThrottle: enabled
target concurrency: 1.5
```

Điều chỉnh dựa trên 429/403/timeout, latency, CPU, bandwidth và disk I/O.

Map rõ `max_response_bytes → DOWNLOAD_MAXSIZE=52428800`, `warn_response_bytes → DOWNLOAD_WARNSIZE=10485760`; `CONCURRENT_REQUESTS_PER_IP=0` để dùng slot theo domain. `64 × 50 MiB = 3.125 GiB` chỉ là lượng body có thể cùng tồn tại, chưa tính bản sao, decompression, queues và extraction workers; không phải trần RAM của process. Giới hạn queue raw-write và đo peak RSS ở Stage B.

Sau Stage A cho phép `DOWNLOAD_SLOTS` override **hostname thực tế** của top domain, ví dụ `{"example.org": {"concurrency": 4, "delay": 0.5, "randomize_delay": True}}`. Giá trị này chỉ là ví dụ; tăng từng bước dựa trên latency/block rate và robots policy. AutoThrottle target 1.5 vẫn có thể giữ throughput thấp dù tăng concurrency. [Scrapy — Download slots và size limits](https://docs.scrapy.org/en/2.13/topics/settings.html#download-slots), [AutoThrottle](https://docs.scrapy.org/en/2.13/topics/autothrottle.html)

DNS cache vốn mặc định bật; vẫn set rõ trong config. Threadpool mặc định 10, pilot thử 32 và benchmark 10/32/64 nếu resolver/blocking I/O là bottleneck; không khẳng định 10 luôn quá thấp. [Scrapy — DNS và reactor threadpool](https://docs.scrapy.org/en/2.13/topics/settings.html#reactor-threadpool-maxsize)

### 429 và Retry-After

RetryMiddleware baseline không đọc `Retry-After`; AutoThrottle điều chỉnh delay theo latency, không thực hiện cooldown theo header. [RetryMiddleware source](https://docs.scrapy.org/en/2.13/_modules/scrapy/downloadermiddlewares/retry.html), [AutoThrottle algorithm](https://docs.scrapy.org/en/2.13/topics/autothrottle.html#throttling-algorithm)

P0 loại 429 khỏi retry tức thì. Callback/middleware ghi `HTTP_429`, header gốc và `retry_not_before`; parse cả delta-seconds và HTTP-date. Nếu header thiếu/sai, dùng exponential backoff + jitter, bắt đầu 60 giây. Không rút ngắn một Retry-After hợp lệ theo delay cap của AutoThrottle.

Lưu cooldown theo slot trong trạng thái bền vững. Scheduler/gate hoãn request **chưa gửi** của slot đó, vẫn phục vụ domain khác; không `sleep()` trong reactor. Các request đã in-flight có thể hoàn tất. Retry ở batch mới khi hết cooldown, tối đa 3 batch trước review; account riêng số lần retry của middleware và batch. 503 có Retry-After cũng đi qua cooldown trước RetryMiddleware.

Default plan: `ROBOTSTXT_OBEY=True`. Nếu robots từ chối thì ghi `ROBOTS_DENIED`, không tự bypass. Nếu BTC có cache chính thức hoặc quyền crawl đặc biệt, cập nhật theo rule chính thức.

---

# 10. Response handling

## HTTP success

200–299: lưu raw nếu content type support.

## Redirect

Cho RedirectMiddleware xử lý, manifest lưu `fetch_url`, `final_url`, redirect count.

## HTTP errors

404/403/429/5xx đều phải tạo terminal/failure record. Không được mất record chỉ vì callback bị HttpErrorMiddleware chặn; implementation phải đảm bảo final HTTP status được accounting.

Baseline đặt **`HTTPERROR_ALLOW_ALL=True` trong settings**, để HttpErrorMiddleware chuyển HTTP lỗi vào callback. **Không đặt `meta["handle_httpstatus_all"]=True`**: RedirectMiddleware của Scrapy 2.13.4 cũng kiểm tra cờ này và sẽ bỏ qua redirect, gây mất coverage. Regression test phải kiểm tra URL redirect có `SUCCESS_RAW`, đúng `final_url/redirect_count` và giữ cả outcome khi target cũng nằm trong frontier. Callback ghi status cuối sau RetryMiddleware, không extract error body như tài liệu thành công. Phân biệt `HTTP_OTHER`, `HTTP_429` chờ batch và `retry_exhausted` boolean; không dùng một status chung làm mất nguyên nhân HTTP/network.

### Robots-denied

RobotsTxtMiddleware đưa request bị chặn vào errback với `IgnoreRequest("Forbidden by robots.txt")`. Map trường hợp này riêng thành `ROBOTS_DENIED`, `http_status=null`, `attempt_count=0` nếu chưa gửi request corpus. Không gán mọi `IgnoreRequest` là robots-denied: middleware khác cũng dùng exception này. Với baseline có thể match type + reason đã kiểm chứng; tốt hơn là gắn reason riêng trong adapter middleware. [RobotsTxtMiddleware source](https://docs.scrapy.org/en/2.13/_modules/scrapy/downloadermiddlewares/robotstxt.html)

## Network failures

Errback map về:

```text
TIMEOUT
DNS_ERROR
SSL_ERROR
CONNECTION_ERROR
OTHER_NETWORK_ERROR
REQUEST_IGNORED
```

Retryable status để RetryMiddleware xử lý trước khi ghi final failure.

Download vượt size cap có thể fail trước callback: adapter download handler/signals phải đánh dấu nguyên nhân và map `RESPONSE_TOO_LARGE`, kể cả stream không có Content-Length. Không coi mọi cancellation là quá cỡ; xác nhận đường đi exception trên đúng pin bằng mock `/large` và `/large-stream`.

---

# 11. Content-Type Router

Support P0:

```text
text/html
application/xhtml+xml
application/pdf
text/plain
```

Optional XML sau Stage A.

Không tải/lưu vô hạn image/video/audio/archive không cần thiết. Download cap pilot **50 MiB**; vượt ngưỡng ghi `RESPONSE_TOO_LARGE`.

Router dùng **header + sniffing bytes**, không chỉ `Content-Type` hay đuôi URL:

1. Đọc prefix giới hạn (ví dụ 8 KiB), nhận `%PDF-` gần đầu file → PDF dù header khai HTML/octet-stream.
2. Nhận BOM, `<!doctype html>`, `<html`, `<head>/<body>` hoặc fragment HTML đáng tin cậy → HTML; xử lý probe UTF-16/32 theo BOM nếu cần.
3. Plain text chỉ khi BOM/charset probe hợp lý và không có dấu hiệu binary; unknown/binary không ép decode thành HTML.
4. Giữ `declared_content_type`, `effective_content_type`, `sniff_reason`, `content_type_mismatch`, `declared_charset`, `detected_charset`, `charset_detection_method`. Probe charset là metadata, không thay raw bytes; HTML truyền bytes cho Trafilatura, plain text decode ở worker (§17).

Charset detection là heuristic, không đảm bảo VI/ZH luôn đúng. Đo replacement-character ratio và kiểm tra fixture UTF-8, Windows-1258, GB18030/Big5. Header sai nhưng HTML hợp lệ vẫn được giữ; HTML soft-404/challenge HTTP 200 đi qua quality checks, không được coi thành công usable chỉ vì sniff đúng HTML.

---

# 12. Raw Cache Pipeline

Spider callback yield `CrawlItem` với:

```text
crawl_url_id
fetch_url
final_url
domain
http_status
content_type
declared_content_type
effective_content_type
declared_charset
sniff_reason
body
elapsed_ms
shard_id
```

`RawCachePipeline`:

1. validate body;
2. chọn extension/storage;
3. compute raw SHA256;
4. write temp file;
5. atomic rename;
6. set `raw_path`;
7. set sizes/checksum;
8. xóa `body` khỏi item;
9. pass item sang ManifestPipeline.

Không ghi success manifest trước khi raw write thành công.

### Raw path

Không để vài triệu file trong một directory. Dùng hash prefix:

```text
raw/html/a8/f3/<raw_sha256>.html.zst
```

### Compression

HTML/text: `zstd level 1` pilot. PDF lưu binary trực tiếp. Stage A đo compression ratio/throughput rồi mới chốt.

Path theo checksum nội dung để retry không ghi đè bytes của lần tải trước; một raw blob có thể được nhiều URL tham chiếu. Lưu checksum trên bytes trước nén, verify lại sau giải nén.

### File lẻ và đóng gói theo shard

Khoảng 4.4M URL không đồng nghĩa chính xác 4.4M raw files (còn URL dedup, failure và shared blobs), nhưng vẫn có thể tạo hàng triệu file nhỏ. Hash directories giải quyết kích thước directory, chưa giải quyết chi phí metadata/copy/upload.

Stage B so sánh file lẻ với immutable container theo shard: tar chứa các member `.zst` độc lập, hoặc WARC có index. Nếu dùng cả `tar.zst`, cần thiết kế seek/index hoặc đọc tuần tự theo shard; không giả định offset trong stream nén cho random access. Giữ `raw_path`, `raw_member`, `raw_locator_version`, container checksum và member `raw_sha256` trong manifest.

Container chỉ được publish sau close + flush + atomic rename + index commit; success manifest chỉ trỏ tới container đã commit. Giữ file lẻ cho recovery trong lúc pack và chỉ garbage-collect sau khi verify mọi reference. Tại 100k URL đo file/directory count, inode usage trên Linux (NTFS đo metadata/file count), pack/unpack throughput và **thời gian copy/upload tới đích thực tế**; chọn layout trước full-scale.

---

# 13. Crawl Manifest Pipeline

Không ghi một JSON lớn. Dùng append-only JSONL shards:

```text
crawl_manifest/part-00000.jsonl
crawl_manifest/part-00001.jsonl
...
```

Rotate khoảng 50k records/part, sau crawl compact sang `crawl_manifest.parquet`.

Schema:

```text
crawl_url_id
event_id
snapshot_id
config_sha256
run_id
attempt_no
event_seq
shard_id
fetch_url
final_url
domain
status
http_status
content_type
declared_content_type
effective_content_type
content_type_mismatch
declared_charset
detected_charset
charset_detection_method
sniff_reason
raw_path
raw_member
raw_locator_version
raw_sha256
raw_size_bytes
compressed_size_bytes
attempt_count
retry_exhausted
retry_after
retry_not_before
redirect_count
elapsed_ms
error_type
error_message
crawl_timestamp
source_provenance
```

Status chuẩn:

```text
SUCCESS_RAW
HTTP_404
HTTP_403
HTTP_429
HTTP_5XX
HTTP_OTHER
ROBOTS_DENIED
TIMEOUT
DNS_ERROR
SSL_ERROR
CONNECTION_ERROR
OTHER_NETWORK_ERROR
REQUEST_IGNORED
RESPONSE_TOO_LARGE
EMPTY_RESPONSE
UNSUPPORTED_CONTENT_TYPE
RAW_WRITE_FAILED
UNKNOWN_ERROR
```

Pipeline order:

```text
RawCachePipeline       100
CrawlManifestPipeline  200
```

### Compaction và crash recovery

- JSONL là event log; retry batch có thể sinh nhiều record cho một `crawl_url_id`. Cấp `attempt_no` tăng đơn điệu từ trạng thái bền vững, `event_seq` tăng trong attempt; `event_id` duy nhất và replay cùng event không tạo attempt mới. Baseline không chạy hai attempt cùng URL đồng thời.
- Compact thành **một trạng thái mới nhất/URL**, lấy max `(attempt_no, event_seq)`; timestamp UTC chỉ dùng audit, không làm khóa thứ tự duy nhất. Bỏ duplicate `event_id`; cùng ordering key nhưng khác payload là lỗi cần report. Không lấy record đầu tiên hay tùy ý ưu tiên SUCCESS cũ. Nếu cần dùng lại success cũ, xuất view `last_success` riêng có provenance.
- Writer ghi UTF-8, mỗi record kết thúc newline; chỉ một writer/part và checkpoint vị trí đã commit. Reader chấp nhận record cuối thiếu newline nếu JSON vẫn hợp lệ; thêm newline trước append tiếp. Chỉ bỏ qua **dòng cuối không hoàn chỉnh ở part đang mở sau crash**, ghi byte offset/count vào recovery report rồi truncate/quarantine tail trước append; JSON lỗi giữa file hoặc ở part đã seal phải báo corruption.
- Compact ghi temp Parquet rồi atomic rename; kết quả rerun phải giống nhau. Reconcile `SUCCESS_RAW` với raw checksum, orphan blobs và frontier; thiếu raw/manifest không được đánh dấu đã hoàn thành.

---

# 14. Manifest-driven recovery

Không phụ thuộc hoàn toàn vào JOBDIR.

Nếu JOBDIR mất/hỏng:

1. đọc shard gốc;
2. đọc crawl manifest;
3. compact latest trạng thái theo §13, verify raw của `SUCCESS_RAW`;
4. loại khỏi pending set;
5. tạo job mới cho phần thiếu/retryable.

Terminal thường gồm:

```text
SUCCESS_RAW
HTTP_404
ROBOTS_DENIED
UNSUPPORTED_CONTENT_TYPE
RESPONSE_TOO_LARGE
```

Có thể retry theo batch:

```text
HTTP_429
HTTP_5XX
TIMEOUT
DNS_ERROR
CONNECTION_ERROR
```

403 không auto-bypass; manual review domain.

`HTTP_403` và `ROBOTS_DENIED` dừng retry trong attempt hiện tại, không có nghĩa chấp nhận mất coverage vĩnh viễn. Quyết định nguồn thay thế hoặc chấp nhận thiếu theo Stage A gate (§24). Retry-exhausted giữ nguyên status cuối và cờ `retry_exhausted=true`.

---

# 15. Stage 2 — Trafilatura extraction offline

Input:

```text
crawl_manifest.parquet
+
raw files
```

Chỉ process `SUCCESS_RAW`.

Trafilatura chạy ở process riêng, không nằm trong Scrapy reactor.

```text
raw HTML queue
     ↓
multiprocessing / ProcessPoolExecutor
     ↓
N extraction workers
```

Worker count benchmark theo CPU/RAM.

### Bảo vệ từng document

Trước parse, kiểm tra input size theo MIME (§4), giải nén có giới hạn output, kiểm checksum; không nạp toàn bộ blob quá cỡ rồi mới kiểm tra. HTML/text pilot 10 MiB, PDF 50 MiB. Quá extraction cap → `EXTRACTION_TOO_LARGE`, vẫn giữ raw và mapping.

Supervisor đo wall-clock từ lúc task bắt đầu: HTML/text 30 giây, PDF 60 giây, tính chung first pass + fallback. Hết hạn phải terminate/kill rồi join **process đang chạy task**, thay worker và ghi `EXTRACTION_TIMEOUT`; không để parser treo giữ slot. Task lỗi/crash không kéo mất các task khác; queue bounded và checkpoint sau từng kết quả commit.

`Future.result(timeout=...)` chỉ dừng chờ, `cancel()` không dừng task đã chạy. P0 dùng supervisor quản lý process để enforce timeout; nếu dùng ProcessPoolExecutor phải thiết kế restart pool và requeue task chưa commit. [Python — Future timeout/cancel](https://docs.python.org/3.12/library/concurrent.futures.html#future-objects)

Recycle worker mỗi 500 doc (`max_tasks_per_child` hoặc bộ đếm tương đương ở supervisor), benchmark RSS theo thời gian. ProcessPoolExecutor có tham số này từ Python 3.11, dùng `spawn` và không tương thích `fork`; multiprocessing.Pool dùng tên `maxtasksperchild`. [Python — ProcessPoolExecutor](https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor)

---

# 16. Extraction HTML, cấu trúc và fallback

Baseline 2.1.0: `fast` hợp lệ; `no_fallback` và tham số `as_dict` đã deprecated. `bare_extraction()` mặc định trả `Document | None`; khi cần dict dùng `doc.as_dict()`. `fast=True` bỏ external backup, nhưng baseline rescue cho text ngắn vẫn có thể chạy. [Trafilatura 2.1.0 — core source](https://github.com/adbar/trafilatura/blob/v2.1.0/trafilatura/core.py)

Truyền **bytes** để extractor còn thông tin encoding, không decode sớm bằng UTF-8 `errors="ignore"`/`"replace"`. Điều này hỗ trợ detection, không đảm bảo đoán charset đúng. [Trafilatura 2.1.0 — encoding utilities](https://github.com/adbar/trafilatura/blob/v2.1.0/trafilatura/utils.py)

Sau audit native 10k, implementation v2 giữ raw bytes bất biến nhưng resolve encoding **strict trong worker**: BOM/UTF-8 hợp lệ/meta/header; GB2312/GBK dùng GB18030. Không dùng `errors="ignore"`. Ghi `extraction_charset`, `encoding_method`, `encoding_conflict`; parser nhận bytes UTF-8 chuẩn hóa, không thay raw/checksum đã lưu.

Adapter production lấy vùng DOM chính đã đối chiếu raw cho các domain lỗi (120ask, Wujue, Cnkang, Familydoctor, Báo Cần Thơ/An Giang, Medlatec, MSD). Nếu không có vùng tin cậy hoặc vùng chỉ chứa comment/không có text, dùng Trafilatura **normal-first** (`fast=False`); nếu normal trả None hoặc text rỗng thì thử primary path `fast=True` trong cùng budget timeout. Trang ngắn hợp lệ không bị bỏ chỉ vì dưới 500 ký tự.

Gọi adapter đang được kiểm thử thay vì copy serializer minh họa cũ:

```python
from src.extraction.html import extract_html

result = extract_html(
    raw_bytes,
    final_url,
    config,
    declared_charset=declared_charset,
    fetch_url=original_fetch_url,
)
```

`bare_extraction` trả Document, không phải chuỗi Markdown; adapter render `doc.body`. Renderer v2 trong `src/extraction/structure.py` giữ H2/H3 lồng strong/span, visible link text, nested lists, table group labels, rowspan/colspan và footnotes; bỏ comment/script/style. `structure_source` ghi nhánh DOM/Trafilatura/fast recovery thực tế; `implementation_version` và code SHA256 trong extraction identity ngăn tái sử dụng checkpoint của code cũ. Hai publisher đã kiểm tra dùng dòng wholly-bold làm section được suy ra H3 và ghi `inferred_heading_count`; không suy heading toàn corpus từ mọi dòng bold.

`text_markdown` giữ heading/list/table trong main body; `title` là metadata riêng, không giả định `<title>` luôn được chèn thành `#`. Downstream có thể thêm title context một lần khi chunk. Markdown không phục hồi mọi cấu trúc của HTML: kiểm tra heading level, nested lists, table cells và biomedical inline symbols. [Trafilatura — Python output formats](https://trafilatura.readthedocs.io/en/latest/usage-python.html#output)

Không chọn candidate bằng độ dài một mình. HTTP 200 cookie/reload challenge, article redirect về homepage, template chưa render và navigation-only được ghi status/reason riêng, QUARANTINE và giữ raw/ID để xử lý sau. `doc_outcomes.handoff_eligible` yêu cầu EXTRACT_SUCCESS và quality policy phù hợp; LOW hợp lệ vẫn giữ trong benchmark khi không có qrels. Kiểm tra source fidelity bằng raw review và paired regression, không coi HIGH/MEDIUM là điểm chất lượng retrieval.

Stage A benchmark thêm `favor_precision` và `favor_recall`; không chọn bằng cảm tính.

Retrieval-oriented policy:

```text
comments: bỏ
navigation/footer: bỏ
outbound URLs: bỏ
table text: giữ
title: giữ
main article body: giữ
```

Title nullable; `text` là plain text dùng cho quality/hash/LID, `text_markdown` là input chunk theo cấu trúc. Cả hai required với HTML valid. PDF/plain text không đảm bảo heading hierarchy, phải ghi `structure_source` và handoff format rõ ràng.

---

# 17. PDF/Text extractors

PDF dùng PyMuPDF:

```text
raw PDF
↓
page text extraction
↓
join pages
```

Lưu `page_count`. Nếu page_count > 0 nhưng extracted text rất thấp → `LIKELY_SCANNED_PDF`; không OCR trong P0.

PyMuPDF chạy trong process có timeout/size cap; close document trong `finally`, extract theo page và giữ page boundaries. PDF hỏng hoặc worker chết ghi status riêng. Không tự suy heading cấp HTML từ font size trong P0; `text_markdown` có thể bằng plain text, `structure_source=pdf_pages`.

`text/plain`: decode robustly, Unicode normalize, whitespace cleanup; không đưa qua Trafilatura.

Decoder ưu tiên BOM, thử charset khai báo nếu hợp lệ và decode strict; khi không phù hợp thử UTF-8/charset detector. Ghi charset/method/confidence và replacement ratio; không `errors="ignore"`. Thay ký tự chỉ là fallback có flag, severe decode failure → QUARANTINE. `structure_source=plain_text`, `text_markdown=text`.

---

# 18. Cleaning v1

Chỉ làm:

```text
Unicode NFC
normalize line endings
collapse excessive blank lines
normalize repeated spaces
strip leading/trailing whitespace
```

Không remove punctuation/digits/diacritics, không lowercase toàn corpus, không translate/stem/lemmatize.

Cleaning áp dụng riêng cho plain text và Markdown. Với Markdown chỉ normalize NFC/line endings/blank lines ngoài block cần bảo toàn; giữ indentation list/code, `#`, dấu `|` và table separators, khoảng trắng có nghĩa và paragraph boundaries. Không chạy `re.sub(r"\s+", " ", ...)` lên `text_markdown`; whitespace normalization mạnh chỉ dùng cho hash view, không ghi đè nội dung handoff.

Phải giữ biomedical tokens như:

```text
H. pylori
HbA1c
BRCA1
HER2
SARS-CoV-2
SpO₂
ICD-10
0.5 mg
mg/kg
Na+
K+
```

---

# 19. Basic quality tiers

Tính:

```text
char_count
line_count
paragraph_count
title_present
replacement_char_ratio
heading_count
list_item_count
table_count
boilerplate_score
error_page_signals
quality_reason
```

Detect obvious error page. Tiers:

```text
HIGH
MEDIUM
LOW
QUARANTINE
```

Threshold pilot ví dụ:

```text
QUARANTINE: no text/error page/severe decode failure
LOW: < 200 chars
MEDIUM: 200–999 chars
HIGH: >= 1000 chars
```

Không freeze threshold trước Stage A manual review.

Độ dài là feature, không đủ để kết luận chất lượng: long cookie/challenge/soft-404 page vẫn QUARANTINE khi có bằng chứng. Tính `char_count` trên **plain main text**, không trên ký hiệu Markdown hay metadata front matter.

Trước khi bỏ LOW khỏi chunk, join gold mapping với quality tier, report `gold_low_doc_ids`, gold bị QUARANTINE và lý do. Stage A audit gold sample, Stage B mở rộng; nếu có gold LOW hợp lệ hoặc phần gold chưa audit đủ, baseline giữ **LOW hợp lệ** trong benchmark handoff, chưa bật lọc production. Chỉ freeze chính sách chung sau khi đo coverage loss và manual review; không tạo ngoại lệ theo từng gold ID. Retain tất cả LOW/raw/mappings để đổi threshold mà không crawl lại.

---

# 20. Language detection

Detect:

```text
vi
en
zh
mixed
other
unknown
```

Không dùng domain làm final label. Language metadata dùng cho EDA, quality report và downstream retrieval routing.

P0 benchmark heuristic nhẹ trên sample text: CJK ratio, dấu tiếng Việt, các stopwords VI/EN và tín hiệu script; không suy Latin không dấu luôn là EN hay mọi CJK là ZH (có thể JA/KO). Short/ambiguous → `unknown`; `mixed` cần đánh giá theo nhiều đoạn, không chỉ top-1 toàn doc.

Ứng viên thay thế là **fastText `lid.176.ftz`** (917 kB) hoặc `lid.176.bin` (126 MB); các model yêu cầu UTF-8. Load một lần/worker, lưu model name + SHA256, benchmark docs/sec, peak RSS và accuracy VI/EN/ZH. [fastText — Language identification models](https://fasttext.cc/docs/en/language-identification.html)

Lingua chỉ là ứng viên benchmark nếu cần; chưa có số đo để khẳng định luôn quá nặng. `language_confidence` là model score khi có; heuristic để null kèm `language_method` và các ratio, không bịa probability. Không dùng `target_language` để âm thầm lọc corpus.

---

# 21. Extraction manifest và storage

Extraction manifest:

```text
crawl_url_id
raw_sha256
snapshot_id
extraction_run_id
extraction_attempt_no
event_seq
event_id
extractor_config_sha256
extract_status
extractor
extractor_version
title
language
language_confidence
language_method
structure_source
heading_count
list_item_count
table_count
char_count
paragraph_count
replacement_char_ratio
quality_tier
quality_reason
content_hash
processed_path
extraction_elapsed_ms
error_type
error_message
extraction_timestamp
```

Statuses:

```text
EXTRACT_SUCCESS
EMPTY_MAIN_TEXT
TRAFILATURA_FAILED
PDF_PARSE_FAILED
TEXT_DECODE_FAILED
LIKELY_SCANNED_PDF
UNSUPPORTED_EXTRACTION
CLEANING_FAILED
EXTRACTION_TOO_LARGE
EXTRACTION_TIMEOUT
WORKER_CRASHED
RAW_CHECKSUM_FAILED
UNKNOWN_ERROR
```

Output structured batches thẳng sang Parquet:

```text
processed/extracted/part-00000.parquet
processed/extracted/part-00001.parquet
...
```

Fields:

```text
crawl_url_id
final_url
domain
title
text
text_markdown
text_format
structure_source
language
language_confidence
language_method
content_type
char_count
paragraph_count
heading_count
list_item_count
table_count
replacement_char_ratio
quality_tier
quality_reason
content_hash
```

Extraction idempotence key: `(crawl_url_id, raw_sha256, extractor_version, extractor_config_sha256)`. Lưu event history; compact latest attempt/event trong **một extraction run/config được chọn**, không trộn kết quả giữa version. Reader JSONL và commit Parquet theo cùng quy tắc crash-safe ở §13. `text_format=markdown` cho HTML, `plain` cho PDF/text P0; luôn giữ `text` plain và `text_markdown` theo contract.

---

# 22. Exact content dedup

Sau cleaning:

```text
content_hash = SHA256(normalized_text_for_hash)
```

Hash normalization chỉ Unicode NFC + whitespace normalization + strip. Không lowercase/remove punctuation.

Hash **plain text**, dùng view riêng và version `hash_normalization_version`; không hash YAML metadata hay Markdown markers, không ghi đè paragraph/heading trong output. Exact dedup ở đây là equality của view đó, không cam kết HTML structure giống hệt nhau.

Group theo content hash và dùng nó làm `canonical_content_id` hoặc map sang stable ID khác.

Hai tầng mapping:

```text
doc_id → crawl_url_id
crawl_url_id → canonical_content_id
```

Cuối cùng:

```text
canonical_content_id → [doc_id...]
```

Downstream có thể embed canonical content một lần nhưng vẫn trả được mọi BTC ID tương ứng.

### Representative xác định

Chọn `representative_crawl_url_id` bằng sort cố định: quality rank `HIGH > MEDIUM > LOW > QUARANTINE`, rồi replacement ratio thấp hơn, title hiện diện, `min(doc_id)` gắn với URL nhỏ hơn, cuối cùng `crawl_url_id` lexicographic. Quy tắc so `doc_id` phải theo kiểu BTC đã khai báo (numeric hay string); không cast tùy tiện. Freeze policy version; không phụ thuộc thứ tự worker hoàn tất. Dùng title/Markdown/metadata của representative và giữ mapping của mọi URL.

### Widespread duplicates và soft-404

Đếm distinct `crawl_url_id` và distinct domain cho mỗi content hash. Pilot `>=100 URL` trên `>=5 domain` → flag `widespread_duplicate` và đưa vào review queue. Khi kèm error/cookie/challenge/template signals hoặc review xác nhận boilerplate, chuyển cả nhóm sang `QUARANTINE` với reason rõ ràng. Tài liệu được mirror hợp lệ có thể xuất hiện rộng rãi, nên không tự coi số lượng duplicate là bằng chứng soft-404.

Report gold doc bị ảnh hưởng trước handoff; không xóa blobs, mapping hay IDs của nhóm bị quarantine. Representative vẫn được chọn deterministic kể cả nhóm bị loại khỏi production chunk.

Quality sau dedup là **effective quality** trong documents/doc_outcomes và gold report; giữ pre-dedup tier trong extraction history để audit. Quarantine override của nhóm phải được propagate tới mọi URL/doc_id trong nhóm, không để report vẫn tính chúng là handoff eligible.

---

# 23. Final output contract

`documents.parquet` — mỗi canonical content một record:

```text
canonical_content_id
representative_crawl_url_id
url
final_url
domain
title
text
text_markdown
text_format
structure_source
language
language_confidence
language_method
content_type
char_count
paragraph_count
quality_tier
quality_reason
content_hash
hash_normalization_version
extractor
extractor_version
extractor_config_sha256
```

`content_doc_map.parquet`:

```text
canonical_content_id
crawl_url_id
doc_id
```

Chunking stage chỉ cần đọc hai dataset này, không cần raw HTML.

HTML chunk theo `text_markdown`, dùng `title` làm context; `text` dành cho plain-text consumers và kiểm tra chất lượng. PDF/plain text dùng page/paragraph boundaries theo `structure_source`. Không đổi ý nghĩa field `text` thành Markdown mà thiếu schema version.

`doc_outcomes.parquet` accounting **mọi BTC doc_id**, kể cả invalid URL, crawl failure, extraction failure, LOW và QUARANTINE: `doc_id`, nullable `crawl_url_id`/`canonical_content_id`, latest crawl/extract status, quality tier, handoff eligibility và reason. `content_doc_map` chỉ có content đã extract/hash được; không thể dùng riêng file này để chứng minh không mất doc_id.

Baseline handoff:

```text
HIGH + MEDIUM → chunk
LOW hợp lệ → benchmark; production policy pending gold audit
QUARANTINE → không chunk production
```

---

# 24. Stage A — 1,000 URLs

Target 1,000 **unique** URLs: khoảng 800 stratified + tối đa 200 gold-associated URLs, dedupe rồi bù đủ sample. Freeze seed và sample inventory; báo số doc_id/query tương ứng vì nhiều ID có thể dùng chung URL. Audit riêng gold invalid/missing trong inventory, không đưa URL invalid vào request set.

Sample phải stratified theo:

```text
top domains
medium domains
long-tail domains
HTML
PDF
VI-like
EN-like
ZH-like
```

Stage A phải trả lời:

1. Bao nhiêu URL request được?
2. Bao nhiêu 404/403/429/timeout?
3. Bao nhiêu robots denied?
4. Bao nhiêu HTML/PDF?
5. Raw trung bình bao nhiêu KB?
6. Compression ratio bao nhiêu?
7. Trafilatura success rate?
8. Fast pass/fallback rate?
9. Nội dung extraction có sạch không?
10. VI/EN/ZH encoding có đúng không?
11. Heading/list/table, title metadata và biomedical tokens có được giữ?
12. Gold doc được tải, extract usable và eligible handoff ở tỷ lệ nào?
13. Top domain có ETA bao lâu theo measured request duration/rate?
14. Content-Type mismatch, worker timeout/size cap và suspicious duplicate có ảnh hưởng gold?

### Gold coverage report

Join `query gold → doc_id → crawl_url_id → latest crawl → latest extraction → handoff`. Ghi riêng count ở mỗi tầng: có trong inventory, URL valid, sample/pending, raw success, usable extraction, HIGH/MEDIUM/LOW/QUARANTINE và eligible chunk.

- `G` là distinct gold IDs của snapshot; report inventory coverage trên toàn `G` và IDs thiếu/invalid.
- `G_A` là gold IDs có URL trong sample cộng gold invalid/missing được chọn audit; `gold_raw_coverage=|raw_success ∩ G_A|/|G_A|`, `gold_usable_coverage` và `gold_handoff_coverage` tương tự. Mẫu số 0 → null, không báo 100%.
- Ghi `gold_audited_count`, `gold_pending_count`, per-domain/per-tier breakdown và danh sách failure reasons. Tỷ lệ quan sát trên sample không được trình bày như coverage toàn corpus.
- Với query có toàn bộ gold IDs đã có outcome (kể cả invalid/missing), report tỷ lệ query có **ít nhất một** gold usable, **toàn bộ** gold usable và mean fraction gold usable/query; query chưa đủ outcome để `pending`, không tính như crawl failure. Gold presence không tự chứng minh nội dung relevant đã được giữ: cần manual review gold samples.

Đây là acceptance metric ưu tiên bên cạnh crawl success chung; không đánh giá coverage bằng số response 200.

`usable` nghĩa là extraction thành công, text có nội dung và effective quality không QUARANTINE; LOW hợp lệ vẫn có thể usable. `handoff_eligible` áp dụng policy/version và mode ghi trong report: Stage A baseline `benchmark`, có thể gồm LOW hợp lệ; khi báo production coverage phải tính lại theo production policy, không dùng tỷ lệ benchmark thay thế.

### Gate khi robots/403 làm mất coverage

`blocked_rate=(ROBOTS_DENIED + HTTP_403)/audited_unique_urls` theo latest status, không đếm retry hai lần. Report raw sample rate và ước lượng có trọng số domain (sample stratified không đại diện đều toàn corpus).

Nếu blocked rate tổng hoặc ở top domain **>10%**, hoặc một nhóm gold quan trọng bị chặn dù tổng <10%, **chưa chuyển Stage B cho phần bị ảnh hưởng**. Lưu `coverage_decisions.json` gồm domain, count/rate, gold impact, lựa chọn và người quyết định:

1. Ưu tiên hỏi BTC về cache/snapshot chính thức hoặc access được cấp; ghi dataset revision và provenance.
2. Nếu BTC cho phép và có bản phù hợp, kiểm tra availability của Common Crawl/Wayback trên sample; đối chiếu original URL, snapshot date, nội dung và quy tắc benchmark. Archive có thể thiếu hoặc lệch phiên bản, không đảm bảo khôi phục gold.
3. Nếu không có nguồn phù hợp, người phụ trách ghi nhận **chấp nhận missing coverage** và tác động tới evaluation. Giữ status/mapping; có thể tiếp tục các domain đã đạt gate.

Không tự bypass robots/403, không tự đổi benchmark sang archive. Nguồn thay thế dùng provenance (`source_kind`, original URL, archive/cache URL, capture timestamp, checksum), không overwrite BTC URL. Quyết định này cần hoàn tất trước scale phần bị chặn; không phải yêu cầu dừng bước viết plan hiện tại.

Manual inspect tối thiểu 100 extracted docs, ưu tiên đủ VI/EN/ZH và problematic docs.

`manual_review.csv`:

```text
crawl_url_id
language
content_ok
title_ok
boilerplate_level
encoding_ok
heading_ok
list_ok
table_ok
biomedical_tokens_ok
is_gold_audit
quality_reason
notes
```

Không scale Stage B nếu resume, raw atomic write, mappings hoặc extraction manual review chưa đạt.

Thêm gate: gold coverage được accounting đầy đủ trên audit set; LOW policy, blocked-domain decisions và baseline dependency lock được ghi nhận. File report phải phân biệt template/chưa đo với kết quả run thật.

---

# 25. Stage B — 10k rồi 100k URLs

### Chốt môi trường trước Stage B

Mục tiêu vận hành P0: **Linux VPS chạy crawler lâu dài, SSD/storage bền vững**. Máy nhà dùng development/pilot; Kaggle chỉ cân nhắc extraction benchmark trên raw snapshot, không chọn mặc định cho crawl/resume dài ngày. Đây là lựa chọn trong plan, chưa có VPS được cấp hay tài nguyên đã đo.

Trước 10k, điền `configs/deployment.yaml`: provider/region, CPU/RAM, OS/Python và lock hash, disk capacity/free space, filesystem/layout, bandwidth/egress limit và chi phí, IP/ASN, nơi backup/transfer, process/worker budgets, checkpoint/shutdown policy. Tài nguyên cụ thể chọn theo Stage A; thiếu projection disk/RAM/network hoặc budget thì chưa chạy Stage B.

Chạy lại một sample Stage A trên **chính môi trường/IP** Stage B để kiểm tra block rate/latency/encoding; kết quả máy nhà không được mặc định áp dụng cho VPS. Lưu environment ID vào report và estimates.

10k đo stability/throughput/storage projection.

Scrapy metrics:

```text
requests/sec
successful responses/sec
terminal unique URLs/sec
requests per terminal URL
MB/sec
P50 latency
P95 latency
mean full request duration per domain
observed concurrency per domain
cooldown time per domain
404/403/429/timeout rates
```

Trafilatura metrics:

```text
docs/sec
CPU utilization
RAM/worker
peak aggregate RSS
timeout/oversize/worker crash rates
fast-pass success %
fallback %
extraction success %
```

100k đo thêm:

```text
JOBDIR growth
raw storage growth
manifest growth
memory stability
restart/resume correctness
exact URL duplicate rate
exact content duplicate rate
gold raw/usable/handoff coverage + pending counts
per-domain ETA and blocked-domain decisions
file/directory count and Linux inode usage
actual copy/upload time, files/sec and MB/sec
packed-shard random/sequential read performance
temporary space during packing/compaction
```

Sau đó mới estimate full corpus.

Đo cùng failure/retry policy và storage layout dự kiến dùng thật. Baseline sequential shards; không nhân throughput theo số process khi chưa có global domain-rate coordination.

---

# 26. Full-scale estimates

### Global throughput và domain bottleneck

```text
estimated_download_hours =
unique_url_count /
measured_terminal_unique_urls_per_second /
3600
```

Rate này accounting cả success/failure của URL gốc và đã gồm retries trong benchmark; không cộng retry multiplier lần nữa. Nếu dự đoán bằng **HTTP attempts/sec**, dùng `U × measured_attempts_per_url / attempts_per_second`. Hai cách phải ghi denominator và giả định rõ ràng.

Per-domain/slot có `U_d` URL, concurrency hiệu dụng `C_d`, mean **full request duration** `L_d` (giây), attempts/URL `A_d`:

```text
nominal_attempts_per_second_d ≈ C_d / L_d
estimated_seconds_d ≈ U_d × A_d / (C_d / L_d)
```

Đây là mô hình lý tưởng; ưu tiên rate đo thực tế, thêm cooldown/delay/bandwidth overhead nếu chưa nằm trong rate. Không dùng riêng latency nhận header của AutoThrottle thay cho full body request time. [Scrapy — AutoThrottle latency definition](https://docs.scrapy.org/en/2.13/topics/autothrottle.html#throttling-algorithm)

Ví dụ **không tính throttle/retry**: `C=2, L=0.5s` → 4 req/s, 500k URL khoảng **1.45 ngày**, 1M khoảng **2.89 ngày**. `L=1s` → 2 req/s, 1M khoảng **5.79 ngày**; `L=2s` → khoảng **11.57 ngày**. Với target concurrency 1.5, throughput có thể thấp hơn. Vì vậy không suy ra 4 req/s chỉ từ `per_domain=2`.

Nếu đủ domain được schedule đồng thời:

```text
lower_bound_wall_time ≈ max(U_total / R_global, max_d(U_d / R_d))
```

Hash shards chạy tuần tự phải tính `sum_s estimated_wall_time_of_shard_s`, vì mỗi shard đều phải chờ domain chậm; không dùng lower bound toàn corpus làm ETA cuối. `domain_stats.parquet` ghi rate, sample size, mean/P95, concurrency/delay config và ETA optimistic/base/pessimistic; top domains thiếu mẫu phải mở rộng benchmark trước freeze.

### Storage, network và extraction

```text
estimated_raw_storage =
unique_url_count
× measured_success_raw_fraction
× measured_mean_compressed_raw_bytes
× 1.3
```

```text
estimated_extraction_hours =
successful_raw_docs /
measured_docs_per_second /
3600
```

Không hard-code trước benchmark.

Disk budget còn gồm extracted/document Parquet, manifests, JOBDIR, indexes và **không gian tạm khi pack/compact**; hệ số 1.3 chỉ là reserve pilot, không thay peak-space measurement. PDF thường nén ít hơn HTML: project theo MIME/domain distribution thay vì chỉ mean của sample thuận tiện.

Kịch bản thô, chưa đo: `4.4M × 120–300 kB/body = 0.53–1.32 TB` **response body trước nén cache**. Không xem đây là wire traffic hay disk nén đã xác nhận; HTTP compression, retries, failure body và archive fetch làm network bytes khác. Stage B phải đo bytes tải thực, compressed bytes, transfer đích và bandwidth/egress cost trên environment đã chốt.

---

# 27. Security constraints

URL list phải coi là untrusted input trong deployment.

Không cho request tới:

```text
localhost
127.0.0.0/8
private RFC1918 networks
link-local
cloud metadata endpoints
```

nếu crawler chạy trong cloud/internal network.

Không forward credentials/cookies giữa domains. Không giả browser User-Agent để bypass protections; dùng UA rõ ràng.

---

# 28. Dependencies

API baseline được đối chiếu cho plan:

```text
Python 3.11.x
scrapy==2.13.4
trafilatura==2.1.0
```

Chọn exact pin để implementation có signature cụ thể; không dùng tài liệu `latest` để suy API của pin. Khi đổi baseline, cập nhật adapter + fixtures + report version và chạy lại Stage A. Renderer phụ thuộc XML body của Trafilatura và DOM của lxml; khóa package và chạy lại fixtures/native source checks khi nâng version.

Core:

```text
scrapy
trafilatura
pyarrow
polars hoặc pandas
zstandard
pymupdf
orjson
pyyaml
```

Optional language ID:

```text
fastText + lid.176.ftz/lid.176.bin + model SHA256
lingua-language-detector (chỉ khi benchmark cần)
```

Tạo lockfile **trước Stage A**, pin exact toàn bộ dependencies/transitives và ghi SHA256 của lockfile, Python patch version, OS/architecture. Chưa chọn version cho các dependency còn lại vì workspace chưa có implementation/lockfile; resolver phải được kiểm tra trên environment mục tiêu. Freeze lock sau Stage A đạt gate. Không đổi Scrapy version trong một resumable JOBDIR; extractor thay đổi phải tạo config/run identity mới.

Rà soát hiện tại kiểm tra tài liệu/source, chưa chạy extractor/crawler runtime. Stage A vẫn bắt buộc smoke-test `inspect.signature(bare_extraction)`, kiểu result (`Document`/None), serializer import, Scrapy start/errback và download-limit behavior trên environment được pin.

---

# 29. Tests bắt buộc

## Unit tests

URL normalizer:

```text
uppercase hostname
fragment
whitespace
invalid URL
query params preserved
```

Inventory:

```text
same URL → one crawl_url_id
multiple doc_ids preserved
```

Raw store:

```text
atomic write
zstd round-trip
checksum
idempotent existing file
```

Trafilatura fixtures:

```text
VI
EN
ZH
broken HTML
empty HTML
header/menu/article/footer
heading h1/h2/h3 retained in Markdown
nested list + table cells + title metadata
short valid article (<250 chars) + fallback structure
UTF-8 / Windows-1258 / GB18030 / Big5 bytes
inline biomedical symbols + emphasis
```

PDF fixtures:

```text
normal text PDF
scanned-like PDF
corrupt PDF
oversize input rejected before parse
worker hang/crash isolated; next document completes
```

## Integration tests

Local mock server endpoints:

```text
/ok-html
/redirect
/not-found
/forbidden
/rate-limit
/server-error
/slow
/pdf
/empty
/large
/large-stream
/wrong-header-html
/wrong-header-pdf
/robots.txt + /robots-blocked
/rate-limit-with-retry-after-seconds
/rate-limit-with-retry-after-date
```

Test full flow:

```text
crawl → raw → manifest → resume → extraction
```

Resume test:

1. start 100 URLs;
2. stop clean after ~30–50;
3. resume same JOBDIR;
4. verify no loss/duplicate;
5. delete JOBDIR;
6. rebuild pending set from manifest;
7. finish remaining URLs.

Crash/recovery fixtures: valid final JSON không newline, final truncated UTF-8/JSON, malformed middle line, replay event, nhiều retry attempts và checksum mismatch. Verify latest-record compaction deterministic, cooldown tồn tại sau restart, không retry trước Retry-After, robots-denied không bị gán network error và mọi failed doc_id vẫn có outcome.

Handoff fixtures: cleaning không phá heading/list/table; dedup representative không đổi khi đảo thứ tự input; widespread mirror hợp lệ không bị tự quarantine, soft-404/cookie duplicates có reason; gold ở LOW/invalid/blocked được report đúng denominator. Đây là tests cần có khi implement pipeline, không phải kết quả tests đã chạy trong lần chỉnh plan.

---

# 30. Performance profiling

Tách benchmark:

## Network-only

Trafilatura không chạy. Đo request throughput, raw write throughput, compression overhead.

## Extraction-only

Không network. Đọc raw cache, đo Trafilatura docs/sec, CPU, RAM, fallback overhead.

Không benchmark trộn rồi không biết bottleneck ở đâu.

---

# 31. P1/P2 features

P1 domain-specific extractor chỉ thêm nếu một domain chiếm tỷ trọng lớn và Trafilatura fail/noisy đáng kể.

P2 Playwright chỉ cho candidate:

```text
HTTP 200
+ meaningful HTML shell
+ Trafilatura empty
+ JS signals
+ domain quan trọng
```

P2 OCR chỉ cho PDF có page_count > 0 nhưng extracted text cực thấp.

Không chạy browser/OCR trên toàn corpus.

---

# 32. Agent implementation tasks

## TASK 1 — Bootstrap

Tạo repo structure, config loader, logging, pyproject, dependency lock, pytest và environment manifest.

DoD: project chạy được, tests baseline pass.

## TASK 2 — Inventory

Snapshot cả corpus/query + SHA256, validate schema/gold join; implement URL normalize, validate, `crawl_url_id`, unique URLs, `url_doc_map`, domain stats.

DoD: source row count accounting 100%, không mất doc_id; gold IDs thiếu/invalid được report, snapshot tái lập được.

## TASK 3 — Crawl shards

Implement unique URL → shards.

DoD: mỗi `crawl_url_id` xuất hiện đúng một lần giữa các shards; stable shard function và sequential execution được freeze.

## TASK 4 — Minimal Scrapy spider

Implement shard arg, async `start()`, Request generation, callback, errback, robots-specific mapping, DOWNLOAD_SLOTS và durable cooldown/Retry-After batch.

DoD: mock 20 URLs được accounting đầy đủ.

## TASK 5 — Raw cache

Implement header+byte-sniff router, download/extraction size accounting, atomic writes, zstd HTML/text, PDF binary, SHA256 và raw locator contract.

DoD: raw round-trip bằng response body.

## TASK 6 — Crawl manifest

Implement JSONL shards, statuses, compaction → Parquet.

DoD: mỗi completed `crawl_url_id` có latest status; deterministic compaction, replay idempotence và truncated-tail recovery pass.

## TASK 7 — Resume

Implement JOBDIR + manifest-driven recovery.

DoD: interrupt pilot không mất URL/doc_id; replay không duplicate event_id, retry history được giữ và compact ra đúng một latest outcome/URL.

## TASK 8 — Trafilatura extractor

Bytes input + strict charset resolution, Document/DOM adapter, normal-first + fast recovery khi rỗng, title metadata + Markdown/text, heading/list/table retention; supervisor timeout + worker recycle.

DoD: VI/EN/ZH bytes/structure fixtures pass; timeout/crash không treo batch, fallback không làm mất word boundaries.

## TASK 9 — PDF/text extractor

PyMuPDF + text decoder.

DoD: text PDF extract được; scanned-like PDF được flag.

## TASK 10 — Cleaner/quality/language

Unicode NFC, whitespace cleanup, basic tier, LID.

DoD: biomedical symbols/numbers và Markdown structure preserved; LID backend được benchmark, gold LOW/QUARANTINE được audit.

## TASK 11 — Exact dedup + mapping

`content_hash`, deterministic representative, widespread-duplicate review, canonical content, content_doc_map và doc_outcomes.

DoD: duplicate content downstream chỉ cần xử lý một lần nhưng mọi doc_id recoverable.

## TASK 12 — Stage A runner

Một command chạy sample 1k từ inventory đến report.

Output:

```text
reports/stage_a/report.json
reports/stage_a/manual_review.csv
reports/stage_a/gold_coverage.parquet
reports/stage_a/domain_estimates.parquet
reports/stage_a/coverage_decisions.json
```

## TASK 13 — Stage B runner

Run 10k rồi 100k, sinh full-scale estimates.

DoD: deployment config chốt trước run; domain ETA, gold coverage, peak RAM/disk, inode/file count và transfer/packing benchmark đủ để chọn full-scale layout.

---

# 33. Agent rules

1. Không xóa source `doc_id`.
2. Không tạo request trùng do nhiều doc_id dùng chung normalized URL; retries phải theo policy và có attempt/event identity.
3. Không dùng `dont_filter=True` làm giải pháp duplicate corpus.
4. Không discover link ngoài BTC frontier.
5. Không gọi Trafilatura trong production Scrapy callback.
6. Không bỏ raw cache.
7. Không ghi success trước raw write thành công.
8. Không `except: pass`.
9. Mọi failure phải có status.
10. Mọi stage phải resume được.
11. Không full-scale trước Stage A/B.
12. Không Playwright/OCR trong P0.
13. Không xóa query params tùy ý.
14. Không aggressive-clean biomedical text.
15. Không chạy nhiều crawler process làm vượt per-domain rate mà không phối hợp.
16. Không đổi Scrapy version khi còn JOBDIR cần resume.
17. Không coi HTTP 200 là content usable.
18. Không cho untrusted URL truy cập internal/private network trong môi trường nguy hiểm.
19. Không decode raw HTML sớm hay làm phẳng Markdown trước handoff.
20. Không compact bằng record đầu tiên hoặc timestamp mơ hồ; bảo toàn event history.
21. Không coi Future timeout là đã kill parser; recycle và supervise worker.
22. Không loại LOW hoặc widespread duplicates chỉ bằng độ dài/tần suất mà thiếu gold-impact audit.
23. Không scale domain bị blocked gate nếu chưa có quyết định coverage; không tự thay BTC snapshot bằng archive.
24. Không gọi template report là số đo; mọi rate phải có denominator và pending count.

---

# 34. Stage A report bắt buộc

Ví dụ **schema/template**, chưa có số đo. `null` là chưa đo/không có denominator; counters điền từ run thật. Artifacts chi tiết chứa IDs và per-domain rows, JSON chỉ tổng hợp.

```json
{
  "report_status": "template_not_measured",
  "snapshot_id": null,
  "config_sha256": null,
  "lockfile_sha256": null,
  "environment_id": null,
  "versions": {"scrapy": "2.13.4", "trafilatura": "2.1.0", "python": "3.11.x"},
  "sample_seed": 42,
  "handoff_mode": "benchmark",
  "planned_unique_urls": 1000,
  "input_urls": 0,
  "unique_urls": 0,
  "crawl": {
    "success_raw": 0,
    "http_404": 0,
    "http_403": 0,
    "http_429": 0,
    "http_5xx": 0,
    "robots_denied": 0,
    "timeout": 0,
    "other_errors": 0,
    "blocked_rate": null,
    "domain_weighted_blocked_rate": null,
    "content_type_mismatches": 0,
    "retry_batches": 0,
    "retry_after_events": 0
  },
  "storage": {
    "raw_bytes": 0,
    "compressed_bytes": 0,
    "compression_ratio": null,
    "raw_file_count": 0,
    "copy_upload_seconds": null
  },
  "extraction": {
    "success": 0,
    "fast_pass_success": 0,
    "fallback_success": 0,
    "failed": 0,
    "likely_scanned_pdf": 0,
    "timeouts": 0,
    "too_large": 0,
    "worker_crashes": 0,
    "structure_review_pass_rate": null
  },
  "quality": {
    "high": 0,
    "medium": 0,
    "low": 0,
    "quarantine": 0
  },
  "languages": {
    "vi": 0,
    "en": 0,
    "zh": 0,
    "mixed": 0,
    "other": 0,
    "unknown": 0
  },
  "gold_coverage": {
    "distinct_gold_doc_ids_total": 0,
    "missing_from_inventory": 0,
    "invalid_url_doc_ids": 0,
    "audited_gold_doc_ids": 0,
    "pending_gold_doc_ids": 0,
    "raw_success_doc_ids": 0,
    "usable_doc_ids": 0,
    "handoff_eligible_doc_ids": 0,
    "low_doc_ids": 0,
    "quarantine_doc_ids": 0,
    "raw_coverage_on_audited_set": null,
    "usable_coverage_on_audited_set": null,
    "handoff_coverage_on_audited_set": null,
    "fully_evaluated_queries": 0,
    "pending_queries": 0,
    "query_any_gold_usable_rate": null,
    "query_all_gold_usable_rate": null,
    "mean_usable_gold_fraction_per_query": null
  },
  "dedup": {"widespread_groups": 0, "confirmed_quarantine_groups": 0},
  "gates": {
    "mapping_accounting_pass": null,
    "resume_crash_recovery_pass": null,
    "manual_structure_review_pass": null,
    "blocked_domains_decision_complete": null,
    "low_handoff_policy": "pending_gold_audit",
    "deployment_ready_for_stage_b": null,
    "stage_b_ready": null
  },
  "artifacts": {
    "gold_coverage": "reports/stage_a/gold_coverage.parquet",
    "domain_estimates": "reports/stage_a/domain_estimates.parquet",
    "coverage_decisions": "reports/stage_a/coverage_decisions.json",
    "manual_review": "reports/stage_a/manual_review.csv"
  }
}
```

---

# 35. Definition of Done

Inventory:

```text
✓ 100% source rows accounted
✓ invalid URLs preserved
✓ unique URL table
✓ doc_id mapping
✓ domain stats
✓ corpus + query snapshot and SHA256 manifest
✓ gold mapping/schema validation
```

Scrapy:

```text
✓ fixed-frontier spider
✓ no link discovery
✓ JOBDIR resume
✓ manifest-driven recovery
✓ retry + AutoThrottle
✓ per-domain concurrency
✓ robots policy
✓ atomic raw cache
✓ success/failure manifest
✓ response size guard
✓ byte sniffing and declared/effective MIME
✓ robots-specific errback accounting
✓ Retry-After cooldown persisted across resume
✓ latest-record compaction and crash-tail recovery
✓ no mapping loss
```

Trafilatura/extraction:

```text
✓ offline from raw cache
✓ fast first pass + fallback
✓ title metadata + Markdown heading/list/table retained
✓ comments/links excluded
✓ VI/EN/ZH preserved
✓ failure manifest
✓ output Parquet shards
✓ decompressed input cap + real per-doc timeout
✓ worker crash isolation and recycling
✓ gold LOW/QUARANTINE impact audited
```

Final ingestion:

```text
✓ documents.parquet
✓ content_doc_map.parquet
✓ crawl_manifest.parquet
✓ extraction_manifest.parquet
✓ doc_outcomes.parquet for every BTC ID
✓ deterministic representative + duplicate review
✓ domain ETA, deployment and transfer benchmark
```

Join chain phải reconstruct được:

```text
BTC doc_id
→ crawl_url_id
→ raw_path
→ canonical_content_id
→ document text
```

Chain nội dung áp dụng cho doc extract thành công; với doc lỗi/invalid, reconstruct được source ID → outcome + reason, các reference chưa có để null. Không tạo content giả để đạt mapping coverage.

---

# 36. Kết quả cuối cùng

```text
links_corpus.parquet
        │
        ▼
URL normalize + mapping
        │
        ▼
unique URLs
        │
        ▼
Scrapy fixed-frontier crawl
        │
        ├── retry
        ├── AutoThrottle
        ├── per-domain limits
        ├── robots policy
        └── JOBDIR
        │
        ▼
raw cache + crawl manifest
        │
        ▼
offline extraction
        │
        ├── Trafilatura HTML
        ├── PyMuPDF PDF
        └── plain text
        │
        ▼
clean + language + quality
        │
        ▼
exact content dedup
        │
        ▼
documents.parquet
+
content_doc_map.parquet
        │
        ▼
READY FOR CHUNKING
```

Phần Scrapy + Trafilatura được xem là hoàn thành khi:

1. Không mất BTC doc_id.
2. Mọi unique URL có trạng thái giải thích được.
3. Crawler pause/resume được.
4. Raw response được cache và verify được.
5. Extraction có thể chạy lại không cần Internet.
6. Main biomedical content được giữ.
7. Boilerplate giảm đáng kể.
8. VI/EN/ZH không lỗi encoding.
9. Duplicate URL/content không gây download/embed lặp vô ích.
10. Output contract ổn định cho chunking.
11. Throughput/storage đã được đo Stage B trước full-scale.
12. Gold coverage/pending/LOW impact và blocked-domain decisions được ghi nhận.
13. Markdown structure được review, worker timeout thực sự enforce được.
14. Snapshot/lockfile/environment tái lập được; storage layout và thời gian transfer đã đo.

Khi đạt các điều kiện trên, freeze thành:

```text
INGESTION_CORPUS_V1
```

rồi mới chuyển sang:

```text
chunking → embedding → indexing → retrieval evaluation
```

---

# 37. Kết luận rà soát các nhận xét

| Nhận xét | Kết luận và thay đổi trong plan |
|---|---|
| Text thuần làm mất heading; cần Markdown | Đúng với config cũ không giữ formatting. §16/18/21/23 giữ `text_markdown`, plain `text`, title riêng và fixtures cấu trúc; extraction vẫn cần kiểm tra thực tế. |
| Truyền bytes thay vì decode sớm | Đúng; §1.3/11/16/17 giữ bytes/headers và kiểm tra charset, không hứa detection luôn chính xác. |
| Tắt fallback bằng `no_fallback`, không phải `fast` | Cần đính chính: baseline dùng `fast`; `no_fallback` deprecated. Rescue nội bộ vẫn có thể chạy (§16, [source 2.1.0](https://github.com/adbar/trafilatura/blob/v2.1.0/trafilatura/core.py)). |
| Trafilatura 2.x trả Document thay dict | Đúng với mặc định; `as_dict=True` vẫn có trong baseline nhưng deprecated. Adapter dùng Document (§16). |
| Per-domain=2 nghĩa là 4 req/s | Chỉ khi full request duration khoảng 0.5s và chưa tính throttle. §26 sửa công thức/ví dụ, ETA per-domain và sequential shards. |
| Threadpool 10 quá thấp; bật DNS cache | Có thể là bottleneck, phải đo. Cache vốn mặc định bật; §4/9/25 set rõ và benchmark threadpool. |
| Robots-denied/403 làm mất coverage | Đúng; §10/24 map IgnoreRequest đúng nguyên nhân, gold coverage và >10% review gate với quyết định nguồn/accept missing. |
| Header MIME có thể sai | Đúng; §11 sniff PDF/HTML, charset probe, mismatch metadata và fixtures. |
| Retry cần latest compaction; JSONL chịu crash tail | Đúng; §13/14/21 định nghĩa ordering/event replay, latest view và corruption handling. |
| Worker có thể treo/rò RAM | Cần guard; §15 thêm cap, supervisor timeout thực, recycling; chưa có benchmark lỗi/RAM của corpus này. |
| Hàng triệu file nhỏ khó transfer | Hợp lý; §12/25 đo file metadata/transfer và chọn loose/packed layout có index/commit contract. |
| Representative deterministic; duplicate đa-domain là cookie/soft-404 | Phần deterministic đúng. Tần suất chỉ là tín hiệu nghi ngờ; §22 quarantine khi có thêm bằng chứng để giữ mirror hợp lệ. |
| LOW cần gold audit; LID cần nhẹ | Đúng hướng; §19/20/24 audit impact và benchmark heuristic/fastText, chưa kết luận Lingua luôn chậm. |
| Thiếu dataset snapshot/version pin | Đúng; §5.1/28 thêm corpus+query SHA256, schema adapter và lock trước pilot. |
| 429 cần AutoThrottle và retry batch vì RetryMiddleware bỏ qua Retry-After | Đúng hướng trên pin; §9/10 bổ sung durable slot cooldown theo header, hỗ trợ delta-seconds/HTTP-date. |
| `start()` vs `start_requests()`; size/RAM cap | Đúng; §7/9/29 kiểm tra async start và downloader limits trên Scrapy 2.13.4. 64 body ×50 MiB chưa phải tổng RAM. |
| Cần chốt nơi crawl; raw khoảng 0.5–1.3 TB | Đúng hướng; §25 chọn mục tiêu VPS Linux và gate tài nguyên. §26 ghi TB là kịch bản theo mean body size, cần đo wire/cache/disk riêng. |

Các liên kết chính thức được đặt cạnh phần tương ứng. Chưa có dữ liệu Parquet, dependency lock, VPS hay số đo Stage A/B trong workspace; những con số pilot/threshold ở đây là **giả định để kiểm chứng**, không phải kết quả crawl. Phần sửa này hoàn thiện plan, chưa triển khai pipeline.
