# R2AI Stage 3 — Data Pipeline Supplement
## Những bổ sung bắt buộc so với kiến trúc gốc để xử lý ViBioMIR an toàn ở quy mô lớn

> **Vai trò của tài liệu này**  
> Đây là tài liệu **bổ sung (supplement / implementation delta)** cho `R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md`. Nó **không thay thế** kiến trúc Stage 3 gốc. Kiến trúc gốc vẫn là source of truth ở mức chiến lược: canonical corpus → child/parent chunks → BM25 + multilingual dense → hybrid retrieval → rerank → child→parent output → scorer-aware selection.  
> Tài liệu này zoom sâu riêng vào **offline data pipeline** và bổ sung các lớp còn thiếu để có thể xử lý hàng triệu URL mà không rơi vào tình trạng “chạy một ngày rồi mới phát hiện toàn bộ corpus/chunk sai”.

---

# 0. Kết luận ngắn trước khi đọc chi tiết

Bản `.md` gốc đã đúng về **architecture**. Các phần không thay đổi:

```text
Provided biomedical URLs
        ↓
raw snapshot / crawl
        ↓
canonical documents
        ↓
dedup
        ↓
structure-aware segmentation
        ↓
child retrieval chunks
        ↓
parent output chunks
        ↓
BM25 + multilingual dense index
        ↓
hybrid retrieval / reranking
```

Những gì cần bổ sung là một lớp **Data Engineering Safety System** nằm xuyên suốt pipeline:

```text
PROCESS
   ↓
VERIFY
   ↓
QUALITY GATE
   ↓
ATOMIC COMMIT
   ↓
PROCESS NEXT SHARD
```

Không chạy theo mô hình:

```text
PROCESS EVERYTHING
       ↓
CHECK AT THE END
```

Mục tiêu ưu tiên của data pipeline phải là:

```text
CORRECTNESS
    ↓
COVERAGE
    ↓
DATA QUALITY
    ↓
RETRIEVABILITY
    ↓
SCALE
    ↓
SPEED / COST
```

Ba bổ sung P0 quan trọng nhất:

1. **Immutable source text + exact source-span provenance**: mọi child/parent output phải round-trip chính xác về source document.
2. **Continuous verification + shard-level atomic commit**: xử lý từng shard, tự validate, chỉ commit khi pass; hard failure tự dừng.
3. **Data Quality Gate + Retrieval Quality Gate**: dữ liệu không chỉ “đúng schema” mà phải thật sự là nội dung biomedical usable và không làm giảm candidate recall.

---

# 1. Quan hệ giữa tài liệu này và `.md` gốc

Bản gốc đã có các ý quan trọng sau và **giữ nguyên**:

- raw snapshot layer;
- parser per source;
- canonical document schema;
- raw/normalized text và source checksum;
- stable `doc_id`;
- duplicate detection;
- structure-aware segmentation;
- child chunk khoảng 120–220 tokens;
- parent output chunk khoảng 350–650 tokens;
- document representation riêng;
- exact raw/source offset provenance;
- artifact manifests;
- corpus/index hashes;
- release gate;
- local scorer và candidate-recall diagnostics.

Tài liệu này thêm phần implementation còn thiếu.

## 1.1. Delta matrix

| Chủ đề | Bản gốc đã có | Bổ sung cần triển khai |
|---|---|---|
| Source identity | stable `doc_id` | assert `doc_id ∈ links_corpus`, never reindex, FK checks ở mọi artifact |
| Raw data | raw snapshot layer | sharded raw archive + crawl manifest + checksums + retry provenance |
| Canonical text | raw + normalized | formalize `source_text` immutable làm source of truth cho submission |
| Normalization | safe surface normalization | tách source text khỏi retrieval representation; không overwrite source |
| Provenance | raw char offsets | exact round-trip invariant cho mọi child/parent chunk |
| Chunking | child ~180, parent ~512 | child-parent containment, structural boundaries, chunk regression tests |
| Dedup | corpus duplicate | tách `content_hash` và `retrieval_representation_hash`; preserve all BTC doc IDs |
| Quality | corpus metrics | extraction/structure/chunk/retrieval quality gates + reason codes |
| Freeze | manifest/hash | shard validation → atomic commit → corpus freeze manifest |
| Scale | general architecture | Stage A/B/C ramp-up + kill-switch thresholds + resource gate |
| Evaluation | candidate recall | canary/golden dataset + retrieval regression before promotion |
| Observability | reports | per-domain/language/type health report + adversarial sample audit |
| Failure recovery | immutable runs | shard checkpoint/resume + side-by-side artifacts + rollback |

---

# 2. Hard principle: `source_text` phải trở thành source of truth bất biến

Đây là bổ sung quan trọng nhất so với cách diễn đạt ở bản gốc.

Stage 3 yêu cầu `chunk_text` là nội dung lấy từ tài liệu gốc, không phải text được viết lại hoặc sinh mới. Vì vậy data layer cần có một representation được coi là **nguồn chân lý duy nhất** cho mọi output chunk.

Đề xuất:

```text
RAW RESPONSE
     ↓
EXTRACTION
     ↓
IMMUTABLE SOURCE_TEXT
     ↓
STRUCTURE / CHUNKS / RETRIEVAL REPRESENTATIONS
```

Sau khi `source_text` được tạo và shard được commit:

- không được silently mutate;
- mọi chunk phải trỏ bằng offsets vào `source_text`;
- title/section aliases/search fields có thể thay đổi theo experiment;
- output chunk không được reconstruct từ normalized/search text.

## 2.1. Canonical source contract

Mỗi document tối thiểu cần:

```text
doc_id: int64
original_url: string
final_url: string
source_text: string
source_text_sha256: string
extractor_name: string
extractor_version: string
raw_asset_id/path: string
crawl_timestamp: timestamp
content_type: string
language: string
quality_tier: string
```

Khuyến nghị thêm:

```text
title_source_start: int | null
title_source_end: int | null
section_map_version: string
normalizer_version: string
schema_version: string
```

## 2.2. Exact-span invariant

Mỗi child chunk:

```text
child_text == source_text[child_start_char:child_end_char]
```

Mỗi parent chunk:

```text
parent_text == source_text[parent_start_char:parent_end_char]
```

Và:

```text
parent_start_char <= child_start_char
child_end_char <= parent_end_char
```

Đây là **HARD invariant**, không phải quality score.

Nếu một chunk fail exact equality:

```text
FAIL SHARD
DO NOT EMBED
DO NOT COMMIT
```

Không dùng fuzzy matching để hợp thức hóa lỗi offset.

---

# 3. Tách ba khái niệm text: source, normalized, retrieval representation

Bản gốc đã nhấn mạnh không normalize phá raw output. Cần formalize thành ba lớp logic.

## 3.1. `source_text`

Dùng cho:

- provenance;
- exact offsets;
- `chunk_text` submission;
- manual audit;
- replay.

Không prepend title/section giả vào body.

## 3.2. `normalized_text`

Dùng cho:

- lexical normalization;
- dedup hỗ trợ;
- diagnostics;
- optional sparse representation.

Có thể áp:

```text
Unicode normalization
HTML entity decode
NBSP → space
whitespace collapse
safe hyphen normalization
```

Nhưng không được:

```text
remove digits
remove all punctuation
strip Vietnamese accents
translate corpus
rewrite biomedical entities
```

## 3.3. `retrieval_text`

Không nhất thiết lưu materialized cho mọi row. Có thể build on demand:

```text
retrieval_text = title + section heading + child source span + optional alias field
```

Dense và sparse có thể có builder khác nhau:

```python
build_dense_text(chunk)
build_sparse_text(chunk)
```

### Critical rule

`retrieval_text` **không bao giờ** được dùng trực tiếp làm `chunk_text` submission.

---

# 4. Raw crawl layer phải là replayable, không phải temporary cache

Bản gốc nói raw snapshot; supplement này yêu cầu biến nó thành artifact có thể replay.

## 4.1. Crawl manifest

Mỗi URL:

```text
doc_id
original_url
final_url
domain
crawl_status
http_status
content_type
raw_asset_location
raw_size_bytes
compressed_size_bytes
attempts
elapsed_ms
crawl_timestamp
response_sha256
```

Status nên đủ chi tiết:

```text
PENDING
SUCCESS
HTTP_404
HTTP_403
HTTP_429
HTTP_5XX
TIMEOUT
DNS_ERROR
SSL_ERROR
CONNECTION_ERROR
BLOCKED
EMPTY_RESPONSE
UNSUPPORTED_TYPE
RETRY_EXHAUSTED
```

## 4.2. Không tạo hàng triệu file nhỏ vô tổ chức

Ưu tiên:

```text
raw/
  shard_00000.tar.zst
  shard_00001.tar.zst
  ...
```

hoặc một archive format tương đương phù hợp implementation.

Manifest giữ mapping:

```text
doc_id → archive/member hoặc offset
```

PDF/binary lớn có thể lưu riêng nếu archive strategy không phù hợp.

Mục tiêu:

- tránh inode explosion;
- replay extractor không crawl lại;
- dễ checksum;
- dễ copy/snapshot;
- shard recovery rẻ.

---

# 5. Scale-up phải theo Stage A/B/C, không full-run ngay

## 5.1. Golden set — trước Stage A

Tạo một bộ 100–500 documents cố định, cố tình chứa case khó:

```text
Vietnamese
English
Chinese
HTML sạch
HTML nhiều boilerplate
PDF text
PDF scan-like
long article
short article
tables
Unicode/Greek symbols
biomedical dosage
mirrors/duplicates
redirects
JS-like failure pages
```

Golden set không cần đại diện thống kê hoàn hảo. Vai trò của nó là **regression detector**.

Mỗi document có expected assertions, ví dụ:

```text
expected_doc_id
expected_title_contains
expected_language
expected_min_text_length
expected_key_snippets
expected_quality_not_quarantine
expected_chunk_count_range
expected_biomedical_tokens_preserved
```

Mỗi lần thay extractor/cleaner/chunker:

```text
run golden suite
```

Nếu fail → không được chạy Stage A/B/full.

## 5.2. Stage A — ~1,000 URL stratified

Không random thuần.

Stratify theo:

- top domains;
- medium domains;
- long-tail;
- likely VI/EN/ZH;
- HTML/PDF/other;
- suspicious/blocked domains.

Mục tiêu:

```text
crawl feasibility
extraction feasibility
raw storage distribution
content-type distribution
language distribution
blocked/JS/scan incidence
quality feature calibration
```

## 5.3. Stage B1 — ~10k

Mục tiêu:

- checkpoint/resume;
- shard validation;
- chunk distributions;
- initial BM25/dense prototype;
- canary retrieval evaluation;
- storage/throughput measurements.

## 5.4. Stage B2 — ~100k

Mục tiêu:

- long-run distribution stability;
- domain-level failure discovery;
- realistic duplicate rate;
- realistic chunk count;
- embedding throughput;
- ANN prototype;
- retrieval quality regression.

## 5.5. Stage C — 100k→1M

Chỉ chạy khi B2 pass.

Mục tiêu:

- memory leak;
- I/O bottlenecks;
- archive pressure;
- Lucene growth;
- FAISS build behavior;
- multi-hour stability;
- resume after forced interruption.

## 5.6. Full corpus gate

Chỉ chạy full khi đã trả lời được:

```text
valid docs estimate?
raw storage estimate?
processed storage estimate?
child chunks estimate?
parent mapping size?
embedding throughput?
GPU-hours?
vector/index size?
BM25 index size?
ANN recall loss?
deadline margin?
```

---

# 6. Continuous Verification — QA phải nằm bên trong pipeline

Không thiết kế:

```text
process 4M docs
↓
QA cuối ngày
```

Mà:

```text
PROCESS SHARD
     ↓
HARD VALIDATION
     ↓
QUALITY CHECK
     ↓
PASS?
 ┌───┴───┐
YES     NO
 ↓        ↓
COMMIT   STOP/QUARANTINE
 ↓
NEXT SHARD
```

## 6.1. Shard size

Không có một con số tối ưu universal. Starting point có thể là:

```text
5k–20k docs/shard
```

Sau Stage A/B chọn dựa trên:

- file sizes;
- average document length;
- retry cost;
- worker memory;
- parquet row group size;
- operational convenience.

## 6.2. Atomic commit

Không ghi trực tiếp artifact final trong lúc processing.

Flow:

```text
part_00042.tmp
     ↓
validate
     ↓
checksum
     ↓
atomic rename / commit
     ↓
part_00042.parquet
```

Manifest:

```text
shard_id
input_count
output_count
status
schema_version
extractor_version
chunker_version
sha256
started_at
finished_at
```

Status:

```text
PENDING
RUNNING
VALIDATING
DONE
FAILED
QUARANTINED
```

---

# 7. HARD integrity gates — lỗi nào phải tự dừng

Các invariant dưới đây phải được encode bằng code/tests, không dựa vào manual inspection.

## 7.1. Dataset identity

```text
query snapshot hash matches manifest
links snapshot hash matches manifest
```

## 7.2. `doc_id`

```text
every processed doc_id ∈ official links_corpus IDs
no unexpected remapping
no array-index-derived doc ID
```

## 7.3. Document uniqueness

Nếu pipeline quy định một processed record cho mỗi BTC `doc_id`:

```text
unique(doc_id)
```

Nếu có multi-attempt/intermediate tables, final promoted table vẫn phải deterministic.

## 7.4. Chunk identity

```text
unique(child_chunk_id)
unique(parent_span_id)
```

## 7.5. Foreign-key integrity

```text
child.doc_id exists in documents
parent.doc_id exists in documents
child.parent_id exists in parents
```

## 7.6. Exact span

```python
assert child.text == source_text[child.start:child.end]
assert parent.text == source_text[parent.start:parent.end]
```

## 7.7. Containment

```python
assert parent.start <= child.start < child.end <= parent.end
```

## 7.8. Bounds

```text
0 <= start < end <= len(source_text)
```

## 7.9. Canonical dedup mapping

```text
every official doc_id represented exactly once in alias/mapping universe
no canonical object without source docs
no source doc silently dropped due to dedup
```

## 7.10. Embedding/index mapping

Khi đến embedding:

```text
vector_count == manifest.vector_count
all vector IDs map to promoted chunks
no duplicate vector IDs
no missing shard
model/tokenizer revision identical across shards
```

**HARD invariant target = 100%, không chấp nhận 99.999%.**

---

# 8. DATA QUALITY GATE — dữ liệu đúng kỹ thuật chưa chắc đã tốt

Tách rõ:

```text
INTEGRITY
= representation có đúng contract không?

QUALITY
= representation có hữu ích cho retrieval không?
```

Data Quality Gate có bốn lớp:

```text
Extraction Quality
Structure Quality
Chunk Quality
Retrieval Quality
```

---

# 9. Extraction Quality Gate

## 9.1. Feature set

Mỗi processed document nên log tối thiểu:

```text
source_char_count
paragraph_count
sentence_count
alphabetic_or_cjk_ratio
duplicate_line_ratio
link_text_ratio (nếu tính được)
title_present
heading_count
possible_error_page
possible_truncation
possible_scan_pdf
possible_js_required
```

Không cần ML classifier ở v1. Rule-based + per-domain audit đủ để bắt phần lớn lỗi thô.

## 9.2. Error-page detection

Các pattern như:

```text
403 Forbidden
Access Denied
Page Not Found
Enable JavaScript
Cloudflare challenge
Sign in to continue
```

không được đi thẳng vào index như biomedical article.

Gán reason code:

```text
ACCESS_DENIED
NOT_FOUND
JS_REQUIRED
BOT_CHALLENGE
LOGIN_REQUIRED
EMPTY_ARTICLE
```

## 9.3. Truncation detection

Heuristics có thể gồm:

```text
very short extraction vs raw asset size
abrupt ending
missing expected body container on known large domain
PDF pages high but extracted chars extremely low
```

Chỉ dùng làm warning/quarantine signal; threshold calibrate ở Stage A/B.

## 9.4. Biomedical sanity signal

Không dùng để loại document cứng vì BTC đã chọn URL universe.

Dùng làm anomaly detector:

```text
biomedical_identifier_count
medical_term_density
dosage_pattern_count
gene/drug/disease alias hits
```

Nếu một domain biomedical nhưng extraction ra hầu hết `Login / Home / Cookie`, anomaly sẽ rõ.


# 10. Structure Quality Gate

Document extraction tốt nhưng structure parsing vẫn có thể sai. Structure quality cần theo dõi riêng.

## 10.1. Structure schema

Khuyến nghị:

```text
section_id
heading_text
section_start_char
section_end_char
paragraph_ids[]
```

Paragraph:

```text
paragraph_id
doc_id
section_id
start_char
end_char
```

Sentence boundary có thể materialize hoặc tính on demand, nhưng version của sentence splitter phải được pin.

## 10.2. Structural invariants

```text
section spans nằm trong source bounds
section order monotonic
paragraph span nằm trong section nếu section known
paragraph order monotonic
không overlap bất hợp lý
```

## 10.3. Structural quality warnings

```text
MISSING_TITLE
NO_SECTIONS
ONE_GIANT_PARAGRAPH
TOO_MANY_MICRO_PARAGRAPHS
SECTION_ORDER_SUSPICIOUS
TABLE_TEXT_FLATTENED
REFERENCE_HEAVY
```

Không phải warning nào cũng là lỗi. Nhiều article không có headings rõ. Vai trò reason code là giúp debug theo domain.

## 10.4. Table handling

V1 không cần perfect table reconstruction.

Nhưng cần detect:

```text
table_present
possible_table_loss
```

Nếu extractor phá association của table, đừng giả vờ quality HIGH. Có thể giữ text linearized và gắn warning.

---

# 11. Chunking architecture — giữ child/parent của bản gốc nhưng formalize contract

Bản gốc đã đúng khi tách child retrieval chunk và parent output chunk. Supplement này chốt contract chi tiết.

## 11.1. Child chunk

Vai trò:

```text
BM25 indexing
multilingual embedding
candidate retrieval
reranking
```

Starting hypothesis từ bản gốc:

```text
~120–220 BGE-M3 tokens
```

Trong implementation có thể benchmark:

```text
128
180
220
256
384
```

Không chốt bằng intuition.

## 11.2. Parent span

Vai trò:

```text
submission evidence candidate
context recovery
metric-aware expansion
```

Starting hypothesis:

```text
~350–650 BGE-M3 tokens
```

Sweep có thể gồm:

```text
384
512
640
```

Parent không bắt buộc phải được embed toàn bộ.

## 11.3. Structural priority

Chunker nên ưu tiên:

```text
section
  ↓
paragraph
  ↓
sentence
  ↓
token fallback
```

Không default raw fixed-window split nếu structural boundaries có sẵn.

## 11.4. Child construction rules

Một child tốt nên:

- không start/end giữa token;
- cố gắng snap về sentence boundary;
- không vượt section nếu tránh được;
- không nhét quá nhiều unrelated paragraphs;
- không prepend text synthetic vào `child_text` source field.

Search representation có thể prepend title/heading nhưng source span vẫn giữ riêng.

## 11.5. Parent construction rules

Parent được tạo quanh child anchor:

```text
left expansion
+ child
+ right expansion
```

Constraints:

```text
stay inside same document
prefer same section
snap to sentence/paragraph boundaries
respect max token budget
exact source slice
```

## 11.6. Parent selection không cần materialize mọi possible window

Để tiết kiệm storage, có thể lưu:

```text
child → section span + sentence positions
```

và build parent on demand theo config `parent_tokens`.

Ưu điểm:

- sweep parent 384/512/640 không cần rechunk toàn corpus;
- giảm duplicate storage;
- phù hợp bài học provenance/replay từ Stage 2.

Nếu implementation đơn giản hơn cần materialized parents, vẫn phải giữ deterministic mapping và version.

---

# 12. Chunk Quality Gate

## 12.1. Hard checks

```text
exact source round-trip
valid offsets
child ⊂ parent
valid doc_id
unique IDs
```

## 12.2. Soft quality features

Mỗi child:

```text
token_count
sentence_count
paragraph_count
starts_mid_sentence
ends_mid_sentence
crosses_section
contains_heading_only
mostly_reference_text
mostly_navigation_like_text
```

## 12.3. Fragment quality

Một chunk kiểu:

```text
"This reduced mortality by 32%."
```

có thể thiếu antecedent.

Có thể log heuristics:

```text
pronoun-heavy opening
numeric-result-without-entity
very-short-context
```

Không hard reject. Có thể dùng để:

- expand thêm previous sentence;
- lower quality flag;
- inspect trong chunk experiments.

## 12.4. Near-duplicate chunk rate

Overlap cao có thể tạo hàng triệu chunk gần giống nhau.

Log:

```text
near_duplicate_chunk_rate
```

Tách rõ:

- corpus/chunk dedup threshold cho index optimization;
- scorer near-duplicate logic cho prediction.

Không reuse một threshold mù quáng cho hai mục tiêu.

---

# 13. Dedup semantics — cần sửa sâu hơn so với bản gốc

Bản gốc có `duplicate detection` đúng hướng. Tuy nhiên cần formalize hai identity khác nhau.

## 13.1. `content_hash`

Hash của canonical body/source content sau normalization được định nghĩa rõ.

Dùng để phát hiện:

```text
exact mirror
same article copied across URLs
```

## 13.2. `retrieval_representation_hash`

Embedding input có thể là:

```text
title + section + chunk body
```

Nếu hai documents có cùng body nhưng khác title/section metadata thì retrieval representation có thể khác.

Do đó:

```text
same content_hash
```

**không suy ra**:

```text
same retrieval_representation_hash
```

Chỉ reuse embedding khi representation thực sự identical hoặc experiment chứng minh policy khác là an toàn.

## 13.3. Preserve all BTC document IDs

Ví dụ:

```text
doc_id 100 ─┐
            ├─ canonical_content C17
doc_id 300 ─┘
```

Canonicalization được phép giảm compute/storage nhưng **không được xóa identity của 100 hoặc 300**.

Mapping:

```text
canonical_content_id → [doc_id...]
```

Hard validation:

```text
set(mapped_doc_ids) == set(promoted_official_doc_ids)
```

sau khi loại các crawl failures theo policy rõ ràng.

## 13.4. Không fan-out alias mù quáng khi submission

Nếu một canonical hit map tới nhiều BTC doc IDs, không mặc định submit tất cả aliases vì precision có thể giảm.

Alias resolution là retrieval/selection problem riêng và phải benchmark.

---

# 14. Language / domain / content-type monitoring

Global aggregate có thể che lỗi nghiêm trọng.

Ví dụ:

```text
overall extraction success = 95%
```

nhưng:

```text
VI 98%
EN 98%
ZH 42%
```

thì cross-lingual retrieval bị hỏng.

Mọi health report phải stratify ít nhất theo:

```text
domain
language
content_type
quality_tier
```

## 14.1. Required reports

```text
crawl_success_by_domain
extract_success_by_domain
quality_distribution_by_domain
language_distribution_by_domain
mean_text_length_by_domain
chunks_per_doc_by_domain
```

Tương tự:

```text
..._by_language
..._by_content_type
```

## 14.2. Top failure report

Luôn xuất:

```text
top failed domains by document count
top domains by quarantine count
top domains by suspicious extraction rate
```

P0 audit tập trung top domains trước vì một lỗi parser ở domain chiếm lớn có tác động corpus rất mạnh.

---

# 15. Human QA không bị loại bỏ — chỉ chuyển sang milestone và adversarial audit

Continuous auto-QA không có nghĩa bỏ human inspection.

Human không cần ngồi canh full run. Thay vào đó:

```text
Golden
→ inspect
Stage A 1k
→ inspect
Stage B 10k
→ inspect
Stage B 100k
→ inspect
Full run
→ periodic sample reports
```

## 15.1. Stratified random audit

Ví dụ mỗi milestone:

```text
20–50 samples / major domain
samples / VI/EN/ZH
samples / HTML/PDF
samples / quality tier
```

Reviewer score đơn giản:

```text
0 = unusable
1 = partially usable
2 = good
```

## 15.2. Adversarial audit

Ngoài random sample, cố tình inspect:

```text
longest documents
shortest documents
highest chunk count
lowest quality
highest boilerplate score
weird language detection
PDF low text density
high duplicate-line ratio
Unicode-heavy docs
table-heavy docs
```

Adversarial audit có xác suất tìm bug cao hơn random-only.

## 15.3. Generated HTML QA report

Sau mỗi milestone hoặc selected shard, tạo report:

```text
URL
crawl status
raw preview
source text preview
section boundaries
child boundaries
parent boundaries
quality reason codes
```

Highlight child/parent spans trực quan để người kiểm nhìn vài phút là thấy parser/chunker có vấn đề hay không.

---

# 16. HARD FAIL vs SOFT WARNING

Phải phân loại từ đầu để pipeline biết tự dừng khi nào.

## 16.1. HARD FAIL

Ví dụ:

```text
INVALID_DOC_ID
DUPLICATE_FINAL_DOC_ID
DUPLICATE_CHUNK_ID
ORPHAN_CHILD
ORPHAN_PARENT
SOURCE_SPAN_MISMATCH
CHILD_OUTSIDE_PARENT
OFFSET_OUT_OF_BOUNDS
CORPUS_HASH_MISMATCH
MISSING_EMBEDDING_SHARD
VECTOR_ID_MISMATCH
TOKENIZER_REVISION_MISMATCH
```

Behavior:

```text
mark shard FAILED
stop downstream promotion
preserve logs/intermediate artifacts
```

## 16.2. SOFT WARNING

Ví dụ:

```text
EXTRACT_SUCCESS_DROP
CHUNKS_PER_DOC_DRIFT
TOKEN_LENGTH_DRIFT
QUALITY_LOW_SPIKE
LANGUAGE_UNKNOWN_SPIKE
DOMAIN_FAILURE_SPIKE
NEAR_DUP_RATE_SPIKE
```

Behavior configurable:

```text
WARN_ONLY
PAUSE_AFTER_SHARD
QUARANTINE_DOMAIN
REQUIRE_HUMAN_REVIEW
```

Threshold không hard-code trước Stage A/B.

---

# 17. Distribution drift detector

Một data pipeline có thể không vi phạm schema nhưng vẫn âm thầm đổi behavior.

Ví dụ hôm qua:

```text
mean chunks/doc = 5.4
```

sau một commit:

```text
mean chunks/doc = 47.1
```

Không có assertion nào fail, nhưng chunker chắc chắn đáng kiểm tra.

## 17.1. Metrics cần snapshot theo shard

```text
source chars/doc
paragraphs/doc
sections/doc
child chunks/doc
child tokens
parent tokens
quality tiers
language distribution
exact duplicate rate
```

Log:

```text
mean
median
P05
P50
P95
P99
```

## 17.2. Baseline reference

Stage A/B tạo baseline distribution.

Full run compare mỗi shard với:

- prior shard window;
- same domain historical distribution;
- Stage B baseline.

## 17.3. Drift policy

Không nhất thiết dùng ML anomaly detection.

V1 có thể dùng:

```text
relative percentage change
IQR-based bounds
z-score-like alert where appropriate
```

Quan trọng là detection tồn tại và reason được log.

---

# 18. Golden regression suite chi tiết

Golden suite phải là artifact versioned trong repo.

Suggested structure:

```text
tests/golden/
├── manifest.yaml
├── expected/
│   ├── doc_001.yaml
│   ├── doc_002.yaml
│   └── ...
└── snapshots/
```

Không nhất thiết commit copyrighted full raw content nếu licensing/storage không phù hợp; có thể lưu expected hashes/snippets hoặc fixtures được phép.

## 18.1. Cleaner regression cases

Phải có examples chứa:

```text
HbA1c
BRCA1
HER2
SARS-CoV-2
SpO₂
0.5 mg
5 mg/kg
95% CI
p < 0.05
TNF-α
H. pylori
ICD-10
```

Assert các signal biomedical quan trọng không bị phá.

## 18.2. Chunk regression cases

Assert:

```text
expected chunk count range
no mid-sentence split where avoidable
parent contains child
source span exact
```

## 18.3. Extractor regression

Assert known key snippets tồn tại trong `source_text`.

Nếu đổi extractor mà key content biến mất → reject change.

---

# 19. Retrieval Quality Gate — quality cuối cùng phải đo bằng khả năng retrieve

Một corpus nhìn sạch chưa chắc retrieval tốt.

Từ bài học Stage 1, especially data normalization/chunk fixes và candidate recall bottleneck, promotion phải dựa thêm retrieval quality.

## 19.1. Nếu có official qrels

Đo:

```text
Recall@10
Recall@50
Recall@100
Recall@1000
nDCG@10
nDCG@100
BTC-style P/R/F2 nơi phù hợp
```

## 19.2. Nếu không có qrels

Tạo mini-dev 50–100 queries.

Pooling:

```text
BM25 top20
Dense top20
Hybrid top20
```

Union và annotate relevance.

Không gọi `query.parquet` là gold nếu nó chỉ có `id + query`.

## 19.3. Regression gate

Ví dụ:

```text
Corpus/chunker v1 → Hybrid Recall@100 = R1
Corpus/chunker v2 → Hybrid Recall@100 = R2
```

Nếu v2 đẹp hơn về clean text nhưng candidate recall giảm đáng kể, không auto-promote.

## 19.4. Self-retrieval smoke test

Dùng chunk text làm query và kiểm index mapping.

Mục đích:

- bắt vector-ID mismatch;
- bắt Lucene ID mismatch;
- bắt shard merge bug.

Không dùng self-retrieval làm semantic benchmark chính thức.

---

# 20. Resource Gate và full-vs-selective embedding

Bản gốc nói xây dense global index. Supplement này thêm fallback khi scale quá lớn.

## 20.1. Full chunk embedding

Chọn khi:

```text
GPU-hours within budget
storage within budget
index build within RAM/I/O budget
deadline margin adequate
retrieval benefit justifies cost
```

## 20.2. Query-conditioned selective chunk embedding

Nếu full child embedding quá đắt, ưu tiên fallback:

```text
ALL DOCS
  ↓
BM25 all docs
+
cheap/document-level multilingual dense
  ↓
run all 1,200 competition queries
  ↓
union top-N document candidates
  ↓
chunk + embed selected document universe deeply
```

Điểm quan trọng:

- không loại document chỉ vì `quality_tier=LOW`;
- LOW extraction quality không đồng nghĩa low relevance;
- cross-lingual gold có thể không được raw Vietnamese BM25 cứu.

## 20.3. Quality-conditioned embedding chỉ là secondary fallback

Có thể ưu tiên HIGH/MEDIUM nếu budget cực thấp, nhưng phải đo recall impact. Không coi quality tier là relevance prior cứng.

---

# 21. Detailed schemas cần bổ sung

## 21.1. `documents` dataset

Khuyến nghị partitioned Parquet dataset, không một giant file.

```text
doc_id: int64
original_url: string
final_url: string
domain: string
content_type: string
crawl_status: string
raw_asset_ref: string
raw_sha256: string
source_text: string
source_text_sha256: string
title: string | null
language: string
language_confidence: float | null
quality_tier: string
quality_reason_codes: list<string>
content_hash: string
canonical_content_id: string
extractor_name: string
extractor_version: string
normalizer_version: string
schema_version: string
char_count: int
paragraph_count: int
section_count: int
created_at: timestamp
```

## 21.2. `sections` dataset

```text
section_id: string
doc_id: int64
heading_text: string | null
start_char: int
end_char: int
section_index: int
parser_version: string
```

## 21.3. `child_chunks` dataset

```text
child_chunk_id: string
doc_id: int64
canonical_content_id: string
section_id: string | null
chunk_index: int
start_char: int
end_char: int
source_text_slice: string        # optional materialization; offsets remain truth
bge_token_count_source: int
bge_token_count_dense_repr: int
language: string
quality_flags: list<string>
retrieval_representation_hash: string
chunker_version: string
```

## 21.4. `parent_spans` dataset

Nếu materialized:

```text
parent_id: string
doc_id: int64
section_id: string | null
start_char: int
end_char: int
source_text_slice: string
bge_token_count: int
parent_builder_version: string
```

Nếu dynamic parent:

```text
child_chunk_id
section_start_char
section_end_char
sentence_boundary_map_ref
```

để reconstruct deterministic parent theo config.

## 21.5. `canonical_aliases`

```text
canonical_content_id
doc_id
content_hash
retrieval_representation_hash | null
alias_rank | null
```

Không mất BTC identity.

---

# 22. Partitioning / sharding strategy

Không dùng một giant `documents.parquet` hoặc `chunks.parquet` ở full scale.

Ví dụ:

```text
data/processed/documents/
  part-00000.parquet
  part-00001.parquet

data/chunks/child/
  part-00000.parquet
  part-00001.parquet
```

Partition có thể theo stable doc ID range hoặc stable hash.

Yêu cầu:

```text
deterministic shard assignment
reproducible mapping
no shard depends on runtime ordering
```

Lợi ích:

- resume;
- parallel build;
- partial rebuild;
- checksum riêng;
- rollback shard;
- cheap audit.

---

# 23. Manifest hierarchy

Không chỉ một `manifest.json` cuối cùng.

## 23.1. Dataset manifest

```json
{
  "dataset_version": "...",
  "links_sha256": "...",
  "query_sha256": "...",
  "links_row_count": 0,
  "query_row_count": 0
}
```

## 23.2. Crawl manifest

Per doc/shard crawl metadata.

## 23.3. Corpus manifest

```json
{
  "corpus_version": "v1",
  "schema_version": "...",
  "extractor_version": "...",
  "normalizer_version": "...",
  "document_count": 0,
  "canonical_content_count": 0,
  "source_text_hash_rollup": "...",
  "shards": []
}
```

## 23.4. Chunk manifest

```json
{
  "chunking_version": "child-v3-parent512",
  "tokenizer_name": "BAAI/bge-m3",
  "tokenizer_revision": "...",
  "child_config": {},
  "parent_config": {},
  "child_count": 0,
  "span_integrity_failures": 0,
  "shards": []
}
```

## 23.5. Embedding manifest

```text
model name
model revision
tokenizer revision
precision
normalize flag
max_length
input representation builder version
vector dimension
shard checksums
```

## 23.6. Index manifest

Dense:

```text
FAISS type
training sample hash
nlist/M/nbits/nprobe defaults
vector manifest hash
index sha256
```

Sparse:

```text
Lucene/Pyserini version
analyzer config
source chunk manifest hash
index sha256
```

Loaders phải assert compatibility thay vì assume.

---

# 24. Tokenizer/version contract

Bản gốc đã pin model revisions. Bổ sung bắt buộc:

```text
tokenizer_name
tokenizer_revision
tokenizer_config_hash
```

Lý do tokenizer ảnh hưởng:

- chunk length;
- max input packing;
- embedding representation;
- scorer tokenization behavior;
- parent-size experiments.

Không để two workers dùng hai tokenizer revisions khác nhau.

---

# 25. Quality reason-code taxonomy

Không chỉ lưu `quality_tier=LOW`.

Suggested taxonomy:

## Crawl

```text
HTTP_403
HTTP_404
HTTP_429
HTTP_5XX
TIMEOUT
DNS_ERROR
SSL_ERROR
BLOCKED
```

## Extraction

```text
EMPTY_CONTENT
ACCESS_DENIED_PAGE
BOT_CHALLENGE
LOGIN_REQUIRED
JS_REQUIRED
SCAN_PDF
EXTRACTION_TRUNCATED
TOO_SHORT
HIGH_BOILERPLATE_RATIO
HIGH_DUPLICATE_LINE_RATIO
MISSING_TITLE
```

## Structure

```text
NO_SECTIONS
ONE_GIANT_PARAGRAPH
MICRO_PARAGRAPH_EXPLOSION
TABLE_FLATTENED
REFERENCE_HEAVY
```

## Chunk

```text
TOO_SHORT_CHUNK
TOO_LONG_CHUNK
MID_SENTENCE_BOUNDARY
CROSS_SECTION
CONTEXT_FRAGMENT
NEAR_DUPLICATE
```

## Integrity

Integrity reason codes là HARD errors, ví dụ:

```text
SOURCE_SPAN_MISMATCH
INVALID_DOC_ID
ORPHAN_CHUNK
OFFSET_OUT_OF_BOUNDS
```

Reason codes giúp trả lời câu hỏi:

```text
"Tại sao 300k docs LOW?"
```

thay vì chỉ biết một scalar score.

---

# 26. Corpus Health Report — artifact bắt buộc

Sau Stage A/B/C và định kỳ full run, sinh report machine-readable + human-readable.

Ví dụ:

```text
CORPUS HEALTH
=============

Coverage
--------
URLs total                  ...
crawl success               ...
extract success             ...
promoted documents          ...

Quality
-------
HIGH                         ...
MEDIUM                       ...
LOW                          ...
QUARANTINE                   ...

Languages
---------
VI                           ...
EN                           ...
ZH                           ...
MIXED                        ...
UNKNOWN                      ...

Extraction warnings
-------------------
JS required                  ...
scan PDF                     ...
possible truncation          ...
high boilerplate             ...

Chunking
--------
children total               ...
mean chunks/doc              ...
P95 chunks/doc               ...
mean child tokens            ...
P95 child tokens             ...
near-duplicate rate          ...
source span failures         0
orphan chunks                0

Retrieval
---------
BM25 Recall@100              ...
Dense Recall@100             ...
Hybrid Recall@100            ...
```

Phải có breakdown theo language/domain.

---

# 27. Kill switch / pause policy

Pipeline full-scale cần auto-stop hoặc auto-pause.

## 27.1. Immediate STOP

```text
source span failure > 0
invalid doc_id > 0
orphan child > 0
parent containment failure > 0
schema incompatibility
input hash mismatch
model/tokenizer revision mismatch
```

## 27.2. PAUSE after shard

Threshold calibrate Stage A/B, ví dụ categories:

```text
extract success drops sharply
quality LOW/QUARANTINE spikes
chunks/doc distribution shifts strongly
language UNKNOWN spikes
specific major domain collapses
near-duplicate rate jumps
```

Không ghi fixed percentage trong spec như fact. Threshold phải dựa distribution thật.

---

# 28. Failure recovery protocol

Khi shard fail:

```text
1. stop promoting downstream artifacts
2. preserve .tmp/intermediate/logs
3. mark manifest FAILED
4. attach reason code
5. fix code/config
6. rerun golden regression
7. rerun failed shard only
8. compare distributions with previous build
9. commit if pass
```

Không rerun toàn corpus nếu issue local và upstream immutable artifacts vẫn compatible.

Nếu extractor semantics thay đổi globally, bump:

```text
extractor_version
corpus_version
```

và rebuild affected documents deterministically.

---

# 29. Side-by-side artifacts và promotion

Học discipline từ Stage 2: không overwrite artifact “best” ngay.

Ví dụ:

```text
corpus_v1/
corpus_v2/
chunks_child180_v1/
chunks_child256_v1/
index_dense_exp017/
```

Promotion symlink/manifest:

```text
promoted_corpus → corpus_v2
promoted_chunk_manifest → child180_parent512_v3
```

Chỉ promote khi:

```text
hard validations pass
health report acceptable
retrieval regression non-regressive
resource budget acceptable
human milestone audit passed where required
```

---

# 30. Repository changes so với cấu trúc gốc

Bản gốc đã có `ingestion/chunking/indexing/evaluation`. Bổ sung các module QA/observability rõ hơn.

```text
src/r2ai3/
├── ingestion/
│   ├── crawler.py
│   ├── raw_archive.py
│   ├── extract_html.py
│   ├── extract_pdf.py
│   ├── source_text.py
│   ├── normalize.py
│   ├── language.py
│   ├── quality.py
│   ├── dedup.py
│   └── schema.py
│
├── chunking/
│   ├── structure.py
│   ├── sentences.py
│   ├── child.py
│   ├── parent.py
│   ├── provenance.py
│   └── quality.py
│
├── validation/
│   ├── dataset.py
│   ├── document.py
│   ├── chunk.py
│   ├── dedup.py
│   ├── embedding.py
│   ├── index.py
│   └── gates.py
│
├── monitoring/
│   ├── distributions.py
│   ├── domain_report.py
│   ├── corpus_health.py
│   └── html_audit.py
│
├── manifests/
│   ├── dataset.py
│   ├── corpus.py
│   ├── chunks.py
│   ├── embeddings.py
│   └── index.py
│
└── evaluation/
    ├── retrieval_regression.py
    ├── candidate_recall.py
    └── scorer_stage3.py
```

Scripts:

```text
scripts/
01_snapshot.py
02_inventory.py
03_golden_test.py
04_stage_a_crawl.py
05_extract.py
06_validate_documents.py
07_build_structure.py
08_build_chunks.py
09_validate_chunks.py
10_corpus_health.py
11_freeze_corpus.py
12_build_bm25.py
13_benchmark_embed.py
14_build_dense.py
15_retrieval_regression.py
16_promote.py
```

---

# 31. Test suite bắt buộc

## Unit tests

```text
test_doc_id_preservation.py
test_source_text_hash.py
test_cleaner_biomedical_tokens.py
test_section_offsets.py
test_child_offsets.py
test_parent_contains_child.py
test_dedup_alias_mapping.py
test_dense_text_builder.py
test_sparse_text_builder.py
```

## Golden regression

```text
test_golden_extraction.py
test_golden_cleaning.py
test_golden_chunking.py
```

## Integration tests

```text
test_shard_end_to_end.py
test_manifest_compatibility.py
test_embedding_id_mapping.py
test_index_self_retrieval.py
test_resume_after_failure.py
```

## Competition-specific

```text
test_chunk_submission_is_source_slice.py
test_predicted_doc_ids_in_official_universe.py
test_stage3_scorer_clone.py
test_submission_schema.py
```

---

# 32. Pseudocode: shard processor

```python
def process_shard(shard):
    verify_input_snapshot(shard)

    raw_records = crawl_or_load_raw(shard)
    write_crawl_manifest(raw_records)

    docs = []
    for raw in raw_records:
        doc = extract_to_source_document(raw)
        validate_document_integrity(doc)
        score_document_quality(doc)
        docs.append(doc)

    write_temp_documents(docs)

    canonical_map = build_exact_dedup_mapping(docs)
    validate_dedup_identity(canonical_map, docs)

    chunks = []
    for doc in promotable_documents(docs):
        structure = parse_structure(doc)
        children = build_child_chunks(doc, structure)
        parent_meta = build_parent_mapping(doc, structure, children)

        validate_chunk_integrity(doc, children, parent_meta)
        score_chunk_quality(children)
        chunks.extend(children)

    stats = compute_shard_distributions(docs, chunks)
    hard_result = run_hard_gates(docs, chunks, canonical_map)
    soft_result = run_soft_gates(stats)

    if not hard_result.ok:
        mark_failed(shard, hard_result)
        raise PipelineStop()

    if soft_result.requires_pause:
        save_reports(shard, stats, soft_result)
        mark_needs_review(shard)
        raise PipelinePause()

    checksums = checksum_temp_artifacts()
    atomic_commit(shard, checksums)
    update_global_manifest(shard, stats, checksums)
```

---

# 33. Pseudocode: exact chunk validator

```python
def validate_child(doc, chunk):
    assert chunk.doc_id == doc.doc_id
    assert 0 <= chunk.start_char < chunk.end_char <= len(doc.source_text)

    recovered = doc.source_text[chunk.start_char:chunk.end_char]
    assert recovered == chunk.text

    if chunk.parent is not None:
        p = chunk.parent
        assert p.doc_id == doc.doc_id
        assert p.start_char <= chunk.start_char
        assert chunk.end_char <= p.end_char
        assert doc.source_text[p.start_char:p.end_char] == p.text
```

Không tolerance.

---

# 34. Pseudocode: dedup identity validator

```python
def validate_dedup(docs, canonical_alias_rows):
    official = {d.doc_id for d in docs}
    mapped = {row.doc_id for row in canonical_alias_rows}

    assert mapped == official

    for row in canonical_alias_rows:
        assert row.canonical_content_id is not None
        assert row.content_hash is not None
```

Nếu một document crawl/extract fail và policy exclude khỏi processed corpus, phải xuất explicit failed-doc manifest. Không được “biến mất”.

---

# 35. Config bổ sung

```yaml
pipeline:
  schema_version: "data-v1"
  shard_docs: 10000
  atomic_commit: true

source_text:
  immutable_after_commit: true
  store_sha256: true

validation:
  hard_fail:
    exact_source_span: true
    doc_id_membership: true
    unique_chunk_id: true
    foreign_keys: true
    child_parent_containment: true

  soft_monitoring:
    distribution_drift: true
    domain_breakdown: true
    language_breakdown: true
    content_type_breakdown: true

quality:
  rule_based_v1: true
  store_reason_codes: true
  error_page_detection: true
  scan_pdf_detection: true
  js_required_detection: true

chunking:
  tokenizer: "BAAI/bge-m3"
  tokenizer_revision: "PIN_ME"
  child_tokens: 180
  child_overlap_tokens: 32
  parent_tokens: 512
  prefer_sentence_boundary: true
  prefer_section_boundary: true
  dynamic_parent: true

dedup:
  exact_content: true
  preserve_doc_aliases: true
  distinguish_retrieval_representation_hash: true
  near_duplicate: false

reports:
  corpus_health: true
  html_audit: true
  top_failed_domains: 30

promotion:
  require_golden_pass: true
  require_hard_gate_pass: true
  require_retrieval_regression_pass: true
```

Các con số 10k/180/512 là starting defaults/hypotheses, không phải kết quả tối ưu đã được chứng minh.

---

# 36. Stage-by-stage Definition of Done

## Golden DoD

```text
[ ] fixtures cover VI/EN/ZH
[ ] HTML/PDF cases exist
[ ] biomedical symbol regression exists
[ ] extraction key snippets pass
[ ] exact child/parent span pass
[ ] dedup alias test pass
```

## Stage A DoD

```text
[ ] ~1k stratified URLs processed
[ ] crawl statuses summarized
[ ] top domains inspected
[ ] raw storage measured
[ ] extraction success by domain/language/type reported
[ ] scan/JS incidence estimated
[ ] health report generated
[ ] human sample audit completed
[ ] no hard integrity failures
```

## Stage B1 DoD

```text
[ ] ~10k processed
[ ] shard resume works
[ ] atomic commit works
[ ] chunk distributions stable enough to inspect
[ ] exact span failures = 0
[ ] dedup mapping validated
[ ] first BM25/dense retrieval prototype runs
[ ] mini-dev/golden retrieval test available
```

## Stage B2 DoD

```text
[ ] ~100k processed
[ ] domain-level failure report reviewed
[ ] chunk size candidates compared
[ ] duplicate rate measured
[ ] BGE throughput measured on actual hardware
[ ] vector/storage estimates computed
[ ] retrieval regression report generated
[ ] child config selected provisionally
[ ] parent config selected provisionally
```

## Stage C DoD

```text
[ ] 100k→1M long run stable
[ ] no memory leak/blocking issue
[ ] recovery from forced stop tested
[ ] Lucene growth measured
[ ] dense index prototype stable
[ ] ANN recall measured vs exact subset
[ ] full-scale budget updated
```

## Full Corpus DoD

```text
[ ] all shards DONE or explicitly failed/quarantined with reasons
[ ] zero hard integrity failures in promoted artifacts
[ ] source provenance complete
[ ] corpus manifest frozen
[ ] child chunk manifest frozen
[ ] health report acceptable
[ ] retrieval regression non-regressive
[ ] model/tokenizer revisions pinned
[ ] indexes trace to exact frozen manifests
```

---

# 37. Promotion policy

Artifact chỉ được gắn `PROMOTED` nếu:

```text
1. integrity pass
2. quality report reviewed/within gate
3. retrieval regression pass
4. resource budget pass
5. provenance/manifests complete
```

Không promote chỉ vì:

```text
"job finished successfully"
```

Job success chỉ chứng minh code chạy hết, không chứng minh corpus tốt.

---

# 38. Observability metrics tối thiểu cần log liên tục

## Crawl

```text
requests/s
success rate
429 rate
403 rate
timeout rate
bytes/s
latency P50/P95
```

## Extract

```text
extract docs/s
extract success
chars/doc
paragraphs/doc
possible error pages
scan PDF count
JS-required count
```

## Chunk

```text
chunks/s
chunks/doc
child tokens
parent tokens
source-span failures
near-duplicate rate
```

## Embedding

```text
chunks/s
tokens/s
GPU utilization
VRAM peak
batch size
P50/P95 batch latency
```

## Index

```text
indexed count
missing IDs
duplicate IDs
index size
build time
ANN recall
```

---

# 39. Những việc KHÔNG được làm sau supplement này

```text
❌ Crawl full corpus trước Stage A/B
❌ Embed trực tiếp trong crawler
❌ Dùng processed DataFrame index làm doc_id
❌ Overwrite source_text sau commit
❌ Reconstruct submission chunk từ normalized/search text
❌ Dedup rồi vứt alias BTC doc_id
❌ Dùng một giant Parquet file cho toàn corpus nếu scale lớn
❌ Đợi full run xong mới validate
❌ Chỉ xem overall success rate
❌ Gọi quality score là relevance score
❌ Chọn chunk size chỉ vì rẻ hơn
❌ Promote corpus chỉ vì pipeline không crash
❌ Silent overwrite artifact tốt bằng artifact mới chưa benchmark
```

---

# 40. Implementation priority — thứ tự code thực tế

## P0.1 — Contracts trước

```text
Document schema
Child schema
Parent schema
Manifest schema
Reason-code enum
```

## P0.2 — Validators trước khi scale

```text
doc_id membership
source span round-trip
child-parent containment
foreign keys
dedup identity
```

## P0.3 — Golden set

```text
extractor fixtures
cleaner biomedical regression
chunk regression
```

## P0.4 — Crawl/raw replay

```text
inventory
crawler
raw archive
crawl manifest
```

## P0.5 — Extraction/source text

```text
HTML
PDF
source hash
quality features
```

## P0.6 — Structure/chunk

```text
sections
paragraphs
sentences
child
parent mapping
```

## P0.7 — Sharded processing engine

```text
.tmp
validate
atomic commit
checkpoint
resume
```

## P0.8 — Health reports

```text
domain/language/type breakdown
drift
HTML QA report
```

## P0.9 — Stage A/B

Chỉ sau đó mới:

## P1 — Indexing/retrieval

```text
Pyserini BM25
BGE-M3 benchmark
FAISS baseline
hybrid retrieval
```

## P1 — Retrieval regression

```text
mini-dev
Recall@K
nDCG
candidate contribution
```

## P2 — Advanced extraction

Chỉ khi metrics chứng minh cần:

```text
Playwright
OCR
near-duplicate MinHash
advanced table reconstruction
ML quality classifier
```

---

# 41. Decision log bắt buộc

Mọi thay đổi data pipeline đáng kể phải có decision record:

```text
Decision ID:
Date:
Component:
Old behavior:
New behavior:
Why:
Golden regression result:
Stage B quality delta:
Retrieval delta:
Storage/runtime delta:
Promoted? yes/no
```

Ví dụ:

```text
DEC-014
Chunk child 180 → 256

Reason:
reduce fragmentation

Result:
Recall@100: ...
Index size: ...
GPU cost: ...

Decision:
reject / promote
```

Điều này ngăn team đổi config theo cảm giác rồi quên lý do.

---

# 42. Release linkage tới Stage 3 submission

Data pipeline hoàn tất không có nghĩa system ready. Tuy nhiên release submission phải trace được:

```text
submission row
  ↓
retrieved chunk ID
  ↓
child source span
  ↓
parent source span
  ↓
document doc_id
  ↓
source_text hash
  ↓
raw asset / crawl manifest
```

Đối với mọi emitted `chunk_text`, ideally validator có thể chạy:

```python
assert submission_chunk_text == recovered_source_span
```

Không emit synthetic/translated/HyDE text.

---

# 43. Lessons từ Stage 1/2 được encode như thế nào

## Stage 1 — data normalization / candidate recall

Encode thành:

```text
Retrieval Quality Gate
chunk-size ablation
candidate recall regression
```

Không đánh giá data quality chỉ bằng visual cleanliness.

## Stage 1 — canonical corpus

Encode thành:

```text
immutable source schema
stable IDs
dedup mappings
freeze before indexing
```

## Stage 1 — manifests/index lifecycle

Encode thành:

```text
corpus/chunk/model/index manifests
compatibility assertions
```

## Stage 2 — provenance/replay

Encode thành:

```text
raw → source → chunk → index traceability
```

## Stage 2 — release/promotion discipline

Encode thành:

```text
side-by-side artifacts
hard gate
regression gate
promotion only after audit
rollback path
```

---

# 44. Final architecture after supplement

```text
links_corpus.parquet + query.parquet
              │
              ▼
        DATASET SNAPSHOT
        hash / schema / IDs
              │
              ▼
          URL INVENTORY
              │
              ▼
       GOLDEN + STAGE A
              │
              ▼
          ASYNC CRAWL
              │
       ┌──────┴────────┐
       ▼               ▼
  RAW ARCHIVE      CRAWL MANIFEST
       │
       ▼
 CONTENT ROUTER
 HTML / PDF / text
       │
       ▼
 EXTRACTION
       │
       ▼
 IMMUTABLE SOURCE_TEXT
       │
       ├────────► source hash / provenance
       │
       ▼
 SAFE NORMALIZATION
       │
       ├────────► language / quality features
       │
       ▼
 EXACT DEDUP MAPPING
       │
       ├────────► preserve every BTC doc_id
       │
       ▼
 STRUCTURE PARSING
 section → paragraph → sentence
       │
       ▼
 CHILD RETRIEVAL CHUNKS
 ~120–256 token experiments
       │
       ├────────► exact offsets
       │
       ▼
 PARENT SPAN MAPPING
 ~350–650 token experiments
       │
       ▼
 HARD INTEGRITY GATE
       │
       ▼
 DATA QUALITY GATE
       │
       ▼
 DISTRIBUTION / DRIFT GATE
       │
       ▼
 SHARD ATOMIC COMMIT
       │
       ▼
 STAGE B/C EVALUATION
       │
       ├────────► retrieval quality
       ├────────► resource budget
       └────────► human milestone audit
       │
       ▼
 FREEZE CORPUS VERSION
       │
       ├───────────────┐
       ▼               ▼
   LUCENE BM25      DENSE EMBEDDING
                       │
                       ▼
                     FAISS
       │               │
       └───────┬───────┘
               ▼
       RETRIEVAL REGRESSION
               │
               ▼
           PROMOTE
```

---

# 45. Final acceptance checklist

## Correctness

```text
[ ] BTC doc IDs never remapped
[ ] every promoted chunk exact-round-trips to source_text
[ ] every parent contains its child
[ ] no orphan child/parent
[ ] all offsets within bounds
[ ] dedup preserves all official identities
```

## Provenance

```text
[ ] raw asset traceable
[ ] source hash exists
[ ] extractor version pinned
[ ] normalizer version pinned
[ ] chunker version pinned
[ ] tokenizer revision pinned
[ ] index traces to chunk manifest
```

## Quality

```text
[ ] extraction quality report exists
[ ] domain/language/type breakdown exists
[ ] error-page detection active
[ ] scan/JS detection active
[ ] chunk distribution report exists
[ ] adversarial sample audit done at milestones
```

## Regression

```text
[ ] golden suite passes
[ ] retrieval regression passes
[ ] self-retrieval smoke test passes
[ ] ANN fidelity acceptable if ANN used
```

## Operations

```text
[ ] shards atomic
[ ] resume tested
[ ] failure recovery tested
[ ] side-by-side artifacts retained
[ ] rollback path exists
```

## Scale

```text
[ ] GPU/storage estimates based on measured Stage B numbers
[ ] full-vs-selective embedding decision documented
[ ] deadline margin documented
```

---

# 46. Definition of “pipeline xử lý data đã hoàn chỉnh”

Không định nghĩa bằng:

```text
"crawler chạy được"
```

hoặc:

```text
"đã tạo được embeddings"
```

Mà bằng:

> **Một corpus version chỉ được coi là hoàn chỉnh khi nó reproducible, provenance-complete, không vi phạm hard integrity invariants, có quality diagnostics rõ ràng, không regression retrieval đáng kể và có thể rollback/rebuild theo shard.**

Do đó state machine của một corpus artifact nên là:

```text
BUILDING
  ↓
VALIDATED
  ↓
QUALITY_REVIEWED
  ↓
RETRIEVAL_TESTED
  ↓
FROZEN
  ↓
PROMOTED
```

Không jump trực tiếp:

```text
BUILDING → PROMOTED
```

---

# 47. Việc nên làm ngay sau tài liệu này

Thứ tự implementation khuyến nghị:

```text
1. Define schemas + enums
2. Implement hard validators
3. Build golden fixtures
4. Implement snapshot/inventory
5. Implement crawler + raw archive + manifest
6. Implement extraction → immutable source_text
7. Implement basic quality features/reason codes
8. Implement exact dedup mapping preserving doc IDs
9. Implement structure parser
10. Implement child chunks + parent mapping
11. Implement shard runner + atomic commit
12. Implement corpus health + HTML audit reports
13. Run Golden
14. Run Stage A 1k
15. Fix top-domain extraction issues
16. Run Stage B 10k
17. Add BM25/dense retrieval prototype
18. Build mini-dev if official qrels unavailable
19. Run Stage B 100k
20. Select provisional child/parent configs
21. Benchmark embedding/index cost
22. Stage C
23. Freeze and full-scale only after gates pass
```

---

# 48. Chốt

Bản `.md` gốc đã có đúng **retrieval architecture** và đúng intuition quan trọng nhất: canonical corpus, stable IDs, child/parent hierarchy, provenance, manifests và scorer-aware output.

Supplement này bổ sung thứ còn thiếu để biến architecture đó thành một **production-grade competition data pipeline**:

```text
architecture
+
continuous verification
+
data quality gates
+
retrieval regression
+
sharded recovery
+
artifact promotion discipline
```

Mục tiêu không phải đảm bảo “không bao giờ có bug”. Mục tiêu thực tế hơn và quan trọng hơn là:

> **Nếu bug xuất hiện, pipeline phải phát hiện nó sớm, cô lập được shard bị ảnh hưởng, biết vì sao fail, không promote artifact sai, và không buộc team phải chạy lại hàng triệu documents từ đầu.**

Đó là lớp bảo vệ cần có trước khi ViBioMIR data pipeline được phép scale full corpus.
