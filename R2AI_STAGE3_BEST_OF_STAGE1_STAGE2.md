# R2AI Stage 3 — Best-of Stage 1 + Stage 2
## Thiết kế hệ thống truy hồi y sinh đa ngôn ngữ tối ưu cho Document F2 + Chunk F2

> **Mục tiêu của tài liệu**  
> Tài liệu này là một **competition specification + system design + research playbook** cho R2AI Stage 3. Phần đầu ghi lại đầy đủ bối cảnh, dataset, submission schema, Dashboard rules, public/private protocol, model/data constraints và cách chấm; các phần sau đọc lại và chắt lọc pattern kỹ thuật tốt nhất từ hai repository công khai `r2ai-stage-1` và `r2ai-stage-2`, rồi chuyển hóa chúng thành kiến trúc truy hồi y sinh đa ngôn ngữ dành riêng cho Stage 3.  
> Đây **không** phải là việc bê nguyên pipeline Stage 1 hoặc Stage 2 sang Stage 3. Những phần legal-specific, financial-specific hoặc answer-generation không phù hợp sẽ bị loại. Những phần về retrieval, candidate recall, reranking, chunking, provenance, artifact discipline, regression testing và F2 optimization sẽ được giữ lại và điều chỉnh theo metric và hard rules Stage 3.  
> **Cập nhật rule:** 01/10/2026.

---

# A. Đặc tả đầy đủ R2AI Stage 3 — đọc phần này trước khi đọc kiến trúc

> **Trạng thái tài liệu:** cập nhật theo thông tin cuộc thi được cung cấp đến **01/10/2026**.  
> Phần này tách rõ **hard rules của cuộc thi** khỏi **khuyến nghị kỹ thuật** ở các chương sau. Nếu một recommendation ở phần sau mâu thuẫn với hard rule tại đây, **hard rule được ưu tiên tuyệt đối**.

## A.1. Bối cảnh và động cơ bài toán

Tri thức y sinh hiện nay phân bố không đồng đều giữa các ngôn ngữ. Nguồn tiếng Việt ngày càng tăng nhưng vẫn hạn chế về chiều sâu và mức độ chuyên môn; trong khi đó tiếng Anh và tiếng Trung có kho tài nguyên y sinh lớn hơn đáng kể. Điều này tạo ra một **khoảng cách tiếp cận bằng chứng y học** đối với người dùng đặt câu hỏi bằng tiếng Việt.

R2AI Stage 3 vì vậy không yêu cầu một chatbot sinh câu trả lời tự do, mà yêu cầu xây dựng một **hệ thống truy hồi y sinh đa ngôn ngữ**. Với một truy vấn bằng tiếng Việt, hệ thống phải tìm được tài liệu và đoạn nội dung liên quan dù bằng chứng nằm trong nguồn tiếng Việt, tiếng Anh hay tiếng Trung.

Có thể hình thức hóa nhiệm vụ như sau:

```text
Input:
    Q = {q1, q2, ..., qn}
    mỗi qi là truy vấn y khoa bằng tiếng Việt

Corpus space:
    các tài liệu được chỉ định trong links_corpus.parquet
    nội dung có thể ở VI / EN / ZH

Đội thi tự xây dựng:
    crawling / acquisition
    preprocessing
    normalization
    document parsing
    chunking
    indexing
    embedding
    sparse retrieval
    dense retrieval
    reranking
    post-processing
    output selection

Output cho mỗi qi:
    tập doc_id dự đoán liên quan
    tập chunk_text nguyên văn, mỗi chunk gắn với một doc_id
```

Hai cấp độ được đánh giá độc lập:

1. **Document retrieval:** tìm đúng tài liệu.
2. **Chunk retrieval:** định vị đúng phần nội dung chứa bằng chứng cần thiết trong tài liệu.

Điểm cuối cùng là trung bình của hai F2-macro này. Vì vậy hệ thống mạnh ở document retrieval nhưng chunk retrieval yếu — hoặc ngược lại — đều không tối ưu.

---

## A.2. Mục tiêu chính thức của hệ thống

### A.2.1. Truy hồi chính xác ở cấp tài liệu

Hệ thống phải:

- xác định đúng các tài liệu liên quan đến truy vấn;
- hoạt động trên nguồn y sinh đa ngôn ngữ;
- thực hiện **cross-lingual retrieval** giữa query tiếng Việt và evidence tiếng Việt/Anh/Trung;
- không phụ thuộc duy nhất vào exact keyword matching.

### A.2.2. Truy hồi chính xác ở cấp chunk

Hệ thống phải:

- tìm được đoạn nội dung thực sự chứa thông tin y khoa liên quan;
- giữ được liên kết chunk → source document;
- tự thiết kế segmentation/chunking strategy;
- trả `chunk_text` lấy từ tài liệu nguồn, không phải đoạn do model tự viết.

### A.2.3. Hiểu truy vấn y khoa tiếng Việt

Hệ thống cần xử lý được:

- thuật ngữ y sinh và lâm sàng tiếng Việt;
- tên thuốc, bệnh, gene/protein, biomarker, xét nghiệm, thủ thuật;
- viết tắt và synonym;
- câu hỏi có nhiều điều kiện hoặc nhiều ý cùng lúc;
- khác biệt cách diễn đạt giữa ngôn ngữ đời thường tiếng Việt và văn phong biomedical tiếng Anh/Trung.

---

## A.3. Dataset chính thức: ViBioMIR

Nguồn dữ liệu chính thức:

**Hugging Face:** <https://huggingface.co/datasets/AIGuruTinix/ViBioMIR>

Ban Tổ chức cung cấp tối thiểu hai thành phần logic quan trọng:

```text
query.parquet
links_corpus.parquet
```

BTC **không cung cấp sẵn** pipeline crawling, parsing, embedding, indexing hay retrieval. Đội thi phải tự xây toàn bộ knowledge-base pipeline.

### A.3.1. `query.parquet`

Mỗi dòng chứa:

```json
{
  "id": 1,
  "query": "Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?"
}
```

Schema logic:

| Field | Type | Ý nghĩa |
|---|---|---|
| `id` | integer | ID truy vấn, dùng để map prediction với câu hỏi |
| `query` | string | Truy vấn y khoa bằng tiếng Việt |

### A.3.2. `links_corpus.parquet`

Mỗi dòng chứa một nguồn tài liệu:

```json
{
  "id": 1,
  "url": "https://example.com/article"
}
```

Schema logic:

| Field | Type | Ý nghĩa |
|---|---|---|
| `id` | integer | ID tài liệu chính thức dùng trong chấm điểm |
| `url` | string | URL nguồn mà đội phải tự thu thập nội dung |

### A.3.3. Rule cực quan trọng về universe của document labels

Trong phiên bản đánh giá này:

> **Mọi nhãn document đều dùng `id` trong `links_corpus.parquet`. Tài liệu nằm ngoài danh sách này không được tính là nhãn dương.**

Hệ quả kỹ thuật:

```text
External biomedical corpus
        │
        ├── OK: train embedding/reranker
        ├── OK: synonym dictionary / ontology support
        ├── OK: query expansion / terminology mapping
        └── KHÔNG NÊN emit như relevant_docs/relevant_chunks
                vì ngoài official links_corpus universe
```

External data có thể nâng chất lượng model, nhưng **submission space phải được grounded vào official corpus**.

---

## A.4. Thu thập và xây dựng corpus là một phần của bài thi

Đội thi nhận URL chứ không nhận một kho chunk đã index sẵn. Vì vậy data acquisition không phải housekeeping mà là một thành phần của hệ thống.

Một pipeline tối thiểu cần quản lý:

```text
links_corpus.parquet
        ↓
URL fetcher
        ↓
content-type detection
        ↓
HTML / XML / PDF / text parser
        ↓
boilerplate removal
        ↓
main-content extraction
        ↓
language detection
        ↓
Unicode / punctuation / whitespace normalization
        ↓
source-preserving canonical document
        ↓
section-aware segmentation
        ↓
child chunks + parent chunks
        ↓
indexes
```

Với mỗi document nên giữ một provenance record tối thiểu:

```json
{
  "doc_id": 123,
  "url": "...",
  "fetched_at": "...",
  "http_status": 200,
  "mime_type": "text/html",
  "content_sha256": "...",
  "language": "en",
  "parser_version": "...",
  "parse_status": "ok",
  "raw_snapshot_path": "...",
  "canonical_text_path": "..."
}
```

Lý do: một URL có thể đổi nội dung, redirect, lỗi crawl hoặc parse khác nhau giữa hai lần build. Nếu không snapshot + hash, rất khó tái lập public-best trong private phase.

---

## A.5. Dữ liệu ngoài và mô hình được phép sử dụng

### A.5.1. External data

Được phép sử dụng dữ liệu ngoài, với điều kiện:

- nguồn phải được công khai/trình bày rõ;
- provenance phải đủ để BTC kiểm tra khi cần;
- usage phải hợp pháp theo license/điều khoản của nguồn.

External biomedical datasets có thể dùng cho:

```text
fine-tune multilingual embedding
fine-tune reranker
hard-negative mining
query paraphrase training
medical terminology mapping
synonym expansion
language alignment
local auxiliary evaluation
```

Nhưng nhắc lại: **external document không trở thành gold document của benchmark nếu không có trong `links_corpus.parquet`.**

### A.5.2. PLM / LLM rule

Được sử dụng pretrained language models và LLM nếu đáp ứng **đồng thời**:

```text
[1] model/data huấn luyện và/hoặc weights được công khai
[2] model được phát hành trước 01/08/2026 theo giờ Việt Nam
[3] kích thước model <= 15B parameters
[4] không phải closed-weight / closed-model system
```

Ví dụ model đóng như GPT-4o, Gemini không được dùng theo rule đã cung cấp.

Đối với **mọi model đưa vào pipeline**, release manifest nên lưu:

```text
model_id
revision / commit hash
parameter count
release date
license
source URL
role in pipeline
```

Không chỉ LLM chat; embedding, reranker, classifier hoặc query rewriter cũng nên có hồ sơ tương tự để dễ audit.

---

# A.6. Submission schema chính thức

Submission retrieval là một file `.json` có dạng:

```json
[
  {
    "id": 1,
    "relevant_docs": [101, 203],
    "relevant_chunks": [
      {
        "doc_id": 101,
        "chunk_text": "Nội dung nguyên văn được trích từ tài liệu nguồn."
      },
      {
        "doc_id": 203,
        "chunk_text": "Một đoạn nguyên văn khác từ tài liệu tương ứng."
      }
    ]
  }
]
```

> Trong mô tả chính thức, source `id` trong `links_corpus.parquet` là số nguyên. Ví dụ placeholder của BTC có thể biểu diễn ID dưới dạng chuỗi như `"doc_id1"`; implementation không nên tự suy đoán kiểu dữ liệu từ placeholder. Khi build release, hãy validate theo **template/schema thực tế do Dashboard cung cấp** và giữ mapping 1–1 với `links_corpus.parquet`.

## A.6.1. `id`

- ID câu hỏi.
- Phải có đầy đủ tất cả query cần nộp.
- Không được duplicate.

## A.6.2. `relevant_docs`

- Danh sách document IDs dự đoán liên quan.
- ID phải correspond với `links_corpus.parquet.id`.
- Không emit URL thay cho ID.
- Không emit external document ID ngoài official corpus.

## A.6.3. `relevant_chunks`

Mỗi phần tử có:

```json
{
  "doc_id": 101,
  "chunk_text": "...",
  "chunk_order": 0
}
```

Trong đó:

- `doc_id`: ID tài liệu nguồn;
- `chunk_text`: đoạn văn được truy hồi;
- `chunk_order`: **optional**, nếu có phải là integer không âm.

Nếu `chunk_order` bị bỏ qua, hệ thống dùng vị trí chunk trong array `relevant_chunks`.

## A.6.4. `chunk_text` là exact-source evidence — hard rule

`chunk_text`:

> **phải là nội dung được trích xuất từ tài liệu gốc; không phải nội dung được viết lại, dịch lại, tóm tắt hoặc sinh mới bởi model.**

Điều này tạo một ranh giới cực rõ:

```text
ALLOWED internally:
    translated query
    HyDE hypothetical passage
    LLM subquery
    synthetic biomedical paraphrase
    model-generated retrieval description

NOT ALLOWED as submitted chunk_text:
    bất kỳ đoạn nào ở trên nếu nó không tồn tại trong source document
```

Vì vậy Stage 3 là **retrieval + source extraction**, không phải passage generation.

## A.6.5. Empty prediction vẫn phải đúng schema

Nếu hệ thống không dự đoán được ở một mức:

```json
{
  "id": 999,
  "relevant_docs": [],
  "relevant_chunks": []
}
```

Không được bỏ field.

Một `relevant_chunks: []` hợp lệ về schema; điều không nên tồn tại là một object chunk với `chunk_text` rỗng hoặc `doc_id` không hợp lệ.

---

# A.7. Packaging và Dashboard

Các đội nộp prediction trực tiếp trên Dashboard chính thức của cuộc thi.

Trang leaderboard:

<https://leaderboard.aiguru.com.vn/>

Quy trình release:

```text
results.json
    ↓
validate schema + query coverage + provenance
    ↓
ZIP
    ↓
My Submissions
    ↓
upload
```

Hard packaging rules:

```text
submission.zip
└── <duy nhất một file JSON ở archive root>
```

Không được:

```text
submission.zip
└── folder/
    └── results.json
```

hoặc nhét nhiều artifact/log/config vào ZIP thi.

### A.7.1. Full-query requirement

Submission phải bao gồm kết quả cho **toàn bộ 1.200 truy vấn** đã phát hành trong hai giai đoạn.

Hệ thống chấm tự quyết định subset thuộc public hay private phase. Đội thi **không cần biết trước** cách chia public/private để format bài nộp.

Hệ quả:

```text
submission builder phải luôn build full 1,200-query file
không build "public-only.json" hoặc "private-only.json"
```

### A.7.2. Invalid/missing submission

Bài thiếu file, thiếu câu hoặc sai format sẽ không được đánh giá. Theo rule được cung cấp, submission bị thiếu file/thiếu câu cũng không bị tính vào quota tối đa; tuy vậy không nên dựa vào hành vi này như một workflow thử nghiệm — release validator phải chặn lỗi trước upload.

---

# A.8. Giới hạn số lần nộp

## Public phase

- Tối đa **10 submissions mỗi ngày** cho mỗi đội theo quy định đã cung cấp.
- Nên coi mỗi submission như một experiment có hypothesis rõ, không dùng leaderboard làm random hyperparameter search.

## Private phase

- Tối đa **5 submissions tổng cộng cho mỗi user** trong Private Phase.
- Vì đội cần một username đại diện, private quota phải được quản lý như release budget.

Một private submission log nên có:

```text
submission_id
artifact_sha256
config_sha256
corpus_snapshot
model revisions
experiment rationale
expected delta
actual dashboard result
remaining_private_budget
```

Private phase không phải chỗ để “thử xem model nào tốt”. Chỉ promote các variant đã được public/dev evidence ủng hộ.

---

# A.9. Working-notes paper và tính chính thức của kết quả

Kết quả cuối cùng **chưa được xem là chính thức** cho đến khi đội nộp một **working notes paper** mô tả đầy đủ phương pháp.

Do đó từ ngày đầu tiên nên log đủ để viết paper:

```text
data sources + licenses
crawl/preprocess method
chunking strategy
indexing strategy
models + revisions
query transformation
fusion
reranking
cutoff
local evaluation
ablations
public submissions
failure analysis
compute/runtime
reproducibility notes
```

Không nên đến cuối cuộc thi mới cố tái dựng các config đã chạy.

BTC có quyền loại thí sinh nếu submission hoặc hệ thống không tuân thủ yêu cầu.

---

# A.10. Timeline chính thức

Tất cả deadline là **23:59 giờ Việt Nam (UTC+07:00)**:

| Mốc | Thời gian |
|---|---|
| Public test release | **01/10/2026** |
| Public test deadline | **31/10/2026 23:59 UTC+07** |
| Private phase bắt đầu | **01/11/2026** |
| Hạn chót nộp hệ thống | **04/11/2026 23:59 UTC+07** |
| Công bố kết quả chung cuộc | **11/11/2026** |

Vì private phase chỉ kéo dài vài ngày và quota rất nhỏ, kiến trúc/index/model pipeline cần **freeze gần như hoàn toàn trước 31/10**.

---

# A.11. Document-level scoring

Với mỗi query, gọi:

```text
D = tập document IDs hệ thống dự đoán
G = tập document IDs gold
```

Khi đó:

```text
Precision_doc = |D ∩ G| / |D|
Recall_doc    = |D ∩ G| / |G|
```

Nếu query không có nhãn dương ở cấp document, query đó **không tham gia mẫu số macro-average document**.

---

# A.12. Chunk-level scoring — phần quan trọng nhất để hiểu đúng

Với mỗi query:

```text
C = tập predicted chunks
I = các information/reference chunks cần thiết trong gold documents
```

Scorer xử lý theo logic:

### Bước 1 — same-document constraint

Một predicted chunk chỉ được so với reference chunk thuộc **cùng `doc_id`**.

Tức là text có giống hoàn hảo nhưng gắn nhầm document vẫn không được credit theo rule này.

### Bước 2 — text normalization

Trước matching, scorer chuẩn hóa các yếu tố được mô tả gồm:

```text
Unicode
HTML
case
punctuation equivalents
whitespace
```

### Bước 3 — BGE-M3 tokenization

Sau normalize, text được token hóa bằng tokenizer của **BGE-M3**.

Đây là lý do local scorer phải dùng đúng tokenizer/normalization behavior gần scorer nhất có thể; whitespace-token hoặc tokenizer khác có thể dẫn đến local metric sai.

### Bước 4 — relevance bằng LCS

Một predicted `chunk_text` được tính relevant nếu với ít nhất một reference chunk cùng document:

```text
LCS(pred_tokens, ref_tokens) >= 0.40 × len(ref_tokens)
```

Điểm đáng chú ý: ngưỡng được tính theo **độ dài reference chunk**.

### Bước 5 — near-duplicate merge

Các predicted chunks trùng hoặc gần trùng trong cùng document được gộp trước khi tính precision.

Ngưỡng mặc định:

```text
LCS / union >= 0.8
```

Vì vậy spam nhiều sliding windows gần giống nhau không phải chiến lược bền vững để tăng coverage.

### Bước 6 — chunk Precision / Recall

```text
Precision_chunk = số predicted chunks relevant / tổng predicted chunks sau xử lý

Recall_chunk = số information units cần thiết được bao phủ / tổng information units cần thiết
```

Nếu query không có nhãn dương ở cấp chunk, query đó **không tham gia mẫu số macro-average chunk**.

---

# A.13. F2 macro và final leaderboard score

Ở mỗi cấp, với từng query có gold dương:

```text
F2 = 5 × Precision × Recall / (4 × Precision + Recall)
```

Sau đó lấy macro-average qua các query hợp lệ:

```text
Document_F2_macro
Chunk_F2_macro
```

Điểm xếp hạng cuối cùng:

```text
FinalScore = (Document_F2_macro + Chunk_F2_macro) / 2
```

## Tại sao F2 thay đổi cách tối ưu?

F2 đặt recall nặng hơn precision. Điều đó **không** có nghĩa là cứ emit thật nhiều document/chunk; noise vẫn kéo precision xuống. Ý đúng là:

```text
high-recall candidate generation
        ↓
strong reranking
        ↓
recall-safe adaptive cutoff
```

chứ không phải:

```text
very strict selector
→ đẹp precision
→ làm rơi gold
```

Các failure của Stage 1 Hung&Fong cho thấy đây là bài học đã có bằng chứng thực nghiệm rất rõ.

---

# A.14. Rule → engineering consequence matrix

| Hard rule / metric | Hệ quả trực tiếp cho thiết kế |
|---|---|
| Query luôn tiếng Việt, corpus VI/EN/ZH | Cần cross-lingual dense retrieval và/hoặc translated sparse branches |
| Chỉ doc IDs trong `links_corpus.parquet` là gold universe | External corpus chỉ dùng hỗ trợ model/query understanding; output phải official-corpus-bound |
| Đội tự crawl URL | Crawler, content snapshot, parser và provenance trở thành phần cốt lõi |
| `chunk_text` phải lấy từ tài liệu gốc | HyDE/LLM translation chỉ là retrieval aid; không bao giờ submit text sinh |
| Chunk match chỉ trong cùng `doc_id` | Provenance mapping chunk → doc phải tuyệt đối đúng |
| BGE-M3 tokenizer trong scorer | Chunk-size sweep và local scorer nên tính theo BGE-M3 tokens |
| LCS ≥ 40% reference | Có lý do mạnh để decouple child retrieval chunk và parent submission chunk |
| Near-duplicate LCS/union ≥ 0.8 | Phải dedup/overlap-suppress prediction bằng scorer-aware logic |
| Document F2 và Chunk F2 chấm riêng | Cần hai scoring heads/cutoff policy, không ép `relevant_docs = unique(final chunks)` mặc định |
| F2 nặng recall | Ưu tiên candidate recall trước aggressive selection |
| Macro average theo query | Không được để một nhóm query khó/ngôn ngữ hiếm bị chìm trong micro statistics |
| Query không có positive không vào macro denominator | Local evaluator phải replicate rule này, tránh tính sai score |
| Full 1,200 queries mỗi submission | Builder phải enforce exact coverage + unique IDs |
| ZIP chỉ một JSON ở root | Release packaging phải có automated structural gate |
| Public max 10/day | Mỗi submit phải gắn hypothesis/ablation ID |
| Private max 5/user total | Private phase là release selection, không phải exploration |
| Model public, ≤15B, released before 01/08/2026 | Model registry/compliance manifest bắt buộc |
| External data phải khai báo nguồn | Data registry + license/source documentation ngay từ đầu |
| Working-notes paper bắt buộc | Experiment logging và reproducibility là deliverable, không phải optional cleanup |

---

# A.15. Những hard invariants nên encode thẳng vào code

Thay vì ghi checklist bằng tay, submission builder nên fail-fast nếu vi phạm:

```python
assert len(predictions) == 1200
assert len({x["id"] for x in predictions}) == 1200
assert set(prediction_ids) == set(query_ids)

for row in predictions:
    assert isinstance(row["relevant_docs"], list)
    assert isinstance(row["relevant_chunks"], list)

    for doc_id in row["relevant_docs"]:
        assert doc_id in official_doc_ids

    for rank, chunk in enumerate(row["relevant_chunks"]):
        assert chunk["doc_id"] in official_doc_ids
        assert isinstance(chunk["chunk_text"], str)
        assert chunk["chunk_text"].strip()
        assert source_verifier.exists_verbatim_or_normalized_source_span(
            chunk["doc_id"], chunk["chunk_text"]
        )
        if "chunk_order" in chunk:
            assert isinstance(chunk["chunk_order"], int)
            assert chunk["chunk_order"] >= 0
```

Archive validator:

```text
[ ] exactly one member in ZIP
[ ] member is JSON
[ ] JSON is at archive root
[ ] JSON parses
[ ] 1,200 query IDs exactly
[ ] no duplicate IDs
[ ] no unknown doc IDs
[ ] every emitted chunk binds to its source document
[ ] every chunk_text is source-derived
[ ] optional chunk_order values valid
```

Các check này nên chạy tự động trước **mọi** leaderboard upload.

---

# A.16. Cách đọc phần còn lại của tài liệu

Từ đây trở đi:

- **Stage 1** được dùng để học retrieval, F2 optimization, candidate recall, reranking, query expansion và hard negatives.
- **Stage 2** được dùng để học provenance, manifests, replay, release gates, regression safety và rollback.
- Các kiến trúc được đề xuất cho Stage 3 đều phải nằm trong submission/data/model constraints ở Chương A.

Một số kỹ thuật như HyDE, translated queries hoặc synthetic medical passages xuất hiện ở các chương sau **chỉ là representation dùng nội bộ để search/rerank**; output cuối vẫn phải là `doc_id` chính thức và `chunk_text` nguyên văn từ source.

---


# 0. Executive summary

Nếu phải chốt ngay một câu:

> **Stage 3 nên dùng “retrieval brain” của Stage 1 + “engineering discipline” của Stage 2.**

Pipeline đề xuất ở mức cao:

```text
VI biomedical query
        │
        ├── deterministic normalization + biomedical entity parsing
        │
        ├── original Vietnamese query                     [always kept]
        ├── conservative English translation/query form  [optional branch]
        ├── conservative Chinese translation/query form  [optional branch]
        └── 0–2 safe subqueries / optional HyDE          [additive only]
                       │
                       ▼
              MULTI-CHANNEL RETRIEVAL
                       │
      ┌────────────────┼────────────────┐
      │                │                │
   BM25/lexical     multilingual     optional specialist
   per language     dense retrieval  biomedical dense branch
      │                │                │
      └────────────────┼────────────────┘
                       ▼
                  weighted RRF
                       ▼
          high-recall document candidates
                       ▼
              document reranker
                       ▼
             selected document pool
                       ▼
       child-passage retrieval inside docs
                       ▼
       multilingual cross-encoder reranker
                       ▼
         child anchor → parent expansion
                       ▼
     scorer-aware near-duplicate suppression
                       ▼
        independent doc/chunk F2 cutoffs
                       ▼
       doc_ids + exact source chunk_text
```

Ba nguyên tắc quan trọng nhất:

1. **Không để gold biến mất khỏi candidate pool.** Với F2, recall là tài sản chính. Không reranker/LLM nào cứu được gold chưa từng được retrieve.
2. **Retrieve bằng đoạn nhỏ, submit bằng đoạn cha lớn hơn.** Metric chunk dùng LCS bất đối xứng so với reference chunk; parent-child retrieval có khả năng cân bằng semantic precision và scorer coverage tốt hơn chunk cố định duy nhất.
3. **Mỗi thử nghiệm phải phân biệt lỗi ở corpus, document retrieval, chunk retrieval, reranking hay cutoff.** Không tối ưu một con số leaderboard tổng hợp mà không biết bottleneck nằm đâu.

Từ hai stage trước, các pattern đáng giữ nhất là:

- **Hung&Fong / Stage 1:** kỷ luật mổ Precision–Recall–F2, candidate-recall-first, query normalization đối xứng, bài học thất bại của deep decomposition và aggressive LLM filtering.
- **Nguyễn Văn Nghiêm / Stage 1:** HyDE, multi-source candidate generation, hard-negative reranker fine-tuning, đo rank của gold trong candidate pool.
- **TQD / Stage 1:** retrieval ở granularity nhỏ + aggregation lên đơn vị lớn; sliding-window MaxP cho tài liệu dài.
- **FAI / Stage 1:** original query là anchor, subquery chỉ bổ sung; weighted RRF; hierarchical parsing.
- **mscAI / Stage 1:** manifest-aware index lifecycle, cache, stable RRF implementation.
- **Agentic Builders / Stage 1:** canonical corpus, deterministic IDs, build flow rõ ràng.
- **Trần Thanh Tú / Stage 1:** baseline đơn giản nhưng mạnh, sweep cutoff offline và đo tác động chiều dài reranker.
- **LASTDANCE / Stage 2:** provenance/replay mindset.
- **ARCANE / Stage 2:** tách stage rõ ràng, score-manifest, hash validation, không destructive-prune candidate trước scoring.
- **KINGPRO + SYNERA / Stage 2:** release gate, regression suite, rollback, source audit, artifact signatures.
- **AISOLO / Stage 2:** không thần thánh hóa dense retrieval; nhánh nào không tăng metric thì loại dù nghe “AI” hơn.

---

# 1. Diễn giải kỹ thuật: Stage 3 thực chất là bài gì?

Stage 3 là **multilingual biomedical information retrieval**, không phải QA generation.

Input:

```text
q_i: truy vấn y khoa bằng tiếng Việt
```

Corpus:

```text
D = tài liệu y sinh VI + EN + ZH
```

Output được đánh giá ở hai lớp:

```text
Document retrieval
Chunk retrieval
```

Điểm cuối:

```text
Final = (Document_F2_macro + Chunk_F2_macro) / 2
```

Điều này khác Stage 1 ở ba điểm lớn:

> **Hard constraint cần nhớ xuyên suốt:** final `chunk_text` phải là source-derived text từ document tương ứng. Translation, HyDE, rewrite và synthetic passage chỉ được dùng làm retrieval representation nội bộ.

1. **Cross-lingual là yêu cầu cốt lõi.** Query luôn tiếng Việt nhưng evidence có thể tiếng Anh/Trung.
2. **Chunk text được scorer so khớp gần đúng bằng BGE-M3 tokenization + LCS.** Vì vậy chunk construction không chỉ là engineering choice mà trở thành một phần trực tiếp của optimization target.
3. **Không cần answer generation.** Mọi compute dành cho LLM answer ở Stage 1/2 đều có thể tái phân bổ cho retrieval, reranking và data quality.

Stage 2 gần Stage 3 ít hơn về task, nhưng cực hữu ích về cách tổ chức hệ thống có thể kiểm chứng và tái lập.

---

# 2. Snapshot repository được đọc lại

Tài liệu này dựa trên hai snapshot sau:

## Stage 1

- Repository: `https://github.com/aigurutinix/r2ai-stage-1`
- Default branch: `main`
- Snapshot commit đã đọc: `7aa3955569371c56016744e1dd758b0a656c0c89`
- Commit date: `2026-07-14`
- Root README xác nhận thứ hạng:
  - mscAI — giải nhất
  - Hung&Fong — giải nhì
  - Nguyễn Văn Nghiêm — giải ba
  - các đội khuyến khích gồm TQD, FAI Team, Agentic Builders, NextGen, BeeIT, Trần Thanh Tú, Thanh Khâu Sơn.

## Stage 2

- Repository: `https://github.com/aigurutinix/r2ai-stage-2`
- Default branch: `master`
- Snapshot commit đã đọc: `2bbc3a43cb29bfaf8aa53da15be5699b4af1b8d2`
- Commit date: `2026-09-17`
- Root README xác nhận thứ hạng:
  - LASTDANCE — giải nhất
  - ARCANE — giải nhì
  - KINGPRO — giải ba
  - SYNERA, VILAMIU, Nguyễn Vũ Hoàng Long, OVERFITTING, IDIOT, AISOLO — khuyến khích.

> **Cảnh báo về số liệu trong repo:** các con số F2/Precision/Recall/Execution xuất hiện trong README hoặc experiment log của từng đội có thể là public-score snapshot, internal experiment hoặc một phiên bản sau cuộc thi. Chúng được dùng ở đây để rút kinh nghiệm kỹ thuật, không mặc định coi là điểm chung cuộc chính thức trừ khi root README/BTC nói rõ.

---

# 3. Phân tích sâu Stage 1 — phần nào thật sự chuyển được sang Stage 3

# 3.1. mscAI — kiến trúc retrieval và index lifecycle sạch

Các file quan trọng đã đọc lại:

```text
mscai/README.md
mscai/src/backend/config.yaml
mscai/src/backend/src/services/vector_store/hybrid.py
mscai/src/backend/src/services/vector_store/index_builder.py
mscai/src/backend/src/services/agents/legal_assistant/agent.py
mscai/src/backend/src/services/agents/legal_assistant/node.py
```

## Điểm hay 1 — RRF dùng rank, không cố trộn raw score khác hệ

`HybridLegalStore` thực hiện weighted reciprocal rank fusion:

```text
score(doc) += weight / (rrf_k + rank)
```

Vì BM25 score và dense cosine/embedding score không cùng thang đo, rank fusion giúp tránh bài toán calibration raw score khó kiểm soát.

Điều này cực hợp Stage 3 vì ta có thể có nhiều retrieval channel:

```text
BM25_VI
BM25_EN
BM25_ZH
multilingual_dense_1
multilingual_dense_2
HyDE_dense
translated_query_dense
biomedical_specialist_dense
```

Không cần ép mọi score về cùng một scale ngay từ đầu.

## Điểm hay 2 — manifest ngăn trộn index sai model/corpus

`index_builder.py` giữ manifest gồm:

```text
source database identity
embedding endpoint/model
vector text format
search-space sizes
total record count
```

BM25 cũng có cache/manifest riêng. Khi cấu hình hoặc corpus thay đổi, index được rebuild; khi không đổi, cache được tái sử dụng.

Đây là thứ Stage 3 **bắt buộc phải có** vì ta sẽ tạo nhiều artifact:

```text
raw documents
normalized documents
child chunks
parent chunks
doc embeddings
chunk embeddings
BM25 indexes
reranker scores
candidate cache
```

Một lỗi cực nguy hiểm là:

```text
chunks_v4.jsonl
+
embeddings_v3.npy
```

vẫn load thành công nhưng row mapping sai hoàn toàn. Manifest/hash phải chặn lỗi này.

## Điểm không nên bê nguyên

Snapshot `config.yaml` của mscAI hiện tại bật HyDE, tắt rewrite, tắt reranker, dùng Chroma top-7 + LLM filter. Đây chỉ là **một snapshot cấu hình**, không nên coi là “winning recipe”.

Stage 3 cần top-K rộng hơn rất nhiều để đo candidate recall trước khi cắt.

---

# 3.2. Hung&Fong — nguồn bài học quan trọng nhất cho F2

Các file đặc biệt đáng đọc:

```text
hung&phong/src/docs/POST_SUBMISSION_REVIEW.md
hung&phong/src/backend/rag.py
hung&phong/src/backend/query_analyzer.py
hung&phong/src/backend/textnorm.py
hung&phong/src/docs/kien_truc.drawio
```

Đây là repo đáng tham khảo nhất về **cách debug retrieval benchmark**, không phải chỉ về model.

## 3.2.1. Bài học số 1 — đừng tối ưu nhầm tầng

Experiment log của họ ghi rõ một giai đoạn:

```text
recall kẹt khoảng 0.52–0.57
nhưng team lại tập trung judge/filter để tăng precision
```

Khi gold không vào candidate pool, filter giỏi đến đâu cũng không cứu được.

Công thức tổng quát cho Stage 3:

```text
Recall_final <= Recall_reranker_input <= Recall_candidate_pool <= Recall_corpus
```

Ta nên đo riêng:

```text
Corpus coverage
Doc candidate Recall@20/50/100/200
Chunk candidate Recall@20/50/100/200
Reranked Recall@K
Final F2
```

Nếu `candidate Recall@100` đã thấp, **không được** phí thời gian tune cutoff hoặc LLM selector.

## 3.2.2. Data normalization có thể tăng cả Precision lẫn Recall

Ở v14, log ghi một lỗi chính tả phổ biến trong corpus `khỏan → khoản` được normalize đối xứng ở corpus và query. Kết quả public snapshot của team:

```text
Articles Recall: 0.6320 → 0.7153
Articles Precision: 0.2683 → 0.2883
Articles F2: 0.4657 → 0.5170
```

Điểm quan trọng không nằm ở từ `khỏan`, mà ở nguyên tắc:

> **Normalization phải symmetric và domain-aware.**

Stage 3 có rất nhiều biến thể tương tự:

```text
COVID–19 / COVID-19 / Covid 19
SARS‑CoV‑2 / SARS-CoV-2
TNF‑α / TNF-alpha / TNFα
β blocker / beta-blocker
HbA1c / Hb A1c
H. pylori / Helicobacter pylori
non–small-cell / non-small cell
```

Nếu normalize corpus mà không normalize query, hoặc ngược lại, lexical retrieval có thể mất recall.

## 3.2.3. F2 ghét aggressive filtering

Experiment LLM judge của Hung&Fong:

```text
v15: P 0.298, R 0.7153, F2 0.5255
v16: P 0.440, R 0.5937, F2 0.5285
```

Precision tăng cực mạnh nhưng F2 gần như đứng yên vì recall giảm.

Sau đó `keep-top-2` giúp phục hồi recall và F2 tăng mạnh hơn.

Kết luận Stage 3:

> Không dùng binary LLM judge để “lọc sạch” top candidates trừ khi đã có recall floor.

Nếu dùng learned selector, nên thiết kế:

```text
always keep top-N reranker candidates
+
selector additions/removals only outside recall floor
```

## 3.2.4. Deep decomposition có thể phá cả P và R

v22 deep decomposition so với v20:

```text
v20: F2 0.5985, P 0.4633, R 0.6903
v22: F2 0.5048, P 0.3363, R 0.6480
```

Audit cho thấy các failure mode:

- subquery mất anchor gốc;
- thay thuật ngữ;
- thêm giả định không có trong query;
- biến câu hỏi thành assertion;
- pool phình quá lớn;
- candidate gold bị đẩy xuống khi merge/cut.

Stage 3 y sinh còn nguy hiểm hơn vì query có thể chứa nhiều điều kiện lâm sàng.

Ví dụ:

```text
"Ở bệnh nhân đái tháo đường type 2 có CKD, SGLT2 inhibitor ảnh hưởng thế nào
đến tiến triển suy thận và biến cố tim mạch?"
```

Nếu subquery 1 bỏ `CKD`, hoặc subquery 2 bỏ `type 2 diabetes`, retrieval có thể drift sang quần thể khác.

Rule đề xuất:

```text
ORIGINAL QUERY ALWAYS PRESENT
subquery = additive retrieval leg
never replace original query
n_subqueries <= 2 by default
preserve entities + population + intervention + outcome constraints
```

## 3.2.5. Retrieve rộng trên cùng một retriever không nhất thiết tăng recall

Hung&Fong v25 mở rộng retrieval nhưng recall gần như không tăng trong khi precision sập.

Thông điệp cực quan trọng:

> Khi cùng một retriever đã đạt trần, tăng K chỉ thêm các item cùng distribution; muốn tăng recall thật phải thêm **retrieval diversity**.

Stage 3 vì vậy nên ưu tiên:

```text
BM25 + multilingual dense + translated lexical + second embedding family
```

hơn là chỉ:

```text
same dense top-50 → top-500
```

---

# 3.3. Nguyễn Văn Nghiêm — candidate diversity + hard negatives + reranker training

Các file đã đọc lại:

```text
nguyenvannghiem/README.md
nguyenvannghiem/src/docs/reproduce.md
nguyenvannghiem/src/docs/model_description.md
nguyenvannghiem/src/code/hyde_retrieval.py
nguyenvannghiem/src/code/rerank_intersection.py
nguyenvannghiem/src/code/train_reranker_v2.py
```

## 3.3.1. HyDE dùng đúng chỗ

Pipeline sinh hypothetical legal article rồi dùng nó để dense retrieve.

Stage 3 có query–document style gap còn lớn hơn:

```text
Vietnamese patient-style question
        vs
English/Chinese biomedical abstract language
```

Ví dụ query:

```text
"Thuốc này có dùng được cho người suy gan không?"
```

Hypothetical biomedical text có thể trở thành:

```text
"In patients with hepatic impairment, pharmacokinetic exposure ..."
```

Embedding của hypothetical passage có thể gần style corpus hơn embedding của câu hỏi đời thường.

Tuy nhiên HyDE nên là **một leg bổ sung**, không phải nguồn duy nhất.

## 3.3.2. Hard-negative mining là hướng fine-tune đáng giá nhất

`train_reranker_v2.py` rất đáng học:

- positive = cited article;
- hard negatives lấy từ dense candidates;
- group 1 positive + 7 hard negatives;
- train cross-encoder;
- held-out query split;
- đánh giá NDCG@10/MRR/MAP;
- checkpoint tốt được chọn qua reranking metric, không qua training loss.

Đây là một template gần như hoàn hảo để chuyển sang Stage 3.

Hard negatives Stage 3 nên đặc biệt có:

```text
same disease, wrong intervention
same drug, wrong population
same outcome, wrong disease
same disease + drug, wrong dosage/context
same biomedical term but review/background instead of evidence span
same document, wrong section/chunk
cross-language semantic neighbor but not answer-bearing
```

Các hard negative “gần về y khoa nhưng sai điều kiện” sẽ có giá trị hơn random negatives rất nhiều.

## 3.3.3. Intersection không nên được bê nguyên như hard gate

Nguyễn Văn Nghiêm dùng intersection giữa HyDE candidate và BM25S candidate trước rerank. Trong legal task của họ, điều này có thể tạo candidate pool sạch.

Nhưng Stage 3 cross-lingual có tình huống:

```text
English gold chunk retrieved by multilingual dense
nhưng BM25 Vietnamese không thể match
```

Nếu hard intersection:

```text
Dense ∩ BM25
```

thì gold bị loại.

Stage 3 nên đổi thành:

```text
UNION candidate pool
+
feature/bonus nếu item xuất hiện ở nhiều retrieval legs
```

Ví dụ:

```text
rrf_score
agreement_count
best_dense_rank
best_sparse_rank
language_match flags
```

Intersection chỉ dùng làm high-confidence signal, không dùng làm recall gate.

## 3.3.4. Cảnh báo về transductive/pseudo-label classifier

Repo có classifier Qwen3-1.7B + QLoRA, train trên candidate distribution cụ thể và pseudo-label từ một submission khác. Đây là ý tưởng thú vị cho benchmark engineering nhưng không phải bằng chứng generalization tốt.

Stage 3 không nên vội train selector theo public-test pseudo-label nếu không có protocol chặt. Ưu tiên supervised train/dev gold chính thức hoặc synthetic data tách biệt.

---

# 3.4. TQD — granularity nhỏ để retrieve, granularity lớn để output

README TQD mô tả:

```text
BM25 article-level
Dense clause-level
→ top clauses
→ MaxRank lên article
→ RRF
→ sliding-window reranker MaxP
```

Đây có lẽ là pattern **quan trọng nhất về chunking** cho Stage 3.

## Vì sao?

Một biomedical paper dài có thể có:

```text
Abstract
Introduction
Methods
Results
Adverse events
Subgroup analysis
Discussion
Supplementary notes
```

Nếu embed nguyên section dài:

- signal cần tìm bị loãng;
- model truncate;
- query chỉ liên quan 2–3 câu trong section.

Nếu chunk quá nhỏ:

- semantic retrieval tốt;
- nhưng Stage 3 scorer có thể không đủ LCS coverage so reference chunk.

Giải pháp:

```text
child chunk = nhỏ, dùng retrieval/rerank
parent chunk = lớn hơn, dùng submission
```

Chi tiết sẽ được thiết kế ở phần Stage 3 architecture.

---

# 3.5. FAI Team — subquery bổ sung và hierarchical pipeline

`PIPELINE_OVERVIEW_29-06.md` cho thấy:

```text
subquery planning
→ dense top30 + BM25 top30
→ weighted RRF dense .4 / BM25 .6
→ multi-query RRF
→ cross-encoder rerank
→ top8
```

Rerank có thể kết hợp:

```text
0.7 * score(original_query, chunk)
+
0.3 * mean(score(subqueries, chunk))
```

Đây là cách xử lý tốt hơn việc thay original query bằng subquery.

Stage 3 có thể áp dụng tương tự:

```text
primary score = original VI query vs candidate
auxiliary score = EN/ZH translation/subqueries vs candidate
final score = strong primary + weaker auxiliary
```

Lưu ý: nếu reranker hỗ trợ multilingual tốt, original Vietnamese query vẫn nên là điểm neo chính.

---

# 3.6. Agentic Builders — canonical corpus quan trọng hơn “AI trick”

Data documentation của Agentic Builders ghi rõ build flow:

```text
raw documents
→ web corpus
→ dedup
→ merge
→ canonicalize
→ BM25 bundle
→ dense embeddings
```

Final legal corpus của họ có 71,050 canonical rows trong snapshot tài liệu.

Stage 3 cần học cách tư duy này ở cấp document/chunk:

```text
raw source
→ parsed document
→ normalized document
→ canonical document ID
→ deduplicated paragraphs
→ child chunks
→ parent chunks
→ indexes
```

Biomedical corpora thường có duplicate mạnh:

- abstract xuất hiện ở PubMed và publisher page;
- preprint vs journal version;
- HTML vs XML vs PDF copies;
- Chinese/English mirror pages;
- repeated headers/footers/reference lists.

Nếu không dedup, top-K có thể bị 5 bản gần giống nhau chiếm chỗ, làm mất recall diversity.

---

# 3.7. Trần Thanh Tú — baseline đơn giản, cutoff rẻ, context length đáng đo

Pipeline official snapshot:

```text
Dense + BM25
→ weighted RRF
→ CAND=80
→ cross-encoder rerank
→ save top20 + scores
→ offline cutoff sweep
```

Hành trình điểm trong README cho thấy riêng việc tăng reranker max sequence từ 256 lên 512 giúp F2 tăng đáng kể trong pipeline của họ.

Thông điệp Stage 3:

> **Reranker truncation length là hyperparameter retrieval, không phải implementation detail.**

Biomedical chunks có thể dài. Ta cần ablation:

```text
max_length = 256 / 512 / 1024 / sliding-MaxP
```

và đo:

```text
chunk candidate recall
reranked recall@K
F2
latency
VRAM
```

Điểm hay khác là lưu `top-N + score` rồi sweep cutoff offline. Stage 3 nên làm y hệt để tiết kiệm GPU.

---

# 3.8. BeeIT và NextGen — có ý tưởng hay nhưng không phải lõi Stage 3

## BeeIT

BeeIT ép LLM dùng tool `submit_sub_queries`, sau đó có tool-following loop để tìm văn bản được reference.

Điểm nên giữ:

- structured output/tool schema tốt hơn parse JSON lỏng;
- model error có observation rõ để self-repair;
- checkpoint/resume.

Điểm **không cần** cho Stage 3 baseline:

- multi-round agent loop;
- answer generation;
- referenced-document tool chasing.

Pure retrieval benchmark không cần thêm latency/failure surface này ngay từ đầu.

## NextGen

NextGen lấy citation xuất hiện trong answer rồi intersect với reranker pool.

Đây là grounding pattern hay cho QA, nhưng Stage 3 không sinh answer nên bỏ.

---

# 3.9. Thanh Khâu Sơn — bài học về scale

Pipeline dùng Elasticsearch + Qdrant trên corpus rất lớn. Bài học cho Stage 3:

- khi corpus lên hàng triệu chunk, brute-force numpy cosine không còn phù hợp;
- tách sparse engine và vector engine là hợp lý;
- deterministic `chunk_id` và pre-tokenization giúp build lại được.

Tuy nhiên không nên bắt đầu Stage 3 bằng hệ thống distributed nặng nếu corpus chưa đủ lớn để cần. Benchmark first, scale second.

---

# 4. Phân tích sâu Stage 2 — lấy engineering, không lấy Text-to-Pandas

Stage 2 khác task nhưng có engineering discipline cực đáng giá.

# 4.1. LASTDANCE — evidence/provenance/replay mindset

README mô tả:

```text
warehouse SQLite + FTS5
→ document/table/row retrieval
→ FinancialPlan
→ evidence CSV
→ pandas query
→ sandbox execution
→ schema + replay + provenance validation
```

Release được kiểm tra 1,012/1,012 query replay và toàn bộ evidence có provenance binding.

Stage 3 không có Pandas, nhưng có thể chuyển nguyên triết lý:

```text
prediction chunk
→ must point to exact source document
→ exact source offsets
→ exact raw/normalized text mapping
→ stable chunk id
```

Mỗi chunk output nên truy vết được:

```json
{
  "chunk_id": "pmid_12345::p_007::w_002",
  "doc_id": "pmid_12345",
  "language": "en",
  "section": "Results",
  "raw_start": 8451,
  "raw_end": 9340,
  "normalized_start": 8120,
  "normalized_end": 8990,
  "source_sha256": "...",
  "chunk_text": "..."
}
```

Nếu prediction không map ngược về source, pipeline phải coi đó là bug.

---

# 4.2. ARCANE — stage boundaries, manifests, no destructive prune

Đây là Stage 2 repo có nhiều pattern transferable nhất.

Architecture document ghi:

```text
document gate
→ BM25 + dense + graph expansion
→ question-level candidate union
→ reranker scores every candidate
→ adaptive selection
→ source materialization
→ validation
```

Một nguyên tắc rất đáng lấy:

> **Retrieval có thể widen candidate pool; destructive pruning chỉ làm sau scoring.**

Cho Stage 3:

```text
BAD:
translation filter → drop docs → dense search

GOOD:
retrieve multiple diverse legs → union → score/rerank → then prune
```

## Score manifest

`validate_rerank_scores.py` kiểm:

- SHA-256 của candidate pair file;
- SHA-256 score file;
- scorer configuration invariant;
- mỗi candidate phải có đúng một score;
- không duplicate question/candidate;
- score range hợp lệ.

Stage 3 nên copy gần nguyên concept:

```text
candidates.jsonl
candidates.sha256
reranker_scores.part-000.jsonl
reranker_scores.part-000.manifest.json
```

Manifest nên chứa:

```text
corpus_manifest_sha256
chunk_manifest_sha256
query_file_sha256
embed_model_revision
reranker_model_revision
prompt/template revision if any
tokenizer revision
max_length
retrieval config hash
candidate file hash
score file hash
```

Điều này đặc biệt quan trọng khi chạy GPU job thành nhiều shard.

---

# 4.3. KINGPRO — release engineering và rollback

README KINGPRO cho thấy họ duy trì rất nhiều version, rollback artifact, SHA-256, source audit và regression test.

Bài học Stage 3:

```text
never overwrite best run
```

Nên có:

```text
runs/
  20261003_baseline_bge_m3/
  20261007_parent512/
  20261012_qwen_reranker/
  ...
```

Mỗi run chứa:

```text
config.yaml
manifest.json
metrics.json
predictions.jsonl
candidate_stats.json
error_buckets.json
logs/
```

Và alias:

```text
BEST_PUBLIC -> immutable run ID
BEST_LOCAL  -> immutable run ID
```

Không có chuyện chạy experiment mới rồi ghi đè `best_predictions.json`.

---

# 4.4. SYNERA / IDIOT — error taxonomy + promotion gate

Architecture của họ tách lỗi thành nhiều tầng:

```text
retrieval
table
row
column
unit
period
scope
formula
```

Stage 3 nên có taxonomy tương đương:

```text
CORPUS_MISSING
DOC_RETRIEVAL_MISS
DOC_RERANK_DROP
CHUNK_RETRIEVAL_MISS
CHUNK_RERANK_DROP
CUTOFF_DROP
LANGUAGE_ALIGNMENT
QUERY_DECOMPOSITION_DRIFT
ENTITY_NORMALIZATION
CHUNK_BOUNDARY
DUPLICATE_OCCUPANCY
SOURCE_PARSE_ERROR
```

Không promote một thay đổi chỉ vì score tổng tăng 0.003 nếu nó tạo regression nặng ở một language/domain bucket mà public test chưa phản ánh đủ.

---

# 4.5. AISOLO — “dense không mặc định tốt hơn BM25”

AISOLO ghi lại một experiment Stage 2 nơi dense embedding làm table F2 giảm so với BM25.

Không được suy ra rằng Stage 3 không cần dense — Stage 3 **bắt buộc có semantic cross-lingual retrieval** — nhưng lesson quan trọng là:

> Mọi retrieval leg phải chứng minh contribution qua ablation.

Nếu một dense model:

```text
adds latency + memory
but contributes 0 unique gold candidates
```

thì bỏ.

Ta sẽ đo `unique gold contribution` cho từng leg:

```text
Gold found by BM25 only
Gold found by dense-A only
Gold found by dense-B only
Gold found by translated BM25 only
Gold found by HyDE only
```

Đây là cách đánh giá ensemble đúng hơn chỉ nhìn individual Recall@K.

---

# 5. Bảng quyết định: lấy gì từ Stage 1 và Stage 2

| Pattern | Nguồn | Stage 3 | Quyết định |
|---|---|---:|---|
| BM25 + dense hybrid | Stage 1 | Rất phù hợp | **Giữ** |
| Weighted RRF | Stage 1 | Rất phù hợp | **Giữ** |
| Cross-encoder rerank | Stage 1 | Rất phù hợp | **Giữ** |
| Candidate recall diagnostics | Hung&Fong | Cốt lõi | **Giữ bắt buộc** |
| Symmetric normalization | Hung&Fong | Cốt lõi | **Giữ** |
| HyDE | Nguyễn Văn Nghiêm | Có tiềm năng | **Ablate** |
| Hard-negative reranker FT | Nguyễn Văn Nghiêm | Rất phù hợp | **Ưu tiên phase 2** |
| Hard intersection BM25∩dense | Nguyễn Văn Nghiêm | Nguy hiểm cross-lingual | **Không dùng làm gate** |
| Clause/small-unit retrieval | TQD | Rất phù hợp | **Giữ** |
| Sliding-window MaxP | TQD | Rất phù hợp | **Giữ/Ablate** |
| Original-query anchored subqueries | FAI | Rất phù hợp | **Giữ** |
| Deep decomposition replacement | Hung&Fong v22 | Rủi ro cao | **Loại** |
| LLM judge aggressive filter | Hung&Fong v16/v31 | F2-hostile | **Không dùng mặc định** |
| Canonical corpus + stable IDs | Agentic Builders | Cốt lõi | **Giữ bắt buộc** |
| Offline cutoff sweep | Trần Thanh Tú | Cực hữu ích | **Giữ** |
| Index manifest/cache | mscAI | Cốt lõi | **Giữ** |
| Evidence provenance | LASTDANCE | Cốt lõi | **Giữ** |
| Hash score manifests | ARCANE | Cốt lõi | **Giữ** |
| No destructive pruning before scoring | ARCANE | Cốt lõi | **Giữ** |
| Rollback + immutable runs | KINGPRO | Cốt lõi | **Giữ** |
| Release regression gate | SYNERA | Cốt lõi | **Giữ** |
| Text-to-Pandas / AST sandbox | Stage 2 | Không liên quan | **Loại** |
| Answer LLM generation | Stage 1/2 | Không chấm | **Loại** |
| Agentic multi-round tool loop | BeeIT | Overkill | **Chưa dùng** |
| Legal version/effect graph | Stage 1 | Domain-specific | **Chỉ mượn ý dedup/version nếu corpus cần** |

---

# 6. Phân tích metric Stage 3 — đây là phần phải thiết kế pipeline theo scorer

# 6.1. Document F2

Với mỗi query:

```text
P_doc = |D ∩ G| / |D|
R_doc = |D ∩ G| / |G|
F2_doc = 5 * P_doc * R_doc / (4*P_doc + R_doc)
```

F2 ưu tiên recall hơn precision.

Ví dụ:

```text
System A: P=0.80, R=0.50 → F2 ≈ 0.54
System B: P=0.60, R=0.75 → F2 ≈ 0.71
```

Vì vậy Stage 3 document stage không nên quá keo top-K.

Nhưng document F2 vẫn phạt noise, nên giải pháp là:

```text
high-recall candidate generation
→ strong reranker
→ adaptive cutoff
```

không phải trả vô hạn docs.

---

# 6.2. Chunk metric bất đối xứng — hệ quả rất lớn

Theo mô tả Stage 3:

1. Reference chunk chỉ được xét nếu cùng `doc_id` với prediction.
2. Text được normalize.
3. Tokenizer = BGE-M3 tokenizer.
4. Prediction được tính relevant nếu:

```text
LCS(pred_tokens, ref_tokens) >= 0.4 * len(ref_tokens)
```

Điểm đáng chú ý là denominator là **reference length**, không phải predicted length.

Giả sử reference dài 200 token.

Prediction A dài 80 token nhưng đúng hoàn toàn:

```text
LCS = 80
80 / 200 = 0.40 → pass vừa đủ
```

Prediction B dài 400 token và chứa nguyên reference:

```text
LCS = 200
200 / 200 = 1.00 → pass
```

Vì vậy chunk output dài hơn **không bị phạt trực tiếp theo overlap ratio** nếu nó vẫn là một prediction chunk duy nhất.

Nhưng chunk dài quá gây ba vấn đề:

1. dense embedding bị loãng;
2. reranker truncate;
3. nếu các relevant areas cách xa nhau, một chunk khổng lồ không nhất thiết cover được mọi information unit theo cách scorer tính recall.

Do đó giải pháp tối ưu là **decouple retrieval chunk và output chunk**.

---

# 6.3. Parent-child retrieval là thiết kế gần như bắt buộc

Đề xuất hai representation:

## Child chunk

Dùng để retrieve/rerank.

Target ban đầu:

```text
~120–220 BGE-M3 tokens
sentence/paragraph aware
overlap nhỏ hoặc dynamic
```

## Parent chunk

Dùng để submit.

Target ban đầu:

```text
~350–650 BGE-M3 tokens
section-aware
centered around child anchor
```

Flow:

```text
query
→ retrieve child
→ rerank child
→ map child to parent context
→ submit exact parent text
```

Lợi ích:

- child vector tập trung semantic signal;
- parent có xác suất vượt LCS 40% với reference cao hơn;
- source provenance vẫn rõ;
- có thể sweep parent size mà không re-embed toàn corpus.

---

# 6.4. Near-duplicate merge 0.8 phải được clone trong local scorer

BTC gộp predicted chunks trùng/gần trùng nếu LCS/union ≥ 0.8 trong cùng document.

Do đó pipeline phải tự implement cùng logic để:

- tránh tưởng mình tăng precision bằng việc spam overlapping windows;
- đo local đúng scorer;
- chọn output windows đa dạng hơn.

Quan trọng:

```text
chunk A: sentences 1–5
chunk B: sentences 2–6
chunk C: sentences 3–7
```

có thể gần duplicate. Nếu cả ba đều retrieve, chúng chiếm budget nhưng không tăng coverage thật.

Nên selection cuối dùng MMR/diversity hoặc overlap suppression theo **chính tokenizer/LCS scorer**.

---

# 6.5. Document và chunk cần cutoff riêng

Không nhất thiết:

```text
relevant_docs = unique(doc_id from final chunks)
```

là tối ưu.

Có thể có doc rất đúng nhưng chunk selector chỉ trả một chunk, hoặc chunk confidence thấp nhưng document confidence cao.

Nên maintain hai score:

```text
doc_score
chunk_score
```

và hai cutoff:

```text
doc_keep_policy
chunk_keep_policy
```

Sau đó đảm bảo mọi chunk output có doc_id hợp lệ; nhưng doc output có thể rộng hơn tập docs có final chunks nếu local metric cho thấy có lợi.

---

# 7. Kiến trúc Stage 3 đề xuất — Best-of hai repo

# 7.1. Offline data build

```text
Provided/allowed biomedical sources
            │
            ▼
      raw snapshot layer
            │
            ▼
      parser per source
            │
            ▼
  canonical document schema
            │
            ├── raw text
            ├── normalized text
            ├── language
            ├── title/abstract/sections
            ├── source metadata
            └── checksums
            │
            ▼
     duplicate detection
            │
            ▼
  structure-aware segmentation
            │
            ├── child chunks
            ├── parent chunks
            └── doc representation
            │
            ▼
       index builders
            │
   ┌────────┼────────┐
   │        │        │
 BM25    dense    optional
 VI/EN/ZH global   specialist
```

---

# 7.2. Canonical document schema

Đề xuất JSONL:

```json
{
  "doc_id": "source:123456",
  "source": "pubmed_like_source",
  "source_native_id": "123456",
  "url": "...",
  "language": "en",
  "title_raw": "...",
  "title_norm": "...",
  "abstract_raw": "...",
  "sections": [
    {
      "section_id": "s03",
      "heading_raw": "Results",
      "heading_norm": "results",
      "start": 4210,
      "end": 9180
    }
  ],
  "raw_text": "...",
  "normalized_text": "...",
  "source_sha256": "...",
  "parser_version": "v1.3",
  "created_at": "..."
}
```

Quy tắc:

- `doc_id` stable qua mọi experiment;
- không dùng array index làm identity;
- giữ raw và normalized song song;
- mọi chunk có exact offset mapping về raw source;
- không normalize phá hỏng output text gốc.

---

# 7.3. Biomedical normalization

Normalization nên có hai lớp.

## Safe surface normalization

Áp đối xứng query/corpus cho sparse search:

```text
Unicode NFC/NFKC where safe
HTML entities
non-breaking spaces
hyphen variants
minus/en dash/em dash variants
Greek letters ↔ optional aliases
whitespace collapse
case-folding for lexical index
```

## Alias enrichment, không replace raw text

Tạo searchable aliases:

```text
TNF-α ↔ TNF alpha
IL‑6 ↔ IL6 ↔ interleukin 6
H. pylori ↔ Helicobacter pylori
T2DM ↔ type 2 diabetes mellitus
CKD ↔ chronic kidney disease
```

Đừng overwrite source. Hãy tạo field:

```text
search_text = raw_text + alias_terms
```

hoặc separate sparse fields.

---

# 7.4. Chunking strategy

Đề xuất hierarchy:

```text
Document
  └── Section
       └── Paragraph
            └── Sentence
```

Từ đó build:

## Level A — child retrieval chunk

```text
120–220 tokens
sentence boundary
max 1–2 paragraphs
small overlap
```

## Level B — parent output chunk

```text
350–650 tokens
section constrained
centered on child
```

## Level C — document representation

```text
title + abstract
or title + headings + short summary fields
```

Không embed full 8,000-token paper làm primary doc vector.

---

# 7.5. Document retrieval stage

Mục tiêu là đưa gold docs vào pool với recall rất cao.

Đề xuất channels:

### Channel 1 — Multilingual dense from Vietnamese original query

```text
VI query → multilingual embedding → all-language docs/chunks
```

Đây là channel bắt buộc.

### Channel 2 — BM25 Vietnamese

Đánh trên VI documents và bilingual alias fields.

### Channel 3 — English translated lexical

```text
VI query → conservative EN translation
→ BM25 English corpus
```

Rất hữu ích cho exact biomedical term.

### Channel 4 — Chinese translated lexical

```text
VI query → conservative ZH translation
→ BM25 Chinese corpus
```

### Channel 5 — optional second dense family

Dùng embedding model khác để tạo diversity.

### Channel 6 — optional HyDE

HyDE bằng English biomedical style, chỉ additive.

Merge:

```text
weighted RRF
```

Initial candidate target:

```text
Doc union: 100–300 docs/query
```

Con số phải tune theo corpus scale.

---

# 7.6. Query translation: dùng để mở sparse branch, không làm single source of truth

Translation có thể mất điều kiện y khoa.

Ví dụ:

```text
"không dung nạp metformin"
```

không được translate thành chỉ:

```text
"metformin treatment"
```

Nên output structured translation:

```json
{
  "original_vi": "...",
  "entities": ["metformin", "CKD"],
  "constraints": ["intolerance", "adult"],
  "query_en": "...",
  "query_zh": "..."
}
```

Validator deterministic check entity preservation trước khi dùng translated branch.

Nếu translation không preserve anchor, bỏ branch đó thay vì dùng output lỗi.

---

# 7.7. Query decomposition theo PICO-lite

Không cần full clinical PICO cho mọi câu; dùng PICO-lite:

```text
Population
Condition
Intervention/Exposure
Comparator
Outcome
Time/Context
```

Ví dụ:

```text
Question:
Ở bệnh nhân T2DM kèm CKD, SGLT2 inhibitor ảnh hưởng thế nào đến
suy giảm chức năng thận và biến cố tim mạch?
```

Parse:

```json
{
  "population": ["type 2 diabetes", "chronic kidney disease"],
  "intervention": ["SGLT2 inhibitor"],
  "outcomes": ["kidney disease progression", "cardiovascular events"]
}
```

Subquery tối đa 2:

```text
SGLT2 inhibitor + T2DM + CKD + kidney progression
SGLT2 inhibitor + T2DM + CKD + cardiovascular outcomes
```

Original query vẫn được retrieve riêng.

---

# 7.8. Document reranking

Input:

```text
(query, doc representation)
```

Doc representation không nên chỉ là full abstract nếu paper dài; có thể dùng:

```text
title + abstract
+
retrieval-hit snippets
```

Một trick tốt:

```text
for each doc:
  take top child hits from first-stage retrieval
  build doc evidence summary = title + top 2 child passages
  rerank query vs evidence summary
```

Như vậy doc reranker nhìn đúng phần doc liên quan thay vì abstract chung chung.

---

# 7.9. Chunk retrieval trong top documents

Sau document pool:

```text
for doc in selected_docs:
    retrieve child chunks using:
        dense score
        BM25 local score
        first-stage global rank
```

Lợi ích hierarchical retrieval:

- chunk search space nhỏ;
- tránh global top-K bị vài document dài dominate;
- dễ đảm bảo mỗi promising doc có một số chunk candidate.

Có thể reserve quota:

```text
at least M child candidates per top document
```

để tăng document diversity.

---

# 7.10. Chunk reranking

Strong multilingual cross-encoder:

```text
(original VI query, child chunk)
```

Sau đó auxiliary scoring:

```text
(query_en, child_en)
(query_zh, child_zh)
subqueries
```

Không nhất thiết translate candidate text sang VI; nếu reranker multilingual tốt, score cross-lingual trực tiếp.

Combined score ban đầu:

```text
score = 0.70 * original_query_score
      + 0.20 * best_translation_score
      + 0.10 * best_subquery_score
```

Đây chỉ là starting hypothesis; cần ablation.

---

# 7.11. Sliding-window MaxP cho chunk dài

Nếu parent/section candidate dài hơn reranker max length:

```text
split candidate into windows
score each window
candidate_score = max(window_scores)
```

TQD đã dùng đúng pattern này.

Stage 3 nên benchmark:

```text
plain truncation
vs
head+tail
vs
sliding MaxP
```

Biomedical evidence thường nằm ở cuối Results/Discussion; head-only truncate dễ miss.

---

# 7.12. Child → parent expansion trước submission

Sau khi rerank child:

```text
child = exact sentence/paragraph region
```

Expand:

```text
parent = surrounding sentences/paragraphs within same section
```

Không crossing section boundary trừ khi section cực ngắn.

Các chiến lược sweep:

```text
P256: ~256 BGE tokens
P384
P512
P640
P768
```

Có thể tạo parent “centered window” bằng token offsets nhưng snap về sentence boundaries.

---

# 7.13. Near-duplicate suppression theo scorer

Trước output:

```python
if same_doc and lcs_union(chunk_a, chunk_b) >= 0.8:
    keep higher-scored chunk only
```

Nhưng cần careful với hai child anchors khác nhau map về parent rất overlap. Có thể merge thành một parent span nếu union không quá dài.

---

# 7.14. Adaptive cutoff thay fixed top-K duy nhất

F2 thường tốt hơn khi số output phụ thuộc confidence.

Ví dụ:

```text
always keep top-2 chunks
keep next chunk if score >= top1 - margin
cap max chunks/query
```

Hoặc:

```text
relative threshold
largest score gap
calibrated probability
```

Nhưng phải sweep offline từ cached scores.

Tương tự Trần Thanh Tú/ARCANE:

```text
GPU stage → save scores once
CPU stage → sweep dozens of cutoff policies
```

---

# 8. Model strategy

# 8.1. Nguyên tắc eligibility

Stage 3 cho phép:

```text
public training/model weights
release before 2026-08-01 VN time
parameter count <= 15B
no closed model API
```

Do đó mọi model phải có một compliance record:

```yaml
model_id: ...
revision: ...
parameter_count: ...
release_date_evidence: ...
license: ...
weight_url: ...
role: embedding|reranker|query_expansion
```

Không dùng tên model “latest” không pin revision.

> Repo Stage 1/2 là nguồn ý tưởng, **không phải bằng chứng eligibility cho Stage 3**. Trước freeze submission phải kiểm model card/revision/release date độc lập.

---

# 8.2. Baseline model family

Các family đã xuất hiện trong Stage 1/2 và hợp lý để benchmark:

```text
BAAI/bge-m3
BAAI/bge-reranker-v2-m3
Qwen3 multilingual embedding family
Qwen3 reranker family
```

Stage 3 không nên chốt model theo reputation. Chốt theo:

```text
Doc candidate Recall@K
Chunk candidate Recall@K
Reranker NDCG/MRR
Final doc/chunk F2
latency
VRAM
```

---

# 8.3. Đề xuất ba tier

## Tier A — baseline dễ chạy

```text
Dense: BGE-M3
Sparse: BM25 per language
Fusion: RRF
Rerank: bge-reranker-v2-m3
```

Ưu:

- multilingual;
- đã được nhiều Stage 1 team dùng;
- đơn giản;
- dễ tái lập.

## Tier B — stronger reranker

```text
Dense: BGE-M3 or Qwen embedding
Sparse: BM25
Rerank: Qwen reranker <=15B
```

Dùng nếu GPU budget cho phép.

## Tier C — ensemble

```text
Dense A: BGE-M3
Dense B: Qwen embedding
Sparse VI/EN/ZH
optional HyDE
→ RRF
→ one strong reranker
```

Không nên chạy hai reranker lớn nối tiếp nếu latency quá cao mà recall pool không tăng.

---

# 9. Local scorer phải được implement trước mọi “AI optimization”

Một scorer clone nên là task số 1.

Pseudo-code:

```python
def normalize_for_score(text):
    text = normalize_unicode(text)
    text = normalize_html(text)
    text = normalize_case(text)
    text = normalize_equivalent_punctuation(text)
    text = normalize_whitespace(text)
    return text


def tokenize(text):
    return bge_m3_tokenizer(normalize_for_score(text))


def lcs_len(a, b):
    ...


def predicted_chunk_relevant(pred, refs_same_doc):
    p = tokenize(pred.text)
    for ref in refs_same_doc:
        r = tokenize(ref.text)
        if lcs_len(p, r) >= 0.40 * len(r):
            return True
    return False


def near_duplicate(a, b):
    if a.doc_id != b.doc_id:
        return False
    ta, tb = tokenize(a.text), tokenize(b.text)
    lcs = lcs_len(ta, tb)
    union = len(ta) + len(tb) - lcs
    return lcs / max(union, 1) >= 0.8
```

Phải viết unit tests cho:

```text
Unicode composed/decomposed
HTML entities
hyphen variants
case
whitespace
very short chunks
empty text
identical chunks
one chunk contained in another
cross-doc identical text
```

---

# 10. Metrics ngoài leaderboard cần log

Chỉ log final F2 là không đủ.

## Corpus metrics

```text
#documents by language
#child chunks by language
#parent chunks by language
avg/median/p95 chunk tokens
duplicate rate
parse failure rate
```

## Retrieval metrics

```text
Doc Recall@10/20/50/100/200
Doc MRR
Chunk Recall@10/20/50/100/200
Gold rank distribution
zero-recall query rate
```

## Fusion contribution

```text
BM25-only gold
Dense-A-only gold
Dense-B-only gold
Translated-BM25-only gold
HyDE-only gold
multi-leg agreement
```

## Final

```text
Doc Precision / Recall / F2 macro
Chunk Precision / Recall / F2 macro
Final score
avg docs/query
avg chunks/query
avg predicted tokens/query
```

## Per-language

```text
Gold doc language = VI / EN / ZH
```

Nếu English F2 cao mà Chinese rớt mạnh, final score có thể che bottleneck.

---

# 11. Error taxonomy bắt buộc

Mỗi miss sample nên gắn một trong các label:

```text
E01 CORPUS_MISSING
E02 PARSE_FAILURE
E03 WRONG_DOC_ID_CANONICALIZATION
E04 DOC_NOT_IN_CANDIDATE
E05 DOC_RERANK_DROPPED
E06 DOC_CUTOFF_DROPPED
E07 CHUNK_NOT_IN_CANDIDATE
E08 CHUNK_RERANK_DROPPED
E09 CHUNK_CUTOFF_DROPPED
E10 CHUNK_BOUNDARY_LCS_FAIL
E11 DUPLICATE_POOL_OCCUPANCY
E12 QUERY_TRANSLATION_DRIFT
E13 QUERY_DECOMPOSITION_DRIFT
E14 BIOMEDICAL_ALIAS_MISS
E15 CROSS_LINGUAL_ALIGNMENT
E16 LONG_CONTEXT_TRUNCATION
```

Sau mỗi submission/experiment:

```text
Top 20 changed queries
Top 20 recall regressions
Top 20 precision regressions
new zero-recall queries
fixed zero-recall queries
```

Đây là phiên bản Stage 3 của `POST_SUBMISSION_REVIEW.md` của Hung&Fong.

---

# 12. Ablation plan — thứ tự experiment hợp lý

Không chạy 20 ý tưởng cùng lúc.

## E0 — lexical baseline

```text
BM25 VI only / raw query
```

Mục tiêu: sanity check ingestion + scorer.

## E1 — multilingual dense baseline

```text
BGE-M3 original VI query
```

Đo per-language recall.

## E2 — BM25 + dense RRF

```text
BM25 + BGE-M3 → RRF
```

## E3 — cross-encoder reranker

```text
E2 pool → reranker
```

## E4 — parent-child chunks

So sánh:

```text
retrieve=parent, submit=parent
retrieve=child, submit=child
retrieve=child, submit=parent
```

Prediction: child→parent sẽ tốt nhất cho chunk F2 nếu parent size tune hợp lý.

## E5 — parent size sweep

```text
256 / 384 / 512 / 640 / 768 BGE tokens
```

Không cần re-embed child index.

## E6 — translated BM25 EN

Đo unique gold contribution.

## E7 — translated BM25 ZH

Đo riêng Chinese.

## E8 — conservative subqueries

Chỉ query phức tạp; original always kept.

## E9 — HyDE

Một leg dense bổ sung.

## E10 — second embedding family

Đo diversity, không chỉ individual score.

## E11 — hard-negative fine-tuned reranker

Chỉ sau khi candidate recall đủ cao.

## E12 — adaptive cutoff

Sweep offline.

## E13 — score ensemble / rank features

Chỉ khi E0–E12 ổn định.

---

# 13. Hard-negative reranker fine-tuning plan

Dựa trên pattern của Nguyễn Văn Nghiêm.

# 13.1. Positive pairs

```text
(query, gold child/reference passage)
```

Nếu chỉ có gold chunk text mà không có exact canonical child chunk, map reference về corpus span bằng scorer normalization/LCS.

# 13.2. Hard negative mining

Với mỗi query:

```text
retrieve top 100 with current best hybrid
remove positives
sample from ranks 1–30 preferentially
```

Negative categories nên cân bằng:

```text
25% same document wrong chunk
25% same disease wrong intervention/outcome
20% same entity but wrong context
15% cross-language semantic false friend
15% high BM25 lexical overlap but medically irrelevant
```

# 13.3. Grouped training

Starting config:

```text
1 positive + 7 hard negatives
max_length 512 or 1024
binary/ranking loss
query-level train/dev split
```

Không split random pairs vì cùng query xuất hiện cả train/dev sẽ leak.

# 13.4. Evaluation

```text
MRR@10
NDCG@10
Recall@K after rerank
final Chunk F2
```

Checkpoint có NDCG cao nhất chưa chắc final F2 cao nhất nếu score calibration làm cutoff khác đi. Vì vậy luôn chạy offline cutoff sweep trên checkpoint finalists.

---

# 14. Query expansion plan an toàn

# 14.1. Không expansion cho query đơn giản

Ví dụ:

```text
"metformin gây thiếu vitamin B12 không?"
```

không cần phân rã 3 vế.

# 14.2. Trigger decomposition theo complexity

Có thể rule-based:

```text
>=2 outcomes
>=2 interventions
comparison language
multi-population
multiple clauses connected by và/hoặc/trong khi
```

# 14.3. Preserve anchors

Validator:

```text
all critical biomedical entities from original must appear
or be mapped to approved synonym
```

# 14.4. Expansion branch weight thấp hơn original

RRF weights ví dụ starting point:

```text
original_dense = 1.0
original_sparse = 0.8
translated_sparse = 0.6
subquery_dense = 0.4
HyDE = 0.3
```

Chỉ là initialization, không phải “best values”.

---

# 15. Document vs chunk two-stage retrieval

Một architecture thực dụng:

## Stage A: global document discovery

Indexes:

```text
doc_title_abstract_dense
doc_sparse
child_global_dense
```

Aggregate child hits → doc score:

```text
doc_score = max(child_score)
          + bonus(top2/top3 evidence)
          + doc_title_score
```

Mục tiêu:

```text
Doc Recall@100 > very high target
```

## Stage B: local chunk retrieval

Chỉ search child chunks trong top docs.

Mỗi doc reserve:

```text
at least 2–5 child candidates
```

Sau đó global rerank tất cả child candidates.

Điều này tránh một paper rất dài có 20 snippets chiếm toàn top-K global.

---

# 16. Multi-language index design

Không nên chỉ có một BM25 index trộn VI/EN/ZH nếu tokenizer không xử lý Chinese tốt.

Đề xuất:

```text
BM25_VI: Vietnamese tokenizer / normalized words
BM25_EN: biomedical-aware token normalization
BM25_ZH: Chinese segmentation or character n-gram
```

Sau đó RRF.

Chinese sparse có thể cần thử:

```text
word segmentation
2-char / 3-char ngram
mixed word + char ngram
```

Không assume default whitespace tokenizer.

Dense index có thể chung multilingual space.

---

# 17. Biomedical lexical features nên thêm

BM25 field boosting:

```text
title              high weight
abstract            normal
section heading      medium
body                 normal
entity alias field   high for exact entities
```

Query entity classes:

```text
disease
symptom
drug
procedure
gene/protein
biomarker
population
outcome
```

Exact match bonus nên là **soft boost**, không hard filter, vì multilingual aliases có thể thiếu.

---

# 18. RRF design

Simple weighted RRF:

```python
score[item] += weight_leg / (k + rank_leg)
```

Starting `k=60` là hợp lý vì nhiều Stage 1 team dùng, nhưng vẫn nên sweep:

```text
k = 20, 40, 60, 100
```

Các feature để log:

```text
num_legs_hit
best_rank
mean_rank
rank_variance
language channel
```

Một candidate xuất hiện top-20 ở 4 legs thường đáng tin hơn candidate chỉ xuất hiện rank 1 ở một noisy HyDE leg.

---

# 19. Cutoff optimization cho F2

Không optimize accuracy.

Sweep policies:

```text
Top-K: 2,3,4,5,6,8,10
Score threshold
Top-score margin
Largest gap
Top-K + margin
Minimum keep + adaptive additions
Per-language thresholds
```

Có thể doc và chunk dùng policy khác.

Ví dụ:

```text
Doc: min_keep=3, max_keep=10, margin=m_doc
Chunk: min_keep=2, max_keep=8, margin=m_chunk
```

Nếu query multi-part, tăng max chunk count.

---

# 20. Một trick Stage 3 riêng: output expansion sau rerank

Đây là idea quan trọng do metric LCS.

Không cần embed lại parent chunks cho mọi sweep.

Pipeline:

```text
child anchors fixed
→ generate parent windows at 256/384/512/640/768
→ local scorer sweep
```

Đây là experiment rẻ nhưng có thể tác động mạnh Chunk F2.

Ta có thể giữ child rank y nguyên và chỉ đổi text output.

---

# 21. Một trick khác: reference-style length estimator

Nếu train/dev labels có reference chunks, hãy thống kê:

```text
reference token length distribution
p25 / median / p75 / p90
by source language
by document type
```

Nếu median reference ≈ 420 tokens, parent=512 hợp lý hơn parent=128.

Không tune blind theo intuition.

---

# 22. Dedup strategy

Có hai loại duplicate khác nhau.

## Corpus duplicate

Trước indexing:

```text
same source ID
exact normalized text
near duplicate document
mirror copy
```

## Prediction duplicate

Theo scorer:

```text
same doc + LCS/union >= 0.8
```

Đừng dùng một threshold duy nhất cho cả hai.

---

# 23. Provenance schema — mượn Stage 2

Mỗi child/parent chunk:

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "parent_id": "...",
  "language": "zh",
  "section_id": "s04",
  "sentence_start": 113,
  "sentence_end": 119,
  "token_start_bge": 840,
  "token_end_bge": 1310,
  "raw_char_start": 10220,
  "raw_char_end": 12611,
  "normalized_text": "...",
  "raw_text": "...",
  "source_sha256": "...",
  "chunker_version": "child-v3-parent512"
}
```

Benefits:

- scorer debugging;
- exact source recovery;
- no phantom chunks;
- parent-size sweep cheap;
- parser regression detectable.

---

# 24. Artifact manifest design — mượn mscAI + ARCANE + KINGPRO

`manifest.json`:

```json
{
  "run_id": "20261014_2210_qwen_rerank_parent512",
  "git_commit": "...",
  "query_sha256": "...",
  "raw_corpus_sha256": "...",
  "canonical_corpus_sha256": "...",
  "child_chunks_sha256": "...",
  "parent_chunks_sha256": "...",
  "dense_index": {
    "model": "...",
    "revision": "...",
    "sha256": "..."
  },
  "bm25": {
    "tokenizer": "...",
    "config": {},
    "sha256": "..."
  },
  "reranker": {
    "model": "...",
    "revision": "...",
    "max_length": 512
  },
  "retrieval_config_sha256": "...",
  "scorer_version": "stage3-local-v1"
}
```

Mỗi loader assert hash trước khi chạy.

---

# 25. Suggested repository structure cho Stage 3

```text
r2ai-stage3-system/
├── README.md
├── pyproject.toml
├── configs/
│   ├── baseline.yaml
│   ├── hybrid.yaml
│   ├── rerank.yaml
│   └── release.yaml
│
├── data/
│   ├── raw/                     # gitignored
│   ├── canonical/               # generated
│   ├── manifests/
│   └── queries/
│
├── src/r2ai3/
│   ├── ingestion/
│   │   ├── parse_vi.py
│   │   ├── parse_en.py
│   │   ├── parse_zh.py
│   │   ├── normalize.py
│   │   ├── dedup.py
│   │   └── schema.py
│   │
│   ├── chunking/
│   │   ├── sentences.py
│   │   ├── child.py
│   │   ├── parent.py
│   │   └── provenance.py
│   │
│   ├── indexing/
│   │   ├── build_bm25.py
│   │   ├── build_dense.py
│   │   ├── manifests.py
│   │   └── cache.py
│   │
│   ├── query/
│   │   ├── normalize.py
│   │   ├── entities.py
│   │   ├── translate.py
│   │   ├── decompose.py
│   │   └── hyde.py
│   │
│   ├── retrieval/
│   │   ├── sparse.py
│   │   ├── dense.py
│   │   ├── fusion.py
│   │   ├── document.py
│   │   └── chunk.py
│   │
│   ├── rerank/
│   │   ├── cross_encoder.py
│   │   ├── sliding_maxp.py
│   │   └── calibration.py
│   │
│   ├── selection/
│   │   ├── cutoffs.py
│   │   ├── dedup_lcs.py
│   │   └── parent_expand.py
│   │
│   ├── evaluation/
│   │   ├── scorer_stage3.py
│   │   ├── candidate_recall.py
│   │   ├── ablation.py
│   │   ├── error_taxonomy.py
│   │   └── reports.py
│   │
│   └── submission/
│       ├── schema.py
│       ├── build.py
│       └── validate.py
│
├── scripts/
│   ├── build_corpus.py
│   ├── build_indexes.py
│   ├── retrieve.py
│   ├── rerank.py
│   ├── sweep_cutoff.py
│   ├── evaluate.py
│   └── release_audit.py
│
├── tests/
│   ├── test_normalization.py
│   ├── test_chunk_provenance.py
│   ├── test_lcs_scorer.py
│   ├── test_manifest.py
│   └── test_submission.py
│
└── runs/
    └── <immutable run_id>/
```

---

# 26. Suggested baseline config

```yaml
seed: 42

chunking:
  tokenizer: bge-m3
  child_tokens: 180
  child_overlap_tokens: 40
  parent_tokens: 512
  sentence_boundary: true
  section_boundary: true

retrieval:
  document_candidate_k: 200
  chunk_candidate_k_per_doc: 5
  rrf_k: 60

  bm25_vi:
    enabled: true
    weight: 0.8

  bm25_en:
    enabled: true
    translated_query: true
    weight: 0.6

  bm25_zh:
    enabled: true
    translated_query: true
    weight: 0.6

  dense:
    enabled: true
    model: BAAI/bge-m3
    weight: 1.0

  hyde:
    enabled: false
    weight: 0.3

query:
  preserve_original: true
  max_subqueries: 2
  decomposition: false

reranker:
  enabled: true
  model: BAAI/bge-reranker-v2-m3
  max_length: 512
  sliding_maxp: false

selection:
  doc_min_keep: 3
  doc_max_keep: 10
  chunk_min_keep: 2
  chunk_max_keep: 8
  near_duplicate_lcs_union: 0.8

reproducibility:
  require_manifest_match: true
  immutable_run_directory: true
```

Đây là **baseline config**, không phải final config.

---

# 27. Stronger config sau baseline

```yaml
query:
  preserve_original: true
  translate_en: true
  translate_zh: true
  decompose_complex_only: true
  max_subqueries: 2
  hyde_complex_only: true

retrieval:
  dense_1:
    model: BAAI/bge-m3
  dense_2:
    model: <eligible qwen multilingual embedding>
  sparse_languages: [vi, en, zh]
  candidate_union: true
  hard_intersection: false

reranker:
  model: <eligible strong multilingual reranker>
  fine_tuned_on_hard_negatives: true
  sliding_maxp: true

chunk_output:
  child_retrieval: true
  parent_expansion: 512
  metric_aware_dedup: true
```

---

# 28. Không nên làm những gì

# 28.1. Không bắt đầu bằng agent graph phức tạp

Stage 3 là retrieval benchmark. Một graph:

```text
planner → agent → tool → critic → reflection → retriever
```

không giải quyết candidate recall nếu base index yếu.

---

# 28.2. Không dùng LLM judge làm final filter quá sớm

Stage 1 đã cho thấy precision có thể tăng mạnh nhưng F2 không tăng vì recall bị giết.

---

# 28.3. Không thay original query bằng decomposition

Original query luôn là anchor.

---

# 28.4. Không hard-intersect sparse và dense

Cross-lingual gold có thể chỉ xuất hiện ở dense hoặc translated sparse.

---

# 28.5. Không chunk toàn corpus bằng fixed 512 token rồi coi như xong

Stage 3 scorer làm chunk design trực tiếp ảnh hưởng metric.

---

# 28.6. Không tune leaderboard mà không giữ local error log

Mỗi submission cần post-submission review.

---

# 28.7. Không đổi 4 thứ cùng lúc

Hung&Fong có experiment bị confound khi rebuild thay đồng thời reranker + tuning chain + cutoff. Stage 3 phải ablate một dimension mỗi lần nếu có thể.

---

# 29. Experiment tracking template

Mỗi experiment ghi:

```markdown
## EXP-017 — parent512 vs parent384

Hypothesis:
Parent 512 tăng LCS coverage mà không giảm precision nhiều.

Base:
EXP-014

Change only:
parent_tokens: 384 → 512

Metrics:
Doc P/R/F2:
Chunk P/R/F2:
Final:
Avg chunks/query:
Zero chunk recall queries:

Per-language:
VI:
EN:
ZH:

Regression IDs:
...

Decision:
KEEP / REJECT / NEEDS_FOLLOWUP
```

---

# 30. Public-test strategy

Public leaderboard nên dùng để xác nhận hypothesis, không dùng làm optimizer duy nhất. **Quota public tối đa 10 bài/ngày**, và mỗi file phải chứa đủ **1.200 query**, vì Dashboard tự chọn subset public để chấm.

Mỗi lần nộp phải biết câu hỏi experiment:

```text
Does EN translated BM25 add unique gold?
Does parent512 improve chunk recall?
Does reranker max_len1024 improve long biomedical evidence?
Does HyDE improve EN/ZH without hurting precision?
```

Không nộp kiểu:

```text
"thử thêm model này xem sao"
```

---

# 31. Kế hoạch từ 01/10 đến 31/10/2026

Các deadline đã cung cấp:

```text
01/10/2026: public test released
31/10/2026 23:59 UTC+07: public deadline
01/11/2026: private phase starts
04/11/2026 23:59 UTC+07: system deadline
11/11/2026: final result
Public quota: max 10 submissions/day
Private quota: max 5 submissions/user total
```

Đề xuất lịch:

## 01–03/10 — scorer + corpus integrity

Deliverables:

```text
local scorer clone
canonical schema
source snapshot manifest
parser smoke tests
language stats
```

Không tune model trước khi scorer đáng tin.

## 04–06/10 — baseline indexes

```text
BM25 VI/EN/ZH
BGE-M3 dense
simple RRF
```

Lấy first baseline.

## 07–10/10 — parent-child chunking

Sweep:

```text
child 128/180/256
parent 384/512/640
```

Đây có thể là gain rẻ nhất cho Chunk F2.

## 11–14/10 — reranking

```text
generic multilingual reranker
max_length sweep
sliding MaxP
```

## 15–18/10 — translation + multilingual sparse

```text
EN translation branch
ZH translation branch
entity-preservation validator
```

## 19–21/10 — query decomposition / HyDE

Chỉ sau khi baseline mạnh.

Mỗi branch phải chứng minh unique gold contribution.

## 22–25/10 — hard-negative reranker fine-tune

Nếu có train/dev gold đủ.

## 26–28/10 — fusion + cutoff sweep

Freeze candidate generation, sweep selection offline.

## 29–30/10 — regression + reproducibility

```text
fresh rebuild from raw
hash validation
re-run best config
compare byte/schema
```

## 31/10 — public freeze

Không thêm architecture mới vào phút cuối.

---

# 32. Private-test strategy 01–04/11

Private phase phải xem như **release engineering + submission selection**, không phải research marathon. Theo rule đã cung cấp, bài nộp vẫn chứa đầy đủ **1.200 queries**; hệ thống chấm tự chọn subset private. Quan trọng hơn, **mỗi user chỉ có tối đa 5 private submissions tổng cộng**.

## Trước 01/11 — bắt buộc freeze phần lớn research

Phải có sẵn ít nhất:

```text
public-best stable artifact
1 conservative rollback artifact
1 diversity-oriented challenger (nếu public evidence đủ mạnh)
all model/index/corpus manifests
full 1,200-query submission builder
source-verbatim verifier
ZIP validator
```

## 01/11 — revalidate, không đổi kiến trúc bừa

```text
verify official query set still maps 1:1 to builder
re-run schema/provenance tests
re-run scorer/regression suite
verify model compliance manifest
produce candidate private artifact #1 from frozen best config
```

## 02–03/11 — chỉ promote variant đã có bằng chứng

Không dùng private quota để sweep ngẫu nhiên. Mỗi upload phải trả lời một hypothesis cụ thể, ví dụ:

```text
A: public-best baseline
B: same candidates + safer chunk parent window
C: same candidates + conservative doc cutoff
```

Không upload 5 cấu hình chỉ khác một hyperparameter mà chưa có local/public evidence.

## 04/11 — release final

```text
rebuild or revalidate selected artifact
exact 1,200 IDs
source-bound chunks only
ZIP = exactly one root JSON
archive config + hashes + model revisions
submit only after release gate PASS
```

Private submission budget nên được ghi rõ:

```text
slot 1/5: artifact + hypothesis + hash
slot 2/5: ...
remaining: 3
```

Không dùng private phase để tune theo gold không quan sát được; chỉ dùng kết quả Dashboard theo cách BTC cho phép để chọn giữa các system variants đã được chuẩn bị hợp lệ.

---

# 33. Release gate trước submission

Một run chỉ được promote nếu:

```text
[ ] exactly 1,200 query records
[ ] query ID set exactly equals query.parquet ID set
[ ] unique query IDs
[ ] relevant_docs and relevant_chunks fields always present (empty lists allowed)
[ ] every predicted doc_id exists in links_corpus.parquet
[ ] every predicted chunk has valid doc_id
[ ] every emitted chunk_text exact-maps to/source-derives from that document
[ ] no emitted chunk object has empty chunk_text
[ ] optional chunk_order is integer >= 0
[ ] no phantom/external document in submission
[ ] ZIP contains exactly one JSON file at archive root
[ ] scorer normalization tests pass
[ ] near-duplicate policy deterministic
[ ] no model outside allowlist
[ ] every model pinned revision
[ ] every model release date verified <= 2026-08-01
[ ] every model <= 15B
[ ] corpus sources documented
[ ] external data citations complete
[ ] working-notes method log is up to date
[ ] index manifests match corpus
[ ] reranker score manifests cover every candidate exactly once
[ ] no missing shard
[ ] deterministic seed/config archived
[ ] best run reproducible from clean environment
```

---

# 34. Candidate recall dashboard

Một dashboard/report tối thiểu:

```text
                         @10    @20    @50    @100
Doc BM25 VI
Doc Dense multilingual
Doc Hybrid RRF
Doc Hybrid + translations

Chunk Dense
Chunk Hybrid
Chunk after rerank
```

Và:

```text
Gold contribution matrix
--------------------------------------------------
BM25 only                6.2%
Dense only              14.4%
Translated EN only       3.8%
Translated ZH only       2.1%
HyDE only                0.9%
Multiple channels       72.6%
```

Các số trên chỉ là format minh họa, không phải kết quả thật.

Nếu HyDE-only = 0.1% mà tốn 40% runtime, bỏ.

---

# 35. Per-language failure analysis

Mỗi query gold doc/chunk nên bucket theo language source.

Báo cáo:

```text
VI gold docs:
EN gold docs:
ZH gold docs:
Mixed-language gold set:
```

Đặc biệt kiểm tra:

```text
VI→VI
VI→EN
VI→ZH
```

Nếu VI→ZH yếu, có thể lỗi ở:

- embedding cross-lingual alignment;
- Chinese tokenizer BM25;
- translation;
- segmentation;
- reranker multilingual capacity.

Không thể biết nếu chỉ nhìn aggregate F2.

---

# 36. Domain-specific biomedical hard cases

Nên tạo regression suite riêng cho:

## Abbreviation ambiguity

```text
MS = multiple sclerosis / mitral stenosis / mass spectrometry
AD = Alzheimer disease / atopic dermatitis
```

## Drug generic vs brand

```text
generic names
international nonproprietary names
brand mentions
```

## Gene/protein aliases

```text
HER2 / ERBB2
PD-1 / PDCD1
```

## Greek symbols

```text
TNF-α / TNF-alpha
β-blocker / beta blocker
```

## Clinical constraints

```text
adult vs pediatric
pregnancy
renal impairment
hepatic impairment
first-line vs second-line
prevention vs treatment
```

## Negation

```text
"không làm tăng nguy cơ"
vs
"làm tăng nguy cơ"
```

Dense model có thể coi hai câu quá gần; reranker/hard negatives cần học polarity.

---

# 37. Biomedical query understanding không cần biến thành answer reasoning

Query parser chỉ cần giúp retrieval.

Output:

```json
{
  "entities": [],
  "constraints": [],
  "outcomes": [],
  "comparison": null,
  "needs_decomposition": false,
  "translations": {}
}
```

Không cần chain-of-thought hay clinical answer.

---

# 38. Training data augmentation

Nếu gold ít, có thể tạo synthetic queries từ corpus bằng open model hợp lệ.

Nhưng synthetic pipeline phải tách khỏi eval data.

Ví dụ:

```text
passage → generate Vietnamese question
```

Tạo cross-lingual train pairs:

```text
VI synthetic query ↔ EN passage
VI synthetic query ↔ ZH passage
```

Dùng cho:

```text
reranker fine-tuning
optional embedding contrastive tuning
```

Hard negative vẫn lấy từ current retriever.

---

# 39. Fine-tune embedding hay reranker trước?

Ưu tiên:

```text
1. fine-tune reranker
2. only then consider embedding fine-tune
```

Lý do:

- reranker training đơn giản hơn;
- candidate retrieval diversity đã có từ multiple legs;
- embedding fine-tune có thể làm hỏng multilingual geometry nếu data không đủ cân bằng VI/EN/ZH.

Chỉ fine-tune embedding nếu candidate Recall@K vẫn là bottleneck rõ ràng.

---

# 40. Khi nào dùng HyDE?

HyDE chỉ đáng giữ nếu:

```text
unique gold contribution > meaningful threshold
or
moves gold from deep rank to rerankable range
```

Đo hai thứ:

```text
Recall@K delta
median gold rank delta
```

Nếu recall không tăng nhưng gold rank tăng 70→20, HyDE vẫn có giá trị vì reranker dễ cứu hơn.

---

# 41. Khi nào dùng second dense model?

Không dùng chỉ vì “ensemble mạnh hơn”.

Giữ nếu:

```text
Dense B unique gold contribution
+
rank improvement on EN/ZH
```

đáng hơn cost.

Có thể offline precompute nên inference query cost nhỏ; index storage mới là tradeoff chính.

---

# 42. Reranker scoring strategy

Một candidate có thể có nhiều child windows.

Aggregation options:

```text
MaxP
mean top-2
logsumexp
```

Starting point:

```text
MaxP
```

vì biomedical relevance thường localized.

Document score từ child reranker:

```text
doc_score = max(child_score) + alpha * second_best
```

Điều này giúp doc có hai evidence regions được boost mà không bị length bias quá nhiều.

---

# 43. Avoiding long-document domination

Paper dài tạo nhiều child chunks → xác suất có random high score tăng.

Mitigation:

```text
per-doc candidate quota
score calibration by document length
aggregate top-N instead of raw count
```

Đừng để doc 200 chunks có lợi chỉ vì có nhiều lottery tickets hơn doc 20 chunks.

---

# 44. Sparse retrieval cho biomedical identifiers

Sparse cực mạnh với:

```text
NCT identifiers
PMID-like IDs
gene symbols
drug names
mutation notation
biomarkers
acronyms
```

Nếu query chứa exact identifier, có thể boost lexical branch rất mạnh.

Nhưng vẫn nên soft boost, vì query có typo/alias.

---

# 45. Chinese retrieval considerations

Chinese branch cần riêng:

- Unicode punctuation normalization;
- simplified/traditional handling nếu sources hỗn hợp;
- segmentation hoặc char n-gram;
- Latin biomedical entities giữ nguyên;
- mixed Chinese-English token handling.

Query translated Chinese có thể chứa English drug/gene token; tokenizer sparse không nên phá chúng.

---

# 46. Vietnamese medical query normalization

Cần giữ cả từ có dấu và alias không dấu chỉ như fallback, không replace chính.

Ví dụ:

```text
"đái tháo đường"
"tiểu đường"
"T2DM"
```

nên map vào same concept alias set nhưng exact phrase vẫn được giữ.

Không nên stemming quá mạnh làm biến medical terms.

---

# 47. Document metadata có thể dùng nhưng không hard-gate tùy tiện

Nếu source metadata đáng tin:

```text
publication language
article type
publication year
journal/source
```

có thể dùng rerank features.

Nhưng không hard-filter year/article type trừ khi query nói rõ. Nhiều gold có thể là older guideline/review.

---

# 48. Publication versions / duplicates — phiên bản biomedical của legal versioning

Legal version graph không bê nguyên được, nhưng biomedical corpus có:

```text
preprint
conference abstract
journal article
PMC full text
publisher mirror
translated abstract
```

Nên canonicalize relationships nếu data source cho phép.

Mục tiêu không phải “giữ newest law” mà là tránh duplicate copies chiếm candidate slots.

---

# 49. Scorer-aware output window merger

Nếu hai selected parent chunks cùng doc overlap mạnh nhưng mỗi chunk cover một child anchor, có thể merge:

```text
if combined token length <= MAX_PARENT_MERGE
and same section
and spans overlap/adjacent:
    output merged exact span
```

Điều này có thể:

- giảm denominator số predicted chunks → precision tốt hơn;
- giữ/ tăng information coverage → recall tốt hơn.

Phải validate bằng local scorer vì quá dài có thể có side effects khác.

---

# 50. Document selection from chunk evidence

Một doc confidence function:

```text
S_doc = w1 * doc_retrieval_score
      + w2 * best_chunk_rerank
      + w3 * second_best_chunk
      + w4 * multi-leg agreement
```

Sau đó doc cutoff được tune riêng.

Không chỉ unique doc IDs từ top chunks.

---

# 51. Final “best combo” đề xuất

Nếu cần chọn một architecture để build ngay, đây là lựa chọn của tài liệu này:

```text
OFFLINE
------
1. Snapshot + hash toàn bộ sources.
2. Parse thành canonical documents với raw↔normalized offset mapping.
3. Dedup document/paragraph mirrors.
4. Build child chunks ~180 BGE tokens, sentence-aware.
5. Build parent windows ~512 BGE tokens, section-aware.
6. Build:
   - BM25 VI
   - BM25 EN
   - BM25 ZH
   - multilingual dense child index
   - document title+abstract dense index
7. Pin model/tokenizer revisions + manifests.

ONLINE
------
1. Normalize Vietnamese query.
2. Extract biomedical entities/constraints.
3. Always search original query.
4. Optional safe EN/ZH query forms for lexical branches.
5. Hybrid document retrieval:
      BM25 + multilingual dense + child-hit aggregation
      → weighted RRF
6. Keep high-recall doc pool.
7. Cross-encoder rerank documents using title+best snippets.
8. Within retained docs, retrieve child passages.
9. Cross-encoder rerank child passages.
10. Optional auxiliary score from translations/subqueries.
11. Expand selected child anchors to ~512-token parent chunks.
12. Merge/suppress near duplicates using scorer-equivalent LCS.
13. Apply independently tuned doc/chunk F2 cutoffs.
14. Emit exact source text + doc IDs.
15. Save full trace + manifests.
```

Sau khi baseline này ổn, thêm theo thứ tự:

```text
A. second dense family
B. HyDE
C. hard-negative reranker fine-tune
D. adaptive per-query cutoff
E. optional embedding fine-tune
```

---

# 52. Tại sao đây là combo tốt hơn việc copy team thắng Stage 1

Stage 1 winner pipeline giải bài:

```text
VI legal query → VI legal article
```

Stage 3 cần:

```text
VI query → VI/EN/ZH biomedical docs/chunks
```

Nếu copy nguyên:

- chunk unit pháp lý “Điều” không tồn tại;
- metadata số văn bản vô nghĩa;
- version/effectiveness law graph không phù hợp;
- top-K nhỏ sẽ mất cross-lingual recall;
- LLM answer/filter tốn compute không được chấm.

Combo đề xuất giữ **principles**, không giữ domain assumptions.

---

# 53. Tại sao đây là combo tốt hơn việc copy Stage 2 winner

Stage 2 winner giải:

```text
financial query → exact table/cell → pandas execution
```

Stage 3 không có structured table execution.

Nếu copy nguyên:

- FinancialPlan không cần;
- sandbox Pandas không cần;
- table schema binding không cần;
- numeric auditor không cần.

Nhưng Stage 2 dạy ta:

```text
source traceability
immutable artifacts
manifest/hash
release validation
bounded stage contracts
```

Đây là thứ cực giá trị khi deadline private chỉ vài ngày.

---

# 54. Research priorities xếp theo expected value

## Priority S

```text
Local scorer correctness
Corpus parsing/canonicalization
Parent-child chunking
Hybrid multilingual candidate recall
Strong reranker
Cutoff sweep
```

## Priority A

```text
EN/ZH translated sparse
Hard-negative reranker fine-tuning
Second dense model
Sliding MaxP
```

## Priority B

```text
HyDE
Conservative decomposition
Learned score fusion
```

## Priority C

```text
Agentic planner loops
LLM judge
complex multi-agent orchestration
```

Không đụng Priority C nếu Priority S còn lỗi.

---

# 55. Minimal viable winning-oriented baseline

Nếu thời gian rất gấp:

```text
Canonical corpus
→ BGE-M3 child chunks
→ BM25 + Dense RRF
→ bge-reranker-v2-m3
→ child-to-parent 512
→ LCS dedup
→ F2 cutoff sweep
```

Đây là baseline nên có trước mọi experimentation.

---

# 56. Strong submission candidate

Sau khi baseline ổn:

```text
3-language BM25
+ BGE-M3 dense
+ eligible Qwen embedding second leg
+ safe translation
+ weighted RRF
+ strong eligible Qwen reranker
+ child→parent 512/640 adaptive
+ hard-negative fine-tune
+ per-language diagnostics
+ adaptive F2 cutoff
```

Đây là hướng có khả năng vượt baseline bằng **recall diversity + precision reranking**, đúng bài học xuyên suốt Stage 1.

---

# 57. Những câu hỏi phải trả lời bằng số trước khi freeze

1. Gold document candidate Recall@100 là bao nhiêu?
2. Gold chunk candidate Recall@100 là bao nhiêu?
3. Bao nhiêu gold chỉ BM25 tìm được?
4. Bao nhiêu gold chỉ dense tìm được?
5. EN translated BM25 thêm bao nhiêu unique gold?
6. ZH translated BM25 thêm bao nhiêu unique gold?
7. HyDE thêm bao nhiêu unique gold?
8. Reranker làm rơi bao nhiêu gold đã có trong pool?
9. Parent 384/512/640 cái nào tốt nhất cho chunk F2?
10. Bao nhiêu final FP là near-duplicate?
11. Per-language F2 chênh nhau bao nhiêu?
12. Query multi-clause có recall thấp hơn query đơn bao nhiêu?
13. Long document có bị reranker truncate không?
14. Cutoff nào tối ưu F2 chứ không chỉ precision?
15. Best run có rebuild sạch ra cùng kết quả không?

Nếu chưa trả lời được các câu này, hệ thống chưa đủ “engineering maturity” để vào private test.

---

# 58. Source map — các file nên đọc lại khi implement

## Stage 1

### mscAI

```text
mscai/README.md
mscai/src/backend/config.yaml
mscai/src/backend/src/services/vector_store/hybrid.py
mscai/src/backend/src/services/vector_store/index_builder.py
mscai/src/backend/src/services/agents/legal_assistant/agent.py
mscai/src/backend/src/services/agents/legal_assistant/node.py
```

### Hung&Fong

```text
hung&phong/README.md
hung&phong/src/docs/POST_SUBMISSION_REVIEW.md
hung&phong/src/backend/rag.py
hung&phong/src/backend/query_analyzer.py
hung&phong/src/backend/textnorm.py
hung&phong/src/docs/kien_truc.drawio
```

### Nguyễn Văn Nghiêm

```text
nguyenvannghiem/README.md
nguyenvannghiem/src/docs/reproduce.md
nguyenvannghiem/src/docs/model_description.md
nguyenvannghiem/src/code/hyde_retrieval.py
nguyenvannghiem/src/code/rerank_intersection.py
nguyenvannghiem/src/code/train_reranker_v2.py
```

### TQD

```text
tqd/README.md
pipeline-corpus.ipynb
```

### FAI Team

```text
faiteam/src/docs/PIPELINE_OVERVIEW_29-06.md
faiteam/src/R2AI/scripts/bm25_retrieval.py
faiteam/src/R2AI/scripts/rerank_retrieval.py
```

### Agentic Builders

```text
agentic_builders/README.md
agentic_builders/data/01_data/DATA_DOCUMENTATION.md
agentic_builders/src/03_source_code/retrieval_pipeline/final_pipeline.ipynb
```

### BeeIT

```text
beeit/README.md
beeit/src/main_v2.py
beeit/src/search_v2.py
```

### NextGen

```text
nextgen/README.md
nextgen/src/hybrid_retriever.py
nextgen/src/answer_intersect.py
```

### Trần Thanh Tú

```text
tranthanhtu/README.md
tranthanhtu/backend/local_rag_engine.py
tranthanhtu/backend/retrieval_cutoff.py
tranthanhtu/scratch/sweep_cutoff.py
```

## Stage 2

### LASTDANCE

```text
lastdance/src/README.md
lastdance/src/analysis/
lastdance/src/src/vifinqa/
```

### ARCANE

```text
arcane/src/README.md
arcane/src/ARCHITECTURE.md
arcane/src/SUBMISSION_DOCUMENTATION.md
arcane/src/scripts/select_slot_tables.py
arcane/src/scripts/validate_rerank_scores.py
```

### KINGPRO

```text
kingpro/src/README.md
kingpro/src/docs/DATA_CARD.md
kingpro/src/scripts/
```

### SYNERA / IDIOT

```text
synera/src/README.md
synera/src/docs/architecture.md
synera/src/docs/optimization.md
idiot/src/README.md
```

### AISOLO

```text
aisolo/src/README.md
aisolo/src/pipeline/
```

---

# 59. Kết luận cuối

Stage 3 không cần “một model thần thánh”. Nó cần một retrieval system mà ta biết chính xác gold biến mất ở đâu.

Công thức tư duy tốt nhất rút ra từ Stage 1:

```text
DATA QUALITY
× CANDIDATE RECALL
× RERANKING QUALITY
× F2-AWARE SELECTION
```

Stage 2 thêm vào:

```text
× PROVENANCE
× REPRODUCIBILITY
× RELEASE DISCIPLINE
```

Và Stage 3 bổ sung biến riêng quan trọng nhất:

```text
× MULTILINGUAL ALIGNMENT
× METRIC-AWARE CHUNKING
```

Do đó “combo tốt nhất” không phải là copy mscAI, LASTDANCE hay KINGPRO. Nó là:

```text
Hung&Fong's diagnostic discipline
+ Nguyễn Văn Nghiêm's candidate diversity / hard negatives
+ TQD's fine-grained retrieval & MaxP
+ FAI's anchored multi-query fusion
+ mscAI's RRF/index lifecycle
+ Agentic Builders' canonical corpus
+ Trần Thanh Tú's cheap cutoff sweeps
+ LASTDANCE's provenance
+ ARCANE's manifests/non-destructive candidate flow
+ KINGPRO/SYNERA's release gates
```

được thiết kế lại quanh scorer Stage 3 bằng:

```text
child retrieval → parent output
3-language sparse + multilingual dense
union before pruning
strong reranker
scorer-equivalent LCS dedup
independent document/chunk F2 cutoff
```

Nếu phải ưu tiên duy nhất một chiến lược nghiên cứu trong tháng public test, hãy ưu tiên:

> **Đẩy gold vào candidate pool bằng retrieval diversity, sau đó dùng reranker mạnh để lấy precision; đừng cố lấy precision bằng cách lọc một candidate pool nghèo recall.**

Đó là bài học nhất quán nhất khi đọc lại cả hai stage, và cũng là bài học khớp nhất với F2 của Stage 3.

