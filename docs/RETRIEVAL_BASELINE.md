# Retrieval cascade v2 trên Colab — 06/10/2026

Mở [notebook 04](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_retrieval_baseline.ipynb)
trong **runtime GPU mới**, chạy từ đầu. Bootstrap tự nâng retrieval code lock một
lần lên workflow `document-child-parent-cascade-v2`, rồi ghim commit cho resume.
Các notebook data 00–03 và candidate đã freeze không cần chạy lại.

Mặc định `RUN_NAME=stage-a-retrieval-v2`, candidate team cũ với 864 documents,
9.076 children và 8.760 unique dense inputs. `EMBEDDING_CACHE_RUN=stage-a-retrieval-v1`
chỉ đọc vectors cũ. Reuse kiểm exact inputs/order, pinned BGE model/budget,
pooling/precision, config/manifest/part hashes và done markers. Nếu cache hoàn
tất, không nạp embedding model. Provenance giữ đúng producer/runtime cũ.
Không bỏ qua cache corruption hoặc tự gán vectors cũ vào model identity mới.

Luồng mới kế thừa các phần phù hợp từ Stage 1/2; mapping code cụ thể ở
[DECISIONS](DECISIONS.md). Tham số lấy từ master plan để bắt đầu benchmark,
chưa phải cấu hình đã tối ưu cho ViBioMIR:

1. **CPU/GPU khi cần encode:** BGE-M3 vectors cho original VI query và children.
   FAISS exact cosine trên pilot; dense child MaxP gộp lên documents.
2. **GPU:** Qwen3-4B-Instruct-2507 dịch toàn query sang EN/ZH dạng JSON gồm
   original/entities/constraints. Giữ raw output/rejections, kiểm số/acronym/
   comparator và một số constraint cues. Nhánh lỗi bị tắt, original dense/VI
   giữ lại. Surface checks không chứng nhận dịch đúng ngữ nghĩa.
3. **CPU:** VI/EN/ZH BM25 postings, positive Lucene-style IDF, CJK unigram/bigram.
   Language suy từ source content, bỏ phụ thuộc declared labels sai của build
   cũ; unknown đi vào cả ba sparse branches. Weighted RRF lấy 200 documents.
   VI vẫn dùng Unicode syllable/word tokens; chưa phải Vietnamese word segmenter
   hoặc hệ Lucene language analyzers. Chưa thêm medical bilingual alias glossary.
4. **GPU:** BGE rerank mỗi document bằng title + top-2 child hits, chia passage
   budget công bằng. Giữ top-30 documents; local dense + BM25 + global dense
   rank chọn tối đa 5 children/document. Rerank original VI query với sliding
   MaxP khi passage dài. Model chỉ cho điểm, không sinh evidence.
5. **CPU:** Expand exact frozen parent; same-document BGE token LCS/union >=0.8
   dedup. Caps mặc định 10 docs/8 chunks, tối đa 2 chunks/doc. Optional score
   floors/margins chưa bật. Raw negative logits không tự có nghĩa irrelevant.
6. **CPU, khi có reviewed reference labels:** local plan-derived Doc/Chunk macro F2,
   40% reference LCS relevance. Notebook chỉ đánh giá, không tune contest queries.
   Đây **chưa phải scorer BTC**. Không có labels thì NOT_EVALUATED, không fake
   nhãn/F2, không promote. API top-k/margin sweep riêng chỉ nhận reviewed dev
   labels trên dev query set độc lập đã chấm candidates; không tự áp dụng policy.

Embedding, LLM và reranker được nạp lần lượt để tiết kiệm VRAM. Qwen có
4.022.468.096 parameters, public Apache-2.0 weights; pinned revision có trước
01/08/2026. Evidence nằm trong `configs/query_translation_model_manifest.json`,
kiểm từ [model API](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507)
và [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Đây là kiểm
theo plan nội bộ, chưa xác nhận BTC chấp nhận cấu hình. Chưa fine-tune model.

## Chạy canary và resume

Đặt `MAX_NEW_TRANSLATIONS=25`, `MAX_NEW_QUERIES=25` ở cell 2 để kiểm một lượt nhỏ.
Notebook chạy retrieval cho phần prefix queries đã dịch, lưu IN_PROGRESS và
dừng trước export. Sau khi xem translations/rankings/pairs/runtime evidence,
đặt hai biến thành None và chạy lại từ đầu cùng RUN_NAME để hoàn tất 1.200 query.
MAX_NEW_* giới hạn số query **mới mỗi phiên**; những query đã xong không tính lại.

Drive lưu `translations/query-<id>.done.json`, `queries/query-<id>-documents.done.json`,
`queries/query-<id>-children.done.json`, `queries/query-<id>.done.json`. Hai scored
stages giữ TẤT CẢ pair inputs và finite scores, kiểm completeness và hashes trước
resume. Ngắt sau document stage không làm rerank lại stage đó. Cache translation
đầy đủ/hợp lệ được reuse mà không nạp LLM lại. Chỉ export khi đủ official queries.
Scored pairs chiếm thêm dung lượng Drive, GPU work lớn hơn v1; chưa đo throughput
Colab thực của cascade mới. Có thể giảm inference batch, không đổi policy khi resume.

Để benchmark fusion theo Stage 2, chạy namespace mới với `rerank_fusion='rrf'` và
retrieval_share khác; không đổi policy trong một run đã có checkpoints. Labels file
tùy chọn `data/labels/retrieval_reference.json` có schema hướng dẫn trong notebook;
chỉ đánh giá outputs hiện tại. Cutoff sweep cần dev query set độc lập và held-out
validation ở namespace riêng; không lấy 1.200 contest queries làm tập tune.

## Quality holds và giới hạn còn lại

`index/quarantine.json` ghi homepage redirects, error pages và nguồn có encoded
payload bị giữ khỏi retrieval. Đây là hold để review/re-extract, không xóa source,
official IDs hoặc sửa candidate manifest. Docs ngắn/LOW quality không tự bị loại.
Sidebar, title/body mismatch và template placeholders của build cũ có thể vẫn
còn; retrieval tốt hơn không sửa được nguồn crawl/parse sai hoặc thiếu coverage.

Chưa triển khai HyDE/PICO/subqueries, language analyzer/alias glossary, reranker
fine-tune/hard-negative mining, embedding fine-tune hoặc sharded/ANN full-scale.
Những phần này cần ablation/labels/source QA; không gọi “best of Stage1/2” là đã
chứng minh tốt nhất. Pilot vẫn giới hạn 2.000 documents/50.000 children.

## Lưu triển khai v1 và audit artifact thực

Các phần bên dưới mô tả **v1**, giữ để đối chiếu với Colab run Sếp đã chạy;
default v2 mới được mô tả ở trên. Audit v1 không phải phép đo chất lượng v2.

### Baseline retrieval v1 trên Colab

Mở [notebook 04](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_retrieval_baseline.ipynb),
chọn GPU trong Runtime > Change runtime type rồi chạy từ đầu. Bootstrap tự clone
repo, cài extra retrieval và mount Drive. Các notebook data 00–03 không cần chạy
lại khi đã có candidate frozen hợp lệ.

Mặc định `DATA_ROOT=/content/drive/MyDrive/VietMedBridge/data`,
`BUILD_RUN=stage-a-data-v3-laodong`, `CANDIDATE_NAME=candidate-1cd220a4be956d5a.json`,
`RUN_NAME=stage-a-retrieval-v1`. Reader kiểm candidate/config/file checksums,
source spans, aliases và representation hashes bằng tokenizer đã pin. Nó đọc
build cũ, không đòi extractor fingerprint trùng với code retrieval mới.

Candidate đã kiểm có 1.000 input IDs, 864 documents, 136 failures, 9.076 children,
6.906 parents và 8.760 representations dense duy nhất. Embedding chỉ encode
representations duy nhất; output giữ official IDs, kể cả nội dung trùng nhau.
Không gọi 9.076 chunks là 9.076 URLs, hoặc coi integrity pass là relevance pass.

## Các bước và GPU

1. Đọc/check frozen data và query Parquet: CPU, chỉ tải tokenizer.
2. BGE-M3 corpus/query embeddings: **GPU**; normalized CLS vectors, không thêm
   query instruction, không truncate input; budget dense 512 gồm special tokens.
3. BM25 + FAISS IndexFlatIP và RRF: CPU; exact cosine search cho pilot nhỏ.
4. BGE reranker v2 m3: **GPU**; tối đa 40 child pairs/query, raw logits. Model dense
   được giải phóng trước khi nạp reranker. Reranker truncates passage khi pair
   vượt 512 token; final source chunk luôn nguyên vẹn vì chọn từ frozen parents.
5. Parent expansion, validation và ZIP export: CPU.

Hai pretrained models/revisions được khóa trong `configs/retrieval_baseline.json`.
Thực thi qua Transformers; local chỉ kiểm CPU/tokenizer, không tải model weights.
BM25 dùng full title/heading/body của sparse builder, gồm Unicode words và CJK
unigrams/bigrams. Dense search dùng representation builder đã freeze. RRF gộp
top-100 dense/sparse; baseline chọn top-10 docs và tối đa 8 parents, 2 parents/doc.
Các số này chưa tune bằng relevance labels. Document list và chunk list có giới
hạn riêng; không ép documents bằng tập doc IDs của chunks đã chọn.

## Resume và thay đổi cấu hình

Drive lưu `retrieval/<RUN_NAME>/corpus_embeddings`, `query_embeddings`, `index`,
`queries`, `submission`. Mỗi vector part có NPY/checksum và done marker ghi cuối;
part chưa có marker được tính lại. Mỗi query có done JSON/checksum; resume bỏ qua
query đã hoàn tất. Có thể giới hạn MAX_NEW_EMBEDDING_PARTS/MAX_NEW_QUERIES mỗi phiên.
Submission chỉ xuất khi đủ toàn bộ 1.200 queries.

Khi Colab ngắt, mở lại notebook, chọn GPU và chạy từ đầu với cùng RUN_NAME/config.
Có thể giảm batch_size; model cũng tự giảm batch khi CUDA OOM. Các part đã xác
nhận giữ nguyên. Không sửa/xóa marker để bỏ qua checksum lỗi. Nếu thay candidate,
model revision, precision/runtime versions, part_size hoặc retrieval selection,
dùng RUN_NAME mới; thông báo lỗi sẽ chỉ ra checkpoint không còn tương thích.
Không để hai runtime cùng ghi vào một run.

`retrieval_code_lock.json` độc lập `code_lock.json`; runtime evidence cũng tách.
HF weights cache nằm ổ local Colab, nên runtime mới có thể phải tải model lại.
Checkpoint vectors/predictions vẫn ở Drive. Cần đọc code revision mới thì đặt
CODE_REVISION trong Bootstrap và chọn run mới; restart nếu runtime đã import code.

## Submission và giới hạn

`submission/submission.zip` chứa đúng `results.json` ở root. JSON là một array,
mỗi query có integer id, relevant_docs và relevant_chunks; mỗi chunk chỉ có doc_id
và chunk_text. Validator kiểm query coverage, official IDs, child/parent anchors,
source hash và exact source slice. Evidence/model/data hashes lưu riêng trong
submission_manifest.json, không đưa vào ZIP. Notebook không upload lên hệ thống BTC.

Đây là thử đường chạy trên corpus pilot 864 documents, trả kết quả cho đủ 1.200
queries. Corpus nhỏ khiến coverage relevance thấp; chưa có score thật. Schema và
source validation theo plan nội bộ chưa chứng minh BTC nhận/chấm file thành công.
Query dataset chưa có nhãn để tính F2, nên không báo score offline hoặc tự promote.
Chưa fine-tune. Cần kết quả Colab và score/feedback BTC trước khi so sánh baseline,
nhập bản crawl 100k, chọn hard negatives và chuẩn bị train/dev theo query.

Index giữ catalog/vectors/BM25 trong RAM, giới hạn pilot 2.000 documents/50.000
children. Không dùng notebook này để index trực tiếp corpus 100k hoặc 1 triệu URL
khi chưa xây và benchmark sharded/ANN indexing. Lỗi crawl và chất lượng nguồn là
backlog riêng, không bị xóa hoặc thay bằng text do model sinh.

## Kiểm chứng ngày 05/10/2026

- 21 tests pass cho retrieval mới và các regression data/supplement liên quan.
  Kiểm FAISS/BM25 thực trên CPU; encoder/reranker dùng test doubles, không chứng
  minh GPU inference hay chất lượng model.
- Disconnect giữa vector parts, file thiếu marker, resume không encode/rerank lại,
  checksum corruption, sai thứ tự inputs/query và đổi policy đều được kiểm.
- OOM backoff được kiểm bằng exception giả: giảm batch, giữ thứ tự inputs và
  tái sử dụng giới hạn batch an toàn ở part tiếp theo; chưa phải phép đo GPU thật.
- Official aliases giữ đủ, parent output là exact source và docs/chunks có giới
  hạn riêng. ZIP đủ 1.200 query không liên tục được kiểm schema/coverage/source;
  dữ liệu inference trong test là synthetic.
- Reader mới chạy với toàn candidate thật: 864 documents, 9.076 children, 6.906
  parents, 8.760 representations; toàn aliases giữ lại, không rebuild extractor.
  Tokenizer pin thật: dense input dài nhất 291 tokens gồm special tokens.
- Query Parquet tải đúng revision/hash: 1.200 unique official queries, query dài
  nhất 350 tokens gồm special tokens. Smoke test chạy các cell 04 với candidate
  thật, query thật và FAISS/BM25 thật; inference dùng test doubles, chỉ chọn 3
  queries để kiểm parent output/ZIP/resume. Không dùng ZIP test để submit.
- Năm notebook active qua nbformat, syntax và empty-output checks. Các kiểm tra
  local trên dùng test doubles; artifact GPU thực được kiểm riêng bên dưới.

## Kiểm tra kết quả Colab thực ngày 05/10/2026

Sếp đã chạy `stage-a-retrieval-v1` bằng commit `e95e3a1`. ZIP và manifests được
đọc lại từ Drive; mọi phép kiểm dưới đây chạy CPU trên artifact đã tải, không
chạy lại inference hoặc sửa dữ liệu/predictions trên Drive. Prediction signature
`193d6cf647a1cec9fbc5b6fbe2a89acc39f0f5abf28c95af24190f70d627d272`.

- ZIP chỉ chứa `results.json`, CRC và SHA-256 khớp manifest. Đủ 1.200 query
  chính thức theo đúng thứ tự/text hash; 9.593 output chunks đều khớp parent và
  exact source spans của frozen candidate. Không phát hiện lỗi kỹ thuật trong
  các kiểm tra đã thực hiện.
- Đã tải/kiểm toàn bộ 35 corpus vector parts và 5 query parts: lần lượt
  8.760×1.024 và 1.200×1.024, float32 hữu hạn, L2-normalized. Checksum và input
  order khớp; mọi vector reconstruct từ FAISS bằng chính xác corpus matrix.
- Candidate, index, encoder, query vectors, reranker và selection/code hashes
  cùng khớp prediction signature. Kiểm thêm 14 query checkpoints: self-hash,
  prediction, provenance và child anchors đều đúng; tái lập BM25/dense/RRF trên
  CPU cho các query này giữ anchors trong candidate pool 40 representations.
  Chưa đọc self-hash riêng của tất cả 1.200 query checkpoint files.
- Runtime evidence ghi Tesla T4, CUDA fp16 và hai model BGE đã pin, mỗi model
  khoảng 568 triệu parameters; chưa fine-tune. `seconds_this_call` là lần gọi
  cuối có thể dùng lại checkpoint, không phải throughput toàn bộ inference.

**Chất lượng retrieval còn yếu trong mẫu đã xem; chưa đạt relevance/quality
gate.** Đọc query, top document titles và phần đầu các chunks dẫn đầu của 30
query mẫu, kiểm thêm các trường hợp lệch chủ đề. Có ví dụ query 2 hỏi về răng
nhưng kết quả đứng đầu là bài sốt xuất huyết; query 3 hỏi Beta-HCG nhưng kết quả
đứng đầu là bài HCY. Đây là ví dụ mismatch, không phải đánh giá y khoa hoặc
ước lượng Precision/Recall/F2 của toàn bộ 1.200 query.

- 147 chunk occurrences trong 39 query thuộc bốn official IDs mà raw capture
  đã xác nhận redirect từ bài viết sang trang chủ. Source-span pass không xác
  nhận trang chủ là nội dung của bài được yêu cầu.
- 360 chunk occurrences trong 264 query chứa marker liên quan nội dung phụ;
  10 occurrences trong 10 query có template placeholders. Marker là tín hiệu
  audit, không có nghĩa toàn bộ mỗi parent chứa marker đều không liên quan.
- 49 output parent occurrences dài tối đa 10 BGE tokens, ảnh hưởng 38 query.
  1.195/1.200 query trả đủ 8 chunks; fixed top-k chưa được calibrate bằng nhãn.
- Corpus vẫn chỉ có 864 documents. Kiểm literal anchors SCC, Yqh+, PDW và
  CA19-9 không thấy chuỗi tương ứng; đây là diagnostic coverage, không chứng
  minh không có tài liệu liên quan diễn đạt bằng ngôn ngữ hoặc ký hiệu khác.

Giữ run này làm baseline kỹ thuật để so sánh. Ưu tiên tạo build sạch từ raw
hiện có, nhập bản crawl 100k qua mapping official IDs và pipeline chuẩn, đồng
thời bổ sung nhãn/scorer để đo candidate recall và tune cutoff. Corpus 100k
cần sharded/ANN indexing thay cho giới hạn RAM của notebook 04 hiện tại.
LLM translation hoặc reranker mạnh hơn là experiment cần benchmark trên cùng
snapshot; chúng không tự sửa nguồn bị lấy nhầm hay thiếu coverage.
Chưa có official score, chưa PROMOTED và chưa duyệt human QA bằng audit này.

Evidence audit local nằm trong `artifacts/notebook04-quality-20261005/`:
`audit_summary.json`, `review_examples.json`, `audit_submission.py`, vector
parts, ZIP và 14 query checkpoints. Các artifact dữ liệu không được commit.
