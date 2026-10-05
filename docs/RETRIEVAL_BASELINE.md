# Baseline retrieval trên Colab

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
- Năm notebook active qua nbformat, syntax và empty-output checks. GPU model
  execution, OOM thực, tốc độ Colab, fine-tuning và official score chưa được đo.
