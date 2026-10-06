# Quyết định kỹ thuật của lớp data

## Adapt kỹ thuật từ hai repo tham chiếu

Repo Stage 1 đang ở commit `7aa3955569371c56016744e1dd758b0a656c0c89`.
[legal_chunker.py](https://github.com/aigurutinix/r2ai-stage-1/blob/7aa3955569371c56016744e1dd758b0a656c0c89/nguyenvannghiem/src/code/vbpl_dataset/legal_chunker.py)
chunk theo Điều/Article và mang hierarchy/citation metadata. VietMedBridge giữ
nguyên tắc ưu tiên cấu trúc và provenance, dùng heading y sinh từ HTML/JATS,
không mang regex Điều/Chương hay citation keys pháp luật sang corpus mới.

Repo Stage 2 đang ở commit `2bbc3a43cb29bfaf8aa53da15be5699b4af1b8d2`.
[chunked_corpus_builder.py](https://github.com/aigurutinix/r2ai-stage-2/blob/2bbc3a43cb29bfaf8aa53da15be5699b4af1b8d2/vilamiu/src/vifinqa-official/src/vifinqa/retrieval/chunked_corpus_builder.py)
giữ canonical parent table refs cho dense row chunks.
[row_chunks.py](https://github.com/aigurutinix/r2ai-stage-2/blob/2bbc3a43cb29bfaf8aa53da15be5699b4af1b8d2/vilamiu/src/vifinqa-official/src/vifinqa/encoding/row_chunks.py)
tách representation/config namespace và thống kê safety split/truncation.
VietMedBridge dùng explicit child→parent/source mapping và versioned text builder,
nhưng reference là official doc_id cùng exact character spans. Financial row/ticker
schema, generated semantic labels và pathological cell truncation không phải
submission evidence phù hợp cho Stage 3.

Hai repo là nguồn kỹ thuật; không cài chúng thành dependency hoặc chỉnh sửa chúng.
Master plan Stage 3 quyết định contract/metric, supplement bổ sung QA và scale gates.

## Retrieval cascade v2 — 06/10/2026

Đối chiếu implementation, không chỉ README:

- Stage 1 MSC-AI `services/vector_store/hybrid.py`, `index_builder.py`: weighted
  rank fusion, source/model/tokenizer cache manifests. Stage 1 Nguyễn Văn Nghiêm
  `retrieval_bm25s.py`: BM25 với Lucene-style IDF và top-k/threshold sweep.
  Stage 1 Trần Thanh Tú `retrieval_cutoff.py`: absolute floor và relative margin.
  V2 dùng positive-IDF postings BM25, weighted RRF và optional raw-score cutoffs;
  không sao chép synthetic legal QA/Điều/citation rules sang y sinh.
- Stage 2 Vilamiu `retrieval/cascade_reranked.py`: summary → detail cascade,
  equal detail quota cho mỗi parent. V2 dùng title + top-2 retrieval-hit passages
  để rerank documents, quota 5 children cho mỗi document trong top-30.
- Stage 2 ARCANE `fusion.py`, `rerank.py`,
  `scripts/validate_rerank_scores.py`: shared representation, cùng scored pool
  cho ablation, kiểm candidate-score completeness và pair/model/prompt hashes.
  V2 giữ toàn bộ document/child pair inputs + scores, checksum từng stage;
  có `rerank_fusion='rrf'` để benchmark so với raw reranker order.
  Không coi fusion thắng ở table retrieval là bằng chứng nó sẽ thắng Stage 3.

Master plan §§7.5–7.14, §26 và §55 quyết định adaptation: original multilingual
dense + additive conservative EN/ZH translations, weighted document retrieval,
document → child → parent, same-doc token LCS/union >=0.8 dedup. Sliding MaxP
tránh mất bằng chứng ở tail khi query dài. Qwen3-4B chỉ dịch query; không tạo
source/submission text. Model pin/public/date/parameter evidence được ghi trong
`configs/query_translation_model_manifest.json`; inference thực cần Colab.

Scorer local áp dụng F2 theo từng query rồi macro, relevant chunk phải cùng doc
và LCS >=40% reference BGE tokens. Đây là proxy từ plan; thiếu normalization/
implementation chính thức của BTC nên không tự gán nhãn “official compatible”.
Cutoff calibration chỉ nhận reviewed dev labels, không tune trên contest/test,
không tự đổi submission bằng best dev result. HyDE, PICO/subqueries, reranker/
embedding fine-tune và full-scale ANN vẫn cần dữ liệu/ablation ở bước sau.

## End-to-end trên candidate hiện có

Master plan `R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md` là nguồn
chính. §§26/55 cho phép bắt đầu bằng pretrained baseline; fine-tune/hard negatives
ở giai đoạn có train/dev gold đủ. Theo yêu cầu lấy điểm sớm, notebook 04 mặc định
chạy hết 1.200 official queries trên candidate hiện có rồi xuất submission ZIP;
không chờ full corpus, import 100k, reviewed labels hay canary bắt buộc.

Run contract gắn hash plan/data/queries/config với run. Export và readiness được
ghi trước đánh giá proxy tùy chọn để file nhãn thiếu/lỗi không chặn pilot.
ZIP có metadata ổn định khi export lại cùng predictions; score feedback gắn ZIP
hash và giữ nguyên khi resume. Chỉ ghi score thật sau upload Dashboard BTC;
không coi schema/source validation là đã đạt relevance hoặc promote corpus.

## Tách source và search formatting

Normalization hoặc thêm title/heading giúp search nhưng thay đổi chuỗi nguồn.
Vì vậy source_text/source hash/spans giữ riêng; retrieval representation có builder
version/hash. Parent-only experiment không đổi child ID và embedding input.
Representation alias chỉ cho phép reuse sau khi encoding policy/model cũng khớp.

## Data freeze và release promotion

Index cần một input snapshot cố định để so sánh experiments. Nếu chờ index xong
mới freeze thì không xác định rõ dữ liệu benchmark. FROZEN_CANDIDATE tạo trước,
PROMOTED cần retrieval/resource/human evidence sau. Pipeline data không giả lập
release promotion khi chưa có index và nhãn đánh giá.

## Coverage và dedup

URL/content duplicate không làm official doc_id interchangeable tự động. Mọi ID
giữ source/outcome; aliases dùng cho grouping index sau này. Integrity checks xét
toàn selected snapshot và bảo toàn input/outcomes. Milestone xét requested range,
tránh duyệt một run bị ngắt chỉ vì mọi shard đã có đều build thành công.
