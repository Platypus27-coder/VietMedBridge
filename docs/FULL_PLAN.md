# Chạy kiến trúc master plan trên corpus hiện có

Nguồn quyết định là [master plan Stage 3](../R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md).
[Notebook 04](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_retrieval_baseline.ipynb)
dùng `configs/retrieval_full.json`, namespace `stage-a-full-plan-v1`. Tên file
giữ nguyên để link cũ mở đúng notebook chính. Chọn runtime GPU mới rồi **Run all**;
không cần ACTION hoặc chạy lại 00–03. Bootstrap clone repo, cài extras và nâng
code lock một lần sang `full-master-plan-strong-v1`.

Input là candidate `candidate-1cd220a4be956d5a.json` trong build
`stage-a-data-v3-laodong`: 864 documents, 9.076 children, 8.760 unique dense
representations. Quarantine giữ 9 documents khỏi retrieval và giữ nguyên ledger.
Corpus pilot chưa gồm crawl 100k và chưa đủ toàn dataset.

## Luồng inference và vai trò model

1. **BGE-M3** encode original queries/child representations. Cache Colab cũ chỉ
   reuse khi exact inputs/order/model/budget/pooling/code và hashes khớp.
2. **Qwen3-4B-Instruct-2507** dịch query sang EN/ZH. Với câu hỏi phức tạp, model
   có thể thêm PICO-lite, tối đa hai subqueries và HyDE ngắn. Original query
   luôn giữ; nhánh sai schema/số/Latin entity/comparator/constraint cue bị loại.
   PICO chỉ nhận substrings câu hỏi gốc. Surface checks chưa chứng nhận ngữ nghĩa
   dịch/HyDE; cần reviewed dev ablation.
3. **Qwen3-Embedding-8B** tạo nhánh dense thứ hai. Query dùng instruction,
   corpus không dùng instruction; last attended token được L2-normalize.
   Original/accepted subqueries/HyDE là các nhánh additive.
4. **BM25 VI/EN/ZH** dùng PyVI, Jieba, original Latin/acronym tokens, CJK
   n-grams; title/heading/body/alias fields có soft boosts. Glossary tùy chọn
   cần `data/labels/medical_aliases.json`, `reviewed=true`, entries chứa aliases
   và nguồn. Chưa có glossary thì alias field rỗng, không giả lập ontology.
5. Weighted RRF gộp candidates; **Qwen3-Reranker-8B** chấm title và hai child
   hits với budget chia đều, giữ top-30 documents. Local dual-dense + sparse
   lấy tối đa năm children/document; Qwen tiếp tục child rerank. Score là raw
   `yes-logit − no-logit` theo prompt model, không phải xác suất. MaxP giữ nguyên
   query và tail passage, không truncate ngầm.
6. Chọn source parents 512/640 BGE tokens theo độ phức tạp query; kiểm số token
   của slice sau khi cắt, giữ trọn anchor child. Parents derive từ source frozen
   trong bộ nhớ, không rewrite candidate cũ. Same-document LCS dedup và doc/chunk
   cutoffs áp dụng trước export.
7. Validate exact source spans/hashes/IDs và đủ **1.200 official queries**;
   ZIP chỉ chứa `results.json` ở root. Manifest/runtime/diagnostics nằm ngoài ZIP.
   Upload lên BTC để lấy score; không tạo score trước upload.

Stage 3 xuất IDs và bằng chứng nguồn, nên không có model tạo câu trả lời trong
submission. Model lớn đảm nhiệm embedding/reranking. Qwen revisions, parameter
counts trước quantization, release/license evidence nằm trong
`configs/strong_model_manifest.json`; đây là kiểm nội bộ theo plan, không phải
xác nhận BTC đã duyệt.

## GPU và checkpoint

Hai model 8B mặc định NF4 4-bit, SDPA, batch2; nạp/giải phóng lần lượt. T4 là
cấu hình mục tiêu, L4/A100 có dư địa hơn. Chưa đo GPU mới nên không cam kết thời
gian hoặc NF4 có chất lượng bằng fp16. Không tải weights local.
Weights/HF cache ở `/content/hf_cache`, vector parts/checkpoints ở Drive.

`MAX_NEW_EMBEDDING_PARTS`, `MAX_NEW_TRANSLATIONS`, `MAX_NEW_QUERIES` mặc định
`None`. Khi ngắt, mở runtime GPU mới và Run all cùng DATA_ROOT/RUN_NAME.
Vector parts có checksum/done; translation/expansion lưu từng query;
document/child rerank lưu pairs + scores; query cuối có marker riêng.
Ngắt giữa scored stage thì stage đó chạy lại; stages hoàn tất được reuse.
OOM giảm batch; batch1 vẫn thiếu memory thì giữ checkpoints và báo lỗi.
Không chạy hai runtime ghi cùng run. Đổi data/model/policy/code cần run mới.

`diagnostics.json` ghi languages, branch counts, output counts, raw-score
percentiles. `full_plan_status.json` phân biệt inference/fine-tune/calibration/
coverage. Source/schema pass chưa chứng minh relevance.

## Supervised workflow khi có nhãn

Sếp xác nhận hiện **chưa có reviewed train/dev labels**. 04 chạy pretrained và
cutoffs chưa tune. [Notebook 05](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/05_colab_supervised_training.ipynb)
chỉ dùng khi có hai file dưới đây; thiếu file trả
`WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS` trước khi nạp model.
Không lấy predictions của 1.200 contest queries làm gold.

```json
{
  "reviewed": true,
  "split": "train",
  "exhaustive_chunks": false,
  "queries": [{
    "id": 100001,
    "query": "Câu hỏi độc lập do team tạo và review",
    "relevant_docs": [583],
    "relevant_chunks": [{"doc_id": 583, "chunk_text": "Slice chính xác từ source frozen"}],
    "negative_child_ids": ["child-ID-đã-review"],
    "negative_categories": {"child-ID-đã-review": "same_document_wrong_chunk"}
  }]
}
```

Lưu vào `data/labels/retrieval_train.json`; dev cùng schema nhưng `split=dev`.
Đây là ví dụ schema, không phải gold dùng train. ID và nội dung train/dev không
trùng nhau hoặc contest queries. Chỉ bật `exhaustive_chunks=true` khi annotation
đầy đủ; nếu không, negatives phải nằm trong reviewed list.

`training_workflow.py` mine full cascade, tạo group 1 positive + 7 negatives.
Gold là source slices; training positive cần same-document gold bao phủ ≥80%
child tokens, riêng với scorer F2 proxy. Quotas nguyên 2/2/1/1/1 xấp xỉ taxonomy
25/25/20/15/15 trong plan; category thiếu ghi rõ unclassified, không tự phán đoán
y khoa. Gold ngoài pool/thiếu negatives vào holds, không thành negative giả.

QLoRA reranker rank16, weighted BCE yes/no logits, gradient checkpointing,
batch1/accumulation8. Epoch checkpoints giữ adapter/optimizer/scheduler/RNG và
hash manifest; resume kiểm data/config/files. Dev NDCG/MRR chọn finalist, rồi
full retrieval + F2 cutoff sweep trên reviewed dev chọn adapter/policy. Trạng
thái vẫn held-out pending. 04 tự đọc `training/stage-a-qlora-v1/selected_adapter`
và `calibrated_policy.json`, kiểm corpus/analyzer/model/scorer/query hashes,
tạo namespace `-ft-` mới.

## Các thí nghiệm có điều kiện

- `retrieval_diagnostics.run_dev_ablations(...)`: bảy cấu hình full, bỏ second
  dense, bỏ translated sparse, bỏ HyDE, bỏ subqueries, parent512, rerank RRF.
  Nhận index, vectors/manifest, translations, reranker, config, reviewed dev và
  contest queries để chặn tuning trên test; lưu score stages và cutoff sweeps.
- `embedding_training.train_embedding_qlora(...)`: chỉ nhận verified independent
  mining bundles và report có measured candidate-recall bottleneck. Contrastive
  QLoRA dùng một positive/bảy negatives. Nạp adapter bằng
  `TorchQwenEncoder(spec, adapter_path=...)`; encode lại corpus/query dưới model
  identity mới. Không tự áp adapter vào 04 trước retrieval/F2 benchmark.
- `sharded_dense.build_dense_shards(...)`: verified NPY parts qua mmap, giới hạn
  rows mỗi FAISS shard, checksum/done để resume. `ShardedDenseIndex.search(...)`
  mở từng shard, gộp global top-k. Đây là exact-search toolkit có kiểm thử CPU;
  chưa là full-scale backend của 04. 04 giới hạn 2.000 documents/50.000 children.
  ANN và budget benchmark tại 100k/1M cần đo sau khi import crawl qua data pipeline.

Master §§7/8/13/17/27/39/56 có module cho kiến trúc/supervised/ablation trên đây;
§§26/55 giữ baseline làm comparator. Relevance, glossary, fine-tune thực,
hyperparameters, scale benchmark và official score cần dữ liệu/đo đạc tương ứng.
Full inference code chưa phải cấu hình đã tối ưu.
