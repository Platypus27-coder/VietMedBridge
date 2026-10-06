# Chạy kiến trúc master plan trên corpus hiện có

Nguồn quyết định là [master plan Stage 3](../R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md).
[Notebook 04](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_retrieval_baseline.ipynb)
dùng `configs/retrieval_full.json`, namespace `stage-a-full-plan-v3-per-model-15b`. Tên file
giữ nguyên để link cũ mở đúng notebook chính. Chọn runtime GPU mới rồi **Run all**;
không cần ACTION hoặc chạy lại 00–03. Bootstrap clone repo, cài extras và nâng
code lock một lần sang `full-master-plan-strong-v3-per-model-15b`.

## Giới hạn từng model đã được Sếp xác nhận với BTC

Ngày 06/10/2026, Sếp xác nhận sau khi hỏi BTC: giới hạn **≤15B áp dụng cho từng
model**, không phải tổng. Bộ model chính được khôi phục:

- BGE-M3: 567.754.752 parameters.
- Qwen3-Embedding-8B: 7.567.295.488 parameters, 4.096 dimensions.
- Qwen3-4B-Instruct-2507: 4.022.468.096 parameters.
- Qwen3-Reranker-8B: 8.188.548.096 parameters.

Tổng pretrained **20.346.066.432** chỉ là inventory. Model lớn nhất là reranker
8.188.548.096 parameters, dưới 15B. BM25/FAISS không thêm neural parameters.
Gate kiểm từng checkpoint và cộng adapter vào base model tương ứng, trước
quantization; embedding adapter không tiêu budget của reranker. Header/hash
safetensors được kiểm mà không load weights. Báo cáo giữ counts từng model
và tổng inventory ở `model_parameter_budget.json`. Xác nhận của Sếp chỉ giải
quyết cách tính giới hạn; không ghi rằng BTC đã duyệt cả bộ checkpoint/license.

Run v3 tách khỏi v1/v2. Giữ DATA_ROOT và candidate; không trộn vectors 0.6B/8B
hoặc thay provenance. BGE vectors cũ vẫn được kiểm để reuse. Hai model 8B dùng
NF4, nạp lần lượt; tài nguyên và chất lượng vẫn cần đo trên GPU/dev.

Input là candidate `candidate-1cd220a4be956d5a.json` trong build
`stage-a-data-v3-laodong`: 864 documents, 9.076 children, 8.760 unique dense
representations. Quarantine giữ 9 documents khỏi retrieval và giữ nguyên ledger.
Corpus pilot chưa gồm crawl 100k và chưa đủ toàn dataset.

## Luồng inference và vai trò model

1. **BGE-M3** encode original queries/child representations. Cache Colab cũ chỉ
   reuse khi exact inputs/order/model/budget/pooling/code và hashes khớp.
   Có nhánh document dense riêng: title + bounded source opening, FAISS theo
   official doc IDs. Khi nguồn không có abstract riêng, opening không được
   gọi là abstract đã trích. Representations fit 512 tokens kể cả special tokens;
   chúng chỉ dùng search, output vẫn lấy source parents.
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

Embedding/reranker 8B dùng NF4 4-bit, SDPA, batch2;
nạp/giải phóng lần lượt. T4 là
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

## Tạo draft, duyệt nhãn và supervised workflow

Sếp xác nhận hiện **chưa có reviewed train/dev labels**. 04 chạy pretrained và
cutoffs chưa tune. [Notebook 05](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/05_colab_supervised_training.ipynb)
đã có luồng tạo draft và review theo master §38 trong cùng notebook, không thêm
notebook/ACTION:

1. Đọc frozen candidate, loại source quarantine; gom source/đoạn trùng rồi tách
   nhóm train/dev/held-out trước khi sinh câu hỏi. Chọn tối đa 256 train/40 dev/
   40 held-out sources,
   giới hạn hai children/document; sample cân đối ngôn ngữ khi có nguồn.
2. Qwen 4B hiện có tạo **draft VI train query**, có quote nguyên văn và checkpoint
   từng mẫu. Không thêm teacher model mới. Quote/schema/contest overlap checks
   chỉ là kiểm bề mặt, không chứng nhận clinical faithfulness.
3. `source_review.json` chứa source spans, hashes, draft và review fields. Dev/held-out
   query để trống để người review viết độc lập. Team đánh ACCEPT/REJECT, reviewer,
   query và exact quote; dev/held-out đánh `independently_written=true` sau khi tự viết.
   Giữ mọi source fields; lưu file về đúng `review_path` trên Drive và Run all.
   Tối thiểu 32 train/8 dev/8 held-out được accept, hết PENDING mới xuất labels
   trong chế độ người duyệt mặc định.
   Sếp đã giao Codex duyệt pilot ngày 06/10/2026: bản được cấp phép có
   `review_mode=AI_ASSISTED_PILOT`, reviewer/author type là AI. Chế độ này xuất
   riêng các mẫu ACCEPT khi đủ tối thiểu; PENDING được giữ để làm sau, không
   thành nhãn âm. Labels ghi `AI_REVIEWED_PILOT_*`, `human_validated=false`
   và `SOURCE_DISJOINT_AI_LABELS_LOCAL_PROXY`. Không đặt
   `independently_written=true` cho câu hỏi AI. Các kiểm tra nguồn, hash,
   source split, contest overlap và minimum counts vẫn áp dụng.
   Notebook 05 v5 nhận `/content/source_review_ai_pilot.json`; helper import
   giữ backup bản Drive, kiểm mọi source/draft fields và từ chối ghi đè các
   quyết định đã được team sửa. Kết quả dev/held-out AI là proxy pilot, cần
   human audit trước khi dùng để khẳng định chất lượng hoặc promote corpus.
4. Workflow chạy hybrid candidates. Nếu chưa có đủ positive/7 reviewed negatives
   cho từng query, xuất `candidate_reviews/<hash>/review.json`. Team đánh
   POSITIVE/NEGATIVE/SKIP + reviewer/category; unknown vẫn PENDING. Source spans,
   query/hash/ID coverage và positive-negative conflicts được kiểm khi nhập lại.
   Additional positives không vượt nhóm source đã reserve cho từng split.
5. Run all sau khi lưu review về Drive: import judgments, reuse verified vectors
   và scored stages. Training chưa chạy khi còn mining holds. Groups-ready mới
   vào QLoRA, dev NDCG/MRR và finalist F2 selection. Doc/chunk margins được sweep
   độc lập. Namespace tách theo label version. Không đưa held-out queries/labels vào mining,
   loss, dev sweep hoặc checkpoint selection; chỉ encode các query để inference.
6. Tự chạy tám dev ablations, lưu báo cáo nhưng không tự promote variant. Chấm
   held-out sau khi adapter/cutoff đã freeze; giữ calibration/adapter/label hashes.
   Không dùng held-out score để chọn lại. 04 đọc adapter/dev policy đã chọn.

Labels train/dev có `reviewed=true` chỉ sau explicit review. Source-derived
synthetic train và independently authored dev có `label_kind`/reviewer/provenance
riêng; đây không phải nhãn BTC. Không tự bật `exhaustive_chunks=true` hoặc coi
teacher/negative scores là gold. Tất cả các bước dừng review là input cần team
thực hiện, không phải inference/training đã được chứng minh. Chưa có nhãn thật,
GPU QLoRA hay held-out score thật/official score ở local.

`configs/training_data.json` là policy chuẩn bị draft; nó không thay đổi frozen
candidate hoặc cấu hình inference 04. 05 nâng code lock sang
`full-master-plan-supervised-v4-per-model-15b`; 04 giữ `full-master-plan-strong-v3-per-model-15b`.
API `run_training_workflow` gọi trực tiếp khi thiếu file vẫn trả
`WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS`; notebook gọi preparation trước.
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

Lưu vào `data/labels/retrieval_train.json`; dev cùng schema với `split=dev`,
held-out ở `retrieval_heldout.json` với `split=heldout`.
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
full retrieval + F2 cutoff sweep trên reviewed dev chọn adapter/policy.
Held-out thiếu thì báo pending; có reviewed held-out thì đánh giá frozen policy
và lưu `heldout_evaluation.json`. 04 tự đọc `training/stage-a-qlora-v3-per-model-15b/selected_adapter`
và `calibrated_policy.json`, kiểm corpus/analyzer/model/scorer/query hashes,
tạo namespace `-ft-` mới.

## Các thí nghiệm có điều kiện

- `retrieval_diagnostics.run_dev_ablations(...)`: tám cấu hình full, bỏ document
  dense, bỏ second dense, bỏ translated sparse, bỏ HyDE, bỏ subqueries, parent512,
  rerank RRF. Notebook 05 gọi tự động sau khi chọn adapter; báo cáo chưa auto-apply.
  Nhận index, vectors/manifest, translations, reranker, config, reviewed dev và
  contest queries để chặn tuning trên test; lưu score stages và cutoff sweeps.
- `embedding_training.train_embedding_qlora(...)`: chỉ nhận verified independent
  mining bundles và report có measured candidate-recall bottleneck. Contrastive
  QLoRA dùng một positive/bảy negatives. Nạp adapter bằng
  `TorchQwenEncoder(spec, adapter_path=...)`; encode lại corpus/query dưới model
  identity mới. Không tự áp adapter vào 04 trước retrieval/F2 benchmark.
  API training nhận `model_budget=model_budget_report(config, registry, ...)`
  gồm mọi adapter đang dùng. Embedding QLoRA cần spec NF4 trong một run mới;
  inference embedding 8B mặc định NF4.
- `sharded_dense.build_dense_shards(...)`: verified NPY parts qua mmap, giới hạn
  rows mỗi FAISS shard, checksum/done để resume. `ShardedDenseIndex.search(...)`
  mở từng shard, gộp global top-k. Đây là exact-search toolkit có kiểm thử CPU;
  chưa là full-scale backend của 04. 04 giới hạn 2.000 documents/50.000 children.
  ANN và budget benchmark tại 100k/1M cần đo sau khi import crawl qua data pipeline.

Master §§7/8/13/17/27/39/56 có module cho kiến trúc/supervised/ablation trên đây;
§§26/55 giữ baseline làm comparator. Relevance, glossary, fine-tune thực,
hyperparameters, scale benchmark và official score cần dữ liệu/đo đạc tương ứng.
Full inference code chưa phải cấu hình đã tối ưu.
