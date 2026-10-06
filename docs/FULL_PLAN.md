# Chạy kiến trúc master plan trên corpus hiện có

Nguồn quyết định là [master plan Stage 3](../R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md).
[Notebook 04](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_retrieval_baseline.ipynb)
dùng `configs/retrieval_full.json`, namespace `stage-a-full-plan-v2-15b`. Tên file
giữ nguyên để link cũ mở đúng notebook chính. Chọn runtime GPU mới rồi **Run all**;
không cần ACTION hoặc chạy lại 00–03. Bootstrap clone repo, cài extras và nâng
code lock một lần sang `full-master-plan-strong-v2-15b`.

## Giới hạn tổng model theo yêu cầu Sếp

Bộ model v1 có tổng **20.346.066.432 parameters**; gate cũ chỉ kiểm từng
model dưới 15B. Câu chữ trong plan chưa xác định giới hạn theo từng model hay
toàn hệ thống; chưa có xác nhận BTC về cách cộng tổng. Theo yêu cầu Sếp,
v2 áp **tổng mọi model đã dùng ≤15B**, trước quantization, kể cả nạp lần lượt:

- BGE-M3: 567.754.752 parameters.
- Qwen3-Embedding-0.6B: 595.776.512 parameters, 1.024 dimensions.
- Qwen3-4B-Instruct-2507: 4.022.468.096 parameters.
- Qwen3-Reranker-8B: 8.188.548.096 parameters.

Tổng pretrained **13.374.547.456**; dư **1.625.452.544** cho adapter.
BM25/FAISS không thêm neural-model parameters. Adapter không merge được
đếm từ shapes trong safetensors đã kiểm hash; training kiểm tham số LoRA
ngay sau tạo model. Gate chặn tổng vượt 15B trước chạy inference/training;
báo cáo nằm tại `model_parameter_budget.json` và submission evidence.
NF4/FP16 và giải phóng VRAM không làm thay đổi tổng tham số này.

Run v2 tách khỏi v1. Giữ DATA_ROOT và candidate hiện có; Qwen vectors 8B
không được reuse cho 0.6B. BGE vectors cũ vẫn được kiểm để reuse.
Đổi model có thể thay đổi chất lượng truy hồi; cần GPU/dev measurements.

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
3. **Qwen3-Embedding-0.6B** tạo nhánh dense thứ hai. Query dùng instruction,
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

Embedding 0.6B dùng fp16; reranker 8B dùng NF4 4-bit, SDPA, batch2;
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
   nhóm train/dev trước khi sinh câu hỏi. Chọn tối đa 256 train/40 dev sources,
   giới hạn hai children/document; sample cân đối ngôn ngữ khi có nguồn.
2. Qwen 4B hiện có tạo **draft VI train query**, có quote nguyên văn và checkpoint
   từng mẫu. Không thêm model vào bộ 13.375B. Quote/schema/contest overlap checks
   chỉ là kiểm bề mặt, không chứng nhận clinical faithfulness.
3. `source_review.json` chứa source spans, hashes, draft và review fields. Dev
   query để trống để người review viết độc lập. Team đánh ACCEPT/REJECT, reviewer,
   query và exact quote; dev đánh `independently_written=true` sau khi tự viết.
   Giữ mọi source fields; lưu file về đúng `review_path` trên Drive và Run all.
   Tối thiểu 32 train/8 dev được accept và hết PENDING mới xuất hai labels files.
4. Workflow chạy hybrid candidates. Nếu chưa có đủ positive/7 reviewed negatives
   cho từng query, xuất `candidate_reviews/<hash>/review.json`. Team đánh
   POSITIVE/NEGATIVE/SKIP + reviewer/category; unknown vẫn PENDING. Source spans,
   query/hash/ID coverage và positive-negative conflicts được kiểm khi nhập lại.
   Additional positives không vượt nhóm source đã reserve cho từng split.
5. Run all sau khi lưu review về Drive: import judgments, reuse verified vectors
   và scored stages. Training chưa chạy khi còn mining holds. Groups-ready mới
   vào QLoRA, dev NDCG/MRR và finalist F2 selection; checkpoint namespace tách
   theo label version. 04 đọc adapter/dev policy đã chọn như trước.

Labels train/dev có `reviewed=true` chỉ sau explicit review. Source-derived
synthetic train và independently authored dev có `label_kind`/reviewer/provenance
riêng; đây không phải nhãn BTC. Không tự bật `exhaustive_chunks=true` hoặc coi
teacher/negative scores là gold. Tất cả các bước dừng review là input cần team
thực hiện, không phải inference/training đã được chứng minh. Chưa có nhãn thật,
GPU QLoRA hay independent held-out test/official score ở local.

`configs/training_data.json` là policy chuẩn bị draft; nó không thay đổi frozen
candidate hoặc cấu hình inference 04. 05 nâng code lock sang
`full-master-plan-supervised-v3`; 04 giữ `full-master-plan-strong-v2-15b`.
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
thái vẫn held-out pending. 04 tự đọc `training/stage-a-qlora-v2-15b/selected_adapter`
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
  API training nhận `model_budget=model_budget_report(config, registry, ...)`
  gồm mọi adapter đang dùng. Embedding QLoRA cần spec NF4 trong một run mới;
  inference 0.6B mặc định fp16.
- `sharded_dense.build_dense_shards(...)`: verified NPY parts qua mmap, giới hạn
  rows mỗi FAISS shard, checksum/done để resume. `ShardedDenseIndex.search(...)`
  mở từng shard, gộp global top-k. Đây là exact-search toolkit có kiểm thử CPU;
  chưa là full-scale backend của 04. 04 giới hạn 2.000 documents/50.000 children.
  ANN và budget benchmark tại 100k/1M cần đo sau khi import crawl qua data pipeline.

Master §§7/8/13/17/27/39/56 có module cho kiến trúc/supervised/ablation trên đây;
§§26/55 giữ baseline làm comparator. Relevance, glossary, fine-tune thực,
hyperparameters, scale benchmark và official score cần dữ liệu/đo đạc tương ứng.
Full inference code chưa phải cấu hình đã tối ưu.
