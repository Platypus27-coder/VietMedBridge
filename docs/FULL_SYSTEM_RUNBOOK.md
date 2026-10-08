# Vận hành hệ thống theo master plan

Nguồn quyết định: [R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md](../R2AI_STAGE3_FULL_COMPETITION_AND_BEST_OF_STAGE1_STAGE2.md).
Notebook chính giữ nguyên 00–05. 04 và 05 dùng code lock retrieval/training riêng;
không nâng code lock xử lý dữ liệu 02–03 đang chạy dở.

## Chạy ngay với candidate 100k hiện có

[Mở 04 trên Colab](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_retrieval_baseline.ipynb).
Chọn GPU, giữ đúng `DATA_ROOT`, chạy Bootstrap ở runtime mới. Workflow API
`full-master-plan-strong-v8-auto-cumulative-candidates` nâng code lock một lần. Không cần
chạy lại 02–03 cho candidate `ce6987985fb015ca` đã hoàn tất.

Trong cell cấu hình:

- Ba thành viên cùng đặt `TEAM_SIZE=3`; `TEAM_WORKER_ID` lần lượt là `0`, `1`, `2`.
- Các tài khoản phải cùng đọc/ghi **một thư mục dữ liệu được chia sẻ**, không phải
  ba bản sao riêng của `MyDrive/VietMedBridge/data`. Sửa `DATA_ROOT` theo vị trí
  mount thực tế nếu cần. Giữ cùng phiên bản code, model và thư viện.
- Mỗi worker chỉ encode các input parts có `part_index % 3 == worker_id`.
  `MAX_NEW_EMBEDDING_PARTS` có thể giới hạn một phiên; mặc định `None` chạy hết phần được giao.
- Sau khi ba worker xong, một người đặt `TEAM_WORKER_ID=None`, giữ `TEAM_SIZE=3`,
  Run all để tổng hợp và chạy phần còn lại. Nếu thiếu phần, coordinator trả trạng
  thái `WAITING_FOR_*_CORPUS_PARTS`; không công bố index đầy đủ.
- Chỉ một phiên hoạt động trên mỗi worker ID, và một coordinator. Chạy một mình
  thì dùng `TEAM_SIZE=1`, `TEAM_WORKER_ID=None`.

Ba worker chia **corpus embeddings**. Query LLM, tìm kiếm, reranking và QLoRA
hiện do một coordinator chạy; đây không phải distributed QLoRA.

## Những bước 04 thực thi

1. Kiểm candidate, official snapshot, tokenizer, input ordering và SHA-256.
2. Dựng hoặc dùng lại SQLite chứa nguồn, aliases, offsets và BM25 VI/EN/ZH.
3. BGE-M3 child dense và title + source opening document dense; Qwen3-Embedding-8B
   tạo nhánh dense thứ hai. BGE 100k đã chạy được đăng ký vào cache theo nội dung,
   không nhân bản vector; cả baseline của các candidate nguồn khi ghép cũng được tìm lại.
4. Qwen3-4B dịch truy vấn EN/ZH, tạo PICO/subqueries/HyDE có kiểm tra ràng buộc.
   Truy vấn gốc luôn được giữ. Query cache độc lập với candidate corpus.
5. Dense search theo block trên toàn bộ inputs, kết hợp document dense và BM25
   bằng weighted RRF. Giữ tối đa 4.096 child hits mỗi nhánh/query trước khi gom
   official aliases; đây là candidate pool có giới hạn, cần đo recall trên dev.
6. Qwen3-Reranker-8B chấm document, rồi child trong các document được chọn,
   xử lý cửa sổ MaxP. Tạo parent 512/640 từ source gốc khi cần và dedup bằng LCS.
7. Kiểm đủ 1.200 query IDs, official document IDs, source hashes, anchor/parent
   và exact offsets, rồi xuất `submission.zip` có `results.json` tại root.

Mỗi model được kiểm giới hạn 15B riêng, trước quantization và gồm adapter của nó.
Model nạp lần lượt. Không cần answer generation để xuất các source chunks.

Thư mục chính:

- `data/model_cache/`: vector blocks bất biến theo encoder và nội dung.
- `data/retrieval/full_resources/<candidate>/<model-policy>/`: mappings, document
  inputs, checkpoints tìm kiếm, profile riêng coordinator/worker.
- `data/retrieval/full_query_cache/`: query vectors, translations và expansions.
- `data/retrieval/full_system/<run>/`: scored stages, query checkpoints,
  `full_plan_status.json`, `diagnostics.json` và `submission/ready_to_submit.json`.

Chỉ tải ZIP khi trạng thái là `READY_FOR_MANUAL_UPLOAD`. `fine_tuned=false`
nghĩa là đang dùng checkpoint pretrained. Các báo cáo integrity không phải điểm relevance.

## Khi đợt dữ liệu mới về

1. Với batch folder mới, nhập đường dẫn folder một lần ở `EXTERNAL_SOURCE` trong 02;
   notebook tự xử lý mọi archive `.tar` bên trong và checkpoint riêng từng archive.
   Sau đó chạy 03 một lần để kiểm/freeze toàn batch. 03 tự ghi các candidate mới nối
   tiếp candidate đang active vào `data/candidate_lineage.json`; không cần chép hash.
   Giữ snapshot ID/URL BTC, source text, offsets và outcomes lỗi trong các file chuẩn.
2. Coordinator chạy 04 một lần với `TEAM_WORKER_ID=None`. Nếu lineage có nhiều
   candidate, notebook tự ghép chúng, kiểm tra xung đột và chuyển active pointer
   sang corpus tích lũy. Khi coordinator báo `WAITING_FOR_BGE_CORPUS_PARTS`, ba
   worker chạy 04 theo ID 0/1/2. Sau khi đủ phần, coordinator Run all lại để
   tổng hợp và tiếp tục inference. Nếu không dùng chia worker, đặt `TEAM_SIZE=1`;
   Run all của coordinator tự xử lý toàn bộ.
3. Nội dung và encoder không đổi sẽ dùng lại vectors;
   nội dung mới mới cần encode. BM25/mappings được xây cho candidate mới, và truy vấn
   phải xếp hạng lại vì candidate pool đã thay đổi.

`compose_candidates` sao chép các hàng đã xử lý vào build mới; không crawl,
extract hoặc chunk lại. Mọi official alias được giữ. Cùng ID, bản thành công
được ưu tiên trước outcome lỗi. Hai bản thành công khác nội dung/title/language
hoặc source spans sẽ báo xung đột; không âm thầm chọn một bản. Các nguồn phải
cùng official snapshot, tokenizer và chunking policy.

Mỗi output part có checkpoint; active pointer chỉ đổi sau khi toàn bộ union đã
qua integrity, freeze và chuẩn bị input. Sau khi gộp thành công, lineage được thu
gọn về candidate tích lũy duy nhất để batch kế tiếp nối vào đó. Ghép tạo thêm bản processed data, vì vậy
cần tính dung lượng lưu trữ. Raw paths trong provenance vẫn chỉ về nguồn ban đầu;
không xóa raw nguồn nếu cần audit/replay. Các vector được tham chiếu từ baseline
cũ vẫn cần file baseline đó; `model_cache` không phải bản sao lưu vector.

## 05: huấn luyện và lựa chọn có kiểm chứng

[Mở 05 trên Colab](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/05_colab_supervised_training.ipynb).
05 đọc candidate lớn đang active và dùng lại corpus embeddings/index resources
của 04. Không quay về corpus 864 documents khi đang dùng candidate 100k.

Khi chưa có nhãn, source pool có giới hạn và phân bổ theo ngôn ngữ được chọn trên
disk để tạo draft/review; không nạp tất cả source texts vào RAM. Mining sau đó dùng
toàn index. Quy trình là source review → train/dev/held-out độc lập → review hard
negatives → 1 positive + 7 negatives → QLoRA reranker → chọn checkpoint/ngưỡng
doc và chunk theo dev F2 → đánh giá held-out. Dev ablations vẫn có thể chạy trong 05.

Nhãn AI-assisted giữ đúng provenance và chỉ cho local proxy. Không suy nhãn từ
1.200 query thi hoặc lấy câu trả lời tự sinh làm gold. Nếu reference source không
còn trong candidate active, 05 dừng trước GPU với
`WAITING_FOR_LABEL_SOURCE_RECONCILIATION`; ghép lại candidate nguồn hoặc review
nhãn mới phù hợp. Những review/mining run của corpus khác được giữ riêng.

Sau training, `training/stage-a-qlora-v3-per-model-15b/active_adapter.json` trỏ đến
adapter được chọn trong namespace của corpus đó. Chạy lại 04 để nhận adapter.
Khi tăng corpus, adapter hợp lệ có thể được dùng lại nhưng ngưỡng dev cũ không
tự áp dụng; trạng thái là `UNTUNED_ON_CURRENT_CORPUS` cho đến khi đánh giá lại.
Embedding fine-tune là nhánh nghiên cứu có điều kiện của plan, chỉ xem xét sau
khi đo được thiếu candidate recall; nó không tự thay vectors pretrained trong 04.

## Checkpoint và chi phí cần tính

Vectors có checkpoint theo input part. Dense search lưu top-k mỗi tám parts,
nên khi ngắt chỉ mất phần sau checkpoint gần nhất. Translations/expansions,
scored document/child stages, từng query và checkpoint QLoRA đều được lưu.
SQLite catalog được công bố khi hoàn tất; nếu ngắt trong lúc tạo catalog, bước
CPU này sẽ dựng lại. Bootstrap chạy lại sau mỗi runtime mới là bình thường.

Cache vector trên ổ local có giới hạn mặc định 4 GiB mỗi nhánh, ngoài tối đa bốn
blocks đang dùng. Việc loại cache chỉ xóa bản sao local, không xóa vectors trên
Drive. Có thể điều chỉnh `local_vector_cache_bytes_per_branch` trong
`configs/retrieval_scale.json` theo dung lượng máy; cache nhỏ sẽ tăng I/O.
SQLite nguồn, model weights, files tạm và processed data vẫn cần dung lượng riêng.

Số liệu benchmark Sếp đã chạy trên T4: BGE khoảng 46,2 texts/s, Qwen embedding
8B khoảng 6,65 texts/s; ngoại suy child-only cho 573.854 inputs tương ứng khoảng
3,45 giờ và 24 giờ. BGE cũ được dùng lại khi identity khớp. Các số này từ mẫu
64 texts, chưa gồm I/O, document dense, query LLM hay reranking, và không bảo đảm
thời gian. Corpus vài triệu URL vẫn cần ngân sách GPU và storage tương ứng;
checkpoint không loại bỏ phần tính toán cho dữ liệu mới.

Baseline 100k Sếp đã submit đạt 0,0081. Bản full system mới chưa có điểm BTC hoặc
benchmark GPU toàn corpus; không coi kiểm thử CPU là bằng chứng tăng điểm.
