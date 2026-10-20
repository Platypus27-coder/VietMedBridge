# Nộp bài từ dữ liệu hiện có, giữ URL cũ

Mở [04_colab_submit_available_data.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_submit_available_data.ipynb).
Đây là notebook dùng cho lượt nộp hiện tại, thay cho việc cấu hình nhiều chế độ trong notebook 04 cũ.
Notebook v9 lấy bản tối ưu bằng code lock riêng khi mở bản mới; giữ nguyên candidate, runtime model
và checkpoint dữ liệu/vector/query. Code lock v8 vẫn giữ để phiên đang chạy không bị đổi code.
Không dừng phiên đang chạy để nâng cấp. Lần resume sau dùng notebook mới cùng DATA_ROOT;
CPU preparation đã COMPLETE được khôi phục từ report, không dựng lại catalog.

Theo yêu cầu ngày 14/10, giữ các URL thuộc snapshot nguồn ban đầu, kể cả URL đã bị BTC loại trong dataset mới.
Không áp danh sách `01_already_chunked` của audit dataset mới làm bộ lọc. Không đổi ID, không xóa source,
chunk, raw archive hoặc checkpoint cũ. ZIP của lượt này được kiểm tra với **snapshot nguồn ban đầu**;
chưa được chứng nhận phù hợp danh sách ID của dataset BTC mới. Điểm chỉ biết sau khi BTC chấm.

## Cách chạy

1. Khi resume, chọn Colab **T4 GPU → Run all**, giữ `ALLOW_CPU_PREPARATION=False` ở cell 2.
   Cell 2 chỉ đọc manifest/báo cáo JSON, kiểm đúng candidate/config/snapshot/source và khôi phục handoff.
   Không mở lại corpus Parquet, vector NPY, model cache hay catalog/BM25 trong cell này.
   Nếu báo cáo thiếu/không khớp, báo `RESUME_ONLY_STOP` và dừng, không tự chạy preparation.
   Chỉ khi chủ động chuẩn bị dữ liệu mới, bật `ALLOW_CPU_PREPARATION=True` và dùng CPU trước.
   Giữ `DATA_ROOT_PATH` trỏ đến data dùng chung.
   Notebook tự nhận candidate đang active, các nguồn của candidate đã ghép, batch hoàn tất gần nhất
   và pilot `stage-a-data-v5-clean-text` nếu còn. Nguồn phải hoàn tất, cùng snapshot gốc và cùng chính sách chunk/tokenizer.
2. Chờ `RESUME_READY` và `CPU_PREPARATION_COMPLETE`. Output gọn ghi build/candidate, số document và unique dense input;
   danh sách source build và số cache tại lúc chuẩn bị vẫn có trong report trên Drive. Đây là số đo của lượt ghép,
   không suy từ số tar hoặc từ tổng số URL đã từng thử crawl.
3. Nếu đã chuẩn bị bằng CPU, tắt `ALLOW_CPU_PREPARATION`, đổi runtime sang **T4 GPU** rồi **Run all cùng notebook**. Không cần đổi ACTION, worker ID hoặc tài khoản Kaggle.
   GPU xác minh CPU preparation, code/model producer/runtime trước khi nạp weights.
   Chế độ chỉ đọc JSON áp dụng cho cell 2; cell 3–4 vẫn kiểm tra payload checkpoint khi thực sự sử dụng.
4. Cell 3 bổ sung **BGE còn thiếu** và xuất ZIP BGE/BM25 đủ 1.200 query. ZIP được giữ trên Drive trước khi nạp LLM
   cho cell 4. Đây là bản đầu để lấy phản hồi từ BTC, chưa có dịch, document dense hoặc neural reranking.
5. Cell 4 tiếp tục bản nâng cấp: dịch EN/ZH, PICO/subqueries/HyDE, BGE document dense,
   weighted RRF, Qwen3-Reranker-8B document/child MaxP, source parent 512/640 và dedup.
   Qwen3-Embedding-8B child dense chỉ dùng khi cache đủ cho **toàn bộ candidate đang chạy**.
   Cache Qwen chưa đủ vẫn giữ; notebook này không dành thời gian encode tiếp toàn bộ Qwen corpus trước lần nộp.
6. Cell cuối xác minh và hiển thị vị trí `submission.zip` mới nhất trên Drive, không tự tải về máy tính.
   Nếu cell 4 còn chạy hoặc bị ngắt, ZIP baseline đã hoàn tất vẫn nằm trên Drive.

Model được nạp lần lượt; mỗi model dưới 15B theo chính sách đã xác nhận. Lượt này dùng weights pretrained,
không gọi việc thiếu nhãn train/dev là đã fine-tune. Cache Qwen chưa đầy đủ cũng không được báo thành nhánh đã hoàn tất.

## Checkpoint và nơi lưu

- `/content/vmb_submit_work` là ổ tạm của máy chủ Colab, không phải ổ C:/D: trên máy Sếp.
  Dữ liệu gốc, vector, checkpoint và ZIP đã publish lưu bền trên Drive. Colab dùng bản cache
  tạm để tính toán; notebook không tự tải dữ liệu hoặc ZIP về máy tính người dùng.
- Nguồn vẫn ở `data/processed/<source-build>`. Không extract/chunk lại nguồn.
- Candidate ghép ở `data/processed/available-<snapshot>-<signature>`: có receipt cho từng source part,
  freeze receipt ổn định và checkpoint từng part model input trên Drive.
- Model input dedup dùng SQLite trên ổ local với page cache 32 MiB và batch 1.024 child;
  không gom toàn bộ hash trong RAM hoặc quét lại toàn corpus cho mỗi part đầu ra.
  Giữ cách chọn `min(chunk_id)` và thứ tự hash như bản trước để tiếp tục checkpoint cũ.
  Spool local commit từng source part, giữ khi gián đoạn và giải phóng sau khi manifest input đã publish;
  checkpoint input trên Drive vẫn còn khi runtime bị thu hồi.
- Nếu đã có `scope_freeze.json`, resume khôi phục metadata đã xác minh và đi thẳng tới Index inputs,
  không duyệt/chép lại toàn bộ source receipts. Chỉ part document/child chưa nằm trong spool local mới được tải.
  Hai báo cáo alias vẫn giữ trên Drive cùng hash trong candidate; không tải chúng để resume Index inputs.
  Runtime mới vẫn cần dựng lại spool local; đó là đọc dữ liệu nguồn cho dedup, không phải tính embedding.
  Log `Index inputs` ghi rõ số part `reused` và `new`.
- Workflow submission dùng `index_inputs/resume.json` có digest, signature và khóa hash cuối để tiếp tục
  từ prefix đã publish. Lần chuyển từ checkpoint cũ đọc các `.done.json` nhỏ bằng 8 luồng và kiểm một part biên;
  các lần sau dùng cursor. Không quét lại nội dung toàn bộ input part cũ trong bước resume.
  File đã lưu vẫn được xác minh SHA256 khi native cache/embedding/catalog reader thực sự dùng file;
  manifest input hoàn tất thể hiện đủ metadata/range, không thay thế kiểm tra payload của consumer.
  API mặc định ngoài workflow vẫn kiểm ngay (`verify_now`); workflow gọi rõ `verify_on_use`.
  Runtime mới vẫn đọc và kiểm checksum source document/child, nhưng SQLite chỉ giữ đại diện có hash sau cursor;
  số text đã lưu cộng số đại diện còn lại phải bằng distinct count trong health của candidate đã freeze.
- Nếu đọc JSON trên Drive gặp Errno 103, retry riêng file tối đa hai lần trước khi phục hồi mount.
  Guard nhận Errno 103 khi xác định đường dẫn thuộc Drive; không coi lỗi socket khác, file hỏng hoặc hết đĩa là mất mount.
  Inventory CPU từng model đã hoàn tất được nhận lại theo input/model contract, không quét lại corpus khi model tiếp theo bị ngắt.
  Đây là snapshot cache tại thời điểm inventory; GPU vẫn tra cache nội dung hiện tại để dùng lại vector được thêm sau đó.
- Parent chỉ được tải cùng document/child khi cần xây catalog mới. Catalog hoàn tất được khôi phục
  từ phần nén, không tải lại source payload. Section, ledger và failure không được tải cho hai bước này.
  Mọi payload thực sự dùng vẫn được kiểm SHA256; identity và checkpoint cũ được giữ.
- Embedding dùng `data/model_cache` theo nội dung và đúng model/runtime; không gán lại vector của producer khác.
  `cache.cached_inputs` trong CPU preparation là snapshot lúc inventory, không phải bộ đếm GPU trực tiếp.
  `Index inputs: COMPLETE` xác nhận đầu vào model đã đủ; `Catalog unit map` là ánh xạ SQL.
  Hoàn tất embedding phải đối chiếu manifest/map BGE của đúng candidate, không suy từ hai mốc trên.
  Bước BGE còn thiếu dùng batch riêng `retrieval_scale.json:bge_batch_size` (hiện là 32),
  hoặc batch lớn hơn của GPU profile; T4 không bị giới hạn bởi batch Qwen dùng chung là 16.
  Encoder giữ cơ chế giảm một nửa batch khi CUDA OOM. Vector theo nội dung đã có được dùng lại;
  chỉ sửa batch thực thi, giữ nguyên model/revision/pooling/runtime và model producer code.
  Cấu hình này chưa phải số đo tốc độ trên GPU thực tế.
  Cell 3–4 in tên stage ngay khi bắt đầu và heartbeat mỗi 15 giây, kể cả khi đang chờ I/O.
  Kiểm tra mapping BGE dùng tối đa 8 reader, giữ nguyên identity, SHA256 và thứ tự input;
  đây là đọc checkpoint để dùng lại, không phải encode lại vector đã lưu.
  Adapter được nhúng trong notebook, không sửa source producer hoặc thay code/model/candidate lock.
  `data/retrieval/submission_progress.json` lưu trạng thái khi chuyển stage và khi kết thúc;
  heartbeat chi tiết lưu local ở `WORK_DIR/submission_progress.json`.
  Hai file này chỉ là chẩn đoán tiến trình, không chứng nhận embedding hoặc ZIP đã hoàn tất.
  Bảng tra `ContentBlocks` dùng SQLite bền trên ổ local ở `WORK_DIR/receipt_lookups`, commit từng
  receipt cùng key/path/hash/row. Khi Drive ngắt, phần đã commit được giữ; lần khôi phục chỉ đọc
  receipt chưa nhập, với tối đa 2 lượt đọc đồng thời và hàng đợi giới hạn 2 file.
  Sau khi inventory đủ, bản nén có checksum được lưu trong `data/retrieval/content_lookups`;
  runtime mới có thể khôi phục bảng tra rồi đọc phần receipt mới thêm.
  Nếu bản nén chưa lưu thành công thì vẫn giữ checkpoint local trong runtime hiện tại.
  Bảng tra chỉ trỏ tới vector gốc: native reader tiếp tục kiểm SHA256/shape khi dùng vector;
  không dùng bảng tra hoặc số receipt để tuyên bố corpus đã embedding đủ.
  Nếu lối tắt `MyDrive/VietMedBridge/data` không hiện trên tài khoản đang chạy, preflight thử
  đường dẫn target ID đã biết của thư mục dùng chung và trả lại `DATA_ROOT` thực tế.
  Cell 3–4 gán lại đường dẫn này, chỉ chấp nhận đúng candidate SHA từ `PREPARATION`;
  không tìm toàn Drive, tạo handoff rỗng hoặc tự chạy lại preparation khi thiếu file.
  Khi Colab từ chối mount vì `/content/drive` còn file local, helper chỉ giữ nguyên thư mục đó
  dưới `/content/vmb_drive_unmounted_<id>` sau khi bảng mount của kernel xác nhận không còn
  mount tại đó hoặc bên trong. Sau đó mount lại đúng `/content/drive` và xác minh candidate cũ.
  Không xóa hoặc tự nhập lại file trong thư mục được giữ; lỗi quyền/auth vẫn được báo riêng.
  Khi BGE chưa có manifest COMPLETE, receipt inventory dùng Drive API GET/list, có phân trang đầy đủ và kiểm size/MD5
  cùng digest JSON/producer. Bước này chạy trước lượt kiểm mapping và nạp weights, commit vào
  đúng SQLite local đã có; không đọc/ghi receipt hoặc diagnostics qua Drive mount trong inventory.
  Colab yêu cầu xác thực tài khoản có quyền đọc data và ghi checkpoint vào thư mục dùng chung;
  không cần gửi token. API chỉ ghi bản nén bảng tra và con trỏ trong `retrieval/content_lookups`,
  không ghi receipt vector hoặc thay đổi nguồn. Bản nén được upload theo cơ chế resumable của
  [Drive API](https://developers.google.com/workspace/drive/api/guides/manage-uploads), kiểm size/MD5
  của payload trước khi publish con trỏ. Snapshot khóa ID folder dùng chung và model producer,
  nên tài khoản khác/alias MyDrive khác có thể khôi phục. Vẫn giữ tương thích bản nén schema cũ.
  Trong inventory, tự lưu tại các mốc 256/512/1024/... receipt; lưu tiếp khi inventory hoàn tất
  và khi đóng stage có dữ liệu mới. Lỗi upload không hủy vector hoặc bảng tra local; log `deferred`
  nghĩa là chưa có xác nhận lưu bền. Chỉ `SAVED_ON_DRIVE` xác nhận snapshot đã publish.
  Mất VM trước mốc lưu thành công vẫn có thể phải đọc lại phần receipt sau snapshot gần nhất.
  Runtime mới tải một bản nén, kiểm checksum/producer, rồi chỉ GET các receipt mới trong danh sách
  API đầy đủ. Bảng tra không chứng nhận payload vector hoặc embedding coverage đã hoàn tất.
  Các bước catalog và xuất kết quả vẫn dùng mount; transport vector của cell GPU mới dùng API khi cần tải.
  Phiên đang chạy code cũ không tự nhận bản sửa này. Sau khi cell xử lý trả quyền điều khiển,
  có thể chạy riêng [save_submission_lookup_cell.py](../scripts/save_submission_lookup_cell.py)
  trong cùng kernel để lưu bảng tra đã commit, trước khi chuyển account/runtime. Helper từ chối
  chạy đồng thời với stage đang hoạt động; không gọi inventory, nạp weights hoặc encode lại.
  Cần giữ runtime cũ đến khi có `SAVED_ON_DRIVE`. Nếu runtime đã mất trước khi có snapshot,
  không thể khôi phục bảng tra local đã mất; vẫn dùng các vector gốc và đọc metadata để dựng lại.
  Cùng runtime, BGE dùng lại `WORK_DIR/official_scope/<build>/index_inputs` đã stage trên CPU
  khi manifest còn khớp và đủ file; checksum payload vẫn kiểm trước khi dùng. Thiếu bản local thì đọc Drive.
  Sau khi catalog của đúng candidate đã có manifest COMPLETE trên Drive, bước GPU giải phóng
  riêng các bản chép Parquet documents/children/parents được candidate liệt kê trong
  `WORK_DIR/official_scope/<build>/parts/`, để dành chỗ cho vector cache. Log ghi thư mục,
  số file và dung lượng trước khi dọn. Không xóa model inputs, SQL/vector cache, metadata,
  file không liên quan hoặc dữ liệu gốc trên Drive; từ chối thư mục nằm trên DATA_ROOT và symlink.
- CPU catalog được tạo trên ổ local Colab và lưu các phần nén trên Drive theo cơ chế catalog hiện có.
  Bản v7 tự tìm catalog nguồn đã hoàn tất và kế thừa nguyên BM25/document/child/parent khi bộ gộp
  giữ toàn bộ nguồn. SHA của các file giữ nguyên có thể chứng minh trực tiếp; khi nguồn cũ được
  đóng gói Parquet lại, receipt scope phải khóa đúng source snapshot/part, giữ đủ ID và đủ counts.
  Chỉ cho phép kế thừa prefix nguyên vẹn, cùng analyzer/glossary/encoding/chunk/tokenizer/snapshot;
  không áp dụng cho scope lọc whitelist BTC hoặc selection thiếu một phần nguồn cũ.
  Catalog cũ được clone/restore có kiểm SHA, không sửa hay xóa bản gốc. Cập nhật bảng dense position
  theo input hiện tại, giữ các postings/index cũ và chỉ phân tích/index các shard delta.
  Catalog local đang dở ở trong prefix cũ giữ được bảng units khi nâng cấp sang kế thừa;
  nếu đã xử lý đến delta thì giữ tiến độ đó. Log tách `Catalog reuse`, `Catalog delta staging`,
  `Catalog unit map` và `Disk catalog/BM25 delta`; bảng unit là SQL, không phải encode neural.
  `reused_catalog` trong manifest/report ghi nguồn, checksum, số part và counts đã kế thừa.
  Bản v8 tính SHA256 toàn file ngay lúc giải nén 220 phần, không đọc lại file lớn sau dòng cuối.
  Kế thừa dùng page cache SQLite 128 MiB; chuyển ánh xạ đã tính theo khối 65.536 dòng,
  commit cursor cùng dữ liệu. Cùng runtime có thể tiếp tục bản `seed-copy.sqlite` đang chuyển dở,
  hoặc bản v7 đã commit xong trước khi bị ngắt trong kiểm tra/copy backup; không khôi phục lại nguồn.
  Phase và heartbeat mỗi 15 giây ghi SQL steps, elapsed và dung lượng file local.
  Bỏ lượt `quick_check` lặp ở bước kế thừa; kiểm toàn DB vẫn chạy khi catalog cuối hoàn tất.
  Trong cùng tiến trình Linux, catalog local đã kiểm SHA256 được nhận lại khi inode/device,
  size, mtime và ctime còn nguyên. Baseline → strong không băm lại toàn bộ SQL local lần nữa.
  Proof chỉ ở RAM; runtime/process mới, file bị thay đổi hoặc checksum yêu cầu khác đều kiểm lại.
  Không dùng memo này cho file trên `/content/drive`, không bỏ kiểm manifest/part khi restore.
  Giữ SQL checkpoint cũ bằng hard link trên cùng ổ khi có thể, tránh copy thêm một file lớn.
  Bản v6 dùng tối đa 4 process theo CPU quota/affinity, cache hash token và ghi SQL theo lô;
  giữ nguyên PyVi/Jieba, medical aliases, token hash, trọng số BM25 và source span.
  Log ghi riêng thời gian lexical/SQL từng shard để đo nút thắt trên phiên thực tế.
  Kết quả lexical mỗi shard được nén Parquet và publish cùng receipt SHA256 ở
  `retrieval/catalogs/lexical_cache/` trước khi commit SQL. Receipt khóa analyzer/encoding/source hash,
  không khóa candidate: part nguồn còn nguyên dùng lại khi bổ sung batch URL mới.
  Cùng runtime: `building.sqlite` ghi cursor và dữ liệu trong cùng transaction, tiếp tục shard đã commit.
  Runtime mới: kế thừa catalog nguồn hoàn tất, nhập SQL delta còn dở từ nguồn đã kiểm
  và dùng receipt lexical để bỏ qua tách từ đã làm.
  Đây không phải snapshot SQLite đầy đủ mỗi shard trên Drive; không tuyên bố SQL tiếp tục nguyên vị trí khi mất VM.
  Bản cũ dùng TemporaryDirectory không có các receipt này: đổi code giữa lượt có thể mất tiến độ catalog dở;
  model input, chunk, vector và catalog đã hoàn tất vẫn dùng lại. Catalog hoàn tất không bị chuyển định dạng hoặc xây lại do nâng cấp executor.
- Report chính: `data/retrieval/current_submission.json`.
- Các artifact inference mới, query checkpoint và ZIP: **`VietMedBridge/submission_runs/<build-run>`** trên Drive,
  nằm cạnh `data`. Runtime mất không làm mất những checkpoint đã publish xong.
- Con trỏ ZIP: `data/retrieval/baseline_submission.json`, `strong_submission.json`, `latest_submission.json`.
  Bản strong hoàn tất sẽ được ưu tiên; chạy lại baseline không ghi đè con trỏ strong của cùng candidate.

Nếu phiên bị ngắt: mở lại cùng notebook, đúng DATA_ROOT, đúng code lock và Run all.
CPU dùng lại receipt đã ghi; GPU dùng lại vector/query/stage checkpoint. ZIP hoàn tất được đọc kiểm lại và dùng lại,
không nạp model để chạy lại toàn bộ inference. Chỉ một phiên workflow này ghi cùng DATA_ROOT tại một thời điểm.
Không chạy notebook 03 hoặc notebook khác đổi active candidate giữa lượt submission.

Đổi runtime chỉ mất biến và ổ tạm. Report chuẩn bị được lưu ở ba vị trí trên Drive: con trỏ chung,
report trong build và marker CPU preparation theo candidate. Nếu thiếu con trỏ, notebook khôi phục từ report phù hợp.
Nếu không có report phù hợp, mặc định dừng; chỉ tiếp tục preparation khi chủ động bật `ALLOW_CPU_PREPARATION`.
Không tạo report giả để bỏ qua kiểm tra.

Các batch 07–13 đã chia chunk nhưng chưa embedding sẽ làm tăng số BGE cần tính. Giữ URL cũ
không biến những batch đó thành đã embedding. Thời gian phụ thuộc số text thiếu thực tế, GPU và Drive;
không có cam kết hoàn tất trong một phiên hoặc đạt 0,4–0,5 khi chưa đo relevance.

## Resume I/O khi BGE đã COMPLETE

Cell 3–4 nhúng adapter `scripts/submission_vector_io.py`, không sửa `src/vietmedbridge`,
không đổi model identity, thứ tự input, source span hoặc khóa checkpoint đã trả phí.

- Manifest BGE tổng đã seal đủ part và khớp frozen inputs thay thế lượt quét receipt/map
  trước inference. Không mở hàng nghìn `*.done.json` hoặc hash các payload trên Drive chỉ
  để xác minh lại. Manifest thiếu coverage hoặc khác producer bị từ chối.
- Vector/map chỉ được đọc khi consumer cần. Giữ đúng tên cache cũ trong SSD tạm Colab.
  Bản cache cũ chưa có proof được kiểm SHA **một lần trên SSD Colab**. Proof SQLite khóa
  SHA, inode/device, size, mtime, ctime và shape; file không đổi dùng lại proof sau khi chạy
  lại cell trong cùng VM. Không áp dụng proof local cho file Drive hoặc một VM mới.
- Payload thiếu tải qua Drive API, tối đa bốn file đồng thời; SHA được kiểm trong lượt
  streaming download, rồi mới atomic rename. Không hash nguồn Drive rồi copy rồi hash
  thêm lần nữa. Metadata tải theo danh sách thư mục phân trang, tránh GET metadata riêng
  từng vector. Log tách số file, byte thiếu, byte mới và phần dùng lại; ETA chỉ là transfer.
- Mapping được giữ riêng khỏi vòng LRU vector. Khi có đủ map hợp lệ trên SSD, adapter
  gói metadata thành một archive có checksum và lưu trong `retrieval/mapping_bundles`.
  VM sau khôi phục một archive thay vì mở riêng từng map; kiểm từng member lúc giải nén,
  từ chối link/file lạ, thiếu/trùng member và giới hạn metadata 512 MiB.
- Mẫu vector huấn luyện ANN giữ nguyên seed, vị trí và giá trị của native FAISS. Mẫu được
  checkpoint riêng trong `retrieval/index_samples`, khóa đầy đủ manifest và vị trí mẫu.
  Sau log `ANN sample SAVED_ON_DRIVE`, VM mới dùng mẫu này thay vì quét corpus lại để lấy mẫu.
  Upload optional bị lỗi sẽ log `deferred`; không coi đó là đã có checkpoint bền.
- Dựng các shard FAISS, kiểm ANN recall, query inference và provenance/export vẫn chạy
  bằng code native. Đọc vector để thực sự dựng index hoặc xếp hạng vẫn cần thiết;
  adapter không biến corpus chưa có index thành một ZIP tức thì.

Phiên đang chạy bản cũ phải trả quyền điều khiển trước khi nạp adapter mới. Giữ nguyên VM,
không reset runtime hoặc chạy lại CPU preparation. Chạy
`scripts/resume_submission_optimized_cell.py` trong cùng notebook để nhận bản sửa và dùng
cache hiện có. Script từ chối chạy đồng thời với cell xử lý. Artifact bền và ZIP lưu trên
Drive; cache SSD nói trên nằm ở VM Colab, không phải ổ cứng máy tính người dùng.

## Kiểm tra trước khi xuất ZIP

Workflow xác minh checksum nguồn, schema, ID/URL của snapshot nguồn, span và representation,
không đổi source text/offset. ZIP phải chứa đúng `results.json`, đủ 1.200 query đúng thứ tự,
ID duy nhất, document thuộc corpus khai báo và chunk trích đúng source parent có provenance.
Đây là kiểm tra dữ liệu/định dạng, không phải điểm relevance hay xác nhận tương thích dataset BTC mới.

Các test integration chạy bằng model CPU giả lập để kiểm tra đường đi thực, resume và export;
không dùng kết quả test làm bằng chứng về tốc độ GPU, chất lượng model hoặc điểm thi.

Regression bộ nhớ: cùng một triệu text giả lập, phép aggregate cũ lỗi ở giới hạn DuckDB 64 MB;
builder dùng ổ local mới hoàn tất đủ một triệu input, kiểm lại thứ tự và SHA256 của từng text.
Page cache SQLite 32 MiB là cấu hình cache DB, không phải tuyên bố tổng RSS của Python/Arrow chỉ 32 MiB.
Tham khảo [giới hạn spill của DuckDB](https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads)
và [page cache SQLite](https://www.sqlite.org/pragma.html#pragma_cache_size).
