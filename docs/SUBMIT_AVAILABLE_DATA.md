# Nộp bài từ dữ liệu hiện có, giữ URL cũ

Mở [04_colab_submit_available_data.ipynb](https://colab.research.google.com/github/Platypus27-coder/VietMedBridge/blob/main/notebooks/04_colab_submit_available_data.ipynb).
Đây là notebook dùng cho lượt nộp hiện tại, thay cho việc cấu hình nhiều chế độ trong notebook 04 cũ.

Theo yêu cầu ngày 14/10, giữ các URL thuộc snapshot nguồn ban đầu, kể cả URL đã bị BTC loại trong dataset mới.
Không áp danh sách `01_already_chunked` của audit dataset mới làm bộ lọc. Không đổi ID, không xóa source,
chunk, raw archive hoặc checkpoint cũ. ZIP của lượt này được kiểm tra với **snapshot nguồn ban đầu**;
chưa được chứng nhận phù hợp danh sách ID của dataset BTC mới. Điểm chỉ biết sau khi BTC chấm.

## Cách chạy

1. Có thể chọn Colab **T4 GPU → Run all** trực tiếp. Notebook tự tiếp tục chuẩn bị bằng CPU khi chưa có report,
   rồi mới nạp model GPU. Muốn tiết kiệm quota, chọn **CPU** cho lượt chuẩn bị trước.
   Giữ `DATA_ROOT_PATH` trỏ đến data dùng chung.
   Notebook tự nhận candidate đang active, các nguồn của candidate đã ghép, batch hoàn tất gần nhất
   và pilot `stage-a-data-v5-clean-text` nếu còn. Nguồn phải hoàn tất, cùng snapshot gốc và cùng chính sách chunk/tokenizer.
2. Chờ `CPU_PREPARATION_COMPLETE`. Output ghi số document, child, unique dense input,
   danh sách source build và số vector cache được dùng lại/còn thiếu. Đây là số đo thực tế của lượt ghép,
   không suy từ số tar hoặc từ tổng số URL đã từng thử crawl.
3. Nếu đã chuẩn bị bằng CPU, đổi runtime sang **T4 GPU** rồi **Run all cùng notebook**. Không cần đổi ACTION, worker ID hoặc tài khoản Kaggle.
   GPU xác minh CPU preparation, code/model producer/runtime trước khi nạp weights.
4. Cell 3 bổ sung **BGE còn thiếu** và xuất ZIP BGE/BM25 đủ 1.200 query. ZIP được giữ trên Drive trước khi nạp LLM
   cho cell 4. Đây là bản đầu để lấy phản hồi từ BTC, chưa có dịch, document dense hoặc neural reranking.
5. Cell 4 tiếp tục bản nâng cấp: dịch EN/ZH, PICO/subqueries/HyDE, BGE document dense,
   weighted RRF, Qwen3-Reranker-8B document/child MaxP, source parent 512/640 và dedup.
   Qwen3-Embedding-8B child dense chỉ dùng khi cache đủ cho **toàn bộ candidate đang chạy**.
   Cache Qwen chưa đủ vẫn giữ; notebook này không dành thời gian encode tiếp toàn bộ Qwen corpus trước lần nộp.
6. Cell cuối tải `submission.zip` mới nhất đã kiểm tra. Nếu cell 4 còn chạy hoặc bị ngắt, ZIP baseline vẫn nằm trên Drive.
   Có thể dừng cell 4 rồi chạy riêng cell cuối để lấy ZIP baseline đã hoàn tất.

Model được nạp lần lượt; mỗi model dưới 15B theo chính sách đã xác nhận. Lượt này dùng weights pretrained,
không gọi việc thiếu nhãn train/dev là đã fine-tune. Cache Qwen chưa đầy đủ cũng không được báo thành nhánh đã hoàn tất.

## Checkpoint và nơi lưu

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
- Parent chỉ được tải cùng document/child khi cần xây catalog mới. Catalog hoàn tất được khôi phục
  từ phần nén, không tải lại source payload. Section, ledger và failure không được tải cho hai bước này.
  Mọi payload thực sự dùng vẫn được kiểm SHA256; identity và checkpoint cũ được giữ.
- Embedding dùng `data/model_cache` theo nội dung và đúng model/runtime; không gán lại vector của producer khác.
- CPU catalog được tạo trên ổ local Colab và lưu các phần nén trên Drive theo cơ chế catalog hiện có.
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
report trong build và marker CPU preparation theo candidate. Nếu thiếu con trỏ, notebook khôi phục từ report phù hợp;
nếu bước chuẩn bị chưa xong, nó tiếp tục receipt đã ghi ngay trong phiên hiện tại. Không tạo report giả để bỏ qua kiểm tra.

Các batch 07–13 đã chia chunk nhưng chưa embedding sẽ làm tăng số BGE cần tính. Giữ URL cũ
không biến những batch đó thành đã embedding. Thời gian phụ thuộc số text thiếu thực tế, GPU và Drive;
không có cam kết hoàn tất trong một phiên hoặc đạt 0,4–0,5 khi chưa đo relevance.

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
