# Thu gom toàn bộ dữ liệu BTC trong AI Guru

Chạy `notebooks/02_colab_collect_all_btc_ai_guru.ipynb` trên **Colab CPU**, Run all.
Không chạy lại notebook 02/03 cũ. Không cần GPU, HF token hay Kaggle key.

Notebook dùng snapshot BTC `71be0e12` đang được khóa trong hệ thống. Đọc tất cả
archive `.tar` trong AI Guru, gồm thư mục con và shortcut, mọi phiên bản resume,
TAR lồng nhau và package retained. Không suy ra archive trùng chỉ từ số shard,
frontier hoặc thời gian. Mỗi leaf giữ Drive file identity và offset thành viên.

Native leaf dùng đủ bốn manifest dataset/frontier/crawl/extracted làm danh mục
URL của nguồn; đọc đủ thì dừng duyệt header, không liệt kê hàng trăm nghìn member
raw phía sau. TAR container được duyệt hết danh sách để tìm mọi TAR con, gồm bản
resume nằm cuối. Không còn giới hạn 128 header. Layout trộn metadata native và
TAR con trong cùng một nguồn không thuộc contract này và cần kiểm tra riêng.

Nếu bản cũ báo lỗi `Metadata not found within 128 TAR headers`, mở notebook đã
cập nhật và Run all trên CPU, giữ nguyên output folder. Mapping và archive đã
commit được dùng lại; chỉ archive lỗi cần thử tiếp. Không xóa checkpoint. Cell 2
reload bộ đọc mới ngay trong runtime cũ, không cần restart để thay code Python.

Nếu Drive trả 403, bộ đọc ghi `reason`, file ID và byte range vào log/báo cáo.
Chỉ 403 với reason `rateLimitExceeded`/`userRateLimitExceeded` được tự thử lại;
nhận cả `RATE_LIMIT_EXCEEDED`/`USER_RATE_LIMIT_EXCEEDED` trong google.rpc.ErrorInfo.
Google có thể gửi đồng thời tên camelCase và viết hoa trong cùng một lỗi.
429 và lỗi server tạm thời cũng thử lại có giới hạn, giữ nguyên cùng byte range.
Quyền truy cập, quota tải, daily quota và 403 không rõ nguyên nhân không bị retry
mù. `Retry-After` được tôn trọng; cooldown dài hoặc rate limit hết lượt retry
dừng campaign, ghi các archive chưa thử để không tiếp tục dồn request lên Drive.
Không đổi tài khoản, URL hoặc quyền để vượt giới hạn của nhà cung cấp.

Mỗi file metadata tải xong có payload và receipt riêng dưới `metadata_cache/`
trên Drive, trước khi bắt đầu SQL. Lỗi ở file tiếp theo không làm mất metadata
đã tải xong. Phiên khác đọc lại payload đã lưu thay vì mở payload trong TAR.
Một byte range lỗi tạm thời được retry tại chỗ; member chưa hoàn tất sau lỗi
cứng chưa có receipt nên có thể tải lại member đó khi tiếp tục. Không chứng
nhận archive hoàn tất chỉ vì đã có vài member metadata.

Trước hết dùng lại alias mapping của baseline và 10 build đã freeze. Những mapping
này trỏ đến nội dung đã xử lý, không mở lại corpus/text/vector. So khớp với URL BTC
hiện tại bằng đúng quy tắc trim + bỏ fragment đang dùng trong importer; không đổi
host, query string hoặc nội dung URL. Giữ nguyên ID/URL BTC và lưu riêng ID cũ.

Qua Drive Range, chỉ đọc các thành viên metadata/extraction. Không tải nguyên TAR,
không giải nén raw HTML ra ổ Colab hoặc PC. Khi bản mới thất bại nhưng bản khác có
extraction thành công đúng hash/snapshot và vượt gate nội dung, lưu lại text đó.
Original snapshot/crawl record/extraction record được giữ riêng với membership mới;
không giả mạo bản crawl cũ thành crawl của snapshot mới. Retained LOW/QUARANTINE
vẫn cần review, không tự đổi nhãn thành thành công.

Sau mỗi mapping, leaf và archive, một SQLite checkpoint nén được lưu trên Drive.
Đổi runtime phục hồi một snapshot; phần đã commit không mở TAR hoặc đọc mapping lại.
Một leaf bị ngắt giữa chừng có thể cần đọc lại leaf đó. Source thay đổi hoặc metadata
không hỗ trợ được ghi lỗi riêng, không lờ đi rồi báo đủ dữ liệu. Metadata SHA và
source version được kiểm; raw payload chưa được rehash nên báo cáo ghi rõ giới hạn.

Đầu ra tại `VietMedBridge/data_staging/btc-all-ai-guru-71be0e12-v1/`:

- `collection_report.json`: số ID BTC theo từng nhóm trạng thái; lỗi archive;
  checksum của các CSV và các phần text đã khôi phục.
- `btc_membership.csv`: một hàng cho mỗi ID/URL BTC, trỏ về nguồn nội dung.
- `missing_or_unusable.csv`: ID/URL chưa có nội dung dùng được, kèm trạng thái.
- `parts/*.parquet`: text/extraction recovery, giữ provenance và checksum receipt.
- `checkpoint.json` và `sql-*.gz`: checkpoint để tiếp tục qua tài khoản/runtime khác.

Lượt dùng bộ lọc raw cũ cần [cập nhật checkpoint phân loại raw](REPAIR_COLLECTION_RAW_PATHS.md).
Xem `counts_provisional` và `raw_path_classification` trước khi chốt số liệu.

Các trạng thái được đếm theo ID BTC; số URL fetch duy nhất được báo riêng:

- `REUSED_VALIDATED_CONTENT`: có nội dung đã xử lý, dùng mapping/reference hiện có.
- `RECOVERED_EXTRACTION_PENDING_CHUNK`: đã thu được text, còn phải chunk/freeze chuẩn.
- `RAW_AVAILABLE_NEEDS_EXTRACTION`: có bản ghi raw thành công, chưa có text vượt gate.
- `CRAWL_NOT_SUCCESSFUL`: có bản ghi nguồn nhưng chưa có raw thành công.
- `RAW_METADATA_REVIEW`: có trạng thái SUCCESS_RAW nhưng hash, locator hoặc
  snapshot không hợp lệ; không coi đây là lỗi HTTP/crawl.
- `NOT_FOUND_IN_ARCHIVES`: chưa tìm thấy URL trong mọi nguồn đọc được.

`ALL_FOLDER_SOURCES_COLLECTED` chỉ nói đã duyệt hết folder. Xem riêng
`all_official_content_available`; không lấy số record/HTTP 200 làm số tài liệu hợp lệ.
Nếu có archive chưa đọc được thì báo `FOLDER_COLLECTION_INCOMPLETE` và không kết luận
đủ toàn BTC. Không tự recrawl, không encode lại, không đổi active submission candidate.
Chỉ một runtime chạy notebook này trên cùng output folder.

Kiểm thử bao gồm flat/nested TAR, failed-new/valid-old recovery, alias ID mới,
retained provenance và quality review, skip sau khi đổi runtime, file nguồn thiếu,
official checksum mismatch và notebook syntax. Chưa chạy thu gom thực tế trên
Drive chỉ nhờ việc kiểm thử cục bộ; cần phiên Colab đã xác thực để đọc Range.
