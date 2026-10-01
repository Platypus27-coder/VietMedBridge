# Kiểm tra giai đoạn xử lý dữ liệu

Kiểm tra local trên Python 3.11 trong Conda env r2ai-stage3, ngày 02/10/2026.

- 5 tests passed: ID không liên tục, URL trùng nhưng official IDs khác nhau,
  audit/subset validation, source span Unicode, parent containment, resume,
  retry riêng nguồn lỗi, manifest input mismatch, raw checksum corruption,
  robots redirect/disallow, XML và challenge-page quarantine.
- Ba notebook đều qua nbformat validation và kiểm tra syntax Python, gồm
  top-level await. Mỗi notebook có 9 cells, không lưu execution outputs.
- Tokenizer BGE-M3 thật ở revision
  5617a9f61b028005a4858fdac845db406aefb181 đã chạy trên văn bản ngắn chứa
  tiếng Việt, ký tự combining accent và tiếng Trung. 10 child spans và
  8 parent spans đều khớp source offsets; không tải model weights.
- pip check không phát hiện dependency requirement bị thiếu/xung đột trong env.
- Kernel Jupyter Python (r2ai-stage3) đã đăng ký với PYTHONNOUSERSITE=1.

Test crawl dùng HTTP mock và dữ liệu giả. Metadata dataset được đọc trực tiếp
từ Hugging Face để xác nhận configs/query/corpus và schema; chưa chạy crawl
toàn corpus. Chưa thực thi notebook trong runtime Colab của người dùng.

Các kết quả trên xác minh hợp đồng dữ liệu và cơ chế resume. Chất lượng nội
dung theo từng domain, coverage toàn corpus, compatibility với scorer chính
thức và hiệu quả F2 cần đo sau khi có artifact chạy từ Colab.
