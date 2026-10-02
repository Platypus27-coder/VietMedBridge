# Kiểm chứng data pipeline v2

Kiểm tra local trong Conda env `r2ai-stage3`, Python 3.11, ngày 02/10/2026.
Package editable đã cập nhật 0.2.0; PYTHONNOUSERSITE=1.

## Kết quả đã chạy

- **13 pytest tests passed**. Bao gồm ID không liên tục, duplicated URL giữ nhiều
  official IDs, official ID/URL mismatch, deterministic stratified sampling có
  rare formats/long-tail, source Unicode spans, parent containment, child identity
  độc lập parent size, global dedup xuyên shard và representation hash theo title.
- Kiểm tra resume/retry failed, checksum corruption, robots redirect/disallow,
  hard span failure không ghi done marker, parser bug không bị nuốt thành lỗi
  nguồn, duplicate input xuyên shard, byte budget, incomplete requested range
  không được duyệt, golden thiếu human assertions bị từ chối và snippet regression.
- Health baseline-relative warning, code/tokenizer/chunk-policy evidence gates,
  post-health artifact corruption bị bắt lại khi freeze. Không tự tạo qrels/pass.
- Bốn notebook qua nbformat và syntax Python, kể cả top-level await:
  00 có 11 cells; 01 có 9; 02 có 9; 03 có 10. Không lưu execution outputs.
- `pip check`: No broken requirements found.

## Tokenizer và mock workflow thực

Script `scripts/check_tokenizer.py` dùng BGE-M3 fast tokenizer đã cache ở revision
`5617a9f61b028005a4858fdac845db406aefb181`, local_files_only=True, không tải weights.

11 synthetic fixture cases đều pass: VI/EN/ZH, biomedical symbols/dosage, combining
Unicode, HTML table, XML/JATS section/table cells, source ngắn, PDF text/scan-like,
403/login/JS quarantine. Kiểm tra bằng tokenizer thật đã phát hiện và sửa token
SentencePiece gộp newline vào heading qua ranh giới section.

Bốn source dài Việt/Anh/Trung/mixed Unicode được kiểm với parent 256/512/640.
Child IDs không đổi khi chỉ đổi parent size; exact slices và token budgets đều
pass. Cấu hình parent 640 tạo tổng **104 child spans và 88 parent spans**; parent
on-demand 256 vẫn chứa đúng child nguồn.

Workflow offline HTTP mock với tokenizer thật đi qua crawl → extract/chunk →
Parquet → whole-snapshot validation → global aliases → health → freeze:
4 input IDs, 3 parsed documents, 1 HTTP 404; 64 children và 60 parents. Aliases
giữ 3 official document IDs thuộc 2 nhóm normalized content. Snapshot đạt
FROZEN_CANDIDATE; không có human milestone/retrieval approval giả.

## Chạy lại

    conda activate r2ai-stage3
    python -m pytest -q
    python scripts/check_notebooks.py
    python scripts/check_tokenizer.py --local-tokenizer <pinned-cache-directory> --report artifacts/tokenizer-validation.json
    python -m pip check

Token validation report được lưu trong artifacts/ và không commit. Không truyền
--local-tokenizer thì script tải tokenizer pinned từ Hub, vẫn không tải weights.

## Giới hạn bằng chứng

Crawl test dùng HTTP mock, documents giả và fixture synthetic. Chưa chạy Stage A
trong runtime Colab của Sếp, chưa review 100–500 nguồn thật, chưa crawl toàn corpus.
Metadata dataset/query/corpus đã kiểm ở revision pinned trong lần setup trước;
không coi metadata Hub là bằng chứng đã xử lý các URL.

Chất lượng extraction theo domain, coverage thực, storage/GPU forecasts, OCR,
official scorer compatibility, ANN recall và F2 cần artifact/labels/benchmark sau
Colab. Các gates hiện giữ những evidence chưa có ở trạng thái pending.
