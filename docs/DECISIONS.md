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
