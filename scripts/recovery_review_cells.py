"""Colab replay of the verified browser captures; no more HTTP requests."""


def review_cells(bootstrap, md, code):
    return [
        md("""
        # VietMedBridge — 01f: Trích bài Long Châu từ HTML đã lưu

        Dùng **runtime CPU mới** và cùng Drive data root. Notebook đọc experiment
        01e đã chạy đủ 11 URL, kiểm hashes, rồi trích lại các HTML đã có. Giữ
        heading, đoạn dẫn, paragraph, list, table và chú thích trong bài; loại
        cookie, menu, tiểu sử và bài liên quan nằm ngoài vùng nội dung.

        Có checkpoint theo doc_id và gói review riêng. RAW, text cũ và corpus
        Stage A được giữ để so sánh. Chưa tự nhập/gộp vào canonical corpus.
        """),
        code(bootstrap.replace("code_lock.json", "recovery_extract_code_lock.json")),
        md("## 1. Đọc experiment đủ 11 URL và trích lại từ cache"),
        code("""
        import csv
        from vietmedbridge.probe_review import review_browser_captures
        from vietmedbridge.artifacts import read_json

        SOURCE_KEY = "4c49d3358aa818b7"
        EXPECTED_ATTEMPTS_SHA256 = "c4c716d838f4c676e62cc7e9e8e9b8ba366ee04dd5f3e5d2abdc3560b3fdb807"
        REPORT_DIR = DATA_ROOT / "reports/crawl_recovery/stage-a-v2"
        PROBE_DIR = REPORT_DIR / "experiments" / SOURCE_KEY
        with (CHECKOUT / "reports/stage-a-v2-longchau-access-gap.csv").open(encoding="utf-8") as stream:
            OFFICIAL_PAIRS = {int(r["doc_id"]): r["url"] for r in csv.DictReader(stream)}
        REVIEW = review_browser_captures(
            PROBE_DIR, REPORT_DIR / "extraction_review",
            expected_links=OFFICIAL_PAIRS,
            expected_attempts_sha256=EXPECTED_ATTEMPTS_SHA256,
            work_dir=WORK_DIR,
        )
        REVIEW_DIR = Path(REVIEW["output_dir"])
        print(json.dumps({k: REVIEW[k] for k in [
            "input_ids", "ready_for_review", "states", "output_dir",
        ]}, ensure_ascii=False, indent=2))
        """),
        md("## 2. Xem text đã làm sạch và tải gói review"),
        code("""
        import shutil
        import pandas as pd
        from IPython.display import display, HTML
        from google.colab import files
        from vietmedbridge.artifacts import publish_file

        display(pd.DataFrame(REVIEW["records"]).reindex(columns=[
            "doc_id", "state", "capture_kind", "source_chars", "section_count",
            "paragraph_count", "quality_flags",
        ]))
        display(HTML((REVIEW_DIR / "review.html").read_text(encoding="utf-8")))
        local_zip = Path(shutil.make_archive(
            str(WORK_DIR / f"recovery-extraction-{REVIEW['signature'][:16]}"),
            "zip", root_dir=REVIEW_DIR,
        ))
        target = REPORT_DIR / local_zip.name
        publish_file(local_zip, target)
        print("Gói text/structure/hash để review:", target)
        files.download(str(local_zip))
        """),
    ]
