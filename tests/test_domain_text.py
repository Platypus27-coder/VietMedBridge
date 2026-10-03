"""Domain extraction must preserve article evidence without importing site UI."""

import hashlib

import pytest

from vietmedbridge.text import extract_source
from vietmedbridge.structure import parse_structure, validate_structure


URL = "https://nhathuoclongchau.com.vn/bai-viet/source.html"


def page(article, *, url=URL, title="Nghiên cứu", intro="Đoạn dẫn nguồn."):
    return f'''<html lang="vi"><head><title>{title}</title><link rel="canonical" href="{url}"></head>
    <body><nav>Giỏ hàng</nav><div data-lcpr="prr-id-articles-content"><div>
    <h1>{title}</h1><div><p>Kích thước chữ</p></div><div><p>{intro}</p></div>
    <div data-theme-element="article">{article}</div>
    <div><h2>Các bài viết liên quan</h2><p>Tiêu đề bài khác.</p></div>
    <div><p>Tiểu sử chuyên gia.</p></div></div></div>
    <aside>Cookie Consent</aside></body></html>'''.encode("utf-8")


def test_article_order_sections_and_inline_medical_values_are_preserved():
    result = extract_source(page('''<p>Liều 1<span>.5</span>mg é 中文.</p>
        <h2>Triệu chứng</h2><p>Văn bản nguồn một.</p>
        <h3>Điều trị</h3><p>Văn bản nguồn hai.</p>'''), "text/html", source_url=URL)
    assert result["parser"] == "longchau-article-dom-v1"
    assert result["source_text"] == (
        "Nghiên cứu\n\nĐoạn dẫn nguồn.\n\nLiều 1.5mg é 中文.\n\n"
        "Triệu chứng\n\nVăn bản nguồn một.\n\nĐiều trị\n\nVăn bản nguồn hai."
    )
    assert result["source_text_sha256"] == hashlib.sha256(result["source_text"].encode()).hexdigest()
    assert all(v not in result["source_text"] for v in (
        "Giỏ hàng", "Kích thước chữ", "Các bài viết liên quan", "Tiểu sử chuyên gia", "Cookie Consent",
    ))
    doc = {**result, "doc_id": 264425}
    structure = parse_structure(doc, result["heading_hints"])
    validate_structure(doc, structure)
    assert structure["known_heading_count"] == 3


def test_lists_tables_captions_and_same_words_inside_article_are_kept():
    result = extract_source(page('''<h2>Cookie Consent</h2>
        <ul><li>Mục một<ul><li>Mục con</li></ul></li><li>Mục hai</li></ul>
        <table><caption>Đơn vị đo trong bảng.</caption><tr><th>Chỉ số</th><th>Giá trị</th></tr><tr><td>HbA1c</td><td>7.5%</td></tr></table>
        <figure><img alt="Không được tự thêm alt vào text"><figcaption>Chú thích nguồn.</figcaption></figure>
        <p hidden>Nội dung ẩn.</p><!-- Invisible comment -->'''), "text/html", source_url=URL)
    text = result["source_text"]
    assert "Cookie Consent" in text  # No blanket keyword deletion within the article.
    assert text.count("Mục con") == 1
    assert "Chỉ số | Giá trị\nHbA1c | 7.5%" in text
    assert text.index("Đơn vị đo trong bảng.") < text.index("Chỉ số | Giá trị")
    assert "Chú thích nguồn." in text
    assert "Không được tự thêm" not in text and "Nội dung ẩn" not in text
    assert "Invisible comment" not in text
    assert result["raw_has_table"] is True


def test_canonical_mismatch_and_changed_layout_do_not_fallback_to_menus():
    with pytest.raises(ValueError, match="canonical_url_mismatch"):
        extract_source(page("<p>Bài khác.</p>", url=URL.replace("source", "other")),
                       "text/html", source_url=URL)
    changed = page("<p>Văn bản bài.</p>").replace(b'data-theme-element="article"', b'data-new-layout="article"')
    with pytest.raises(ValueError, match="article_layout_changed"):
        extract_source(changed, "text/html", source_url=URL)


def test_cloudflare_error_page_is_rejected_before_domain_adapter():
    with pytest.raises(ValueError, match="blocked_or_challenge_page"):
        extract_source(b"<html><title>Just a moment...</title><body>Enable JavaScript</body></html>",
                       "text/html", source_url=URL)


def test_other_domains_keep_the_generic_extractor():
    result = extract_source(page("<p>Văn bản nguồn y sinh có nội dung cho người đọc.</p>"),
                            "text/html", source_url="https://other.test/bai-viet/source.html")
    assert result["parser"] != "longchau-article-dom-v1"
