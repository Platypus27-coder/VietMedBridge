"""Regression cases for wrong-language, homepage and boilerplate captures."""

import pytest

from vietmedbridge.quality import document_quality, has_encoded_payload
from vietmedbridge.text import extract_source, language_hint


def test_content_language_overrides_wrong_html_declaration():
    chinese = "患者接受治疗后，医生观察症状和检查结果，并记录药物反应。" * 8
    vietnamese = "Người bệnh cần theo dõi triệu chứng và tái khám để bác sĩ đánh giá kết quả điều trị. " * 8
    assert language_hint(chinese, "en")[0] == "zh"
    assert language_hint(vietnamese, "en")[0] == "vi"


def test_article_redirected_to_homepage_is_not_a_document():
    home = b"<html><head><title>Medical site home</title></head><body><main>" + (
        b"Latest health news and medical updates. " * 20
    ) + b"</main></body></html>"
    with pytest.raises(ValueError, match="article_redirected_to_homepage"):
        extract_source(home, "text/html", requested_url="https://example.org/health/article-42",
                       source_url="https://example.org/")


def test_article_with_homepage_canonical_is_not_a_document():
    page = b"<html><head><title>Medical site home</title><link rel='canonical' href='https://example.org/'></head><body><main>" + (
        b"Latest health news and medical updates. " * 20
    ) + b"</main></body></html>"
    with pytest.raises(ValueError, match="article_canonical_points_to_homepage"):
        extract_source(page, "text/html", source_url="https://example.org/health/article-42")


def test_long_encoded_payload_is_not_high_quality_or_extracted():
    encoded = ("AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/" * 7)
    source = "Clinical findings and observations. " * 15 + encoded
    assert has_encoded_payload(source)
    quality = document_quality(source, title="Clinical findings")
    assert quality["quality_tier"] == "LOW"
    assert "ENCODED_PAYLOAD_SUSPECTED" in quality["quality_flags"]
    html = f"<html><body><article><h1>Clinical findings</h1><p>{source}</p></article></body></html>"
    with pytest.raises(ValueError, match="encoded_payload_in_extracted_text"):
        extract_source(html.encode(), "text/html")
    assert not has_encoded_payload("ACGT" * 100)


def test_related_dom_and_recommendation_tail_do_not_enter_article_text():
    article = "Patients received treatment and were monitored for clinical response. " * 12
    html = ("<html lang='en'><head><title>Treatment study</title></head><body>"
            f"<article><h1>Treatment study</h1><p>{article}</p></article>"
            "<div class='related-articles'>Related articles about other conditions. "
            "Recommended doctors and hospitals.</div>"
            "<aside>Latest stories and clinic listings.</aside></body></html>")
    result = extract_source(html.encode(), "text/html")
    assert "Patients received treatment" in result["source_text"]
    assert "Recommended doctors" not in result["source_text"]
    assert "Latest stories" not in result["source_text"]


def test_related_work_and_recommended_treatment_are_article_evidence():
    body = "Clinical treatment outcomes were recorded at follow-up. " * 12
    evidence = "Evidence from previous clinical trials remains important. " * 12
    treatment = "Recommended treatment depends on clinical response. " * 12
    html = ("<html><head><title>Clinical evidence</title></head><body><article>"
            f"<h1>Clinical evidence</h1><p>{body}</p>"
            f"<section class='related-work'><h2>Related work</h2><p>{evidence}</p></section>"
            f"<section class='recommended-treatment'><h2>Treatment</h2><p>{treatment}</p></section>"
            "</article></body></html>")
    result = extract_source(html.encode(), "text/html")
    assert "Evidence from previous clinical trials" in result["source_text"]
    assert "Recommended treatment depends" in result["source_text"]


def test_auxiliary_phrase_in_clinical_sentence_does_not_trim_article():
    from vietmedbridge.text import _trim_auxiliary_tail

    body = "Clinical treatment outcomes were recorded at follow-up. " * 12
    source = body + "The report describes 健康资讯推荐 in a discussion, followed by clinical follow-up."
    assert _trim_auxiliary_tail(source) == source
    assert _trim_auxiliary_tail(body + "\n健康资讯推荐\nUnrelated website links") == body.strip()
