"""Site UI must be removed without dropping clinical article evidence."""

from vietmedbridge.site_cleanup import clean_site_text
from vietmedbridge.text import extract_source
from vietmedbridge.quality import document_quality


def test_youlai_header_doctor_footer_and_repeated_article():
    article = "眩晕的原因包括耳部疾病及全身性疾病，应结合检查判断。" * 8
    page = ("-视频-文章-语音\n眩晕是怎么引起的\n2023-05-09 11:49:39阅读：-\n"
            "手机浏览\n用手机扫描二维码在手机上继续观看\n"
            f"眩晕是怎么引起的\n{article}\n神经内科医生推荐\n"
            f"-视频-文章-语音\n眩晕是怎么引起的\n{article}\n神经内科医生推荐")
    result = clean_site_text(page, "https://www.youlai.cn/yyk/article/123.html")
    assert result == f"眩晕是怎么引起的\n{article}"


def test_familydoctor_related_previews_are_not_article_body():
    article = "白癜风护理应根据患者情况由医生评估。" * 15
    source = (f"白癜风如何护理\n{article}\n- 白癜风诊断的其他文章\n-\n"
              "这篇其他文章的内容不能被当作正文。\n（责任编辑：编辑 ）\n相关推荐\n热门文章")
    result = clean_site_text(source, "https://baidianfeng.familydoctor.com.cn/a/201311/531722.html")
    assert result == f"白癜风如何护理\n{article}"
    # A genuine bulleted article without the site's editorial tail is kept.
    assert clean_site_text(source.split("\n（责任编辑：")[0],
                           "https://baidianfeng.familydoctor.com.cn/a/201311/531722.html") == source.split("\n（责任编辑：")[0]


def test_medlatec_article_keeps_body_without_link_list_or_service_ui():
    article = "Thoái hóa cột sống cần được đánh giá dựa trên triệu chứng và thăm khám. " * 8
    source = ("Tin tức\nThoái hóa cột sống\n"
              "- 14/02/2022 | Gai cột sống là gì?\n"
              "- 16/05/2022 | Một bài viết liên quan\n"
              f"1. Nguyên nhân\n{article}\nBình luận ()\nLựa chọn dịch vụ")
    assert clean_site_text(source, "https://medlatec.vn/tin-tuc/bai-s68-n28336") == (
        f"Thoái hóa cột sống\n1. Nguyên nhân\n{article.strip()}")


def test_phutho_pharmacity_and_nghean_templates():
    article = "Vitamin C từ thực phẩm là một phần của chế độ ăn đa dạng. " * 10
    phutho = (f"{{title}}\n{{publish}}\n{{head}}\n{article}\n"
              "{name} - {time}\n{body}\nÝ kiến của bạn\nTin mới")
    assert clean_site_text(phutho, "https://baophutho.vn/bai-viet.htm") == article.strip()
    pharmacity = article + "Xem chi tiếtCác bài viết liên quan\nMột bài khác"
    assert clean_site_text(pharmacity, "https://www.pharmacity.vn/bai-viet.htm") == article.strip()
    nghean = article + "\n| TIN LIÊN QUAN |\n|---|"
    assert clean_site_text(nghean, "https://baonghean.vn/bai-viet.htm") == article.strip()


def test_other_hosts_and_replacement_character_quality():
    source = "Các bài viết liên quan là chủ đề nghiên cứu. " * 10
    assert clean_site_text(source, "https://example.org/article") == source.strip()
    quality = document_quality("Điều trị nhiễm trùng cần đánh giá lâm sàng. " * 10 + "�", title="Điều trị")
    assert "DECODE_REPLACEMENT_CHAR" in quality["quality_flags"]
    assert quality["quality_tier"] == "MEDIUM"


def test_cleanup_happens_before_source_hash_and_chunks():
    body = ("<html lang='vi'><title>Thoái hóa cột sống</title><body><article>"
            "<h1>Thoái hóa cột sống</h1><p>" + "Người bệnh cần theo dõi triệu chứng. " * 14 +
            "</p><p>Bình luận ()</p><p>Lựa chọn dịch vụ</p></article></body></html>")
    result = extract_source(body.encode(), "text/html",
                            source_url="https://medlatec.vn/tin-tuc/bai-s68-n28336")
    assert "Người bệnh cần theo dõi" in result["source_text"]
    assert "Bình luận" not in result["source_text"]
