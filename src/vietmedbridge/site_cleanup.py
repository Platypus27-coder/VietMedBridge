"""Conservative cleanup for source layouts observed in the Stage A pilot.

Only remove site UI with an anchored, source-specific signature. These rules
run before chunking, so every retained chunk still comes from extracted source.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit


_YOULAI_HEADER = re.compile(
    r"\A-视频-文章-语音\n[^\n]+\n\d{4}-\d{2}-\d{2} [^\n]*阅读：[^\n]*\n"
    r"手机浏览\n用手机扫描二维码在手机上继续观看\n"
)
_YOULAI_DOCTORS = re.compile(r"\n[\u3400-\u9fff\ufffd]{2,16}医生推荐(?:\n|\Z)")
_FAMILYDOCTOR_TEASER = re.compile(r"\n- [^\n]{4,100}\n-\n")
_FAMILYDOCTOR_EDITOR = re.compile(r"\n（责任编辑：[^\n]+）")
_MEDLATEC_HEADER = re.compile(
    r"\ATin tức\n(?P<title>[^\n]+)\n"
    r"(?P<links>(?:- \d{2}/\d{2}/\d{4} \| [^\n]+\n)+)"
)
_PHUTHO_HEADER = re.compile(r"\A\{title\}\n\{publish\}\n\{head\}\n")
_PHUTHO_COMMENTS = re.compile(r"\n\{name\} - \{time\}\n\{body\}")
_NGHEAN_RELATED = re.compile(r"\n\| TIN LIÊN QUAN \|\n\|---\|\s*\Z", re.I)


def clean_site_text(source: str, source_url: str) -> str:
    """Remove verified navigation/related blocks, preserving article wording."""
    host = (urlsplit(source_url).hostname or "").lower()
    cleaned = source.strip()

    if host == "www.youlai.cn":
        header = _YOULAI_HEADER.match(cleaned)
        if header:
            cleaned = cleaned[header.end():]
        doctors = _YOULAI_DOCTORS.search(cleaned)
        if doctors and len(cleaned[:doctors.start()].strip()) >= 120:
            # Some pages repeat the entire article after this heading.
            cleaned = cleaned[:doctors.start()]

    elif host == "baidianfeng.familydoctor.com.cn":
        teaser = _FAMILYDOCTOR_TEASER.search(cleaned)
        editor = _FAMILYDOCTOR_EDITOR.search(cleaned)
        if teaser and editor and teaser.start() < editor.start() and teaser.start() >= 250:
            cleaned = cleaned[:teaser.start()]

    elif host == "medlatec.vn":
        header = _MEDLATEC_HEADER.match(cleaned)
        if header:
            cleaned = header.group("title") + "\n" + cleaned[header.end():]
        comments = re.search(r"\nBình luận \(\)(?:\n|\Z)", cleaned)
        if comments and comments.start() >= 300:
            cleaned = cleaned[:comments.start()]

    elif host == "baophutho.vn":
        header = _PHUTHO_HEADER.match(cleaned)
        if header:
            cleaned = cleaned[header.end():]
        comments = _PHUTHO_COMMENTS.search(cleaned)
        if comments and comments.start() >= 300:
            cleaned = cleaned[:comments.start()]

    elif host == "www.pharmacity.vn":
        related = re.search(r"Xem chi tiếtCác bài viết liên quan", cleaned, re.I)
        if related and related.start() >= 300:
            cleaned = cleaned[:related.start()]

    elif host == "baonghean.vn":
        cleaned = _NGHEAN_RELATED.sub("", cleaned)

    return cleaned.strip()
