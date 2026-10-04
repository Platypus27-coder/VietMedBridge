"""Source-derived extraction. Offsets refer to extracted text, not HTML byte offsets."""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from urllib.parse import urlsplit

import trafilatura
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect_langs
from lxml import etree
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .quality import error_page_reason, has_encoded_payload
from .domain_text import laodong_article, longchau_article
from .source_challenges import laodong_cookie_challenge

DetectorFactory.seed = 0
EXPECTED_PARSE_ERRORS = (ValueError, etree.LxmlError, PyPdfError)


def normalize_for_retrieval(text: str) -> str:
    """This representation never replaces source_text or submission text."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def language_hint(text: str, declared: str = "") -> tuple[str, str]:
    declared = declared.lower().split("-")[0].split("_")[0]
    if len(text.strip()) < 80:
        if declared in ("vi", "en", "zh"):
            return declared, "source-declared-short-text"
        return "unknown", "too-short"
    letters = [char for char in text[:12000] if char.isalpha()]
    han = sum("\u3400" <= char <= "\u9fff" for char in letters)
    if han >= 20 and han / max(1, len(letters)) >= 0.20:
        return "zh", "content-han-script"
    vietnamese_marks = sum(char in (
        "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
        "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
        "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
    ) for char in letters)
    if vietnamese_marks >= 4 and vietnamese_marks / max(1, len(letters)) >= 0.01:
        return "vi", "content-vietnamese-script"
    try:
        prediction = detect_langs(text[:12000])[0]
        language = "zh" if prediction.lang.startswith("zh") else prediction.lang
        if language in ("vi", "en", "zh") and prediction.prob >= 0.80:
            return language, f"langdetect:{prediction.prob:.3f}"
    except LangDetectException:
        pass
    if declared in ("vi", "en", "zh"):
        return declared, "source-declared-fallback"
    return "unknown", "uncertain"


def _redirected_to_homepage(requested_url: str, final_url: str) -> bool:
    requested, final = urlsplit(requested_url), urlsplit(final_url)
    requested_path = requested.path.rstrip("/").lower()
    final_path = final.path.rstrip("/").lower()
    return (requested_path not in ("", "/", "/index.html", "/index.htm")
            and final_path in ("", "/", "/index.html", "/index.htm")
            and not final.query)


_AUXILIARY_CLASS = re.compile(
    r"(?:^|[-_])(?:related|recommend(?:ed)?|sidebar|latest|mostread|"
    r"doctor-list|hospital-list|tin-lien-quan|bai-viet-lien-quan)(?:$|[-_])", re.I,
)
_AUXILIARY_TAIL = re.compile(
    r"(?:健康资讯推荐|推荐专家更多|推荐医院更多|热门问答更多|"
    r"(?:^|\n)\s*(?:Bài viết liên quan|Tin liên quan|Tin mới nhất)\s*(?:\n|:))",
    re.I,
)


def _remove_auxiliary_dom(soup: BeautifulSoup) -> None:
    for node in list(soup.select("nav, aside, footer, [role=navigation], [role=complementary]")):
        node.decompose()
    for node in list(soup.find_all(True)):
        if not node.parent or node.name in ("html", "body", "main", "article"):
            continue
        tokens = [str(node.get("id", "")), *[str(item) for item in node.get("class", [])]]
        if any(_AUXILIARY_CLASS.search(token) for token in tokens) and not node.find("article"):
            node.decompose()


def _trim_auxiliary_tail(source: str) -> str:
    match = _AUXILIARY_TAIL.search(source)
    if match and len(source[:match.start()].strip()) >= 300:
        return source[:match.start()].strip()
    return source


def extract_source(body: bytes, content_type: str = "", *, source_url: str | None = None,
                   requested_url: str | None = None) -> dict:
    if source_url and requested_url and _redirected_to_homepage(requested_url, source_url):
        raise ValueError("article_redirected_to_homepage")
    media = content_type.lower().split(";")[0].strip()
    prefix = body[:4096].lstrip().lower()
    title, declared, headings, raw_has_table = "", "", [], False
    if body.startswith(b"%PDF") or media == "application/pdf":
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("encrypted_pdf")
        # Page order is preserved; scanned PDFs are reported, never replaced by invented text.
        source = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        title = str((reader.metadata or {}).get("/Title", ""))
        parser = "pypdf-text-v1"
    elif media in ("application/xml", "text/xml") or (
        prefix.startswith(b"<?xml") and b"<html" not in prefix
    ):
        root = etree.fromstring(body, parser=etree.XMLParser(
            resolve_entities=False, no_network=True, recover=False,
        ))
        raw_has_table = bool(root.xpath("//*[local-name()='table' or local-name()='table-wrap']"))
        title_nodes = root.xpath("//*[local-name()='article-title' or local-name()='ArticleTitle']")
        title = " ".join("".join(node.itertext()).strip() for node in title_nodes)
        headings = ["".join(node.itertext()).strip() for node in root.xpath(
            "//*[local-name()='sec']/*[local-name()='title']"
        )]
        nodes = root.xpath("//*[local-name()='abstract' or local-name()='Abstract' or local-name()='body']")
        if not nodes:
            nodes = [root]
        pieces = []
        if title:
            pieces.append(title)
        for node in nodes:
            # Avoid processing a nested abstract/body a second time.
            if any(parent in nodes for parent in node.iterancestors()):
                continue
            paragraphs = node.xpath(
                ".//*[local-name()='p' or local-name()='title' or local-name()='AbstractText' or local-name()='table']"
            )
            if paragraphs:
                paragraph_set = set(paragraphs)
                for block in paragraphs:
                    if any(parent in paragraph_set for parent in block.iterancestors()):
                        continue
                    if etree.QName(block).localname == "table":
                        for row in block.xpath(".//*[local-name()='tr']"):
                            cells = row.xpath("./*[local-name()='th' or local-name()='td']")
                            pieces.append(" | ".join(" ".join("".join(cell.itertext()).split()) for cell in cells))
                    else:
                        pieces.append("".join(block.itertext()).strip())
            else:
                pieces.append(" ".join(t.strip() for t in node.itertext() if t.strip()))
        source = "\n\n".join(piece for piece in pieces if piece).strip()
        declared = root.get("{http://www.w3.org/XML/1998/namespace}lang", "")
        parser = "xml-text-v2"
    elif "html" in media or b"<html" in prefix or b"<!doctype html" in prefix:
        if source_url and laodong_cookie_challenge(body, source_url):
            raise ValueError("laodong_cookie_challenge_requires_recrawl")
        soup = BeautifulSoup(body, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        declared = str(soup.html.get("lang", "")) if soup.html else ""
        canonical = soup.select_one('link[rel="canonical"][href]')
        if canonical and source_url and _redirected_to_homepage(source_url, str(canonical.get("href"))):
            raise ValueError("article_canonical_points_to_homepage")
        raw_has_table = soup.find("table") is not None
        if error_page_reason(title, ""):
            raise ValueError("blocked_or_challenge_page:" + (error_page_reason(title, "") or "UNKNOWN"))
        adapted = None
        if source_url:
            for adapter in (longchau_article, laodong_article):
                adapted = adapter(soup, source_url)
                if adapted is not None:
                    break
        if adapted is not None:
            source, title = adapted["source_text"], adapted["title"]
            headings, raw_has_table, parser = adapted["heading_hints"], adapted["raw_has_table"], adapted["parser"]
        else:
            _remove_auxiliary_dom(soup)
            headings = [node.get_text(" ", strip=True) for node in soup.find_all(re.compile(r"^h[1-6]$"))]
            source = trafilatura.extract(
                str(soup), output_format="txt", include_tables=True,
                include_comments=False, favor_recall=True, deduplicate=False,
            ) or ""
            source = _trim_auxiliary_tail(source.strip())
            parser = "trafilatura-text-v1"
        if not source:
            for node in soup.select("script, style, nav, header, footer, aside, form"):
                node.decompose()
            container = soup.find("article") or soup.find("main") or soup.body
            source = container.get_text("\n\n", strip=True) if container else ""
            parser = "beautifulsoup-basic-v1"
    elif media.startswith("text/plain"):
        source = body.decode("utf-8-sig", errors="strict").strip()
        parser = "utf8-plain-v1"
    else:
        raise ValueError(f"unsupported_content_type:{media or 'unknown'}")
    if not source:
        raise ValueError("empty_extracted_text_or_scanned_pdf")
    if has_encoded_payload(source):
        raise ValueError("encoded_payload_in_extracted_text")
    error = error_page_reason(title, source)
    if error:
        raise ValueError("blocked_or_challenge_page:" + error)
    language, method = language_hint(source, declared)
    return {
        "title": title, "source_text": source,
        "source_text_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "language": language, "language_method": method, "parser": parser,
        "language_confidence": float(method.split(":")[1]) if method.startswith("langdetect:") else None,
        "quality_flags": ["short_text"] if len(source) < 80 else [],
        "heading_hints": headings, "raw_has_table": raw_has_table,
    }
