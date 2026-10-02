"""Source-derived extraction. Offsets refer to extracted text, not HTML byte offsets."""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata

import trafilatura
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect_langs
from lxml import etree
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .quality import error_page_reason

DetectorFactory.seed = 0
EXPECTED_PARSE_ERRORS = (ValueError, etree.LxmlError, PyPdfError)


def normalize_for_retrieval(text: str) -> str:
    """This representation never replaces source_text or submission text."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def language_hint(text: str, declared: str = "") -> tuple[str, str]:
    declared = declared.lower().split("-")[0].split("_")[0]
    if declared in ("vi", "en", "zh"):
        return declared, "source-declared"
    if len(text.strip()) < 80:
        return "unknown", "too-short"
    try:
        prediction = detect_langs(text[:12000])[0]
        language = "zh" if prediction.lang.startswith("zh") else prediction.lang
        if language in ("vi", "en", "zh") and prediction.prob >= 0.80:
            return language, f"langdetect:{prediction.prob:.3f}"
    except LangDetectException:
        pass
    return "unknown", "uncertain"


def extract_source(body: bytes, content_type: str = "") -> dict:
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
        soup = BeautifulSoup(body, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        declared = str(soup.html.get("lang", "")) if soup.html else ""
        headings = [node.get_text(" ", strip=True) for node in soup.find_all(re.compile(r"^h[1-6]$"))]
        raw_has_table = soup.find("table") is not None
        if error_page_reason(title, ""):
            raise ValueError("blocked_or_challenge_page:" + (error_page_reason(title, "") or "UNKNOWN"))
        source = trafilatura.extract(
            str(soup), output_format="txt", include_tables=True,
            include_comments=False, favor_recall=True, deduplicate=False,
        ) or ""
        source = source.strip()
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
