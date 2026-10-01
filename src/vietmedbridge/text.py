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

DetectorFactory.seed = 0


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
    title, declared = "", ""
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
        title_nodes = root.xpath("//*[local-name()='article-title' or local-name()='ArticleTitle']")
        title = " ".join("".join(node.itertext()).strip() for node in title_nodes)
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
                ".//*[local-name()='p' or local-name()='title' or local-name()='AbstractText']"
            )
            if paragraphs:
                pieces.extend("".join(p.itertext()).strip() for p in paragraphs)
            else:
                pieces.append(" ".join(t.strip() for t in node.itertext() if t.strip()))
        source = "\n\n".join(piece for piece in pieces if piece).strip()
        declared = root.get("{http://www.w3.org/XML/1998/namespace}lang", "")
        parser = "xml-text-v1"
    elif "html" in media or b"<html" in prefix or b"<!doctype html" in prefix:
        soup = BeautifulSoup(body, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        declared = str(soup.html.get("lang", "")) if soup.html else ""
        if re.search(r"^(just a moment|access denied|attention required|robot check)", title, re.I):
            raise ValueError("blocked_or_challenge_page")
        source = trafilatura.extract(
            str(soup), output_format="txt", include_tables=True,
            include_comments=False, favor_recall=True, deduplicate=False,
        ) or ""
        source = source.strip()
        parser = "trafilatura-text-v1"
    elif media.startswith("text/plain"):
        source = body.decode("utf-8-sig", errors="strict").strip()
        parser = "utf8-plain-v1"
    else:
        raise ValueError(f"unsupported_content_type:{media or 'unknown'}")
    if not source:
        raise ValueError("empty_extracted_text_or_scanned_pdf")
    language, method = language_hint(source, declared)
    return {
        "title": title, "source_text": source,
        "source_text_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "language": language, "language_method": method, "parser": parser,
        "quality_flags": ["short_text"] if len(source) < 80 else [],
    }
