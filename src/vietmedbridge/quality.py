"""Conservative diagnostics: quality is never a document relevance filter."""

from __future__ import annotations

import re
from collections import Counter

QUALITY_VERSION = "quality-rules-v1"
NORMALIZER_VERSION = "nfc-whitespace-v1"
ERROR_TITLES = re.compile(
    r"^(?:just a moment|access denied|attention required|robot check|"
    r"403(?:\s+forbidden)?|404(?:\s+(?:not found|error))?|page not found|"
    r"sign in to continue|enable javascript)(?:\b|$)", re.I,
)


def error_page_reason(title: str, text: str) -> str | None:
    """Avoid rejecting articles merely mentioning an error message in their body."""
    candidate = title.strip() or (text.strip().splitlines() or [""])[0]
    if not ERROR_TITLES.search(candidate):
        return None
    lower = candidate.lower()
    if "javascript" in lower:
        return "JS_REQUIRED"
    if "sign in" in lower:
        return "LOGIN_REQUIRED"
    if "404" in lower or "not found" in lower:
        return "NOT_FOUND_PAGE"
    if "access denied" in lower or "403" in lower:
        return "ACCESS_DENIED_PAGE"
    return "BOT_CHALLENGE"


def document_quality(text: str, *, title: str = "", section_count: int = 0,
                     raw_bytes: int | None = None, content_type: str = "") -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    duplicate_ratio = sum(count - 1 for count in counts.values()) / max(1, len(lines))
    nonspace = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in nonspace) / max(1, len(nonspace))
    paragraphs = len(re.findall(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, re.S))
    reasons = []
    if len(text.strip()) < 80:
        reasons.append("TOO_SHORT")
    if not title.strip():
        reasons.append("MISSING_TITLE")
    if not section_count:
        reasons.append("NO_SECTIONS")
    if len(text) > 2000 and paragraphs <= 1:
        reasons.append("ONE_GIANT_PARAGRAPH")
    if duplicate_ratio > 0.3:
        reasons.append("HIGH_DUPLICATE_LINE_RATIO")
    if alpha_ratio < 0.3:
        reasons.append("LOW_ALPHABETIC_RATIO")
    # This is a warning, never proof of truncation or a reason to drop a document.
    if raw_bytes and raw_bytes > 100_000 and len(text) < 200:
        reasons.append("POSSIBLE_TRUNCATION")
    medical_signals = len(re.findall(
        r"\b(?:HbA1c|BRCA\d|HER2|SARS-CoV-2|ICD-10|TNM|H\.\s*pylori)\b|"
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|mmol|ml)(?:/kg)?\b", text, re.I,
    ))
    low = any(reason in reasons for reason in (
        "TOO_SHORT", "HIGH_DUPLICATE_LINE_RATIO", "LOW_ALPHABETIC_RATIO",
    ))
    tier = "LOW" if low else ("HIGH" if len(text) >= 300 else "MEDIUM")
    return {
        "quality_tier": tier, "quality_flags": reasons,
        "quality_version": QUALITY_VERSION,
        "char_count": len(text), "paragraph_count": paragraphs,
        "alphabetic_ratio": alpha_ratio, "duplicate_line_ratio": duplicate_ratio,
        "biomedical_signal_count": medical_signals,
    }


def reason_code(reason: str) -> str:
    value = reason.split(":", 1)[-1] if reason.startswith("ValueError:") else reason
    if "empty_extracted_text_or_scanned_pdf" in value:
        return "EMPTY_CONTENT_OR_SCAN_PDF"
    if "blocked_or_challenge_page" in value:
        return value.rsplit(":", 1)[-1].upper() if ":" in value else "BOT_CHALLENGE"
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")[:100]
