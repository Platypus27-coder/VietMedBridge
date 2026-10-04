"""Conservative diagnostics: quality is never a document relevance filter."""

from __future__ import annotations

import re
from collections import Counter

QUALITY_VERSION = "quality-rules-v3"
NORMALIZER_VERSION = "nfc-whitespace-v1"
ERROR_TITLES = re.compile(
    r"^(?:just a moment|access denied|attention required|robot check|"
    r"403(?:\s+forbidden)?|404(?:\s+(?:not found|error))?|page not found|"
    r"sign in to continue|enable javascript)(?:\b|$)", re.I,
)

_BASE64_LIKE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{180,}={0,2}(?![A-Za-z0-9+/])")


def has_encoded_payload(text: str) -> bool:
    """Find long, mixed-alphabet encoded blobs without treating DNA as base64."""
    for match in _BASE64_LIKE.finditer(text):
        value = match.group().rstrip("=")
        if (sum(char.islower() for char in value) >= 6
                and sum(char.isupper() for char in value) >= 6
                and sum(char.isdigit() for char in value) >= 4
                and len(set(value)) >= 30):
            return True
    return False


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
    if has_encoded_payload(text):
        reasons.append("ENCODED_PAYLOAD_SUSPECTED")
    if "\ufffd" in text:
        reasons.append("DECODE_REPLACEMENT_CHAR")
    # This is a warning, never proof of truncation or a reason to drop a document.
    if raw_bytes and raw_bytes > 100_000 and len(text) < 200:
        reasons.append("POSSIBLE_TRUNCATION")
    medical_signals = len(re.findall(
        r"\b(?:HbA1c|BRCA\d|HER2|SARS-CoV-2|ICD-10|TNM|H\.\s*pylori)\b|"
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|mmol|ml)(?:/kg)?\b", text, re.I,
    ))
    low = any(reason in reasons for reason in (
        "TOO_SHORT", "HIGH_DUPLICATE_LINE_RATIO", "LOW_ALPHABETIC_RATIO",
        "ENCODED_PAYLOAD_SUSPECTED",
    ))
    tier = "LOW" if low else ("HIGH" if len(text) >= 300 and "DECODE_REPLACEMENT_CHAR" not in reasons else "MEDIUM")
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
