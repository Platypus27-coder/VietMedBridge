"""Section/paragraph/sentence spans refer to immutable extracted source text."""

from __future__ import annotations

import hashlib
import re

STRUCTURE_VERSION = "exact-line-structure-v1"
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "fig", "vs", "etc", "al", "ts", "ths"}


def span_id(doc_id: int, source_sha: str, start: int, end: int, kind: str) -> str:
    return hashlib.sha256(f"{doc_id}:{source_sha}:{start}:{end}:{kind}:{STRUCTURE_VERSION}".encode()).hexdigest()


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    result, cursor = [], start
    for match in re.finditer(r"[。！？]+|[!?]+(?=\s|$)|\.(?=\s|$)", text[start:end]):
        stop = start + match.end()
        if match.group() == ".":
            before = text[start:stop - 1]
            word = re.search(r"([\w.]+)$", before)
            word = word.group(1).lower() if word else ""
            if word in ABBREVIATIONS or len(word) == 1 or word.endswith(("e.g", "i.e")):
                continue
        left, right = cursor, stop
        while left < right and text[left].isspace():
            left += 1
        if left < right:
            result.append((left, right))
        cursor = stop
    while cursor < end and text[cursor].isspace():
        cursor += 1
    right = end
    while right > cursor and text[right - 1].isspace():
        right -= 1
    if cursor < right:
        result.append((cursor, right))
    return result


def parse_structure(doc: dict, heading_hints: list[str] | None = None) -> dict:
    text, doc_id, sha = doc["source_text"], doc["doc_id"], doc["source_text_sha256"]
    headings = set(hint.strip() for hint in (heading_hints or []) if hint.strip())
    anchors = []
    for line in re.finditer(r"[^\n]+", text):
        value = line.group().strip()
        if value in headings:
            left = line.start() + len(line.group()) - len(line.group().lstrip())
            anchors.append((left, value))
    if not anchors or anchors[0][0] > 0:
        anchors.insert(0, (0, ""))
    sections, paragraphs, sentences = [], [], []
    for index, (left, heading) in enumerate(anchors):
        right = anchors[index + 1][0] if index + 1 < len(anchors) else len(text)
        if left >= right:
            continue
        section_id = span_id(doc_id, sha, left, right, "section")
        sections.append({
            "doc_id": doc_id, "source_text_sha256": sha, "section_id": section_id,
            "heading_text": heading, "start_char": left, "end_char": right,
            "section_index": len(sections), "parser_version": STRUCTURE_VERSION,
        })
        for block in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text[left:right], re.S):
            p_start, p_end = left + block.start(), left + block.end()
            while p_end > p_start and text[p_end - 1].isspace():
                p_end -= 1
            p_id = span_id(doc_id, sha, p_start, p_end, "paragraph")
            paragraphs.append({
                "doc_id": doc_id, "section_id": section_id, "paragraph_id": p_id,
                "start_char": p_start, "end_char": p_end,
            })
            sentences.extend({
                "doc_id": doc_id, "section_id": section_id, "paragraph_id": p_id,
                "start_char": a, "end_char": b,
            } for a, b in sentence_spans(text, p_start, p_end))
    return {"version": STRUCTURE_VERSION, "sections": sections,
            "paragraphs": paragraphs, "sentences": sentences,
            "known_heading_count": sum(bool(section["heading_text"]) for section in sections)}


def validate_structure(doc: dict, structure: dict) -> None:
    text = doc["source_text"]
    sections = {row["section_id"]: row for row in structure["sections"]}
    previous = -1
    for row in structure["sections"]:
        if row["doc_id"] != doc["doc_id"] or row["source_text_sha256"] != doc["source_text_sha256"]:
            raise ValueError("Structure source identity mismatch.")
        if not 0 <= row["start_char"] < row["end_char"] <= len(text) or row["start_char"] < previous:
            raise ValueError("Invalid/non-monotonic section bounds.")
        previous = row["end_char"]
    for row in structure["paragraphs"] + structure["sentences"]:
        parent = sections[row["section_id"]]
        if row["doc_id"] != doc["doc_id"] or not (
            parent["start_char"] <= row["start_char"] < row["end_char"] <= parent["end_char"]
        ):
            raise ValueError("Structure span outside its document/section.")
