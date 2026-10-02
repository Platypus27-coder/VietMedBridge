"""BGE-token windows with exact character offsets into source_text."""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from .artifacts import digest_json
from .text import normalize_for_retrieval
from .representation import BUILDER_VERSION, build_dense_text, count_tokens, representation_hash
from .structure import STRUCTURE_VERSION, parse_structure, validate_structure

DEFAULT_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@dataclass(frozen=True)
class ChunkConfig:
    child_tokens: int = 180
    overlap_tokens: int = 40
    parent_tokens: int = 512
    snap_sentences: bool = True
    prefer_sections: bool = True
    dense_max_tokens: int = 512

    def validate(self):
        if self.child_tokens < 1 or not 0 <= self.overlap_tokens < self.child_tokens:
            raise ValueError("Require child_tokens > overlap_tokens >= 0.")
        if self.parent_tokens < self.child_tokens:
            raise ValueError("Parent token budget must be at least the child budget.")
        if self.dense_max_tokens < self.child_tokens + 2:
            raise ValueError("Dense budget must leave room for child body and model special tokens.")


def load_bge_tokenizer(revision: str | None = None):
    model_id = "BAAI/bge-m3"
    pinned = HfApi().model_info(model_id, revision=revision or DEFAULT_BGE_REVISION).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=pinned, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer with character offsets is required.")
    return tokenizer, {"model_id": model_id, "revision": pinned, "special_tokens": False}


def _span_id(doc_id: int, source_sha: str, start: int, end: int, policy: str, kind: str) -> str:
    payload = f"{doc_id}:{source_sha}:{start}:{end}:{policy}:{kind}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_source(doc, tokenizer, structure=None, *, prefer_sections=True):
    text = doc["source_text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != doc["source_text_sha256"]:
        raise ValueError("Source text/hash mismatch.")
    # SentencePiece can attach the preceding newline to a heading token. Encoding
    # each exact section avoids splitting that token at an artificial boundary.
    structure = structure or parse_structure(doc, doc.get("heading_hints"))
    regions = structure["sections"] if prefer_sections else [{"start_char": 0, "end_char": len(text)}]
    offsets = []
    for region in regions:
        lower, upper = region["start_char"], region["end_char"]
        encoded = tokenizer(text[lower:upper], add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)
        offsets.extend((lower + int(a), lower + int(b)) for a, b in encoded["offset_mapping"] if int(b) > int(a))
    starts, ends = [a for a, _ in offsets], [b for _, b in offsets]
    if starts != sorted(starts) or ends != sorted(ends):
        raise ValueError("Tokenizer returned non-monotonic source offsets.")
    return offsets, starts, ends


def _parent(doc, child, tokenizer, tokenizer_spec, config, structure, offsets, starts, ends):
    text, sha = doc["source_text"], doc["source_text_sha256"]
    section = next((row for row in structure["sections"] if
                    row["start_char"] <= child["start_char"] < child["end_char"] <= row["end_char"]),
                   {"start_char": 0, "end_char": len(text), "section_id": None, "heading_text": ""})
    lower = bisect.bisect_left(starts, section["start_char"]) if config.prefer_sections else 0
    upper = bisect.bisect_right(ends, section["end_char"]) if config.prefer_sections else len(offsets)
    anchor_start = bisect.bisect_right(ends, child["start_char"])
    anchor_end = bisect.bisect_left(starts, child["end_char"])
    left = max(lower, (anchor_start + anchor_end) // 2 - config.parent_tokens // 2)
    right = min(upper, left + config.parent_tokens)
    left = max(lower, right - config.parent_tokens)
    a, b = starts[left], ends[right - 1]
    if config.snap_sentences:
        candidates = structure["paragraphs"] + structure["sentences"]
        left_choices = sorted(row["start_char"] for row in candidates
                              if a <= row["start_char"] <= child["start_char"])
        right_choices = sorted(row["end_char"] for row in candidates
                               if child["end_char"] <= row["end_char"] <= b)
        if left_choices:
            a = left_choices[0]
        if right_choices:
            b = right_choices[-1]
    a, b = min(a, child["start_char"]), max(b, child["end_char"])
    # Retokenizing a slice can differ from tokenizing the entire document.
    while count_tokens(tokenizer, text[a:b]) > config.parent_tokens:
        if a < child["start_char"] and left < anchor_start:
            left += 1
            a = max(a, starts[left])
        elif b > child["end_char"] and right > anchor_end:
            right -= 1
            b = min(b, ends[right - 1])
        else:
            raise ValueError("Parent cannot contain child within its token budget.")
    parent_policy = digest_json({
        "parent_tokens": config.parent_tokens, "snap_sentences": config.snap_sentences,
        "prefer_sections": config.prefer_sections, "tokenizer": tokenizer_spec,
        "structure": STRUCTURE_VERSION, "algorithm": "parent-expansion-v3",
    })
    within_section = section["start_char"] <= a < b <= section["end_char"]
    return {
        "doc_id": int(doc["doc_id"]), "source_text_sha256": sha,
        "language": doc.get("language", "unknown"), "policy_sha256": parent_policy,
        "chunk_id": _span_id(doc["doc_id"], sha, a, b, parent_policy, "parent"),
        "section_id": section["section_id"] if within_section else None,
        "heading": section["heading_text"] if within_section else "",
        "start_char": a, "end_char": b, "token_start": bisect.bisect_right(ends, a),
        "token_end": bisect.bisect_left(starts, b), "token_count": count_tokens(tokenizer, text[a:b]),
        "text": text[a:b], "retrieval_text": normalize_for_retrieval(text[a:b]),
    }


def parent_for_child(doc: dict, child: dict, tokenizer, tokenizer_spec: dict,
                     *, parent_tokens: int = 512, structure: dict | None = None) -> dict:
    """Rebuild output context without changing child identity or embedding input."""
    if child["doc_id"] != doc["doc_id"] or child["source_text_sha256"] != doc["source_text_sha256"]:
        raise ValueError("Child/source identity mismatch.")
    if not 0 <= child["start_char"] < child["end_char"] <= len(doc["source_text"]):
        raise ValueError("Invalid child source bounds.")
    if child["text"] != doc["source_text"][child["start_char"]:child["end_char"]]:
        raise ValueError("Child source slice mismatch.")
    if count_tokens(tokenizer, child["text"]) > parent_tokens:
        raise ValueError("Requested parent budget is smaller than child.")
    structure = structure or parse_structure(doc, doc.get("heading_hints"))
    offsets, starts, ends = _encode_source(doc, tokenizer, structure)
    config = ChunkConfig(parent_tokens=parent_tokens)
    return _parent(doc, child, tokenizer, tokenizer_spec, config, structure, offsets, starts, ends)


def chunk_source(doc: dict, tokenizer, tokenizer_spec: dict,
                 config: ChunkConfig | None = None, *, structure: dict | None = None) -> tuple[list[dict], list[dict]]:
    config = config or ChunkConfig()
    config.validate()
    text = doc["source_text"]
    actual_sha = doc["source_text_sha256"]
    structure = structure or parse_structure(doc, doc.get("heading_hints"))
    validate_structure(doc, structure)
    offsets, starts, ends = _encode_source(doc, tokenizer, structure, prefer_sections=config.prefer_sections)
    if not offsets:
        return [], []
    policy = digest_json({
        "child_tokens": config.child_tokens, "overlap_tokens": config.overlap_tokens,
        "snap_sentences": config.snap_sentences, "prefer_sections": config.prefer_sections,
        "tokenizer": tokenizer_spec, "structure": STRUCTURE_VERSION,
        "algorithm": "structure-source-window-v3",
    })
    children, parents, parent_ids, seen = [], [], set(), set()
    index, token_count = 0, len(offsets)
    while index < token_count:
        section = next(row for row in structure["sections"] if row["start_char"] <= starts[index] < row["end_char"])
        section_stop = bisect.bisect_right(ends, section["end_char"]) if config.prefer_sections else token_count
        stop = min(index + config.child_tokens, section_stop)
        if stop <= index:
            raise ValueError("Tokenizer token crosses a section boundary.")
        if config.snap_sentences and stop < token_count:
            for kind in ("paragraphs", "sentences"):
                boundaries = sorted(row["end_char"] for row in structure[kind]
                                    if starts[index] < row["end_char"] <= ends[stop - 1])
                if boundaries:
                    aligned = bisect.bisect_right(ends, boundaries[-1], lo=index, hi=stop)
                    if aligned - index >= max(config.overlap_tokens + 1, config.child_tokens // 2):
                        stop = aligned
                        break
        while stop > index + 1 and count_tokens(tokenizer, text[starts[index]:ends[stop - 1]]) > config.child_tokens:
            stop -= 1
        start_char, end_char = starts[index], ends[stop - 1]
        if not 0 <= start_char < end_char <= len(text):
            raise ValueError("Invalid source character offsets.")
        common = {
            "doc_id": int(doc["doc_id"]), "source_text_sha256": actual_sha,
            "language": doc.get("language", "unknown"), "policy_sha256": policy,
        }
        if (start_char, end_char) not in seen:
            seen.add((start_char, end_char))
            child_text = text[start_char:end_char]
            child = {
                **common, "chunk_id": _span_id(doc["doc_id"], actual_sha, start_char, end_char, policy, "child"),
                "chunk_order": len(children),
                "section_id": section["section_id"], "heading": section["heading_text"],
                "start_char": start_char, "end_char": end_char,
                "token_start": index, "token_end": stop,
                "token_count": count_tokens(tokenizer, child_text),
                "text": child_text, "retrieval_text": normalize_for_retrieval(child_text),
            }
            if child["token_count"] > config.child_tokens:
                raise ValueError("A source token exceeds the child budget.")
            dense = build_dense_text(child, title=doc.get("title", ""), heading=section["heading_text"],
                                     tokenizer=tokenizer, max_tokens=config.dense_max_tokens)
            child.update(dense_token_count=count_tokens(tokenizer, dense, special=True),
                         retrieval_representation_hash=representation_hash(dense),
                         representation_builder_version=BUILDER_VERSION,
                         quality_flags=["CROSS_SECTION"] if end_char > section["end_char"] else [])
            parent = _parent(doc, child, tokenizer, tokenizer_spec, config, structure, offsets, starts, ends)
            child["parent_id"] = parent["chunk_id"]
            if parent["chunk_id"] not in parent_ids:
                parent_ids.add(parent["chunk_id"])
                parents.append({**parent, "chunk_order": len(parents)})
            children.append(child)
        if stop >= token_count:
            break
        index = stop if config.prefer_sections and stop == section_stop else max(index + 1, stop - config.overlap_tokens)
    return children, parents
