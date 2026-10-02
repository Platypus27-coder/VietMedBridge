"""Search formatting is separate from submission/source spans."""

from __future__ import annotations

import hashlib

from .text import normalize_for_retrieval

BUILDER_VERSION = "title-heading-source-v1"


def count_tokens(tokenizer, text: str, *, special: bool = False) -> int:
    encoded = tokenizer(text, add_special_tokens=special, truncation=False, verbose=False)
    return len(encoded.get("input_ids", encoded.get("offset_mapping", [])))


def build_dense_text(chunk: dict, *, title: str = "", heading: str = "",
                     tokenizer=None, max_tokens: int | None = None) -> str:
    body = normalize_for_retrieval(chunk["text"])
    prefix = "\n\n".join(dict.fromkeys(
        normalize_for_retrieval(value) for value in (title, heading) if value.strip()
    ))
    result = prefix + "\n\n" + body if prefix else body
    if tokenizer is None or max_tokens is None:
        return result
    if count_tokens(tokenizer, body, special=True) > max_tokens:
        raise ValueError("Child body alone exceeds dense input budget; rechunk rather than silently truncate.")
    if count_tokens(tokenizer, result, special=True) <= max_tokens:
        return result
    # Only the optional prefix is shortened. The complete child source body stays searchable.
    while prefix:
        prefix = prefix[:len(prefix) // 2].rstrip()
        result = prefix + "\n\n" + body if prefix else body
        if count_tokens(tokenizer, result, special=True) <= max_tokens:
            return result
    return body


def build_sparse_text(chunk: dict, *, title: str = "", heading: str = "") -> str:
    return build_dense_text(chunk, title=title, heading=heading)


def representation_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
