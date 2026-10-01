"""BGE-token windows with exact character offsets into source_text."""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import asdict, dataclass

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from .artifacts import digest_json
from .text import normalize_for_retrieval

DEFAULT_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@dataclass(frozen=True)
class ChunkConfig:
    child_tokens: int = 180
    overlap_tokens: int = 40
    parent_tokens: int = 512
    snap_sentences: bool = True

    def validate(self):
        if self.child_tokens < 1 or not 0 <= self.overlap_tokens < self.child_tokens:
            raise ValueError("Require child_tokens > overlap_tokens >= 0.")
        if self.parent_tokens < self.child_tokens:
            raise ValueError("Parent token budget must be at least the child budget.")


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


def chunk_source(doc: dict, tokenizer, tokenizer_spec: dict,
                 config: ChunkConfig | None = None) -> tuple[list[dict], list[dict]]:
    config = config or ChunkConfig()
    config.validate()
    text = doc["source_text"]
    actual_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if doc["source_text_sha256"] != actual_sha:
        raise ValueError("Source text/hash mismatch.")
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False,
        verbose=False,
    )
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"] if int(b) > int(a)]
    if not offsets:
        return [], []
    starts, ends = [pair[0] for pair in offsets], [pair[1] for pair in offsets]
    if starts != sorted(starts) or ends != sorted(ends):
        raise ValueError("Tokenizer returned non-monotonic source offsets.")
    policy = digest_json({"config": asdict(config), "tokenizer": tokenizer_spec,
                          "algorithm": "source-offset-window-v1"})
    boundaries = [m.end() for m in re.finditer(r"[。！？]|[.!?](?:\s+|$)|\n{2,}", text)]
    children, parents, parent_ids, seen = [], [], set(), set()
    index, token_count = 0, len(offsets)
    while index < token_count:
        stop = min(index + config.child_tokens, token_count)
        if config.snap_sentences and stop < token_count:
            boundary_index = bisect.bisect_right(boundaries, ends[stop - 1]) - 1
            if boundary_index >= 0:
                aligned = bisect.bisect_right(ends, boundaries[boundary_index], lo=index, hi=stop)
                if aligned - index >= max(config.overlap_tokens + 1, config.child_tokens // 2):
                    stop = aligned
        start_char, end_char = starts[index], ends[stop - 1]
        if not 0 <= start_char < end_char <= len(text):
            raise ValueError("Invalid source character offsets.")
        parent_start = max(0, ((index + stop) // 2) - config.parent_tokens // 2)
        parent_stop = min(token_count, parent_start + config.parent_tokens)
        parent_start = max(0, parent_stop - config.parent_tokens)
        p_start, p_end = starts[parent_start], ends[parent_stop - 1]
        parent_id = _span_id(doc["doc_id"], actual_sha, p_start, p_end, policy, "parent")
        common = {
            "doc_id": int(doc["doc_id"]), "source_text_sha256": actual_sha,
            "language": doc.get("language", "unknown"), "policy_sha256": policy,
        }
        if parent_id not in parent_ids:
            parent_ids.add(parent_id)
            parent_text = text[p_start:p_end]
            parents.append({
                **common, "chunk_id": parent_id, "chunk_order": len(parents),
                "start_char": p_start, "end_char": p_end,
                "token_start": parent_start, "token_end": parent_stop,
                "text": parent_text, "retrieval_text": normalize_for_retrieval(parent_text),
            })
        if (start_char, end_char) not in seen:
            seen.add((start_char, end_char))
            child_text = text[start_char:end_char]
            children.append({
                **common, "chunk_id": _span_id(doc["doc_id"], actual_sha, start_char, end_char, policy, "child"),
                "parent_id": parent_id, "chunk_order": len(children),
                "start_char": start_char, "end_char": end_char,
                "token_start": index, "token_end": stop,
                "text": child_text, "retrieval_text": normalize_for_retrieval(child_text),
            })
        if stop >= token_count:
            break
        index = max(index + 1, stop - config.overlap_tokens)
    return children, parents
