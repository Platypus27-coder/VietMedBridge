"""Language-aware lexical analysis and reviewed soft alias fields."""
from __future__ import annotations

import re
import unicodedata
import importlib.metadata
from pathlib import Path

from .artifacts import digest_json, read_json, sha256_file
from .retrieval import lexical_tokens


class MedicalAnalyzer:
    def __init__(self, *, segmentation=True, glossary_path=None):
        self.segmentation = segmentation
        self.entries = []
        if glossary_path is not None:
            document = read_json(glossary_path)
            if document.get("reviewed") is not True:
                raise ValueError("Medical aliases need a reviewed glossary with source attribution.")
            self.entries = document["entries"]
            if any(not e.get("source") or not isinstance(e.get("aliases"), list) or len(e["aliases"]) < 2
                   or any(not isinstance(a, str) or not a.strip() for a in e["aliases"]) for e in self.entries):
                raise ValueError("Invalid medical alias entries.")
        self.identity = {"version": "medical-fields-pyvi-jieba-cjk-v1", "segmentation": segmentation,
            "glossary_sha256": digest_json(self.entries), "aliases": len(self.entries),
            "code_sha256": sha256_file(Path(__file__)),
            "libraries": {k: importlib.metadata.version(k) for k in ("pyvi", "jieba")} if segmentation else {}}
        if segmentation:
            from pyvi import ViTokenizer
            import jieba
            self.vi = ViTokenizer
            self.zh = jieba.Tokenizer()

    def analyze(self, text, language="unknown"):
        text = unicodedata.normalize("NFKC", text)
        # Retain biomedical Latin/punctuation anchors alongside language segmentation.
        tokens = lexical_tokens(text)
        if self.segmentation:
            if language in ("vi", "unknown"):
                tokens += self.vi.tokenize(text).casefold().split()
            if language in ("zh", "unknown"):
                tokens += [t.casefold() for t in self.zh.cut(text) if t.strip()]
        return tokens

    def aliases(self, text):
        matched = []
        for entry in self.entries:
            if any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text, re.I) for a in entry["aliases"]):
                matched.extend(entry["aliases"])
        return " ".join(dict.fromkeys(matched))
