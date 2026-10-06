"""Derived 512/640-token source parents; frozen files and official IDs stay intact."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from copy import copy

from .artifacts import digest_json
from .retrieval_data import Catalog


class SourceCatalog(Catalog):
    @property
    def identity(self):
        return {**super().identity, "derived_parent_policy_sha256": self.parent_policy_sha256}


def derive_parents(catalog, tokenizer, budgets=(512, 640)):
    if not budgets or any(type(n) is not int or n < 180 for n in budgets):
        raise ValueError("Invalid parent token budgets.")
    children = {k: dict(v) for k, v in catalog.children.items()}
    parents = dict(catalog.parents)
    per_doc = {}
    for child_id, child in children.items():
        doc_id = child["doc_id"]
        source = catalog.documents[doc_id]["source_text"]
        if doc_id not in per_doc:
            offsets = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)["offset_mapping"]
            per_doc[doc_id] = offsets, [b for a, b in offsets], [a for a, b in offsets]
        offsets, ends, starts = per_doc[doc_id]
        first, last = bisect_right(ends, child["start_char"]), bisect_left(starts, child["end_char"])
        if last - first > min(budgets):
            raise ValueError("A child cannot fit the parent budget.")
        alternatives = {}
        for budget in budgets:
            start = max(0, first - (budget - (last - first)) // 2)
            end = min(len(offsets), start + budget)
            start = max(0, end - budget)
            while True:
                a, b = min(offsets[start][0], child["start_char"]), max(offsets[end - 1][1], child["end_char"])
                # Retokenizing a slice can differ at its edges. Trim context only,
                # retaining the complete anchor, until the actual slice fits.
                before = source.rfind("\n\n", a, child["start_char"])
                after = source.find("\n\n", child["end_char"], b)
                if before >= a:
                    a = before + 2
                if after >= child["end_char"]:
                    b = after
                if not a <= child["start_char"] < child["end_char"] <= b:
                    raise ValueError("Derived parent lost its anchor child.")
                text = source[a:b]
                if len(tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]) <= budget:
                    break
                if first - start >= end - last and start < first:
                    start += 1
                elif end > last:
                    end -= 1
                elif start < first:
                    start += 1
                else:
                    raise ValueError("Anchor child alone exceeds the parent token budget.")
            parent_id = "source-parent-" + digest_json([doc_id, child["source_text_sha256"], a, b, budget])
            parents[parent_id] = {"chunk_id": parent_id, "doc_id": doc_id, "start_char": a,
                "end_char": b, "source_text_sha256": child["source_text_sha256"], "text": text,
                "token_budget": budget, "derived_from_frozen_source": True}
            alternatives[str(budget)] = parent_id
        child["source_parent_alternatives"] = alternatives
    result = SourceCatalog(copy(catalog.candidate), catalog.documents, children, parents,
                           catalog.units, catalog.aliases, catalog.build_config)
    result.parent_policy_sha256 = digest_json({"budgets": budgets, "policy": "center-anchor-paragraph-inward-retokenize-v2",
        "parents": {k: v for k, v in parents.items() if k.startswith("source-parent-")}})
    return result
