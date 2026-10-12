"""CPU output-policy experiments using immutable saved model scores.

Parent windows are derived only for retrieved children, one query at a time.
No corpus embeddings, vector index, model weights or source files are changed.
"""
from __future__ import annotations

from collections import ChainMap
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .chunks import ChunkConfig, _encode_source, _parent
from .reranker_training import validate_training_labels
from .retrieval_audit import verify_records
from .retrieval_eval import evaluate_predictions
from .retrieval_policy import select_parents
from .structure import parse_structure
from .submission import validate_submission


def _catalog(base, children, parents, identity):
    return SimpleNamespace(documents=base.documents, children=ChainMap(children, base.children),
        parents=ChainMap(parents, base.parents), identity=identity)


def _derive_query(base, ranking, tokenizer, tokenizer_spec, budget, merge_tokens):
    children, parents, documents = {}, {}, {}
    changed = deepcopy(ranking)
    for row in changed["children"]:
        child = dict(base.children[row["child_id"]])
        if child["doc_id"] not in documents:
            doc = base.documents[child["doc_id"]]
            structure = parse_structure(doc, doc.get("heading_hints"))
            documents[child["doc_id"]] = doc, structure, _encode_source(doc, tokenizer, structure)
        doc, structure, encoded = documents[child["doc_id"]]
        if child["source_text_sha256"] != doc["source_text_sha256"] or child["text"] != doc["source_text"][child["start_char"]:child["end_char"]]:
            raise ValueError("Replay anchor/source mismatch.")
        if len(tokenizer(child["text"], add_special_tokens=False)["input_ids"]) > budget:
            raise ValueError("Parent budget cannot contain the complete anchor.")
        parent = _parent(doc, child, tokenizer, tokenizer_spec, ChunkConfig(parent_tokens=budget), structure, *encoded)
        children[row["child_id"]] = child
        parents[parent["chunk_id"]] = parent
        child["source_parent_alternatives"] = {**child.get("source_parent_alternatives", {}), "replay": parent["chunk_id"]}
        row["parent_id"] = parent["chunk_id"]
    if merge_tokens:
        # Merge before near-duplicate suppression, preserving all anchor coverage.
        # Only exact overlapping/adjacent spans within the same known section.
        groups = []
        for row in changed["children"]:
            parent = parents[row["parent_id"]]
            matched = None
            for group in groups:
                prior = group["parent"]
                if (prior["doc_id"] != parent["doc_id"] or not parent.get("section_id")
                    or parent["section_id"] != prior.get("section_id")
                    or max(prior["start_char"], parent["start_char"]) > min(prior["end_char"], parent["end_char"])):
                    continue
                a, b = min(prior["start_char"], parent["start_char"]), max(prior["end_char"], parent["end_char"])
                text = documents[parent["doc_id"]][0]["source_text"][a:b]
                if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= merge_tokens:
                    identifier = "replay-merge-" + digest_json([parent["doc_id"], parent["source_text_sha256"], a, b, merge_tokens])
                    group["parent"] = {**parent, "chunk_id": identifier, "start_char": a, "end_char": b, "text": text}
                    group["rows"].append(row)
                    matched = group
                    break
            if matched is None:
                groups.append({"parent": parent, "rows": [row]})
        for group in groups:
            parent = group["parent"]
            parents[parent["chunk_id"]] = parent
            for row in group["rows"]:
                row["parent_id"] = parent["chunk_id"]
                children[row["child_id"]]["source_parent_alternatives"]["replay"] = parent["chunk_id"]
    return changed, children, parents


def replay_selection(records, queries, catalog, tokenizer, tokenizer_spec, config, output_dir, *,
                     parent_tokens=512, merge_tokens=None, max_new_queries=None):
    """Return source-validated records/report/catalog suitable for export_submission.

    Original ranking stays attached for attribution; selected source overlays are
    saved with each query. Only these small overlays are retained across queries.
    """
    config.validate()
    verify_records(records, queries)
    if type(parent_tokens) is not int or parent_tokens < 180:
        raise ValueError("Parent budget must be an integer >=180.")
    if merge_tokens is not None and (type(merge_tokens) is not int or merge_tokens < parent_tokens):
        raise ValueError("Merge budget must be at least the parent budget.")
    if max_new_queries is not None and (type(max_new_queries) is not int or max_new_queries < 0):
        raise ValueError("Invalid replay query limit.")
    if not tokenizer_spec.get("revision"):
        raise ValueError("Replay needs the pinned BGE tokenizer identity.")
    identity = {"catalog": catalog.identity, "queries_sha256": digest_json(queries),
        "source_records": [r["record_sha256"] for r in records], "tokenizer": tokenizer_spec,
        "selection": asdict(config), "parent_tokens": parent_tokens, "merge_tokens": merge_tokens,
        "code": {name: sha256_file(Path(__file__).with_name(name)) for name in
                 ("selection_replay.py", "retrieval_policy.py", "chunks.py", "structure.py")}}
    signature = digest_json(identity)
    root = Path(output_dir)
    config_path = root / "replay_config.json"
    if config_path.exists() and read_json(config_path) != identity:
        raise ValueError("Replay inputs/policy changed; use a new replay directory.")
    atomic_json(config_path, identity)
    output, all_children, all_parents, written = [], {}, {}, 0
    replay_identity = {**catalog.identity, "selection_replay_signature": signature}
    by_id = {q["id"]: q for q in queries}
    for original in records:
        query = by_id[original["prediction"]["id"]]
        path = root / f"query-{query['id']}.json"
        if path.exists():
            saved = read_json(path)
            if saved.get("signature") != signature or saved.get("source_record_sha256") != original["record_sha256"]:
                raise ValueError("Replay checkpoint belongs to another source run.")
        else:
            if max_new_queries is not None and written >= max_new_queries:
                break
            ranking, children, parents = _derive_query(catalog, original["ranking"], tokenizer, tokenizer_spec, parent_tokens, merge_tokens)
            local = _catalog(catalog, children, parents, replay_identity)
            prediction, provenance = select_parents(local, ranking, config, tokenizer, query["id"])
            used_children = {p["anchor_child_id"]: children[p["anchor_child_id"]] for p in provenance}
            used_parents = {p["parent_id"]: parents[p["parent_id"]] for p in provenance}
            saved = {"signature": signature, "query_sha256": original["query_sha256"],
                "prediction": prediction, "provenance": provenance, "ranking": original["ranking"],
                "ranking_scope": "ORIGINAL_SCORED_CANDIDATES_BEFORE_OUTPUT_REPLAY",
                "source_record_sha256": original["record_sha256"],
                "source_prediction_signature": original["signature"],
                "source_score_checkpoints": original.get("score_checkpoints", {}),
                "replay_sources": {"children": used_children, "parents": used_parents}}
            saved["record_sha256"] = digest_json(saved)
            atomic_json(path, saved)
            written += 1
        source = saved["replay_sources"]
        local = _catalog(catalog, source["children"], source["parents"], replay_identity)
        validate_submission([saved], [query], local,
            {"state": "COMPLETE", "signature": signature, "completed_queries": 1}, expected_count=1)
        for key, value in source["children"].items():
            prior = all_children.get(key)
            if prior:
                # The same anchor can be merged differently by different queries.
                alternatives = dict(prior["source_parent_alternatives"])
                alternatives.update({pid: pid for pid in value["source_parent_alternatives"].values()})
                value = {**value, "source_parent_alternatives": alternatives}
            all_children[key] = value
        all_parents.update(source["parents"])
        output.append(saved)
    report = {"signature": signature, "state": "COMPLETE" if len(output) == len(queries) else "IN_PROGRESS",
        "completed_queries": len(output), "requested_queries": len(queries), "model_calls": 0,
        "scope": "CPU_OUTPUT_REPLAY_OF_SAVED_SCORES", "parent_tokens": parent_tokens, "merge_tokens": merge_tokens}
    atomic_json(root / "predictions.json", report)
    return output, report, _catalog(catalog, all_children, all_parents, replay_identity)


def sweep_parent_windows(records, queries, catalog, tokenizer, tokenizer_spec, config, labels_document,
                         contest_queries, output_dir, *, budgets=(256, 384, 512, 640, 768), include_merge=True):
    """One-factor output sweeps on independent reviewed dev; never auto-promote."""
    labels = validate_training_labels(labels_document, contest_queries, split="dev")
    if {q["id"]: q["query"] for q in labels} != {q["id"]: q["query"] for q in queries}:
        raise ValueError("Parent sweep must use exactly the reviewed dev queries.")
    for q in labels:
        for ref in q["relevant_chunks"]:
            source = catalog.documents.get(ref["doc_id"])
            if source is None or ref["chunk_text"] not in source["source_text"]:
                raise ValueError("Dev gold must be an exact slice of the active frozen source.")
    trials = []
    variants = [(n, None) for n in budgets]
    if include_merge:
        variants.append((512, 768))
    for budget, merge in variants:
        name = f"parent-{budget}" + (f"-merge-{merge}" if merge else "")
        values, report, _ = replay_selection(records, queries, catalog, tokenizer, tokenizer_spec, config,
            Path(output_dir) / name, parent_tokens=budget, merge_tokens=merge)
        evaluation = evaluate_predictions([r["prediction"] for r in values], labels, tokenizer)
        trials.append({"variant": name, "parent_tokens": budget, "merge_tokens": merge,
            "signature": report["signature"], "evaluation": evaluation})
    if not trials or any(t["evaluation"]["combined_f2"] is None for t in trials):
        raise ValueError("Parent sweep needs document and chunk references.")
    trials.sort(key=lambda t: (-t["evaluation"]["combined_f2"], t["parent_tokens"], t["merge_tokens"] or 0))
    result = {"state": "DEV_OUTPUT_SWEEP_REQUIRES_HELD_OUT_VALIDATION", "split": "dev",
        "labels_sha256": digest_json(labels_document), "trials": trials, "best_dev_variant": trials[0]["variant"],
        "model_calls": 0, "auto_applied": False, "official_score": None}
    atomic_json(Path(output_dir) / "parent_sweep.json", result)
    return result


def evaluate_frozen_parent_window(records, queries, catalog, tokenizer, tokenizer_spec, config,
                                  labels_document, train_document, dev_document, contest_queries,
                                  output_dir, *, parent_tokens, merge_tokens=None):
    """One held-out score for the dev-selected window; never selects a policy."""
    from .heldout import validate_heldout_split
    expected = validate_heldout_split(labels_document, train_document, dev_document, contest_queries, catalog)
    if {q["id"]: q["query"] for q in expected} != {q["id"]: q["query"] for q in queries}:
        raise ValueError("Held-out output evaluation must cover its exact reviewed query split.")
    replayed, replay_report, _ = replay_selection(records, queries, catalog, tokenizer, tokenizer_spec, config,
        output_dir, parent_tokens=parent_tokens, merge_tokens=merge_tokens)
    evaluation = evaluate_predictions([r["prediction"] for r in replayed], labels_document["queries"], tokenizer)
    baseline = evaluate_predictions([r["prediction"] for r in records], labels_document["queries"], tokenizer)
    if evaluation["combined_f2"] is None or baseline["combined_f2"] is None:
        raise ValueError("Held-out parent evaluation requires document and chunk references.")
    result = {"state": "HELDOUT_EVALUATED_LOCAL_PROXY_NO_POLICY_SELECTION", "split": "heldout",
        "labels_sha256": digest_json(labels_document), "replay_signature": replay_report["signature"],
        "parent_tokens": parent_tokens, "merge_tokens": merge_tokens, "model_calls": 0,
        "baseline_evaluation": baseline, "evaluation": evaluation,
        "combined_f2_delta_vs_frozen_parent_policy": evaluation["combined_f2"] - baseline["combined_f2"],
        "official_score": None, "used_for_selection": False}
    atomic_json(Path(output_dir) / "heldout_parent_evaluation.json", result)
    return result
