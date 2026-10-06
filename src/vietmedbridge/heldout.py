"""Independent source/query validation and evaluation after selection is frozen."""
from .artifacts import atomic_json, digest_json
from .reranker_training import validate_training_labels, validate_query_split
from .retrieval_eval import evaluate_predictions, evaluate_candidate_recall
from .training_data import query_is_independent


def validate_heldout_split(document, train_document, dev_document, contest_queries, catalog):
    train, dev = validate_query_split(train_document,dev_document,contest_queries)
    heldout = validate_training_labels(document,contest_queries,split="heldout")
    existing = train + dev + contest_queries
    if ({q["id"] for q in heldout} & {q["id"] for q in existing}
        or any(not query_is_independent(q["query"],existing) for q in heldout)):
        raise ValueError("Held-out queries overlap training/dev/contest queries.")
    if any(not query_is_independent(q["query"],heldout[:i]) for i,q in enumerate(heldout)):
        raise ValueError("Duplicate held-out questions.")
    for q in heldout:
        for ref in q["relevant_chunks"]:
            if (ref["doc_id"] not in catalog.documents or ref["doc_id"] not in q["relevant_docs"]
                or not ref["chunk_text"] or ref["chunk_text"] not in catalog.documents[ref["doc_id"]]["source_text"]):
                raise ValueError("Held-out references must be exact frozen source slices.")
            if document.get("source_splits",{}).get(str(ref["doc_id"]),"heldout") != "heldout":
                raise ValueError("Held-out positive crosses a reserved source fold.")
    return heldout


def evaluate_frozen_selection(records, labels_document, catalog, tokenizer, calibration, output_path):
    if labels_document.get("split") != "heldout" or labels_document.get("reviewed") is not True:
        raise ValueError("Frozen evaluation requires reviewed independent held-out labels.")
    expected = labels_document["queries"]
    if {r["prediction"]["id"] for r in records} != {q["id"] for q in expected}:
        raise ValueError("Held-out predictions must cover exactly the independent held-out queries.")
    if (calibration["catalog"] != catalog.identity
        or digest_json({k:v for k,v in calibration.items() if k != "manifest_sha256"}) != calibration["manifest_sha256"]):
        raise ValueError("Held-out evaluation needs a verified frozen dev selection.")
    for r in records:
        q = next(q for q in expected if q["id"] == r["prediction"]["id"])
        if r["query_sha256"] != digest_json({"id":q["id"],"query":q["query"]}):
            raise ValueError("Held-out prediction/query checkpoint mismatch.")
    report = {"state":"FROZEN_POLICY_HELD_OUT_EVALUATED_LOCAL_PROXY", "split":"heldout",
        "label_review_mode":labels_document.get("review_mode","HUMAN_REVIEW"),
        "label_evaluation_scope":labels_document.get("evaluation_scope","HUMAN_REVIEWED_LOCAL_PROXY"),
        "calibration_manifest_sha256":calibration["manifest_sha256"],
        "adapter_manifest_sha256":calibration["adapter_manifest_sha256"],
        "labels_sha256":digest_json(labels_document),"catalog":catalog.identity,
        "evaluation":evaluate_predictions([r["prediction"] for r in records],expected,tokenizer),
        "candidate_recall":evaluate_candidate_recall(records,expected,catalog,tokenizer),
        "used_for_checkpoint_or_cutoff_selection":False,"official_btc_score":None,"corpus_promoted":False}
    report["manifest_sha256"] = digest_json(report)
    atomic_json(output_path,report)
    return report
