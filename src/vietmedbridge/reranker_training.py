"""Source-grounded hard negatives, query-disjoint QLoRA and dev checkpoint metrics."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import random

import numpy as np

from .artifacts import atomic_json, digest_json, read_json, sha256_file, verify_file
from .qwen_models import RERANKER_ID, RERANKER_REVISION, pack_pair
from .retrieval_policy import lcs_length, score_tokens, scoring_text

CATEGORIES = ("same_document_wrong_chunk", "same_disease_wrong_intervention", "same_entity_wrong_context",
              "cross_language_false_friend", "high_lexical_overlap")
QUOTAS = (2, 2, 1, 1, 1)


def validate_training_labels(document, contest_queries, *, split=None):
    if document.get("reviewed") is not True or document.get("split") not in ("train", "dev") or (split and document["split"] != split):
        raise ValueError("Training/calibration requires reviewed train/dev labels.")
    queries = document.get("queries", [])
    if not queries or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Training queries need unique IDs.")
    test_ids = {q["id"] for q in contest_queries}
    test_texts = {scoring_text(q["query"]) for q in contest_queries}
    for query in queries:
        if type(query.get("id")) is not int or not isinstance(query.get("query"), str) or not query["query"].strip():
            raise ValueError("Training queries need integer IDs and nonempty source text.")
        if query["id"] in test_ids or scoring_text(query["query"]) in test_texts:
            raise ValueError("Contest/test queries cannot enter training or cutoff calibration.")
        if not query.get("relevant_chunks") or not isinstance(query.get("relevant_docs"), list):
            raise ValueError("Training needs reviewed source chunk/document references.")
    return queries


def validate_query_split(train_document, dev_document, contest_queries):
    train = validate_training_labels(train_document, contest_queries, split="train")
    dev = validate_training_labels(dev_document, contest_queries, split="dev")
    if {q["id"] for q in train} & {q["id"] for q in dev} or {scoring_text(q["query"]) for q in train} & {scoring_text(q["query"]) for q in dev}:
        raise ValueError("Train/dev overlap; split by query, never by candidate pairs.")
    return train, dev


def mine_hard_negatives(records, labels_document, catalog, tokenizer, contest_queries, *, seed=42):
    labels = validate_training_labels(labels_document, contest_queries)
    by_id = {r["prediction"]["id"]: r for r in records}
    groups, holds, evaluation = [], [], []
    rng = random.Random(seed)
    for query in labels:
        record = by_id.get(query["id"])
        if record is None or record["query_sha256"] != digest_json({"id": query["id"], "query": query["query"]}):
            raise ValueError("Mining candidates belong to another labeled query.")
        for ref in query["relevant_chunks"]:
            if (type(ref.get("doc_id")) is not int or ref["doc_id"] not in catalog.documents
                or ref["doc_id"] not in query["relevant_docs"] or not isinstance(ref.get("chunk_text"), str)
                or not ref["chunk_text"].strip() or ref["chunk_text"] not in catalog.documents[ref["doc_id"]]["source_text"]):
                raise ValueError("Reviewed gold chunks must be exact source slices in the frozen catalog.")
        refs = [(c["doc_id"], score_tokens(tokenizer, c["chunk_text"])) for c in query["relevant_chunks"]]
        candidates = record["ranking"]["children"]
        positives, negative_buckets = [], defaultdict(list)
        categories = query.get("negative_categories", {})
        judgments = query.get("negative_child_ids", [])
        for rank, row in enumerate(candidates, 1):
            child = catalog.children[row["child_id"]]
            tokens = score_tokens(tokenizer, child["text"])
            # Training children are shorter than gold parents. Judge coverage of
            # the child, not 40% of a much longer gold parent (the F2 proxy).
            positive = bool(tokens) and any(doc == child["doc_id"] and lcs_length(tokens, ref) >= .8 * len(tokens) for doc, ref in refs)
            item = {"query_id": query["id"], "query": query["query"], "child_id": row["child_id"],
                "doc_id": child["doc_id"], "text": child["text"], "rank": rank, "label": int(positive)}
            reviewed_negative = labels_document.get("exhaustive_chunks") is True or row["child_id"] in judgments
            if positive or reviewed_negative:
                evaluation.append(item)
            if positive:
                positives.append(item)
            elif reviewed_negative:
                category = categories.get(row["child_id"], "same_document_wrong_chunk" if child["doc_id"] in query["relevant_docs"] else "unclassified_retrieved")
                negative_buckets[category].append(item | {"category": category})
        if not positives:
            holds.append({"id": query["id"], "reason": "NO_GOLD_CHILD_IN_RETRIEVED_POOL"})
            continue
        selected, used = [], set()
        for category, count in zip(CATEGORIES, QUOTAS, strict=True):
            for item in negative_buckets[category][:count]:
                selected.append(item)
                used.add(item["child_id"])
        remaining = sorted([n for bucket in negative_buckets.values() for n in bucket if n["child_id"] not in used], key=lambda n: (n["rank"] > 30, n["rank"], n["child_id"]))
        selected += remaining[:max(0, 7 - len(selected))]
        if len(selected) < 7:
            holds.append({"id": query["id"], "reason": "INSUFFICIENT_REVIEWED_HARD_NEGATIVES", "negatives": len(selected)})
            continue
        groups.append({"query_id": query["id"], "pairs": [rng.choice(positives), *selected[:7]]})
    bundle = {"split": labels_document["split"], "labels_sha256": digest_json(labels_document),
        "catalog": catalog.identity, "seed": seed, "groups": groups, "evaluation_pairs": evaluation,
        "holds": holds, "category_targets": dict(zip(CATEGORIES, QUOTAS, strict=True)),
        "positive_mapping": "same-document-gold-covers-80-percent-of-child-tokens",
        "category_scope": "REVIEWED_CATEGORY_ANNOTATIONS_ELSE_UNCLASSIFIED_NOT_MEDICAL_SEMANTIC_CERTIFICATION",
        "state": "READY" if groups else "NEEDS_REVIEWED_GOLD_AND_NEGATIVES"}
    bundle["manifest_sha256"] = digest_json(bundle)
    return bundle


def ranking_metrics(scores, pairs):
    by_query = defaultdict(list)
    for score, pair in zip(scores, pairs, strict=True):
        by_query[pair["query_id"]].append((float(score), int(pair["label"])))
    mrr, ndcg = [], []
    for rows in by_query.values():
        total = sum(label for score, label in rows)
        if not total:
            continue
        rows.sort(key=lambda row: -row[0])
        labels = [label for score, label in rows[:10]]
        mrr.append(next((1 / (idx + 1) for idx, label in enumerate(labels) if label), 0.0))
        dcg = sum(label / math.log2(idx + 2) for idx, label in enumerate(labels))
        ideal = sum(1 / math.log2(idx + 2) for idx in range(min(total, 10)))
        ndcg.append(dcg / ideal)
    return {"mrr_at_10": float(np.mean(mrr)) if mrr else 0., "ndcg_at_10": float(np.mean(ndcg)) if ndcg else 0.,
        "queries_with_gold_in_pool": len(mrr), "scope": "REVIEWED_RETRIEVED_CHILD_SUBSET_METRICS_NOT_OFFICIAL_F2"}


def write_adapter_manifest(directory, *, labels, training_contract):
    root = Path(directory)
    files = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file() and p.name != "adapter_manifest.json"}
    document = {"base_model": RERANKER_ID, "base_revision": RERANKER_REVISION,
        "training_contract": training_contract, "labels_sha256": digest_json(labels), "files": files,
        "fine_tuned": True, "selection": "DEV_NDCG_FINALISTS_REQUIRE_FULL_PIPELINE_F2_SWEEP"}
    document["manifest_sha256"] = digest_json(document)
    atomic_json(root / "adapter_manifest.json", document)


def verify_training_checkpoint(directory, contract):
    root = Path(directory).resolve()
    manifest = read_json(root / "adapter_manifest.json")
    if (manifest["training_contract"] != contract or
        digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]):
        raise ValueError("Training checkpoint contract/hash mismatch.")
    for name, checksum in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe training checkpoint path.")
        verify_file(path, checksum)
    return manifest


def train_qlora(reranker, train_bundle, dev_bundle, output_dir, *, model_budget, epochs=2, learning_rate=2e-5, seed=42):
    """BCE yes/no training, 1 positive + 7 negatives; only CUDA/verified train/dev."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    for bundle, split in ((train_bundle, "train"), (dev_bundle, "dev")):
        if bundle.get("state") != "READY" or bundle["split"] != split or digest_json({k: v for k, v in bundle.items() if k != "manifest_sha256"}) != bundle["manifest_sha256"]:
            raise ValueError("Training requires complete verified train/dev mining bundles.")
    if train_bundle["catalog"] != dev_bundle["catalog"]:
        raise ValueError("Train/dev bundles must use the same frozen catalog.")
    if not torch.cuda.is_available() or reranker.spec["quantization"] != "nf4":
        raise ValueError("QLoRA requires CUDA and the pinned NF4 base model.")
    train_pairs = [p for g in train_bundle["groups"] for p in g["pairs"]]
    dev_pairs = dev_bundle["evaluation_pairs"]
    if {p["query_id"] for p in train_pairs} & {p["query_id"] for p in dev_pairs}:
        raise ValueError("Training/dev query leakage.")
    contract = {"train": train_bundle["manifest_sha256"], "dev": dev_bundle["manifest_sha256"],
        "base": reranker.identity, "epochs": epochs, "learning_rate": learning_rate, "seed": seed,
        "lora_rank": 16, "loss": "weighted-bce-yes-minus-no", "code_sha256": sha256_file(Path(__file__))}
    root = Path(output_dir)
    if (root / "training_contract.json").exists() and read_json(root / "training_contract.json") != contract:
        raise ValueError("Training policy/data changed; use a new training run.")
    atomic_json(root / "training_contract.json", contract)
    model = prepare_model_for_kbit_training(reranker.model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    from .model_budget import add_adapter_parameters
    budget = add_adapter_parameters(model_budget, sum(p.numel() for p in model.parameters() if p.requires_grad))
    atomic_json(root / "model_parameter_budget.json", budget)
    model.config.use_cache = False
    tokenizer = reranker.tokenizer

    class PairDataset:
        def __init__(self, pairs):
            self.rows = [{"input_ids": pack_pair(tokenizer, p["query"], p["text"], reranker.spec["max_length"]),
                          "target": p["label"]} for p in pairs]
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return self.rows[idx]

    def collate(rows):
        inputs = tokenizer.pad({"input_ids": [r["input_ids"] for r in rows]}, padding=True, return_tensors="pt")
        inputs["labels"] = torch.tensor([r["target"] for r in rows], dtype=torch.float32)
        return inputs

    class RankingTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            targets = inputs.pop("labels")
            logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            difference = logits[:, reranker.yes] - logits[:, reranker.no]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(difference, targets,
                pos_weight=torch.tensor(7., device=difference.device))
            return (loss, {"logits": difference[:, None]}) if return_outputs else loss

    class ManifestCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            write_adapter_manifest(Path(args.output_dir) / f"checkpoint-{state.global_step}",
                labels={"train": train_bundle["labels_sha256"], "dev": dev_bundle["labels_sha256"]}, training_contract=contract)

    def metrics(prediction):
        result = ranking_metrics(np.asarray(prediction.predictions).reshape(-1), dev_pairs)
        return {k: v for k, v in result.items() if isinstance(v, (int, float))}

    arguments = TrainingArguments(output_dir=str(root), num_train_epochs=epochs, learning_rate=learning_rate,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        fp16=True, gradient_checkpointing=True, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="ndcg_at_10", greater_is_better=True,
        save_total_limit=3, seed=seed, data_seed=seed, report_to="none", remove_unused_columns=False,
        optim="paged_adamw_8bit", logging_steps=10)
    trainer = RankingTrainer(model=model, args=arguments, train_dataset=PairDataset(train_pairs),
        eval_dataset=PairDataset(dev_pairs), data_collator=collate, compute_metrics=metrics, callbacks=[ManifestCallback()])
    latest = get_last_checkpoint(str(root))
    if latest is not None:
        verify_training_checkpoint(latest, contract)
    trainer.train(resume_from_checkpoint=latest)
    trainer.save_model(str(root / "best-ndcg-adapter"))
    write_adapter_manifest(root / "best-ndcg-adapter", labels={"train": train_bundle["labels_sha256"], "dev": dev_bundle["labels_sha256"]}, training_contract=contract)
    report = {"state": "TRAINED_REQUIRES_DEV_F2_AND_HELD_OUT_VALIDATION", "best_ndcg_checkpoint": trainer.state.best_model_checkpoint,
        "adapter": str(root / "best-ndcg-adapter"), "metrics": trainer.evaluate(), "contract": contract}
    atomic_json(root / "training_report.json", report)
    reranker.model = model
    return report
