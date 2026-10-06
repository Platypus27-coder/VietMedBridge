"""Optional Qwen embedding QLoRA, after measured candidate-recall bottlenecks."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .artifacts import atomic_json, digest_json, read_json, sha256_file
from .qwen_models import EMBEDDING_ID, EMBEDDING_REVISION, QUERY_INSTRUCTION
from .reranker_training import ranking_metrics, verify_training_checkpoint


def train_embedding_qlora(encoder, train_bundle, dev_bundle, output_dir, *, recall_report,
                         model_budget=None, epochs=2, learning_rate=1e-5, temperature=.05, seed=42):
    """Train 1 positive + 7 negatives with contrastive loss, not answer generation."""
    recall = recall_report.get("fused_document_pool_recall_macro")
    if type(recall) not in (int, float) or not np.isfinite(recall) or not 0 <= recall < .95:
        raise ValueError("Embedding fine-tune needs a measured candidate-recall bottleneck.")
    for bundle, split in ((train_bundle, "train"), (dev_bundle, "dev")):
        if bundle.get("state") != "READY" or bundle["split"] != split or digest_json({k: v for k, v in bundle.items() if k != "manifest_sha256"}) != bundle["manifest_sha256"]:
            raise ValueError("Embedding training needs verified independent train/dev bundles.")
    if train_bundle["catalog"] != dev_bundle["catalog"]:
        raise ValueError("Embedding train/dev frozen catalogs differ.")
    if {g["query_id"] for g in train_bundle["groups"]} & {g["query_id"] for g in dev_bundle["groups"]}:
        raise ValueError("Embedding train/dev query overlap.")
    if not 0 < temperature <= 1:
        raise ValueError("Invalid contrastive temperature.")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available() or encoder.spec["quantization"] != "nf4":
        raise ValueError("Embedding QLoRA requires Colab CUDA/NF4.")
    root = Path(output_dir)
    contract = {"train": train_bundle["manifest_sha256"], "dev": dev_bundle["manifest_sha256"],
        "encoder": encoder.identity, "recall_report": digest_json(recall_report), "temperature": temperature,
        "epochs": epochs, "learning_rate": learning_rate, "seed": seed, "code_sha256": sha256_file(Path(__file__))}
    if (root / "training_contract.json").exists() and read_json(root / "training_contract.json") != contract:
        raise ValueError("Embedding training policy changed; choose a new run.")
    atomic_json(root / "training_contract.json", contract)
    model = prepare_model_for_kbit_training(encoder.model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION", bias="none"))
    from .model_budget import add_adapter_parameters
    budget = add_adapter_parameters(model_budget, sum(p.numel() for p in model.parameters() if p.requires_grad), role="second_dense")
    atomic_json(root / "model_parameter_budget.json", budget)

    class Groups:
        def __init__(self, bundle):
            self.groups = bundle["groups"]
        def __len__(self):
            return len(self.groups)
        def __getitem__(self, idx):
            return self.groups[idx]

    def collate(groups):
        texts = []
        for group in groups:
            if len(group["pairs"]) != 8 or group["pairs"][0]["label"] != 1 or any(p["label"] for p in group["pairs"][1:]):
                raise ValueError("Embedding contrastive groups require one positive and seven reviewed negatives.")
            texts += [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {group['pairs'][0]['query']}"] + [p["text"] for p in group["pairs"]]
        inputs = encoder.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if inputs["input_ids"].shape[1] > encoder.spec["max_length"]:
            raise ValueError("Embedding training pair budget exceeded; do not truncate.")
        inputs["labels"] = torch.zeros(len(groups), dtype=torch.long)
        return inputs

    class ContrastiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            targets = inputs.pop("labels")
            hidden = model(**inputs, use_cache=False).last_hidden_state[:, -1].float()
            vectors = torch.nn.functional.normalize(hidden, p=2, dim=1).reshape(-1, 9, hidden.shape[-1])
            logits = torch.einsum("bd,bkd->bk", vectors[:, 0], vectors[:, 1:]) / temperature
            loss = torch.nn.functional.cross_entropy(logits, targets)
            return (loss, {"logits": logits}) if return_outputs else loss

    def save_manifest(directory):
        path = Path(directory)
        files = {str(p.relative_to(path)): sha256_file(p) for p in path.rglob("*") if p.is_file() and p.name != "adapter_manifest.json"}
        manifest = {"base_model": EMBEDDING_ID, "base_revision": EMBEDDING_REVISION, "files": files,
            "training_contract": contract, "fine_tuned": True, "selection": "DEV_CONTRASTIVE_FINALIST_REQUIRES_RETRIEVAL_F2"}
        manifest["manifest_sha256"] = digest_json(manifest)
        atomic_json(path / "adapter_manifest.json", manifest)

    class SaveManifest(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            save_manifest(Path(args.output_dir) / f"checkpoint-{state.global_step}")

    def metrics(prediction):
        values = np.asarray(prediction.predictions).reshape(-1, 8)
        pairs = [{"query_id": group["query_id"], "label": int(idx == 0)} for group in dev_bundle["groups"] for idx in range(8)]
        result = ranking_metrics(values.reshape(-1), pairs)
        return {k: v for k, v in result.items() if isinstance(v, (int, float))}

    args = TrainingArguments(output_dir=str(root), num_train_epochs=epochs, learning_rate=learning_rate,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        fp16=True, gradient_checkpointing=True, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="ndcg_at_10", greater_is_better=True,
        remove_unused_columns=False, save_total_limit=3, report_to="none", seed=seed, optim="paged_adamw_8bit")
    trainer = ContrastiveTrainer(model=model, args=args, train_dataset=Groups(train_bundle), eval_dataset=Groups(dev_bundle),
        data_collator=collate, compute_metrics=metrics, callbacks=[SaveManifest()])
    latest = get_last_checkpoint(str(root))
    if latest is not None:
        verify_training_checkpoint(latest, contract)
    trainer.train(resume_from_checkpoint=latest)
    trainer.save_model(str(root / "best-dev-adapter"))
    save_manifest(root / "best-dev-adapter")
    report = {"state": "TRAINED_REQUIRES_RETRIEVAL_AND_HELD_OUT_F2_VALIDATION", "adapter": str(root / "best-dev-adapter"),
        "metrics": trainer.evaluate(), "auto_applied": False, "official_btc_score": None}
    atomic_json(root / "training_report.json", report)
    encoder.model = model
    return report
