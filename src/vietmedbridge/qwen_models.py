"""Pinned Qwen retrieval models. Weights are loaded only on a Colab CUDA runtime."""
from __future__ import annotations

import gc
import importlib.metadata
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import digest_json, read_json, sha256_file, verify_file
from .embeddings import embedding_matrix, unit_signature
from .model_budget import model_budget_report
from .retrieval_models import _TorchInference

EMBEDDING_ID = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
EMBEDDING_PARAMETERS = 7567295488
EMBEDDING_DIMENSION = 4096
RERANKER_ID = "Qwen/Qwen3-Reranker-8B"
RERANKER_REVISION = "77d193c791ed757ca307ee72715aa132723da912"
QUERY_INSTRUCTION = "Given a Vietnamese biomedical question, retrieve source passages in Vietnamese, English or Chinese that provide relevant evidence."
RERANK_INSTRUCTION = "Given a Vietnamese biomedical question, determine whether the document provides relevant evidence. Respect negation, population, intervention, comparator, outcome, time and numerical constraints."
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def review_model_registry(config, document):
    """Offline evidence gate, before any weights are loaded; not BTC approval."""
    registered = {(m["model_id"], m["revision"]): m for m in document["models"]}
    cutoff = datetime(2026, 7, 31, 17, tzinfo=timezone.utc)
    for key in ("dense", "second_dense", "translation", "reranker"):
        spec = config[key]
        entry = registered.get((spec["model_id"], spec["revision"]))
        if (not entry or type(entry["parameters"]) is not int or not 0 < entry["parameters"] <= 15_000_000_000
            or not entry.get("sources") or not entry.get("license")
            or date.fromisoformat(entry["public_release"]) >= date(2026, 8, 1)
            or datetime.fromisoformat(entry["revision_last_modified"].replace("Z", "+00:00")) >= cutoff):
            raise ValueError("Model/revision lacks eligible public/date/parameter evidence in the reviewed registry.")
    model_budget_report(config, document)
    return digest_json(document)


def review_spec(spec, role):
    expected = ((EMBEDDING_ID, EMBEDDING_REVISION) if role == "embedding" else (RERANKER_ID, RERANKER_REVISION))
    if (spec.get("model_id"), spec.get("revision")) != expected:
        raise ValueError("Use the reviewed pinned Qwen model/revision.")
    if spec.get("quantization") not in ("nf4", "fp16") or type(spec.get("max_length")) is not int or spec["max_length"] < 128:
        raise ValueError("Invalid Qwen precision/token policy.")
    return expected


def cached_qwen_embeddings(root, inputs, spec, role):
    root = Path(root)
    if not (root / "embeddings.json").exists():
        return None
    manifest = read_json(root / "embeddings.json")
    if manifest.get("state") != "COMPLETE":
        return None
    encoder = manifest["encoder"]
    if (manifest["units_sha256"] != unit_signature(inputs) or encoder.get("input_role") != role
        or encoder.get("query_instruction") != QUERY_INSTRUCTION or encoder.get("fine_tuned") is not False
        or encoder.get("pooling") != "last-attended-token-l2" or encoder.get("truncation") is not False
        or encoder.get("role") != "second_dense" or encoder.get("dimension") != EMBEDDING_DIMENSION or manifest["dimension"] != EMBEDDING_DIMENSION
        or encoder.get("precision") != "torch.float16" or encoder.get("device_class") != "cuda"
        or encoder.get("inference_code_sha256") != sha256_file(Path(__file__))
        or any(encoder.get(k) != v for k, v in spec.items())):
        raise ValueError("Qwen embedding cache input/model/policy mismatch.")
    return embedding_matrix(root, manifest), manifest


def format_pair(query, document, instruction=RERANK_INSTRUCTION):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def pack_pair(tokenizer, query, document, max_length, instruction=RERANK_INSTRUCTION):
    ids = (tokenizer.encode(PREFIX, add_special_tokens=False)
           + tokenizer.encode(format_pair(query, document, instruction), add_special_tokens=False)
           + tokenizer.encode(SUFFIX, add_special_tokens=False))
    if len(ids) > max_length:
        raise ValueError("Qwen pair exceeds budget; window the passage without truncating the query.")
    return ids


def pair_windows(tokenizer, query, text, max_length, overlap=32, instruction=RERANK_INSTRUCTION):
    """Exact source windows, with the full chat/query overhead accounted for."""
    overhead = len(pack_pair(tokenizer, query, "", max_length, instruction))
    budget = max_length - overhead - 8
    if budget < 8:
        raise ValueError("Query leaves insufficient Qwen document budget.")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)["offset_mapping"]
    if not offsets:
        raise ValueError("Empty passage.")
    windows, start = [], 0
    while start < len(offsets):
        end = min(len(offsets), start + budget)
        while end > start:
            a, b = offsets[start][0], offsets[end - 1][1]
            try:
                pack_pair(tokenizer, query, text[a:b], max_length, instruction)
                break
            except ValueError:
                end -= 1
        if end == start:
            raise ValueError("Cannot fit a source token in Qwen pair budget.")
        windows.append({"text": text[a:b], "start_char": a, "end_char": b})
        if end == len(offsets):
            break
        start = max(start + 1, end - min(overlap, (end - start) // 2))
    return windows


class _QwenCUDA(_TorchInference):
    def _load(self, spec, role):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_id, revision = review_spec(spec, role)
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen retrieval inference requires a Colab GPU; do not load weights on local CPU.")
        self.torch, self.device, self.spec = torch, "cuda", dict(spec)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
            use_fast=True, padding_side="left", trust_remote_code=False)
        kwargs = {"revision": revision, "torch_dtype": torch.float16,
            "device_map": {"": 0}, "attn_implementation": "sdpa", "trust_remote_code": False}
        if spec["quantization"] == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
        loader = AutoModel if role == "embedding" else AutoModelForCausalLM
        self.model = loader.from_pretrained(model_id, **kwargs).eval()
        self.identity = {**spec, "device_class": "cuda", "precision": "torch.float16",
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes") if spec["quantization"] == "nf4" else None,
            "inference_code_sha256": sha256_file(Path(__file__)), "fine_tuned": False}
        self.identity["parameters_before_quantization"] = EMBEDDING_PARAMETERS if role == "embedding" else 8188548096
        self.oom_backoffs = 0

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


class RoleEncoder:
    """One resident model, two explicit immutable corpus/query encoding identities."""
    def __init__(self, encoder, role):
        if role not in ("corpus", "query"):
            raise ValueError("Invalid embedding role.")
        self.encoder, self.role, self.dimension = encoder, role, encoder.dimension
        self.identity = {**encoder.identity, "input_role": role}

    def encode(self, texts, *, batch_size=2):
        self.encoder.input_role = self.role
        return self.encoder.encode(texts, batch_size=batch_size)


class TorchQwenEncoder(_QwenCUDA):
    def __init__(self, spec, *, adapter_path=None):
        self._load(spec, "embedding")
        self.dimension = int(self.model.config.hidden_size)
        self.input_role = "corpus"
        self.identity.update(pooling="last-attended-token-l2", dimension=self.dimension,
            query_instruction=QUERY_INSTRUCTION, truncation=False, role="second_dense")
        if adapter_path is not None:
            from peft import PeftModel
            root = Path(adapter_path).resolve()
            manifest = read_json(root / "adapter_manifest.json")
            if (manifest["base_model"] != EMBEDDING_ID or manifest["base_revision"] != EMBEDDING_REVISION
                or digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]):
                raise ValueError("Embedding adapter base/hash mismatch.")
            for name, checksum in manifest["files"].items():
                path = (root / name).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Unsafe embedding adapter path.")
                verify_file(path, checksum)
            self.model = PeftModel.from_pretrained(self.model, root, is_trainable=False).eval()
            self.identity.update(fine_tuned=True, adapter_manifest_sha256=manifest["manifest_sha256"])

    def for_role(self, role):
        return RoleEncoder(self, role)

    def _batch(self, texts):
        if self.input_role == "query":
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in texts]
        inputs = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.spec["max_length"]:
            raise ValueError("Qwen embedding input exceeds budget; adjust config in a new run.")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            # Left padding puts the last attended token at -1 for every row.
            if not self.torch.all(inputs["attention_mask"][:, -1] == 1):
                raise ValueError("Qwen pooling requires left padding.")
            vectors = self.torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1)
        return vectors.cpu().numpy().astype(np.float32, copy=False)

    def encode(self, texts, *, batch_size=2):
        return self._batches(texts, batch_size) if texts else np.empty((0, self.dimension), np.float32)


class TorchQwenReranker(_QwenCUDA):
    def __init__(self, spec, *, adapter_path=None):
        self._load(spec, "reranker")
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes == self.no or self.yes == self.tokenizer.unk_token_id or self.no == self.tokenizer.unk_token_id:
            raise ValueError("Invalid Qwen yes/no vocabulary IDs.")
        self.identity.update(role="document_and_child_reranker", score="yes-minus-no-raw-logits",
                             instruction=RERANK_INSTRUCTION, prompt_sha256=digest_json([PREFIX, SUFFIX]))
        if adapter_path is not None:
            from peft import PeftModel
            root = Path(adapter_path)
            manifest = read_json(root / "adapter_manifest.json")
            if digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
                raise ValueError("Adapter manifest hash mismatch.")
            if manifest["base_model"] != RERANKER_ID or manifest["base_revision"] != RERANKER_REVISION:
                raise ValueError("Adapter base model/revision mismatch.")
            for name, checksum in manifest["files"].items():
                path = (root / name).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Unsafe adapter path.")
                verify_file(path, checksum)
            self.model = PeftModel.from_pretrained(self.model, root, is_trainable=False).eval()
            self.identity.update(fine_tuned=True, adapter_manifest_sha256=manifest["manifest_sha256"])

    def windows(self, query, text, overlap=32):
        return pair_windows(self.tokenizer, query, text, self.spec["max_length"], overlap)

    def _batch(self, pairs):
        ids = [pack_pair(self.tokenizer, q, p, self.spec["max_length"]) for q, p in pairs]
        inputs = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            scores = logits[:, self.yes] - logits[:, self.no]
        return scores.cpu().numpy().astype(np.float32, copy=False)

    def score(self, pairs, *, batch_size=2):
        return self._batches(pairs, batch_size)
