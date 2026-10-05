"""Pinned pretrained inference; heavy dependencies are loaded only on Colab."""
from __future__ import annotations

import gc
import importlib.metadata
import re
from pathlib import Path

import numpy as np

from .artifacts import sha256_file

DEFAULT_DENSE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def _check_spec(spec, model_id):
    if spec.get("model_id") != model_id or not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
        raise ValueError("Use the supported model and an immutable 40-character Hub revision.")
    if not isinstance(spec.get("max_length"), int) or spec["max_length"] < 2:
        raise ValueError("Invalid model token budget.")


class _TorchInference:
    def _setup(self, spec, device, *, reranker=False):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        if device not in ("cuda", "cpu") or (device == "cuda" and not torch.cuda.is_available()):
            raise RuntimeError("Chọn GPU trong Colab Runtime > Change runtime type.")
        self.torch, self.device, self.spec = torch, device, dict(spec)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"],
                                                       use_fast=True, trust_remote_code=False)
        loader = AutoModelForSequenceClassification if reranker else AutoModel
        self.model = loader.from_pretrained(spec["model_id"], revision=spec["revision"],
                                           torch_dtype=dtype, trust_remote_code=False)
        self.model.to(device).eval()
        self.identity = {**spec, "precision": str(dtype), "device_class": device,
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "torch": torch.__version__,
                         "transformers": importlib.metadata.version("transformers"),
                         "inference_code_sha256": sha256_file(Path(__file__))}
        self.oom_backoffs = 0

    def _batches(self, items, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        batch_size = min(batch_size, getattr(self, "safe_batch_size", batch_size))
        cursor, outputs = 0, []
        while cursor < len(items):
            try:
                value = self._batch(items[cursor:cursor+batch_size])
            except self.torch.OutOfMemoryError:
                if self.device != "cuda" or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.safe_batch_size = batch_size
                self.oom_backoffs += 1
                gc.collect()
                self.torch.cuda.empty_cache()
                continue
            outputs.append(value)
            cursor += len(value)
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)

    def close(self):
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


class TorchDenseEncoder(_TorchInference):
    """BGE-M3 dense CLS pooling; no query instruction and no text truncation."""
    def __init__(self, spec, *, device="cuda"):
        _check_spec(spec, "BAAI/bge-m3")
        self._setup(spec, device)
        self.dimension = int(self.model.config.hidden_size)
        self.identity.update(pooling="cls-l2-v1", dimension=self.dimension, query_instruction=None,
                             truncation=False, implementation="transformers-bge-m3-dense-v1")

    def _batch(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.spec["max_length"]:
            raise ValueError("Dense input exceeds max_length; do not silently truncate source or query.")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            vectors = self.model(**inputs).last_hidden_state[:, 0].float()
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().numpy().astype(np.float32, copy=False)

    def encode(self, texts, *, batch_size=16):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._batches(texts, batch_size)


class TorchReranker(_TorchInference):
    """Score bounded query/child pairs; parent expansion happens after scoring."""
    def __init__(self, spec, *, device="cuda"):
        _check_spec(spec, "BAAI/bge-reranker-v2-m3")
        self._setup(spec, device, reranker=True)
        self.identity.update(implementation="transformers-bge-reranker-v1", pair_truncation="only_second",
                             score="raw_logit", fine_tuned=False)

    def _batch(self, pairs):
        queries, passages = zip(*pairs)
        inputs = self.tokenizer(list(queries), list(passages), padding=True, truncation="only_second",
                                max_length=self.spec["max_length"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError("Expected one relevance logit per pair.")
        return logits[:, 0].float().cpu().numpy()

    def score(self, pairs, *, batch_size=16):
        return self._batches(pairs, batch_size)
