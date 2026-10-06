"""Pinned open 4B LLM for additive query translation, loaded on Colab only."""
from __future__ import annotations

import gc
import importlib.metadata
import json
from pathlib import Path

from .artifacts import sha256_file

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


class TorchQueryTranslator:
    def __init__(self, spec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if spec.get("model_id") != MODEL_ID or spec.get("revision") != REVISION:
            raise ValueError("Use the reviewed pinned translation model.")
        if type(spec.get("max_input_tokens")) is not int or spec["max_input_tokens"] < 1 or type(spec.get("max_new_tokens")) is not int or spec["max_new_tokens"] < 1:
            raise ValueError("Invalid LLM token budgets.")
        if not torch.cuda.is_available():
            raise RuntimeError("Translation LLM requires a Colab GPU.")
        self.torch, self.spec = torch, dict(spec)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
            torch_dtype=torch.float16, trust_remote_code=False, attn_implementation="sdpa")
        self.model.to("cuda").eval()
        parameters = sum(p.numel() for p in self.model.parameters())
        if parameters > 15_000_000_000:
            self.close()
            raise ValueError("Model exceeds the plan's 15B limit.")
        self.identity = {**spec, "parameters": parameters, "precision": "torch.float16",
            "device_class": "cuda", "do_sample": False, "fine_tuned": False,
            "role": "query_translation_only", "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "inference_code_sha256": sha256_file(Path(__file__))}

    def translate(self, query, prompt):
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(query, ensure_ascii=False)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        count = inputs["input_ids"].shape[1]
        if count > self.spec["max_input_tokens"]:
            # Disable translated branches; never silently truncate the original query.
            return "INPUT_BUDGET_EXCEEDED"
        inputs = {k:v.to("cuda") for k,v in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False,
                max_new_tokens=self.spec["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(outputs[0, count:], skip_special_tokens=True)

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
