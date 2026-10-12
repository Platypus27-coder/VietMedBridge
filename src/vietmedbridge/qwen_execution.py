"""Execution settings that leave the frozen Qwen embedding producer unchanged."""


def configure_embedding_execution(encoder):
    """Avoid retaining generation KV tensors during a single embedding forward.

    The frozen model code, weights, tokenization, pooling and output dtype stay
    unchanged. No past_key_values are supplied or consumed by this encoder.
    Keeping this allocation policy separate preserves existing vector identities.
    """
    identity = encoder.identity
    if (identity.get("model_id") != "Qwen/Qwen3-Embedding-8B"
        or identity.get("input_role") != "corpus"):
        return {}
    owner = getattr(encoder, "encoder", encoder)
    model = getattr(owner, "model", None)
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "model_type", None) == "qwen3":
        config.use_cache = False
        print("Qwen corpus execution: generation KV cache disabled; weights/tokenization/pooling unchanged.", flush=True)
        return {"use_cache":False}
    return {}
