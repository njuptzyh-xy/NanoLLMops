from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ConvertedModelSummary:
    model_dir: str
    tensor_count: int
    checked_keys: int


def _require_safetensors_safe_open():
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Validating converted weights requires safetensors. Install safetensors in the current environment."
        ) from exc
    return safe_open


def load_converted_config(model_dir: str) -> dict[str, object]:
    config_path = Path(model_dir) / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_tensor_shapes(config: dict[str, object]) -> dict[str, tuple[int, ...]]:
    n_layer = int(config["n_layer"])
    n_head = int(config["n_head"])
    n_embd = int(config["n_embd"])
    vocab_size = int(config["vocab_size"])
    block_size = int(config["n_positions"])
    bias = bool(config.get("bias", False))
    hidden4 = 4 * n_embd
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab_size, n_embd),
        "model.position_embeddings.weight": (block_size, n_embd),
        "model.norm.weight": (n_embd,),
        "lm_head.weight": (vocab_size, n_embd),
    }
    if bias:
        shapes["model.norm.bias"] = (n_embd,)

    for layer_id in range(n_layer):
        prefix = f"model.layers.{layer_id}"
        shapes[f"{prefix}.input_layernorm.weight"] = (n_embd,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (n_embd,)
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (n_embd, n_embd)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (n_embd, n_embd)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (n_embd, n_embd)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (n_embd, n_embd)
        shapes[f"{prefix}.mlp.fc_in.weight"] = (hidden4, n_embd)
        shapes[f"{prefix}.mlp.fc_out.weight"] = (n_embd, hidden4)
        if bias:
            shapes[f"{prefix}.input_layernorm.bias"] = (n_embd,)
            shapes[f"{prefix}.post_attention_layernorm.bias"] = (n_embd,)
            shapes[f"{prefix}.self_attn.q_proj.bias"] = (n_embd,)
            shapes[f"{prefix}.self_attn.k_proj.bias"] = (n_embd,)
            shapes[f"{prefix}.self_attn.v_proj.bias"] = (n_embd,)
            shapes[f"{prefix}.self_attn.o_proj.bias"] = (n_embd,)
            shapes[f"{prefix}.mlp.fc_in.bias"] = (hidden4,)
            shapes[f"{prefix}.mlp.fc_out.bias"] = (n_embd,)
    return shapes


def validate_converted_model(model_dir: str) -> ConvertedModelSummary:
    safe_open = _require_safetensors_safe_open()
    config = load_converted_config(model_dir)
    expected = expected_tensor_shapes(config)
    weights_path = Path(model_dir) / "model.safetensors"

    with safe_open(str(weights_path), framework="pt", device="cpu") as tensors:
        actual_keys = set(tensors.keys())
        missing = sorted(set(expected) - actual_keys)
        unexpected = sorted(actual_keys - set(expected))
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(f"Converted weights are missing expected tensors: {preview}")
        if unexpected:
            preview = ", ".join(unexpected[:8])
            raise ValueError(f"Converted weights contain unexpected tensors: {preview}")
        for key, shape in expected.items():
            actual_shape = tuple(tensors.get_tensor(key).shape)
            if actual_shape != shape:
                raise ValueError(f"Tensor shape mismatch for {key}: expected {shape}, got {actual_shape}")
        return ConvertedModelSummary(
            model_dir=str(Path(model_dir)),
            tensor_count=len(actual_keys),
            checked_keys=len(expected),
        )
