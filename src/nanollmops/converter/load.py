from __future__ import annotations

from pathlib import Path
from typing import Any

from nanollmops.converter.validate import load_converted_config, validate_converted_model


def load_converted_nanogpt_state_dict(
    model_dir: str,
    device: str = "cpu",
) -> tuple[dict[str, object], dict[str, Any]]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading converted weights requires safetensors in the current environment."
        ) from exc

    validate_converted_model(model_dir)
    config = load_converted_config(model_dir)
    converted = load_file(str(Path(model_dir) / "model.safetensors"), device=device)
    return config, build_nanogpt_state_dict(
        converted,
        n_layer=int(config["n_layer"]),
        bias=bool(config.get("bias", False)),
    )


def build_nanogpt_state_dict(
    converted: dict[str, Any],
    n_layer: int,
    bias: bool,
) -> dict[str, Any]:
    import torch

    state_dict = {
        "transformer.wte.weight": converted["model.embed_tokens.weight"],
        "transformer.wpe.weight": converted["model.position_embeddings.weight"],
        "transformer.ln_f.weight": converted["model.norm.weight"],
        "lm_head.weight": converted["lm_head.weight"],
    }
    if bias:
        state_dict["transformer.ln_f.bias"] = converted["model.norm.bias"]

    for layer_id in range(n_layer):
        source = f"model.layers.{layer_id}"
        target = f"transformer.h.{layer_id}"
        state_dict[f"{target}.attn.c_attn.weight"] = torch.cat(
            [
                converted[f"{source}.self_attn.q_proj.weight"],
                converted[f"{source}.self_attn.k_proj.weight"],
                converted[f"{source}.self_attn.v_proj.weight"],
            ],
            dim=0,
        )
        state_dict[f"{target}.attn.c_proj.weight"] = converted[
            f"{source}.self_attn.o_proj.weight"
        ]
        state_dict[f"{target}.mlp.c_fc.weight"] = converted[f"{source}.mlp.fc_in.weight"]
        state_dict[f"{target}.mlp.c_proj.weight"] = converted[f"{source}.mlp.fc_out.weight"]
        state_dict[f"{target}.ln_1.weight"] = converted[f"{source}.input_layernorm.weight"]
        state_dict[f"{target}.ln_2.weight"] = converted[
            f"{source}.post_attention_layernorm.weight"
        ]
        if bias:
            state_dict[f"{target}.attn.c_attn.bias"] = torch.cat(
                [
                    converted[f"{source}.self_attn.q_proj.bias"],
                    converted[f"{source}.self_attn.k_proj.bias"],
                    converted[f"{source}.self_attn.v_proj.bias"],
                ],
                dim=0,
            )
            state_dict[f"{target}.attn.c_proj.bias"] = converted[
                f"{source}.self_attn.o_proj.bias"
            ]
            state_dict[f"{target}.mlp.c_fc.bias"] = converted[f"{source}.mlp.fc_in.bias"]
            state_dict[f"{target}.mlp.c_proj.bias"] = converted[f"{source}.mlp.fc_out.bias"]
            state_dict[f"{target}.ln_1.bias"] = converted[f"{source}.input_layernorm.bias"]
            state_dict[f"{target}.ln_2.bias"] = converted[
                f"{source}.post_attention_layernorm.bias"
            ]
    return state_dict
