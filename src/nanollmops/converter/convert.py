from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanollmops.artifacts import ArtifactLayout, TrainingArtifact, build_artifact_layout
from nanollmops.converter.planner import (
    create_plan_from_files,
    ensure_layout,
    load_meta_pickle,
    write_json,
    write_plan_bundle,
)


@dataclass(slots=True)
class ConvertedArtifactBundle:
    config_path: str
    generation_config_path: str
    tokenizer_json_path: str
    tokenizer_config_path: str
    special_tokens_map_path: str
    weights_path: str | None


def build_hf_config(artifact: TrainingArtifact, model_id: str) -> dict[str, Any]:
    config = artifact.model_config
    return {
        "_name_or_path": model_id,
        "architectures": ["NanoGPTForCausalLM"],
        "model_type": "nanogpt",
        "vocab_size": config.vocab_size,
        "n_positions": config.block_size,
        "n_ctx": config.block_size,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "bos_token_id": None,
        "eos_token_id": None,
        "activation_function": "gelu",
        "embd_pdrop": config.dropout,
        "attn_pdrop": config.dropout,
        "resid_pdrop": config.dropout,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
        "tie_word_embeddings": True,
        "bias": config.bias,
        "nanollmops_source_format": artifact.source_format,
    }


def build_generation_config() -> dict[str, Any]:
    return {
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_k": 200,
        "do_sample": True,
    }


def build_char_tokenizer_json(meta: dict[str, Any]) -> dict[str, Any]:
    stoi = meta["stoi"]
    added_tokens = [
        {
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        }
        for token, token_id in sorted(stoi.items(), key=lambda item: item[1])
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {"type": "Split", "pattern": "", "behavior": "Isolated", "invert": False},
        "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": "\u2581", "content": " "}]},
        "model": {
            "type": "WordLevel",
            "vocab": stoi,
            "unk_token": None,
        },
    }


def build_tokenizer_config(artifact: TrainingArtifact, meta_path: str | None) -> dict[str, Any]:
    tokenizer = artifact.tokenizer
    return {
        "tokenizer_class": "CharLevelTokenizer",
        "model_max_length": artifact.model_config.block_size,
        "padding_side": "right",
        "truncation_side": "right",
        "clean_up_tokenization_spaces": False,
        "nanollmops_tokenizer_type": tokenizer.tokenizer_type if tokenizer else "unknown",
        "nanollmops_meta_path": meta_path,
    }


def build_special_tokens_map() -> dict[str, Any]:
    return {}


def write_metadata_files(
    artifact: TrainingArtifact,
    layout: ArtifactLayout,
    model_id: str,
    meta: dict[str, Any],
    meta_path: str | None,
) -> ConvertedArtifactBundle:
    ensure_layout(layout)
    converted_dir = Path(layout.converted_dir)
    config_path = converted_dir / "config.json"
    generation_config_path = converted_dir / "generation_config.json"
    tokenizer_json_path = converted_dir / "tokenizer.json"
    tokenizer_config_path = converted_dir / "tokenizer_config.json"
    special_tokens_map_path = converted_dir / "special_tokens_map.json"

    write_json(str(config_path), build_hf_config(artifact, model_id))
    write_json(str(generation_config_path), build_generation_config())
    write_json(str(tokenizer_json_path), build_char_tokenizer_json(meta))
    write_json(str(tokenizer_config_path), build_tokenizer_config(artifact, meta_path))
    write_json(str(special_tokens_map_path), build_special_tokens_map())

    return ConvertedArtifactBundle(
        config_path=str(config_path),
        generation_config_path=str(generation_config_path),
        tokenizer_json_path=str(tokenizer_json_path),
        tokenizer_config_path=str(tokenizer_config_path),
        special_tokens_map_path=str(special_tokens_map_path),
        weights_path=None,
    )


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Converting nanoGPT weights requires torch, but it is not installed in the current environment."
        ) from exc
    return torch


def _require_safetensors() -> Any:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing converted weights requires safetensors. Install safetensors to produce model.safetensors."
        ) from exc
    return save_file


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    unwanted_prefix = "_orig_mod."
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized[key[len(unwanted_prefix) :]] = value if key.startswith(unwanted_prefix) else value
        if not key.startswith(unwanted_prefix):
            normalized[key] = value
    return normalized


def _split_qkv_tensor(tensor: Any) -> tuple[Any, Any, Any]:
    chunks = tensor.chunk(3, dim=0)
    if len(chunks) != 3:
        raise ValueError("Expected fused QKV tensor to split into exactly 3 chunks.")
    return chunks[0].contiguous(), chunks[1].contiguous(), chunks[2].contiguous()


def _owned_tensor(tensor: Any) -> Any:
    # safetensors rejects shared-storage tensors, so converted outputs must own storage.
    return tensor.detach().contiguous().clone()


def convert_state_dict(state_dict: dict[str, Any], artifact: TrainingArtifact) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    normalized = normalize_state_dict_keys(state_dict)
    n_layer = artifact.model_config.n_layer

    converted["model.embed_tokens.weight"] = _owned_tensor(normalized["transformer.wte.weight"])
    converted["model.position_embeddings.weight"] = _owned_tensor(normalized["transformer.wpe.weight"])
    converted["model.norm.weight"] = _owned_tensor(normalized["transformer.ln_f.weight"])
    if artifact.model_config.bias:
        converted["model.norm.bias"] = _owned_tensor(normalized["transformer.ln_f.bias"])
    converted["lm_head.weight"] = _owned_tensor(normalized["lm_head.weight"])

    for layer_id in range(n_layer):
        prefix = f"transformer.h.{layer_id}"
        target = f"model.layers.{layer_id}"
        q_weight, k_weight, v_weight = _split_qkv_tensor(normalized[f"{prefix}.attn.c_attn.weight"])
        converted[f"{target}.self_attn.q_proj.weight"] = _owned_tensor(q_weight)
        converted[f"{target}.self_attn.k_proj.weight"] = _owned_tensor(k_weight)
        converted[f"{target}.self_attn.v_proj.weight"] = _owned_tensor(v_weight)
        if artifact.model_config.bias:
            q_bias, k_bias, v_bias = _split_qkv_tensor(normalized[f"{prefix}.attn.c_attn.bias"])
            converted[f"{target}.self_attn.q_proj.bias"] = _owned_tensor(q_bias)
            converted[f"{target}.self_attn.k_proj.bias"] = _owned_tensor(k_bias)
            converted[f"{target}.self_attn.v_proj.bias"] = _owned_tensor(v_bias)
        converted[f"{target}.self_attn.o_proj.weight"] = _owned_tensor(normalized[f"{prefix}.attn.c_proj.weight"])
        converted[f"{target}.mlp.fc_in.weight"] = _owned_tensor(normalized[f"{prefix}.mlp.c_fc.weight"])
        converted[f"{target}.mlp.fc_out.weight"] = _owned_tensor(normalized[f"{prefix}.mlp.c_proj.weight"])
        converted[f"{target}.input_layernorm.weight"] = _owned_tensor(normalized[f"{prefix}.ln_1.weight"])
        converted[f"{target}.post_attention_layernorm.weight"] = _owned_tensor(normalized[f"{prefix}.ln_2.weight"])
        if artifact.model_config.bias:
            converted[f"{target}.self_attn.o_proj.bias"] = _owned_tensor(normalized[f"{prefix}.attn.c_proj.bias"])
            converted[f"{target}.mlp.fc_in.bias"] = _owned_tensor(normalized[f"{prefix}.mlp.c_fc.bias"])
            converted[f"{target}.mlp.fc_out.bias"] = _owned_tensor(normalized[f"{prefix}.mlp.c_proj.bias"])
            converted[f"{target}.input_layernorm.bias"] = _owned_tensor(normalized[f"{prefix}.ln_1.bias"])
            converted[f"{target}.post_attention_layernorm.bias"] = _owned_tensor(normalized[f"{prefix}.ln_2.bias"])

    return converted


def write_safetensors_weights(
    checkpoint_path: str,
    artifact: TrainingArtifact,
    layout: ArtifactLayout,
) -> str:
    torch = _require_torch()
    save_file = _require_safetensors()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError("Checkpoint payload is missing 'model'.")
    converted = convert_state_dict(checkpoint["model"], artifact)
    output_path = Path(layout.converted_dir) / "model.safetensors"
    save_file(converted, str(output_path))
    return str(output_path)


def copy_source_files(layout: ArtifactLayout, checkpoint_path: str, meta_path: str | None) -> None:
    source_dir = Path(layout.source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, source_dir / "ckpt.pt")
    if meta_path:
        shutil.copy2(meta_path, source_dir / "meta.pkl")


def convert_nanogpt_checkpoint(
    checkpoint_path: str,
    meta_path: str,
    root_dir: str,
    model_name: str,
    version: str,
    copy_source: bool = True,
) -> tuple[ArtifactLayout, ConvertedArtifactBundle]:
    layout, plan = create_plan_from_files(
        checkpoint_path=checkpoint_path,
        meta_path=meta_path,
        root_dir=root_dir,
        model_name=model_name,
        version=version,
    )
    write_plan_bundle(plan, layout)
    meta = load_meta_pickle(meta_path)
    bundle = write_metadata_files(plan.source, layout, plan.model_id, meta, meta_path)
    weights_path = write_safetensors_weights(checkpoint_path, plan.source, layout)
    if copy_source:
        copy_source_files(layout, checkpoint_path, meta_path)
    bundle.weights_path = weights_path
    return layout, bundle


def scaffold_conversion_bundle(
    checkpoint_path: str,
    meta_path: str,
    root_dir: str,
    model_name: str,
    version: str,
    copy_source: bool = True,
) -> tuple[ArtifactLayout, ConvertedArtifactBundle]:
    layout, plan = create_plan_from_files(
        checkpoint_path=checkpoint_path,
        meta_path=meta_path,
        root_dir=root_dir,
        model_name=model_name,
        version=version,
    )
    write_plan_bundle(plan, layout)
    meta = load_meta_pickle(meta_path)
    bundle = write_metadata_files(plan.source, layout, plan.model_id, meta, meta_path)
    if copy_source:
        copy_source_files(layout, checkpoint_path, meta_path)
    return layout, bundle
