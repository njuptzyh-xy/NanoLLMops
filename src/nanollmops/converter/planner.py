from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nanollmops.artifacts import (
    ArtifactLayout,
    ConversionPlan,
    ModelStatus,
    NanoGPTModelConfig,
    TokenizerSpec,
    TrainingArtifact,
    WeightRule,
    build_artifact_layout,
)


REQUIRED_MODEL_ARGS = ("block_size", "vocab_size", "n_layer", "n_head", "n_embd")


def to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def load_meta_pickle(meta_path: str) -> dict[str, Any]:
    with open(meta_path, "rb") as handle:
        return pickle.load(handle)


def load_nanogpt_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading nanoGPT checkpoints requires torch. Install torch to parse ckpt.pt files."
        ) from exc
    return torch.load(checkpoint_path, map_location="cpu")


def parse_nanogpt_tokenizer_meta(meta: Mapping[str, Any], meta_path: str | None = None) -> TokenizerSpec:
    vocab_size = int(meta["vocab_size"])
    notes: list[str] = []
    tokenizer_type = "unknown"
    stoi_path = None
    itos_path = None

    if "stoi" in meta and "itos" in meta:
        tokenizer_type = "char"
        notes.append("Character-level tokenizer detected from stoi/itos tables.")
        if meta_path:
            stoi_path = meta_path
            itos_path = meta_path
    else:
        notes.append("Tokenizer metadata does not match the current char-level expectation.")

    return TokenizerSpec(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        stoi_path=stoi_path,
        itos_path=itos_path,
        source_meta_path=meta_path,
        notes=notes,
    )


def parse_nanogpt_checkpoint_payload(
    payload: Mapping[str, Any],
    checkpoint_path: str,
    tokenizer: TokenizerSpec | None = None,
) -> TrainingArtifact:
    if "model_args" not in payload:
        raise ValueError("Checkpoint payload is missing 'model_args'.")
    if "model" not in payload:
        raise ValueError("Checkpoint payload is missing 'model'.")

    model_args = payload["model_args"]
    missing = [name for name in REQUIRED_MODEL_ARGS if name not in model_args]
    if missing:
        raise ValueError(f"Checkpoint model_args missing required fields: {missing}")

    model_config = NanoGPTModelConfig(
        block_size=int(model_args["block_size"]),
        vocab_size=int(model_args["vocab_size"]),
        n_layer=int(model_args["n_layer"]),
        n_head=int(model_args["n_head"]),
        n_embd=int(model_args["n_embd"]),
        dropout=float(model_args.get("dropout", 0.0)),
        bias=bool(model_args.get("bias", False)),
    )

    return TrainingArtifact(
        source_format="nanogpt-ckpt",
        checkpoint_path=checkpoint_path,
        checkpoint_keys=sorted(payload.keys()),
        model_state_key_count=len(payload["model"]),
        model_config=model_config,
        tokenizer=tokenizer,
        train_config_keys=sorted(payload.get("config", {}).keys()),
        iter_num=to_python_scalar(payload.get("iter_num")),
        best_val_loss=to_python_scalar(payload.get("best_val_loss")),
    )


def build_weight_rules() -> list[WeightRule]:
    return [
        WeightRule(
            source="transformer.wte.weight",
            target="model.embed_tokens.weight",
            transform="copy_or_tie",
            notes="Token embedding should align with lm_head for tied weights.",
        ),
        WeightRule(
            source="transformer.wpe.weight",
            target="model.position_embeddings.weight",
            transform="copy",
            notes="Absolute position embeddings must stay within block_size.",
        ),
        WeightRule(
            source="transformer.h.{i}.attn.c_attn.weight",
            target="model.layers.{i}.self_attn.{q_proj,k_proj,v_proj}.weight",
            transform="split_qkv",
            notes="nanoGPT stores fused QKV projection weights in c_attn.",
        ),
        WeightRule(
            source="transformer.h.{i}.attn.c_attn.bias",
            target="model.layers.{i}.self_attn.{q_proj,k_proj,v_proj}.bias",
            transform="split_qkv",
            notes="Bias split is required only when the training model uses bias=True.",
        ),
        WeightRule(
            source="transformer.h.{i}.attn.c_proj.weight",
            target="model.layers.{i}.self_attn.o_proj.weight",
            transform="copy",
            notes="Output projection naming differs but shape should stay compatible.",
        ),
        WeightRule(
            source="transformer.h.{i}.mlp.c_fc.weight",
            target="model.layers.{i}.mlp.fc_in.weight",
            transform="copy",
            notes="MLP expansion layer is a direct rename for GPT-style blocks.",
        ),
        WeightRule(
            source="transformer.h.{i}.mlp.c_proj.weight",
            target="model.layers.{i}.mlp.fc_out.weight",
            transform="copy",
            notes="MLP projection back to hidden size stays dense linear.",
        ),
        WeightRule(
            source="transformer.h.{i}.ln_1.{weight,bias}",
            target="model.layers.{i}.input_layernorm.{weight,bias}",
            transform="copy",
            notes="LayerNorm flavor must stay LayerNorm, not RMSNorm.",
        ),
        WeightRule(
            source="transformer.h.{i}.ln_2.{weight,bias}",
            target="model.layers.{i}.post_attention_layernorm.{weight,bias}",
            transform="copy",
            notes="Second block norm follows the same mapping strategy.",
        ),
        WeightRule(
            source="transformer.ln_f.{weight,bias}",
            target="model.norm.{weight,bias}",
            transform="copy",
            notes="Final norm maps one-to-one if the inference model keeps LayerNorm.",
        ),
        WeightRule(
            source="lm_head.weight",
            target="lm_head.weight",
            transform="copy_or_tie",
            notes="When weights are tied, lm_head should share storage with embeddings.",
        ),
    ]


def build_conversion_plan(
    artifact: TrainingArtifact,
    layout: ArtifactLayout,
    model_name: str,
    version: str,
) -> ConversionPlan:
    model_id = f"{model_name}-{version}"
    output_files = [
        str(Path(layout.converted_dir) / "config.json"),
        str(Path(layout.converted_dir) / "generation_config.json"),
        str(Path(layout.converted_dir) / "model.safetensors"),
        str(Path(layout.converted_dir) / "tokenizer.json"),
        str(Path(layout.converted_dir) / "tokenizer_config.json"),
        str(Path(layout.converted_dir) / "special_tokens_map.json"),
    ]
    assumptions = [
        "Phase 1 starts from nanoGPT char-level checkpoints such as shakespeare_char.",
        "The inference-side GPT implementation should preserve absolute position embeddings.",
        "LayerNorm behavior should stay compatible with nanoGPT instead of switching to RMSNorm.",
        "The converted model directory should be HF-style so nano-vllm Config can load it via AutoConfig.",
    ]
    open_questions = [
        "Whether the inference model should live inside NanoLLMOps or be upstreamed into nano-vllm.",
        "Whether tokenizer output should remain meta.pkl-backed or be exported to tokenizer.json.",
        "Whether safetensors is the sole target artifact or pt should remain as a debugging fallback.",
    ]
    return ConversionPlan(
        model_id=model_id,
        source=artifact,
        target_dir=layout.converted_dir,
        target_format="hf-compatible-nanogpt",
        status=ModelStatus.TRAINED,
        output_files=output_files,
        rules=build_weight_rules(),
        assumptions=assumptions,
        open_questions=open_questions,
    )


def ensure_layout(layout: ArtifactLayout) -> None:
    Path(layout.model_dir).mkdir(parents=True, exist_ok=True)
    Path(layout.source_dir).mkdir(parents=True, exist_ok=True)
    Path(layout.converted_dir).mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def render_plan_markdown(plan: ConversionPlan, layout: ArtifactLayout) -> str:
    config = plan.source.model_config
    tokenizer = plan.source.tokenizer
    lines = [
        f"# Conversion Plan: {plan.model_id}",
        "",
        "## Source Artifact",
        "",
        f"- checkpoint: `{plan.source.checkpoint_path}`",
        f"- checkpoint keys: `{', '.join(plan.source.checkpoint_keys)}`",
        f"- model state key count: `{plan.source.model_state_key_count}`",
        f"- block_size: `{config.block_size}`",
        f"- vocab_size: `{config.vocab_size}`",
        f"- n_layer: `{config.n_layer}`",
        f"- n_head: `{config.n_head}`",
        f"- n_embd: `{config.n_embd}`",
        f"- head_dim: `{config.head_dim}`",
    ]
    if tokenizer:
        lines.extend(
            [
                "",
                "## Tokenizer",
                "",
                f"- tokenizer_type: `{tokenizer.tokenizer_type}`",
                f"- vocab_size: `{tokenizer.vocab_size}`",
                f"- source_meta_path: `{tokenizer.source_meta_path}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target Layout",
            "",
            f"- model_dir: `{layout.model_dir}`",
            f"- source_dir: `{layout.source_dir}`",
            f"- converted_dir: `{layout.converted_dir}`",
            "",
            "## Weight Mapping Rules",
            "",
        ]
    )
    for rule in plan.rules:
        lines.append(f"- `{rule.source}` -> `{rule.target}` [{rule.transform}] {rule.notes}")
    lines.extend(["", "## Assumptions", ""])
    for item in plan.assumptions:
        lines.append(f"- {item}")
    lines.extend(["", "## Open Questions", ""])
    for item in plan.open_questions:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_plan_bundle(plan: ConversionPlan, layout: ArtifactLayout) -> None:
    ensure_layout(layout)
    write_json(layout.manifest_path, plan.source.to_dict())
    write_json(layout.plan_path, plan.to_dict())
    with open(layout.summary_path, "w", encoding="utf-8") as handle:
        handle.write(render_plan_markdown(plan, layout))


def create_plan_from_files(
    checkpoint_path: str,
    meta_path: str | None,
    root_dir: str,
    model_name: str,
    version: str,
) -> tuple[ArtifactLayout, ConversionPlan]:
    tokenizer = None
    if meta_path:
        meta = load_meta_pickle(meta_path)
        tokenizer = parse_nanogpt_tokenizer_meta(meta, meta_path)
    checkpoint = load_nanogpt_checkpoint(checkpoint_path)
    artifact = parse_nanogpt_checkpoint_payload(checkpoint, checkpoint_path, tokenizer)
    layout = build_artifact_layout(root_dir, model_name, version)
    plan = build_conversion_plan(artifact, layout, model_name, version)
    return layout, plan
