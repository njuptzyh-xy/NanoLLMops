from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ModelStatus(str, Enum):
    TRAINED = "TRAINED"
    CONVERTING = "CONVERTING"
    CONVERTED = "CONVERTED"
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"
    FAILED = "FAILED"
    OFFLINE = "OFFLINE"


@dataclass(slots=True)
class NanoGPTModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["head_dim"] = self.head_dim
        return data


@dataclass(slots=True)
class TokenizerSpec:
    tokenizer_type: str
    vocab_size: int
    stoi_path: str | None = None
    itos_path: str | None = None
    source_meta_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingArtifact:
    source_format: str
    checkpoint_path: str
    checkpoint_keys: list[str]
    model_state_key_count: int
    model_config: NanoGPTModelConfig
    tokenizer: TokenizerSpec | None = None
    train_config_keys: list[str] = field(default_factory=list)
    iter_num: int | None = None
    best_val_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_keys": self.checkpoint_keys,
            "model_state_key_count": self.model_state_key_count,
            "model_config": self.model_config.to_dict(),
            "tokenizer": self.tokenizer.to_dict() if self.tokenizer else None,
            "train_config_keys": self.train_config_keys,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
        }


@dataclass(slots=True)
class WeightRule:
    source: str
    target: str
    transform: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConversionPlan:
    model_id: str
    source: TrainingArtifact
    target_dir: str
    target_format: str
    status: ModelStatus
    output_files: list[str]
    rules: list[WeightRule]
    assumptions: list[str]
    open_questions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source": self.source.to_dict(),
            "target_dir": self.target_dir,
            "target_format": self.target_format,
            "status": self.status.value,
            "output_files": self.output_files,
            "rules": [rule.to_dict() for rule in self.rules],
            "assumptions": self.assumptions,
            "open_questions": self.open_questions,
        }


@dataclass(slots=True)
class ArtifactLayout:
    root_dir: str
    model_dir: str
    source_dir: str
    converted_dir: str
    manifest_path: str
    plan_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_artifact_layout(root_dir: str, model_name: str, version: str) -> ArtifactLayout:
    model_id = f"{model_name}-{version}"
    model_dir = Path(root_dir) / model_id
    source_dir = model_dir / "source"
    converted_dir = model_dir / "converted"
    return ArtifactLayout(
        root_dir=str(Path(root_dir)),
        model_dir=str(model_dir),
        source_dir=str(source_dir),
        converted_dir=str(converted_dir),
        manifest_path=str(model_dir / "artifact_manifest.json"),
        plan_path=str(model_dir / "conversion_plan.json"),
        summary_path=str(model_dir / "conversion_plan.md"),
    )
