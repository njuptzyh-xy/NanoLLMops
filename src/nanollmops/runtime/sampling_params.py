from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 1e-10:
            raise ValueError("greedy sampling is not permitted")
