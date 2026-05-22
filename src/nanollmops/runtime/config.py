from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 16
    max_model_len: int = 256
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = 128
    eos: int = -1
