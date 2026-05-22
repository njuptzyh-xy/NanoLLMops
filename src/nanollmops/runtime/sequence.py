from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

from nanollmops.runtime.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(slots=True)
class Sequence:
    token_ids: list[int]
    sampling_params: SamplingParams
    block_size: int
    seq_id: int = field(init=False)
    status: SequenceStatus = field(init=False, default=SequenceStatus.WAITING)
    last_token: int = field(init=False)
    num_tokens: int = field(init=False)
    num_prompt_tokens: int = field(init=False)
    num_cached_tokens: int = field(init=False, default=0)
    block_table: list[int] = field(init=False, default_factory=list)
    temperature: float = field(init=False)
    max_tokens: int = field(init=False)
    ignore_eos: bool = field(init=False)

    _counter = count()

    def __post_init__(self) -> None:
        self.seq_id = next(self._counter)
        self.token_ids = copy(self.token_ids)
        self.last_token = self.token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.temperature = self.sampling_params.temperature
        self.max_tokens = self.sampling_params.max_tokens
        self.ignore_eos = self.sampling_params.ignore_eos

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key: int | slice):
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, index: int) -> list[int]:
        if not 0 <= index < self.num_blocks:
            raise IndexError(index)
        return self.token_ids[index * self.block_size : (index + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
