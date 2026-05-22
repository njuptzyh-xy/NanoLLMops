from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib

from nanollmops.runtime.sequence import Sequence


@dataclass(slots=True)
class Block:
    block_id: int
    ref_count: int = 0
    hash: str = ""
    token_ids: list[int] = field(default_factory=list)

    def update(self, block_hash: str, token_ids: list[int]) -> None:
        self.hash = block_hash
        self.token_ids = token_ids

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = ""
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[str, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: str = "") -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(prefix.encode("utf-8"))
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
        return digest.hexdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise ValueError(f"block {block_id} is not free")
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        if seq.block_table:
            raise ValueError("sequence already allocated")
        prefix_hash = ""
        cache_miss = False
        for index in range(seq.num_blocks):
            token_ids = seq.block(index)
            block_hash = (
                self.compute_hash(token_ids, prefix_hash) if len(token_ids) == self.block_size else ""
            )
            block_id = self.hash_to_block_id.get(block_hash, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if block_hash:
                block.update(block_hash, token_ids)
                self.hash_to_block_id[block_hash] = block_id
                prefix_hash = block_hash
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        needs_block = len(seq) % self.block_size == 1
        return len(self.free_block_ids) >= int(needs_block)

    def may_append(self, seq: Sequence) -> None:
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            if last_block.hash == "":
                raise ValueError("last block should be finalized before extending")
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            if last_block.hash:
                return
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else ""
            block_hash = self.compute_hash(token_ids, prefix)
            last_block.update(block_hash, token_ids)
            self.hash_to_block_id[block_hash] = last_block.block_id
