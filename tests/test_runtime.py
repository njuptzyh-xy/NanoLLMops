from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.runtime.block_manager import BlockManager
from nanollmops.runtime.config import RuntimeConfig
from nanollmops.runtime.sampling_params import SamplingParams
from nanollmops.runtime.scheduler import Scheduler
from nanollmops.runtime.sequence import Sequence, SequenceStatus


def build_sequence(token_ids: list[int], block_size: int = 2, max_tokens: int = 2) -> Sequence:
    return Sequence(
        token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        block_size=block_size,
    )


class SequenceTest(unittest.TestCase):
    def test_tracks_blocks_and_completion_tokens(self) -> None:
        seq = build_sequence([1, 2, 3])

        self.assertEqual(seq.num_blocks, 2)
        self.assertEqual(seq.last_block_num_tokens, 1)
        self.assertEqual(seq.block(0), [1, 2])
        self.assertEqual(seq.completion_token_ids, [])

        seq.append_token(4)

        self.assertEqual(seq.last_token, 4)
        self.assertEqual(seq.num_completion_tokens, 1)
        self.assertEqual(seq.completion_token_ids, [4])


class BlockManagerTest(unittest.TestCase):
    def test_reuses_cached_prefix_after_sequence_is_deallocated(self) -> None:
        manager = BlockManager(num_blocks=3, block_size=2)
        first = build_sequence([1, 2])
        manager.allocate(first)
        cached_block_id = first.block_table[0]
        manager.deallocate(first)

        second = build_sequence([1, 2, 3])
        manager.allocate(second)

        self.assertEqual(second.block_table[0], cached_block_id)
        self.assertEqual(second.num_cached_tokens, 2)
        self.assertEqual(len(manager.used_block_ids), 2)

    def test_finalizes_full_block_before_allocating_decode_extension(self) -> None:
        manager = BlockManager(num_blocks=2, block_size=2)
        seq = build_sequence([1])
        manager.allocate(seq)

        seq.append_token(2)
        manager.may_append(seq)
        self.assertNotEqual(manager.blocks[seq.block_table[0]].hash, "")

        seq.append_token(3)
        manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 2)


class SchedulerTest(unittest.TestCase):
    def test_schedules_prefill_then_releases_finished_sequence(self) -> None:
        scheduler = Scheduler(
            RuntimeConfig(
                max_num_batched_tokens=8,
                max_num_seqs=1,
                kvcache_block_size=2,
                num_kvcache_blocks=2,
            )
        )
        seq = build_sequence([1, 2], max_tokens=1)
        scheduler.add(seq)

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [3])

        self.assertTrue(is_prefill)
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 2)


if __name__ == "__main__":
    unittest.main()
