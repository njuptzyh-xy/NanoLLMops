from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.artifacts import ModelStatus
from nanollmops.converter.planner import (
    build_artifact_layout,
    build_conversion_plan,
    parse_nanogpt_checkpoint_payload,
    parse_nanogpt_tokenizer_meta,
    write_plan_bundle,
)


class ConversionPlannerTest(unittest.TestCase):
    def test_parse_nanogpt_tokenizer_meta_for_char_tokenizer(self) -> None:
        meta = {
            "vocab_size": 4,
            "stoi": {"a": 0, "b": 1},
            "itos": {0: "a", 1: "b"},
        }
        tokenizer = parse_nanogpt_tokenizer_meta(meta, "data/meta.pkl")
        self.assertEqual(tokenizer.tokenizer_type, "char")
        self.assertEqual(tokenizer.vocab_size, 4)
        self.assertEqual(tokenizer.source_meta_path, "data/meta.pkl")

    def test_build_conversion_plan(self) -> None:
        payload = {
            "model_args": {
                "block_size": 256,
                "vocab_size": 65,
                "n_layer": 6,
                "n_head": 6,
                "n_embd": 384,
                "dropout": 0.2,
                "bias": False,
            },
            "model": {
                "transformer.wte.weight": "tensor",
                "lm_head.weight": "tensor",
            },
            "config": {"dataset": "shakespeare_char", "learning_rate": 1e-3},
            "iter_num": 2500,
            "best_val_loss": 1.42,
        }
        tokenizer = parse_nanogpt_tokenizer_meta(
            {"vocab_size": 65, "stoi": {"a": 0}, "itos": {0: "a"}}, "data/meta.pkl"
        )
        artifact = parse_nanogpt_checkpoint_payload(payload, "out/ckpt.pt", tokenizer)
        layout = build_artifact_layout("artifacts", "shakespeare-char", "v1")
        plan = build_conversion_plan(artifact, layout, "shakespeare-char", "v1")

        self.assertEqual(plan.model_id, "shakespeare-char-v1")
        self.assertEqual(plan.status, ModelStatus.TRAINED)
        self.assertEqual(plan.source.model_config.head_dim, 64)
        self.assertGreaterEqual(len(plan.rules), 8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = build_artifact_layout(tmp_dir, "shakespeare-char", "v1")
            write_plan_bundle(plan, layout)
            self.assertTrue(layout.manifest_path.endswith("artifact_manifest.json"))


if __name__ == "__main__":
    unittest.main()
