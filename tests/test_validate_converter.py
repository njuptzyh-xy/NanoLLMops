from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.converter.validate import expected_tensor_shapes, validate_converted_model


class ValidateConvertedModelTest(unittest.TestCase):
    def test_expected_tensor_shapes_without_bias(self) -> None:
        config = {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 8,
            "vocab_size": 16,
            "n_positions": 32,
            "bias": False,
        }
        shapes = expected_tensor_shapes(config)
        self.assertEqual(shapes["model.embed_tokens.weight"], (16, 8))
        self.assertEqual(shapes["model.layers.0.mlp.fc_in.weight"], (32, 8))
        self.assertNotIn("model.norm.bias", shapes)

    def test_validate_converted_model_reports_missing_weights(self) -> None:
        try:
            from safetensors.torch import save_file
            import torch
        except ModuleNotFoundError:
            self.skipTest("safetensors and torch are required for this validation test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            config = {
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 4,
                "vocab_size": 8,
                "n_positions": 16,
                "bias": False,
            }
            (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            save_file(
                {
                    "model.embed_tokens.weight": torch.zeros(8, 4),
                },
                str(model_dir / "model.safetensors"),
            )
            with self.assertRaisesRegex(ValueError, "missing expected tensors"):
                validate_converted_model(str(model_dir))


if __name__ == "__main__":
    unittest.main()
