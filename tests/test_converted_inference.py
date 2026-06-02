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

from nanollmops.artifacts import NanoGPTModelConfig, TrainingArtifact
from nanollmops.converter.convert import (
    build_char_tokenizer_json,
    build_hf_config,
    convert_state_dict,
)
from nanollmops.serving.nanogpt import GPT, GPTConfig, GenerationRequest, NanoGPTDeployment


class ConvertedInferenceTest(unittest.TestCase):
    def test_loads_converted_directory_and_generates_text(self) -> None:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError:
            self.skipTest("safetensors is required for converted inference")

        model_config = NanoGPTModelConfig(
            block_size=8,
            vocab_size=3,
            n_layer=1,
            n_head=1,
            n_embd=4,
            dropout=0.0,
            bias=True,
        )
        artifact = TrainingArtifact(
            source_format="nanogpt-ckpt",
            checkpoint_path="unused",
            checkpoint_keys=["model", "model_args"],
            model_state_key_count=0,
            model_config=model_config,
        )
        model = GPT(
            GPTConfig(
                block_size=model_config.block_size,
                vocab_size=model_config.vocab_size,
                n_layer=model_config.n_layer,
                n_head=model_config.n_head,
                n_embd=model_config.n_embd,
                dropout=model_config.dropout,
                bias=model_config.bias,
            )
        )
        converted = convert_state_dict(model.state_dict(), artifact)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "tiny-v1" / "converted"
            model_dir.mkdir(parents=True)
            config = build_hf_config(artifact, "tiny-v1")
            tokenizer = build_char_tokenizer_json(
                {"stoi": {"a": 0, "b": 1, "c": 2}, "itos": {0: "a", 1: "b", 2: "c"}}
            )
            (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
            save_file(converted, str(model_dir / "model.safetensors"))

            deployment = NanoGPTDeployment.from_converted(str(model_dir))
            response = deployment.generate(
                GenerationRequest(prompt="a", max_new_tokens=2, temperature=1.0, top_k=1)
            )

        self.assertEqual(deployment.model_name, "tiny-v1")
        self.assertEqual(response.input_tokens, 1)
        self.assertEqual(response.output_tokens, 2)
        self.assertEqual(len(response.output), 3)


if __name__ == "__main__":
    unittest.main()
