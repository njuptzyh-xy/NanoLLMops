from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.artifacts import NanoGPTModelConfig, TokenizerSpec, TrainingArtifact, build_artifact_layout
from nanollmops.converter.convert import (
    _require_safetensors,
    _require_torch,
    build_char_tokenizer_json,
    build_generation_config,
    build_hf_config,
    write_metadata_files,
)


class ConverterTest(unittest.TestCase):
    def build_artifact(self) -> TrainingArtifact:
        return TrainingArtifact(
            source_format="nanogpt-ckpt",
            checkpoint_path="out/ckpt.pt",
            checkpoint_keys=["config", "model", "model_args"],
            model_state_key_count=3,
            model_config=NanoGPTModelConfig(
                block_size=256,
                vocab_size=65,
                n_layer=6,
                n_head=6,
                n_embd=384,
                dropout=0.2,
                bias=False,
            ),
            tokenizer=TokenizerSpec(
                tokenizer_type="char",
                vocab_size=65,
                source_meta_path="data/meta.pkl",
            ),
        )

    def test_build_hf_config(self) -> None:
        config = build_hf_config(self.build_artifact(), "shakespeare-char-v1")
        self.assertEqual(config["model_type"], "nanogpt")
        self.assertEqual(config["n_layer"], 6)
        self.assertEqual(config["vocab_size"], 65)
        self.assertTrue(config["tie_word_embeddings"])

    def test_build_generation_config(self) -> None:
        config = build_generation_config()
        self.assertEqual(config["top_k"], 200)
        self.assertTrue(config["do_sample"])

    def test_build_char_tokenizer_json(self) -> None:
        payload = build_char_tokenizer_json(
            {"stoi": {"a": 0, "\n": 1}, "itos": {0: "a", 1: "\n"}, "vocab_size": 2}
        )
        self.assertEqual(payload["model"]["type"], "WordLevel")
        self.assertEqual(payload["model"]["vocab"]["a"], 0)
        self.assertEqual(len(payload["added_tokens"]), 2)

    def test_write_metadata_files(self) -> None:
        artifact = self.build_artifact()
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = build_artifact_layout(tmp_dir, "shakespeare-char", "v1")
            bundle = write_metadata_files(
                artifact=artifact,
                layout=layout,
                model_id="shakespeare-char-v1",
                meta={"stoi": {"a": 0}, "itos": {0: "a"}, "vocab_size": 1},
                meta_path="data/meta.pkl",
            )
            config = json.loads(Path(bundle.config_path).read_text(encoding="utf-8"))
            tokenizer = json.loads(Path(bundle.tokenizer_json_path).read_text(encoding="utf-8"))
            self.assertEqual(config["n_positions"], 256)
            self.assertEqual(tokenizer["model"]["vocab"]["a"], 0)

    def test_dependency_errors_are_actionable(self) -> None:
        if importlib.util.find_spec("torch") is None:
            with self.assertRaisesRegex(RuntimeError, "requires torch"):
                _require_torch()
        else:
            self.assertIsNotNone(_require_torch())

        if importlib.util.find_spec("safetensors") is None:
            with self.assertRaisesRegex(RuntimeError, "requires safetensors"):
                _require_safetensors()
        else:
            self.assertIsNotNone(_require_safetensors())


if __name__ == "__main__":
    unittest.main()
