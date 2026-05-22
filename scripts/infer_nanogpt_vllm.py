from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.runtime.config import RuntimeConfig
from nanollmops.runtime.nanogpt_vllm import VLLMStyleNanoGPT
from nanollmops.runtime.sampling_params import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM-style NanoGPT inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--kvcache-block-size", type=int, default=16)
    parser.add_argument("--num-kvcache-blocks", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = VLLMStyleNanoGPT(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
        device=args.device,
        dtype=args.dtype,
        runtime_config=RuntimeConfig(
            max_model_len=args.max_model_len,
            kvcache_block_size=args.kvcache_block_size,
            num_kvcache_blocks=args.num_kvcache_blocks,
        ),
    )
    result = engine.generate(
        prompt=args.prompt,
        sampling_params=SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
