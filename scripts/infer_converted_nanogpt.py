from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.serving.nanogpt import GenerationRequest, NanoGPTDeployment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference from a converted NanoGPT model.")
    parser.add_argument("--model-dir", required=True, help="Path to converted/ directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-k", default=200, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deployment = NanoGPTDeployment.from_converted(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    result = deployment.generate(
        GenerationRequest(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
