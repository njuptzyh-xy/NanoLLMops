from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.converter.validate import validate_converted_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a converted NanoLLMOps model bundle.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the converted/ directory that contains config.json and model.safetensors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_converted_model(args.model_dir)
    print(f"validated -> {summary.model_dir}")
    print(f"tensor_count -> {summary.tensor_count}")
    print(f"checked_keys -> {summary.checked_keys}")


if __name__ == "__main__":
    main()
