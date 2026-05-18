from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.train.nanogpt_runner import NANOGPT_REPO, run_nanogpt_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nanoGPT training through NanoLLMOps.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "train_shakespeare_char.py"),
        help="Path to a nanoGPT-style config file.",
    )
    parser.add_argument(
        "--nanogpt-repo",
        default=str(NANOGPT_REPO),
        help="Path to the reference nanoGPT repository.",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra nanoGPT overrides such as --device=cpu --compile=False",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = run_nanogpt_train(
        config_path=args.config,
        extra_args=args.extra_args,
        nanogpt_repo=args.nanogpt_repo,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
