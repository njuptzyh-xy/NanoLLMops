from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.converter.planner import create_plan_from_files, write_plan_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a nanoGPT checkpoint and emit a NanoLLMOps conversion plan."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to nanoGPT ckpt.pt")
    parser.add_argument("--meta", help="Optional path to nanoGPT meta.pkl")
    parser.add_argument("--model-name", required=True, help="Logical model name, e.g. shakespeare-char")
    parser.add_argument("--version", required=True, help="Model version, e.g. v1")
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root directory where manifest and conversion plan files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout, plan = create_plan_from_files(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
        root_dir=args.artifacts_root,
        model_name=args.model_name,
        version=args.version,
    )
    write_plan_bundle(plan, layout)
    print(f"wrote manifest: {layout.manifest_path}")
    print(f"wrote plan json: {layout.plan_path}")
    print(f"wrote plan summary: {layout.summary_path}")
    print(f"next target dir: {Path(layout.converted_dir)}")


if __name__ == "__main__":
    main()
