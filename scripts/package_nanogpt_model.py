from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.converter.planner import create_plan_from_files, write_plan_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy nanoGPT checkpoint/meta into the NanoLLMOps artifact layout."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to nanoGPT ckpt.pt")
    parser.add_argument("--meta", required=True, help="Path to nanoGPT meta.pkl")
    parser.add_argument("--model-name", required=True, help="Logical model name")
    parser.add_argument("--version", default="v1", help="Model version")
    parser.add_argument("--artifacts-root", default=str(ROOT / "artifacts"), help="Artifact root dir")
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

    source_dir = Path(layout.source_dir)
    checkpoint_dst = source_dir / "ckpt.pt"
    meta_dst = source_dir / "meta.pkl"
    shutil.copy2(args.checkpoint, checkpoint_dst)
    shutil.copy2(args.meta, meta_dst)
    print(f"copied checkpoint -> {checkpoint_dst}")
    print(f"copied tokenizer meta -> {meta_dst}")
    print(f"plan summary -> {layout.summary_path}")


if __name__ == "__main__":
    main()
