from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.converter.convert import convert_nanogpt_checkpoint, scaffold_conversion_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a nanoGPT checkpoint into a NanoLLMOps converted artifact bundle."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to nanoGPT ckpt.pt")
    parser.add_argument("--meta", required=True, help="Path to nanoGPT meta.pkl")
    parser.add_argument("--model-name", required=True, help="Logical model name")
    parser.add_argument("--version", default="v1", help="Model version")
    parser.add_argument("--artifacts-root", default=str(ROOT / "artifacts"), help="Artifact root dir")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write plan/config/tokenizer files without converting model weights.",
    )
    parser.add_argument(
        "--skip-copy-source",
        action="store_true",
        help="Do not copy ckpt.pt/meta.pkl into the artifact source/ directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_fn = scaffold_conversion_bundle if args.metadata_only else convert_nanogpt_checkpoint
    layout, bundle = convert_fn(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
        root_dir=args.artifacts_root,
        model_name=args.model_name,
        version=args.version,
        copy_source=not args.skip_copy_source,
    )
    print(f"plan -> {layout.plan_path}")
    print(f"config -> {bundle.config_path}")
    print(f"tokenizer -> {bundle.tokenizer_json_path}")
    if bundle.weights_path:
        print(f"weights -> {bundle.weights_path}")
    else:
        print("weights -> skipped")


if __name__ == "__main__":
    main()
