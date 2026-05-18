from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanollmops.serving.api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a nanoGPT checkpoint through FastAPI.")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pt")
    parser.add_argument("--meta", required=True, help="Path to meta.pkl")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    parser.add_argument("--dtype", default="float32", help="torch dtype name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta,
        device=args.device,
        dtype=args.dtype,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
