from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end demo with nanoGPT shakespeare_char.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompt", default="To be or not to be")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def wait_for_health(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            response.raise_for_status()
            return
        except Exception as exc:  # pragma: no cover - integration polling
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Service did not become ready: {last_error}")


def main() -> None:
    args = parse_args()
    checkpoint = ROOT / "artifacts" / "shakespeare-char-v1" / "source" / "ckpt.pt"
    meta = ROOT / "artifacts" / "shakespeare-char-v1" / "source" / "meta.pkl"
    if not checkpoint.exists() or not meta.exists():
        raise SystemExit(
            "Expected packaged artifact at artifacts/shakespeare-char-v1/source/. "
            "Run scripts/package_nanogpt_model.py first."
        )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "serve_nanogpt_simple.py"),
        "--checkpoint",
        str(checkpoint),
        "--meta",
        str(meta),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    process = subprocess.Popen(cmd, cwd=ROOT)
    base_url = f"http://{args.host}:{args.port}"
    try:
        wait_for_health(base_url)
        payload = {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.8,
            "top_k": 200,
        }
        response = httpx.post(f"{base_url}/generate", json=payload, timeout=60.0)
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
