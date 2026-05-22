from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    parser = argparse.ArgumentParser(description="Serve a vLLM-style NanoGPT runtime.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--kvcache-block-size", type=int, default=16)
    parser.add_argument("--num-kvcache-blocks", type=int, default=128)
    return parser.parse_args()


def make_handler(engine: VLLMStyleNanoGPT) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(
                    {
                        "status": "ok",
                        "model": engine.model_name,
                        "engine": "vllm-style-gpt",
                        "kvcache_block_size": engine.runtime_config.kvcache_block_size,
                        "num_kvcache_blocks": engine.runtime_config.num_kvcache_blocks,
                    }
                )
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self._send_json({"error": "not found"}, status=404)
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                payload = json.loads(raw.decode("utf-8"))
                result = engine.generate(
                    prompt=payload["prompt"],
                    sampling_params=SamplingParams(
                        temperature=payload.get("temperature", 0.8),
                        max_tokens=payload.get("max_tokens", payload.get("max_new_tokens", 32)),
                        ignore_eos=True,
                    ),
                )
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return Handler


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
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(
        f"serving {engine.model_name} with vllm-style runtime on http://{args.host}:{args.port}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
