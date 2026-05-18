from __future__ import annotations

from fastapi import FastAPI

from nanollmops.serving.nanogpt import GenerationRequest, GenerationResponse, NanoGPTDeployment


def create_app(checkpoint_path: str, meta_path: str, device: str = "cpu", dtype: str = "float32") -> FastAPI:
    deployment = NanoGPTDeployment(
        checkpoint_path=checkpoint_path,
        meta_path=meta_path,
        device=device,
        dtype=dtype,
    )
    app = FastAPI(title="NanoLLMOps NanoGPT Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": deployment.model_name}

    @app.post("/generate", response_model=GenerationResponse)
    def generate(request: GenerationRequest) -> GenerationResponse:
        return deployment.generate(request)

    return app
