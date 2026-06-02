# NanoLLMOps

NanoLLMOps is a lightweight train-infer integrated platform built around `nanoGPT` and `nano-vllm`, with the long-term goal of combining the two codebases coherently and iteratively optimizing the full training-to-serving workflow.

Current repository status on 2026-06-02:

- Product scope is defined in `docs/product/PRD.md`.
- Phase 1 MVP is complete. Converted artifacts can run through local inference and the CPU vLLM-style runtime.
- The active task plan lives in `docs/planning/tasks.md`.
- Task specifications and logs are stored under `docs/tasks/`.
- Read `docs/index.md` before starting a new development task.

Project direction:

1. Keep `nanoGPT` as the main reference for training flow, checkpoint structure, and lightweight GPT experimentation.
2. Keep `nano-vllm` as the main reference for runtime scheduling, KV-cache management, and high-throughput inference design.
3. Use NanoLLMOps as the integration layer that gradually connects the two and absorbs validated optimizations back into a clear, reproducible pipeline.

Near-term focus:

1. Build the MVP closed loop from `nanoGPT` training artifacts to `nano-vllm` inference.
2. Add lightweight model registry, job management, and metrics collection.
3. Keep the project readable and educational instead of prematurely over-engineering it.

Current implemented entrypoints:

- `scripts/train_nanogpt.py`: trigger `nanoGPT` training from the NanoLLMOps repo.
- `scripts/package_nanogpt_model.py`: package `ckpt.pt` and `meta.pkl` into the artifact layout.
- `scripts/convert_nanogpt_to_nanovllm.py`: write the `converted/` bundle with config/tokenizer metadata and, when dependencies are present, `model.safetensors`.
- `scripts/infer_nanogpt.py` / `scripts/serve_nanogpt.py`: run the minimal local inference path.
- `scripts/infer_nanogpt_vllm.py` / `scripts/serve_nanogpt_vllm.py`: run the vLLM-style runtime path.
- `scripts/infer_converted_nanogpt.py`: load and infer directly from a converted artifact.
- `scripts/infer_converted_nanogpt_vllm.py`: load a converted artifact into the CPU vLLM-style runtime.

Reference repositories:

- Training: `/home/zyh-ub/PyRepos/nanoGPT`
- Inference: `/home/zyh-ub/PyRepos/nano-vllm`

Notes:

- A curated Codex skill, `cli-creator`, was installed on 2026-05-18 to support the planned CLI entrypoint.
- Restart Codex to pick up newly installed skills.
