# NanoLLMOps

NanoLLMOps is a lightweight train-infer integrated platform built around `nanoGPT` and `nano-vllm`.

Current repository status on 2026-05-18:

- Product scope is defined in `nanoLLMops产品文档.md`.
- Initial project skeleton is ready for phased implementation.
- The active planning and execution log lives in `docs/开发规划与实时记录.md`.

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

Reference repositories:

- Training: `/home/zyh-ub/PyRepos/nanoGPT`
- Inference: `/home/zyh-ub/PyRepos/nano-vllm`

Notes:

- A curated Codex skill, `cli-creator`, was installed on 2026-05-18 to support the planned CLI entrypoint.
- Restart Codex to pick up newly installed skills.
