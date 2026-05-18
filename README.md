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

Reference repositories:

- Training: `/home/zyh-ub/PyRepos/nanoGPT`
- Inference: `/home/zyh-ub/PyRepos/nano-vllm`

Notes:

- A curated Codex skill, `cli-creator`, was installed on 2026-05-18 to support the planned CLI entrypoint.
- Restart Codex to pick up newly installed skills.
