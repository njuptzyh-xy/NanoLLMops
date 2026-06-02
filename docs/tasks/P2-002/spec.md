# P2-002：接入 NanoGPT 原生推理路径

## 目标

依据 `P2-001` 设计，在正式 `nano-vllm` 路径加载 NanoGPT converted 产物并完成最小生成。

## 前置任务

- `P2-001`

## 验收标准

- 正式后端可以加载 converted 模型。
- 至少完成一次真实文本生成烟测。
- 不破坏现有 CPU runtime 回归基线。
