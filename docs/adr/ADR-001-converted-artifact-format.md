# ADR-001：converted 产物采用 safetensors 目录格式

状态：`ACCEPTED`

日期：2026-06-02

## 背景

原始 `nanoGPT` checkpoint 同时包含训练状态和模型权重，不适合直接作为稳定部署接口。

## 决策

推理侧统一优先读取独立 `converted/` 目录：

```text
converted/
├── config.json
├── tokenizer.json
├── generation_config.json
├── tokenizer_config.json
├── special_tokens_map.json
└── model.safetensors
```

## 原因

- 将部署产物与训练状态解耦。
- 允许在推理前校验 tensor key 和 shape。
- 便于后续接入正式推理后端。
- 避免持续依赖原始 `ckpt.pt` 和 `meta.pkl`。

## 影响

- 转换模块必须维护权重映射。
- 普通推理和 runtime 应复用同一套 converted 加载逻辑。
- 正式 `nano-vllm` 接入时需要评估当前命名是否需要调整。

## 关联任务

- `P1-002`
- `P1-003`
- `P1-005`
- `P2-001`
