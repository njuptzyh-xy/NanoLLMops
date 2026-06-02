# P2-001：调研正式 nano-vllm 原生接入点

## 目标

明确 NanoGPT converted 产物接入正式 `nano-vllm` 的模型注册、权重加载、执行器和 tokenizer 边界。

## 范围

- 阅读 `/home/zyh-ub/PyRepos/nano-vllm` 模型加载和执行流程。
- 输出适配设计。
- 判断当前 converted 权重命名是否需要调整。

## 非目标

- 不实现完整原生后端。
- 不引入模型注册数据库。

## 验收标准

- 输出接入点清单。
- 输出最小实现步骤。
- 标明风险、依赖和验证路径。

## 关联文档

- `docs/product/architecture.md`
- `docs/adr/ADR-001-converted-artifact-format.md`
