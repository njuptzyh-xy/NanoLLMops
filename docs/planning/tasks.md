# NanoLLMOps 任务清单

更新时间：2026-06-02

## 使用规则

本文件是任务状态的唯一事实源，只维护任务编号、状态、验收目标和任务规格链接。

允许状态：

- `TODO`
- `IN_PROGRESS`
- `BLOCKED`
- `DONE`
- `CANCELLED`

详细开发过程必须写入对应任务目录：

```text
docs/tasks/<任务编号>/logs/YYYY-MM-DD-HHmm.md
```

## 当前任务

| 任务编号 | 阶段 | 状态 | 任务名称 | 验收目标 | 任务规格 |
| --- | --- | --- | --- | --- | --- |
| `P2-001` | `PHASE-02` | `TODO` | 调研正式 `nano-vllm` 原生接入点 | 明确模型注册、权重加载、执行器和 tokenizer 的接入边界 | `docs/tasks/P2-001/spec.md` |

## 后续任务

| 任务编号 | 阶段 | 状态 | 任务名称 | 前置任务 | 任务规格 |
| --- | --- | --- | --- | --- | --- |
| `P2-002` | `PHASE-02` | `TODO` | 接入 NanoGPT 原生推理路径 | `P2-001` | `docs/tasks/P2-002/spec.md` |
| `P2-003` | `PHASE-02` | `TODO` | 增加原生后端回归测试 | `P2-002` | `docs/tasks/P2-003/spec.md` |
| `P2-004` | `PHASE-02` | `TODO` | 设计模型注册表 | `P2-003` | `docs/tasks/P2-004/spec.md` |
| `P2-005` | `PHASE-02` | `TODO` | 统一 CLI 入口 | `P2-004` | `docs/tasks/P2-005/spec.md` |
| `P3-001` | `PHASE-03` | `TODO` | 建立 benchmark 报告 | `P2-003` | `docs/tasks/P3-001/spec.md` |

## 已完成任务

| 任务编号 | 阶段 | 状态 | 任务名称 | 结果摘要 | 任务规格 |
| --- | --- | --- | --- | --- | --- |
| `P0-001` | `PHASE-00` | `DONE` | 建立项目骨架与产物规范 | 已定义 artifact、转换计划和基础目录 | `docs/tasks/P0-001/spec.md` |
| `P1-001` | `PHASE-01` | `DONE` | 打通最小训推闭环 | 已完成训练入口、产物打包、推理和 HTTP demo | `docs/tasks/P1-001/spec.md` |
| `P1-002` | `PHASE-01` | `DONE` | 实现真实权重转换与校验 | 已生成并校验 `model.safetensors` | `docs/tasks/P1-002/spec.md` |
| `P1-003` | `PHASE-01` | `DONE` | converted 目录直接推理 | 已脱离原始 checkpoint 完成加载和生成 | `docs/tasks/P1-003/spec.md` |
| `P1-004` | `PHASE-01` | `DONE` | runtime 独立测试 | 已覆盖调度和 KV Cache block 管理 | `docs/tasks/P1-004/spec.md` |
| `P1-005` | `PHASE-01` | `DONE` | converted 目录接入 vLLM 风格 runtime | 已完成 CPU runtime 推理和回归测试 | `docs/tasks/P1-005/spec.md` |
| `DOC-001` | `DOCS` | `DONE` | 初步拆分规划与日志 | 已建立独立规划入口和日志目录 | `docs/tasks/DOC-001/spec.md` |
| `DOC-002` | `DOCS` | `DONE` | 建立严格任务编号驱动文档体系 | 已完成 PRD、规划、任务、阶段、手册、ADR 和归档分层 | `docs/tasks/DOC-002/spec.md` |
