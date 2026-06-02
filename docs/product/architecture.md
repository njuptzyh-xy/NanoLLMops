# NanoLLMOps 技术架构

更新时间：2026-06-02

## 架构目标

将 `nanoGPT` 训练产物稳定转换为可复用模型目录，并逐步接入 `nano-vllm` 原生推理路径。

## 当前链路

```text
nanoGPT 训练
-> ckpt.pt + meta.pkl
-> NanoLLMOps artifact 打包
-> converted/config.json + tokenizer.json + model.safetensors
-> 完整性校验
-> 普通推理或 CPU vLLM 风格 runtime
```

## 核心模块

| 模块 | 位置 | 职责 |
| --- | --- | --- |
| 训练入口 | `src/nanollmops/train/` | 调用 `nanoGPT/train.py` |
| 产物规范 | `src/nanollmops/artifacts.py` | 描述训练产物、转换计划和目录布局 |
| 转换模块 | `src/nanollmops/converter/` | 转换、校验和加载 safetensors |
| 普通推理 | `src/nanollmops/serving/` | 提供最小生成和 HTTP 服务 |
| CPU runtime | `src/nanollmops/runtime/` | 验证调度、KV Cache、Prefill 和 Decode |

## 当前边界

已经完成：

- 训练产物固化
- converted 目录生成与校验
- converted 目录普通推理
- converted 目录进入 CPU vLLM 风格 runtime

尚未完成：

- 正式接入 `/home/zyh-ub/PyRepos/nano-vllm`
- 原生 GPU 高性能路径
- 模型注册表和统一 CLI
- benchmark 报告

## 关键决策

- converted 产物格式见 `docs/adr/ADR-001-converted-artifact-format.md`
- 正式 `nano-vllm` 接入方案由 `P2-001` 输出
