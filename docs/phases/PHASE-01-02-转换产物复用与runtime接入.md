# Phase 1 到 Phase 2 过渡总结：转换产物复用与 runtime 接入

更新时间：2026-06-02

关联任务：`P1-003`、`P1-004`、`P1-005`

## 当前阶段

`Phase 1` 的最小训推闭环已经完成。当前处于 `Phase 1 -> Phase 2` 过渡阶段。

本轮开发解决的核心问题是：

```text
converted/ 目录能否脱离原始 ckpt.pt 和 meta.pkl，
独立完成加载、校验、普通推理和 vLLM 风格 runtime 推理
```

当前答案是：已经可以。

下一开发重点不再是补齐本仓库 CPU demo，而是调研并接入正式 `nano-vllm` 原生推理路径。

## 本轮完成内容

### 1. converted 目录直接加载

新增 `NanoGPTDeployment.from_converted(...)`，支持从以下文件恢复模型：

```text
converted/
├── config.json
├── tokenizer.json
└── model.safetensors
```

加载过程：

1. 校验 converted 目录中的权重名称和形状。
2. 从 `config.json` 恢复 GPT 配置。
3. 从 `tokenizer.json` 恢复 char tokenizer。
4. 将独立的 Q、K、V 权重重新拼接为 NanoGPT 使用的 fused QKV 权重。
5. 加载模型并执行文本生成。

### 2. converted 权重共享加载

新增 `src/nanollmops/converter/load.py`。

该模块统一负责：

- 读取 `model.safetensors`
- 执行 converted 产物校验
- 将转换后的权重名称映射回 NanoGPT state dict
- 重组 fused QKV 权重

普通推理与 vLLM 风格 runtime 共用这套加载逻辑，避免两条路径分别维护权重映射。

### 3. converted 目录接入 vLLM 风格 runtime

新增 `VLLMStyleNanoGPT.from_converted(...)`。

该入口支持从 converted 目录初始化：

- NanoGPT 模型
- char tokenizer
- scheduler
- KV Cache block
- Prefill / Decode 执行状态

新增脚本：

- `scripts/infer_converted_nanogpt_vllm.py`

### 4. runtime 独立测试

新增 `tests/test_runtime.py`，覆盖：

- `Sequence` 的 token、block 和 completion 状态
- prefix cache block 释放后的复用
- Decode 跨 block 时的扩展逻辑
- Scheduler 完成请求后的 KV Cache block 释放

扩展 `tests/test_converted_inference.py`，覆盖：

- converted 目录普通推理
- converted 目录进入 vLLM 风格 runtime 推理

## 核心代码位置

| 能力 | 核心代码或脚本 |
| --- | --- |
| converted 目录校验 | `src/nanollmops/converter/validate.py` |
| converted 权重共享加载 | `src/nanollmops/converter/load.py` |
| 普通推理加载 | `src/nanollmops/serving/nanogpt.py` |
| vLLM 风格 runtime 加载 | `src/nanollmops/runtime/nanogpt_vllm.py` |
| 普通 converted 推理脚本 | `scripts/infer_converted_nanogpt.py` |
| converted runtime 推理脚本 | `scripts/infer_converted_nanogpt_vllm.py` |
| converted 推理测试 | `tests/test_converted_inference.py` |
| runtime 独立测试 | `tests/test_runtime.py` |

## 验证结果

完整测试：

```bash
conda run -n nanollmops python -m unittest discover -s tests -v
```

结果：

```text
Ran 14 tests
OK
```

converted 普通推理烟测：

```text
prompt        -> "To be"
input_tokens  -> 5
output_tokens -> 4
output        -> "To be gon"
```

converted vLLM 风格 runtime 烟测：

```text
prompt        -> "To be"
input_tokens  -> 5
output_tokens -> 4
output        -> "To be you"
engine        -> "vllm-style-gpt"
```

原 checkpoint runtime 回归：

```text
prompt        -> "To be"
input_tokens  -> 5
output_tokens -> 2
output        -> "To bed "
engine        -> "vllm-style-gpt"
```

## 当前边界

当前已经完成：

- 训练产物打包
- 权重转换与校验
- converted 目录普通推理
- converted 目录进入本仓库 CPU 版 vLLM 风格 runtime
- runtime 调度和 KV Cache block 管理基础测试

当前尚未完成：

- 正式接入 `/home/zyh-ub/PyRepos/nano-vllm` 原生模型注册和执行器
- 接入原生 `flash-attn`、`triton` 和 GPU 高性能路径
- 建立正式 benchmark 与回归指标报告

## 下一步

1. 阅读 `/home/zyh-ub/PyRepos/nano-vllm` 当前模型加载、模型注册和执行器实现。
2. 明确 NanoGPT 模型结构适配点，以及当前 converted 权重格式是否需要调整。
3. 保留本仓库 CPU demo 和现有 `14` 项测试作为后续接入回归基线。
