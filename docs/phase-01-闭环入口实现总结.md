# Phase 1 总结：闭环入口实现

更新时间：2026-05-22

## 本阶段目标

尽快把训练到推理部署的最小链路从“规划”推进到“可执行入口”。

这一阶段的策略不是先做 `nano-vllm` 深度改造，而是先把最小闭环入口搭起来：

```text
nanoGPT 训练 -> checkpoint/meta 打包 -> 本地推理 -> FastAPI 服务
```

## 本阶段完成的内容

### 1. 训练入口封装

新增：

- `src/nanollmops/train/nanogpt_runner.py`
- `scripts/train_nanogpt.py`
- `configs/train_shakespeare_char.py`

作用：

- 通过 NanoLLMOps 统一入口触发 `/home/zyh-ub/PyRepos/nanoGPT/train.py`
- 保持训练配置仍然兼容 nanoGPT 原生方式
- 为后面接任务管理和 CLI 提供稳定脚本入口

### 2. 训练产物打包

新增：

- `scripts/package_nanogpt_model.py`

作用：

- 接收 `ckpt.pt` 和 `meta.pkl`
- 生成标准 artifact 目录
- 输出 manifest、conversion plan 和 Markdown 摘要
- 将原始 checkpoint/tokenizer 元数据复制到平台目录下

当前产物布局示例：

```text
artifacts/shakespeare-char-v1/
├── artifact_manifest.json
├── conversion_plan.json
├── conversion_plan.md
└── source/
    ├── ckpt.pt
    └── meta.pkl
```

### 3. 最小推理加载与生成

新增：

- `src/nanollmops/serving/nanogpt.py`
- `scripts/infer_nanogpt.py`

作用：

- 在本仓库内实现最小 GPT 推理模型
- 直接加载 `nanoGPT` checkpoint
- 使用 `meta.pkl` 中的 char tokenizer 做 encode/decode
- 输出生成文本和基础指标：
  - `latency_ms`
  - `tokens_per_second`

### 4. 最小服务部署

新增：

- `src/nanollmops/serving/api.py`
- `scripts/serve_nanogpt.py`
- `scripts/serve_nanogpt_simple.py`

作用：

- 提供 FastAPI 服务入口
- 暴露 `/health`
- 暴露 `/generate`
- 在依赖受限时，提供标准库 HTTP fallback 服务

这让当前项目第一次具备了“部署为服务”的基本形态。

### 5. 端到端运行文档

新增：

- `docs/MVP端到端运行说明.md`

作用：

- 明确训练、打包、推理、启动服务的执行顺序
- 给后续验证和演示提供固定命令模板

## 本阶段验证结果

已通过的真实验证：

- 成功使用 `nanoGPT` 现有 `ckpt.pt` 进行本地推理
- 成功生成文本结果
- 成功将真实 `ckpt.pt` 和 `meta.pkl` 打包进 `artifacts/shakespeare-char-v1/`
- 成功启动本地 HTTP demo 服务
- 成功调用 `/health`
- 成功调用 `/generate`

验证样例说明：

- checkpoint 来源：`/home/zyh-ub/PyRepos/nanoGPT/out-shakespeare-char/ckpt.pt`
- tokenizer 来源：`/home/zyh-ub/PyRepos/nanoGPT/data/shakespeare_char/meta.pkl`

## 本阶段修复的问题

在真实 checkpoint 打包过程中发现：

- `best_val_loss` 在 checkpoint 中可能是 tensor，而不是纯 Python float

已修复：

- 在 `converter/planner.py` 中增加标量归一化逻辑，保证 manifest/plan 可正确序列化为 JSON

## 当前状态评估

现在仓库已经不再只是“规划 + 转换草案”，而是具备了实际闭环入口：

- 能触发训练
- 能整理产物
- 能做本地推理
- 能启动服务

这已经满足“尽快跑通训练到推理部署流程”的第一阶段要求。

## 仍未完成的关键点

最初缺的三块里，前两块已经补齐：

1. `nanollmops` 环境已补齐 `torch`、`safetensors`、`numpy`
2. 已实现真实 `ckpt.pt -> converted/config.json + model.safetensors`
3. `vLLM-style runtime` 也已补到仓库内，但仍不是 `nano-vllm` 原生模型接入

## 新进展：已接入 vLLM 风格核心链路

在后续开发中，已经补上了一条 CPU 可运行的 `vLLM-style` GPT runtime，用于把 `nano-vllm` 的核心思想真正落进当前项目。

新增模块：

- `src/nanollmops/runtime/config.py`
- `src/nanollmops/runtime/sampling_params.py`
- `src/nanollmops/runtime/sequence.py`
- `src/nanollmops/runtime/block_manager.py`
- `src/nanollmops/runtime/scheduler.py`
- `src/nanollmops/runtime/nanogpt_vllm.py`

这条 runtime 已经包含：

- `Sequence`
- `BlockManager`
- `Scheduler`
- `KV Cache` block 分配
- `Prefill / Decode` 分离
- 基于 prompt/request 的调度推进

这意味着当前 demo 已经不只是“直接调用 `model.generate()`”，而是具备了 `nano-vllm` 风格的核心执行链。

## vLLM-style demo 验证结果

已通过的真实验证新增如下：

- 成功使用 `scripts/infer_nanogpt_vllm.py` 运行离线推理
- 成功使用 `scripts/serve_nanogpt_vllm.py` 启动本地 HTTP 服务
- 成功调用 `GET /health`
- 成功调用 `POST /generate`

健康检查返回：

```json
{
  "status": "ok",
  "model": "shakespeare-char-v1",
  "engine": "vllm-style-gpt",
  "kvcache_block_size": 16,
  "num_kvcache_blocks": 64
}
```

生成接口样例返回：

```json
{
  "prompt": "To be or not to be",
  "output": "To be or not to belowing you.\n\nRING EEN LO",
  "input_tokens": 18,
  "output_tokens": 24,
  "latency_ms": 87.651,
  "tokens_per_second": 273.812,
  "model": "shakespeare-char-v1",
  "engine": "vllm-style-gpt",
  "kvcache_block_size": 16,
  "num_kvcache_blocks": 64
}
```

## 下一步开发重点

下一阶段应直接进入以下内容：

1. 补齐 `nanollmops` 环境中的 `torch/transformers/safetensors`
2. 编写真正的 `ckpt.pt -> converted/` 转换器
3. 设计 `NanoGPTForCausalLM` 的 `nano-vllm` 接入实现
4. 让 `converted/` 产物可以被后续推理后端稳定加载

## 新进展：已补齐真实转换与转换后校验

在最新开发中，Phase 1 里原本最关键的缺口已经补上：项目现在不仅能生成 conversion plan，也能实际输出 converted model bundle。

新增模块与脚本：

- `src/nanollmops/converter/convert.py`
- `scripts/convert_nanogpt_to_nanovllm.py`
- `src/nanollmops/converter/validate.py`
- `scripts/validate_converted_nanogpt.py`
- `tests/test_converter.py`
- `tests/test_validate_converter.py`

新增能力：

- 从真实 `nanoGPT ckpt.pt` 输出：
  - `converted/config.json`
  - `converted/generation_config.json`
  - `converted/tokenizer.json`
  - `converted/tokenizer_config.json`
  - `converted/special_tokens_map.json`
  - `converted/model.safetensors`
- 支持 `--metadata-only` 模式，便于在依赖不完整时先生成骨架
- 对转换后的权重执行 key/shape 校验，提前发现坏包

真实验证结果：

- 已在 `nanollmops` 环境中安装 `torch 2.12.0`、`safetensors 0.7.0`、`numpy 2.2.6`
- 已使用真实 checkpoint 成功生成：
  - `artifacts/shakespeare-char-v1/converted/model.safetensors`
- 已使用校验脚本验证真实 converted 目录：
  - `tensor_count = 36`
  - `checked_keys = 36`

本阶段新增修复：

- `lm_head.weight` 与 `embed_tokens.weight` 在原始 checkpoint 中共享底层存储
- `safetensors` 默认拒绝直接保存共享存储 tensor
- 已在转换阶段显式复制输出 tensor，保证写盘稳定

最新阶段结论：

- `Phase 1` 不再只是“闭环入口可执行”
- 当前已经具备：
  - 原始训练产物整理
  - 本地 ckpt 直接推理
  - `vLLM-style runtime` demo
  - 真实 converted bundle 输出
  - converted bundle 自检
- 因此下一步的技术重心应切换为：
  - 从 `converted/` 目录加载模型并推理
  - 将 converted 产物真正接到后续推理后端
