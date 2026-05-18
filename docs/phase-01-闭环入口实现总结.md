# Phase 1 总结：闭环入口实现

更新时间：2026-05-18

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

当前仍有三块没有完成：

1. `nanollmops` 解释器环境中的核心依赖还不完整
2. 还没有把 `ckpt.pt` 转成 HF-compatible `config.json + safetensors`
3. 还没有把模型真正接入 `nano-vllm` 的调度和 KV Cache 运行时

## 下一步开发重点

下一阶段应直接进入以下内容：

1. 补齐 `nanollmops` 环境中的 `torch/transformers/safetensors`
2. 编写真正的 `ckpt.pt -> converted/` 转换器
3. 设计 `NanoGPTForCausalLM` 的 `nano-vllm` 接入实现
4. 让 `converted/` 产物可以被后续推理后端稳定加载
