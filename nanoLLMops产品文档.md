# NanoLLMOps：轻量级训练推理一体化平台产品文档

## 1. 项目定位

NanoLLMOps 是一个面向 AI Infra 学习、验证与简历展示的轻量级训练推理一体化平台。平台基于 nanoGPT 与 nano-vLLM 构建，目标是打通从模型训练、模型产物管理、权重转换、推理部署、在线评估到指标反馈的完整链路。

该项目不是单纯复现 nanoGPT 或 nano-vLLM，而是将二者连接成一条完整的模型生命周期闭环：

```text
数据准备 → 模型训练 → Checkpoint 管理 → 权重/配置转换 → 推理服务部署 → 请求生成 → 性能评估 → 监控反馈
```

项目重点关注三个问题：

1. 训练侧产物如何被推理侧稳定加载与服务化；
2. 推理框架内部如何完成模型加载、KV Cache 管理、Prefill/Decode 与请求调度；
3. 如何基于推理指标反向指导模型部署、推理参数调节和后续系统优化。

---

## 2. 项目背景

在大模型工程实践中，训练与推理往往由不同系统负责。训练系统关注数据加载、模型结构、Loss、Optimizer、Checkpoint 保存；推理系统关注权重加载、Batch 调度、KV Cache、请求排队、显存管理和服务指标。

如果训练与推理之间缺少统一链路，就会出现以下问题：

- 训练完成的模型产物难以直接部署；
- Checkpoint、Config、Tokenizer、权重命名不统一；
- 推理侧性能指标无法反向指导训练和部署策略；
- 模型版本、评估结果、部署状态缺少统一管理；
- 难以系统性理解“模型从训练到上线服务”的完整工程流程。

因此，NanoLLMOps 希望构建一个小而完整的训推一体化实验平台，帮助理解大模型训练与推理之间的工程连接点，并作为后续研究推理框架优化、动态调度和运行时参数调优的基础平台。

---

## 3. 项目目标

### 3.1 核心目标

构建一个基于 nanoGPT 与 nano-vLLM 的轻量级训推一体平台，完成以下闭环：

```text
nanoGPT 训练小型 GPT 模型
        ↓
保存训练产物与模型配置
        ↓
转换为 nano-vLLM 可加载格式
        ↓
通过 nano-vLLM 启动推理服务
        ↓
接收请求并完成文本生成
        ↓
采集 TTFT、TPOT、吞吐、显存等推理指标
        ↓
形成模型训练、部署与推理性能记录
```

### 3.2 能力目标

平台需要体现以下能力：

- 训练流程理解：数据处理、模型训练、Loss 计算、Optimizer 更新、Checkpoint 保存；
- 模型结构理解：Embedding、Attention、MLP、Norm、RoPE、LM Head 等核心模块；
- 权重适配能力：训练框架 Checkpoint 到推理框架权重格式的转换；
- 推理框架理解：模型加载、KV Cache 分配、Prefill/Decode、请求调度、采样；
- 服务化能力：提供统一推理接口，支持模型部署与在线请求；
- 指标评估能力：统计 TTFT、TPOT、吞吐、请求延迟、显存占用等指标；
- 后续扩展能力：支持多模型部署、动态调度、运行时参数调优和训练反馈。

---

## 4. 项目边界

### 4.1 当前阶段定位

当前阶段建议将项目定位为：

> 面向学习和验证的轻量级训推一体化实验平台。

也就是说，平台当前不追求完整工业级 MLOps 能力，而是优先打通核心链路，突出模型训练与推理框架之间的连接。

### 4.2 当前阶段做什么

当前阶段重点做：

- nanoGPT 训练流程接入；
- Checkpoint 与 Config 管理；
- Tokenizer 与模型配置对齐；
- nanoGPT 权重到 nano-vLLM 权重格式的转换；
- nano-vLLM 推理服务接入；
- 简单模型注册与版本管理；
- 推理请求接口；
- 推理性能指标采集；
- 单模型、单实例场景下的端到端闭环。

### 4.3 当前阶段暂不做什么

当前阶段暂不重点实现：

- 大规模分布式训练；
- 完整 Kubernetes 云原生调度；
- 多租户权限系统；
- 企业级可视化平台；
- 完整自动扩缩容系统；
- 复杂 RLHF / DPO / 强化学习训练链路；
- 工业级模型仓库与模型市场。

这些能力可以作为后续扩展方向，而不应成为第一阶段的实现负担。

---

## 5. 用户与使用场景

### 5.1 目标用户

NanoLLMOps 的目标用户主要包括：

- AI Infra 学习者：希望理解模型训练到推理部署完整链路；
- 推理框架研究者：希望基于轻量系统研究 KV Cache、Batch 调度、Prefill/Decode；
- 简历项目开发者：希望构建一个具有系统完整性的 AI Infra 项目；
- 教学实验用户：希望通过小模型理解 LLMOps / MLOps 的核心流程。

### 5.2 典型使用场景

#### 场景一：训练并部署一个小型 GPT 模型

用户使用 nanoGPT 在小型文本数据集上训练模型，训练完成后平台自动保存 checkpoint、config、tokenizer 信息，并通过转换模块生成 nano-vLLM 可加载的推理产物，最终部署为推理服务。

#### 场景二：分析推理框架执行流程

用户输入 prompt 后，平台记录请求在推理侧的执行路径，包括模型加载、Prefill、Decode、KV Cache 分配与更新、采样输出等流程，帮助理解推理框架内部机制。

#### 场景三：评估不同推理参数对服务性能的影响

用户调整 max_batch_size、max_model_len、temperature、top_p 等参数，平台记录 TTFT、TPOT、吞吐、显存占用等指标，辅助分析推理服务性能变化。

#### 场景四：为后续动态调度系统提供实验底座

平台采集请求级指标和 GPU 指标，为后续实现实例数量调节、运行时参数调节、请求路由、多实例协同等能力提供基础。

---

## 6. 总体架构

### 6.1 架构概览

```text
┌──────────────────────────────────────────┐
│              Web / CLI / API              │
│  提交训练任务 / 部署模型 / 发起推理请求      │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│              任务管理模块                  │
│  Train Job / Convert Job / Deploy Job      │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│              训练模块 nanoGPT              │
│  数据加载 / 模型训练 / 评估 / Checkpoint    │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│              模型管理模块                  │
│  模型版本 / Config / Tokenizer / 指标记录   │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│              权重转换模块                  │
│  Key Rename / Weight Transpose / 格式适配   │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│              推理服务模块 nano-vLLM         │
│  模型加载 / KV Cache / Prefill / Decode     │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│              监控与评估模块                │
│  TTFT / TPOT / 吞吐 / 显存 / 请求日志        │
└──────────────────────────────────────────┘
```

### 6.2 核心链路

平台最核心的链路是：

```text
Train Job → Checkpoint → Model Registry → Convert Job → Inference Engine → Metrics Feedback
```

这条链路体现了训推一体平台的本质：训练产物能够进入推理系统，推理系统产生的数据能够沉淀为后续优化依据。

---

## 7. 功能模块设计

## 7.1 数据管理模块

### 模块目标

负责训练数据集的导入、预处理、切分与元信息记录。

### 核心功能

- 支持小型文本数据集导入；
- 支持 train / val 数据切分；
- 支持字符级 tokenizer 或 BPE tokenizer；
- 保存 tokenizer 元信息；
- 记录数据集版本、样本数量、词表大小等信息。

### 关键产物

```text
data/train.bin
data/val.bin
data/meta.pkl
tokenizer.json
```

### 需要关注的问题

- 训练侧 tokenizer 必须与推理侧 tokenizer 保持一致；
- vocab_size 需要与模型 config 对齐；
- 字符级 tokenizer 虽然简单，但推理侧需要额外实现 encode/decode 逻辑。

---

## 7.2 训练任务模块

### 模块目标

基于 nanoGPT 完成小型 GPT 模型训练，生成可管理的 checkpoint 与训练指标。

### 核心功能

- 创建训练任务；
- 配置模型参数，如 n_layer、n_head、n_embd、block_size；
- 配置训练参数，如 batch_size、learning_rate、max_iters；
- 启动 nanoGPT 训练；
- 保存 checkpoint；
- 记录 train_loss、val_loss 等训练指标；
- 支持从 checkpoint 恢复训练。

### 关键产物

```text
checkpoint.pt
config.json
train_log.json
metrics.json
```

### 关键设计点

训练任务模块不只是调用 nanoGPT 脚本，而是需要将训练过程中的关键元信息保存下来，为后续推理部署提供依据。

例如：

```json
{
  "model_name": "nanogpt-shakespeare",
  "version": "v1",
  "n_layer": 6,
  "n_head": 6,
  "n_embd": 384,
  "block_size": 256,
  "vocab_size": 65,
  "train_loss": 1.42,
  "val_loss": 1.56
}
```

---

## 7.3 模型管理模块

### 模块目标

统一管理训练产物、模型配置、模型版本、转换状态与部署状态。

### 核心功能

- 注册模型；
- 管理模型版本；
- 保存 checkpoint 路径；
- 保存 tokenizer 路径；
- 保存 config 信息；
- 记录模型评估结果；
- 记录模型部署状态；
- 支持查询模型详情。

### 模型状态设计

```text
TRAINED       已完成训练
CONVERTED     已完成权重转换
DEPLOYING     正在部署
DEPLOYED      已部署
FAILED        任务失败
OFFLINE       已下线
```

### 示例模型记录

```json
{
  "model_id": "nanogpt-shakespeare-v1",
  "base_framework": "nanoGPT",
  "inference_framework": "nano-vLLM",
  "checkpoint_path": "./runs/nanogpt-shakespeare/checkpoint.pt",
  "converted_path": "./models/nanogpt-shakespeare-v1/",
  "tokenizer_path": "./models/nanogpt-shakespeare-v1/tokenizer.json",
  "status": "DEPLOYED"
}
```

---

## 7.4 权重转换模块

### 模块目标

将 nanoGPT 训练得到的 checkpoint 转换为 nano-vLLM 可加载的模型格式。

这是整个训推一体平台中最关键的连接模块。

### 核心功能

- 读取 nanoGPT checkpoint；
- 解析模型 config；
- 对齐 nanoGPT 与 nano-vLLM 的模型结构；
- 重命名权重 key；
- 必要时转置 Linear 权重；
- 拆分或合并 QKV 权重；
- 保存转换后的权重文件；
- 输出转换报告。

### 典型转换流程

```text
读取 checkpoint.pt
        ↓
提取 model_state_dict
        ↓
读取 config
        ↓
检查模型结构是否兼容
        ↓
执行权重 key 映射
        ↓
处理 Linear 权重方向
        ↓
保存为 nano-vLLM 模型目录
        ↓
生成转换报告
```

### 需要重点处理的问题

#### 1. 权重命名差异

nanoGPT 中可能存在如下权重：

```text
transformer.wte.weight
transformer.wpe.weight
transformer.h.0.attn.c_attn.weight
transformer.h.0.attn.c_proj.weight
transformer.h.0.mlp.c_fc.weight
transformer.h.0.mlp.c_proj.weight
lm_head.weight
```

而推理侧模型可能希望使用另一套命名方式。因此需要建立 key mapping。

#### 2. QKV 权重结构差异

nanoGPT 中常见做法是将 Q、K、V 合并在一个 c_attn 中：

```text
c_attn = [Wq, Wk, Wv]
```

推理侧可能需要拆分为：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

或者反过来合并。因此转换模块需要明确 QKV 的排列方式。

#### 3. Linear 权重方向差异

不同框架中 Linear 权重可能存在转置差异，需要检查：

```text
PyTorch Linear: y = x @ W.T + b
```

转换时必须确认推理侧期望的 weight shape。

#### 4. Tokenizer 与 vocab 对齐

如果训练侧使用字符级 tokenizer，推理侧必须加载相同的 char-to-id / id-to-char 映射。

---

## 7.5 推理服务模块

### 模块目标

基于 nano-vLLM 加载转换后的模型，并提供文本生成服务。

### 核心功能

- 加载模型权重；
- 加载 tokenizer；
- 初始化 KV Cache；
- 支持 Prefill 阶段；
- 支持 Decode 阶段；
- 支持采样策略，如 temperature、top_k、top_p；
- 提供 HTTP API 或 CLI 推理接口；
- 返回生成结果与基础指标。

### 推理请求示例

```json
{
  "model": "nanogpt-shakespeare-v1",
  "prompt": "To be or not to be",
  "max_tokens": 128,
  "temperature": 0.8,
  "top_p": 0.95
}
```

### 推理响应示例

```json
{
  "model": "nanogpt-shakespeare-v1",
  "output": "...",
  "input_tokens": 18,
  "output_tokens": 128,
  "latency_ms": 860,
  "tokens_per_second": 148.8
}
```

### 需要关注的问题

- 推理侧模型结构必须与训练侧完全一致；
- max_model_len 不能超过训练时 block_size 或模型位置编码能力；
- tokenizer encode/decode 必须一致；
- KV Cache shape 需要与 n_layer、n_head、head_dim 对齐；
- 单请求流程先打通，再扩展 batch 调度。

---

## 7.6 推理评估模块

### 模块目标

对推理服务进行性能评估，记录服务级指标，为后续调度优化提供依据。

### 核心指标

| 指标 | 含义 | 作用 |
|---|---|---|
| TTFT | Time To First Token，首 token 延迟 | 衡量 Prefill 与排队开销 |
| TPOT | Time Per Output Token，每输出 token 耗时 | 衡量 Decode 性能 |
| Latency | 请求总延迟 | 衡量端到端体验 |
| Throughput | 吞吐量 | 衡量单位时间处理 token 或请求能力 |
| QPS | 每秒请求数 | 衡量服务处理能力 |
| GPU Memory | 显存占用 | 衡量模型与 KV Cache 开销 |
| GPU Utilization | GPU 利用率 | 衡量计算资源利用情况 |
| KV Cache Usage | KV Cache 使用量 | 衡量上下文与并发压力 |

### 评估方式

第一阶段可以采用小型压测脚本：

```text
固定 prompt 长度，调整 max_tokens
固定 max_tokens，调整并发请求数
调整 max_model_len
调整 batch size
记录不同配置下的指标变化
```

### 输出结果

```json
{
  "model": "nanogpt-shakespeare-v1",
  "concurrency": 4,
  "avg_ttft_ms": 42.5,
  "avg_tpot_ms": 8.7,
  "p95_latency_ms": 1180,
  "tokens_per_second": 312.4,
  "gpu_memory_mb": 1860
}
```

---

## 7.7 任务管理模块

### 模块目标

对训练、转换、部署、评估等任务进行统一管理。

### 核心任务类型

```text
TrainJob      训练任务
ConvertJob    权重转换任务
DeployJob     推理部署任务
EvalJob       推理评估任务
OfflineJob    模型下线任务
```

### 任务状态设计

```text
PENDING
RUNNING
SUCCEEDED
FAILED
CANCELLED
```

### 任务记录示例

```json
{
  "job_id": "job-20260201-001",
  "job_type": "ConvertJob",
  "model_id": "nanogpt-shakespeare-v1",
  "status": "SUCCEEDED",
  "start_time": "2026-02-01 10:00:00",
  "end_time": "2026-02-01 10:01:20",
  "log_path": "./logs/job-20260201-001.log"
}
```

---

## 7.8 监控与日志模块

### 模块目标

采集训练、转换、部署和推理过程中的关键日志与指标。

### 核心功能

- 训练日志记录；
- 转换日志记录；
- 推理请求日志记录；
- 推理性能指标记录；
- GPU 指标采集；
- 错误日志与异常追踪。

### 日志类型

```text
train.log
convert.log
serve.log
request.log
metrics.jsonl
```

### 请求日志示例

```json
{
  "request_id": "req-001",
  "model": "nanogpt-shakespeare-v1",
  "prompt_tokens": 18,
  "output_tokens": 128,
  "ttft_ms": 40.2,
  "tpot_ms": 8.4,
  "latency_ms": 1115,
  "timestamp": "2026-02-01 10:15:00"
}
```

---

## 7.9 前端 / CLI / API 入口

### 模块目标

为用户提供统一操作入口。第一阶段可以优先实现 CLI 和 HTTP API，后续再扩展 Web UI。

### CLI 示例

```bash
trainservex train --config configs/train_nanogpt.yaml
trainservex convert --model nanogpt-shakespeare-v1
trainservex deploy --model nanogpt-shakespeare-v1
trainservex infer --model nanogpt-shakespeare-v1 --prompt "To be or not to be"
trainservex benchmark --model nanogpt-shakespeare-v1 --concurrency 4
```

### API 示例

```text
POST /api/train
POST /api/convert
POST /api/deploy
POST /api/generate
GET  /api/models
GET  /api/jobs
GET  /api/metrics
```

---

## 8. 数据模型设计

### 8.1 Model 表

| 字段 | 含义 |
|---|---|
| model_id | 模型唯一 ID |
| model_name | 模型名称 |
| version | 模型版本 |
| framework | 训练框架 |
| checkpoint_path | 原始 checkpoint 路径 |
| converted_path | 转换后模型路径 |
| tokenizer_path | tokenizer 路径 |
| config | 模型配置 |
| status | 模型状态 |
| created_at | 创建时间 |

### 8.2 Job 表

| 字段 | 含义 |
|---|---|
| job_id | 任务 ID |
| job_type | 任务类型 |
| model_id | 关联模型 |
| status | 任务状态 |
| config | 任务配置 |
| log_path | 日志路径 |
| created_at | 创建时间 |
| updated_at | 更新时间 |

### 8.3 Metric 表

| 字段 | 含义 |
|---|---|
| metric_id | 指标 ID |
| model_id | 模型 ID |
| request_id | 请求 ID |
| ttft_ms | 首 token 延迟 |
| tpot_ms | 每 token 延迟 |
| latency_ms | 总延迟 |
| throughput | 吞吐 |
| gpu_memory | 显存占用 |
| timestamp | 记录时间 |

---

## 9. 技术架构建议

### 9.1 第一阶段技术选型

| 模块 | 建议技术 |
|---|---|
| 训练框架 | nanoGPT / PyTorch |
| 推理框架 | nano-vLLM |
| API 服务 | FastAPI |
| 任务管理 | Python subprocess / multiprocessing / Celery 可选 |
| 元数据存储 | SQLite |
| 指标存储 | JSONL / SQLite |
| 日志 | Python logging |
| CLI | Typer / argparse |
| 部署 | Docker |
| GPU 指标 | pynvml / nvidia-smi |

### 9.2 第二阶段技术选型

| 模块 | 可扩展技术 |
|---|---|
| 任务队列 | Redis + Celery |
| 指标监控 | Prometheus + Grafana |
| 模型服务 | FastAPI + Uvicorn / OpenAI Compatible API |
| 多实例管理 | Docker Compose / Kubernetes |
| 模型管理 | MLflow 风格 Model Registry |
| 推理网关 | OpenAI-compatible Router |

---

## 10. 参考设计

NanoLLMOps 可以参考以下系统的设计思想，但不需要完整复刻。

### 10.1 nanoGPT

参考重点：

- 简洁训练循环；
- GPT 模型结构实现；
- Checkpoint 保存方式；
- 数据预处理与 tokenizer 元信息保存；
- 从训练 Loss 到模型权重的完整流程。

在 NanoLLMOps 中，nanoGPT 主要作为训练侧基础组件。

### 10.2 nano-vLLM

参考重点：

- 推理框架基本结构；
- 模型加载流程；
- KV Cache 管理；
- Prefill / Decode 分离；
- Batch 调度；
- Sampling 逻辑。

在 NanoLLMOps 中，nano-vLLM 主要作为推理侧基础组件。

### 10.3 vLLM

参考重点：

- PagedAttention 思想；
- KV Cache block 管理；
- Continuous Batching；
- OpenAI-compatible API；
- 推理服务参数设计，如 max_model_len、max_num_seqs、gpu_memory_utilization。

NanoLLMOps 不需要完整实现 vLLM，但可以借鉴其推理服务指标和运行时参数设计。

### 10.4 SGLang

参考重点：

- 面向 LLM 应用的推理服务接口；
- 请求调度与运行时优化；
- 多模型服务与推理后端集成思路。

可以作为后续扩展多后端推理服务的参考。

### 10.5 MLflow / Model Registry 思想

参考重点：

- 模型版本管理；
- 实验参数记录；
- 评估指标记录；
- 模型状态流转。

NanoLLMOps 可以实现一个轻量级 Model Registry，不需要完整接入 MLflow。

### 10.6 KServe / Triton Inference Server 思想

参考重点：

- 模型服务化；
- 模型部署状态管理；
- 推理接口规范；
- 服务监控与版本切换。

当前阶段只需要参考其服务化思想，不需要引入复杂云原生架构。

---

## 11. 关键技术难点

### 11.1 训练模型结构与推理模型结构对齐

这是项目最核心难点。nanoGPT 训练出的模型必须能够被 nano-vLLM 中的模型定义正确加载。需要保证：

- 层数一致；
- hidden size 一致；
- head 数一致；
- vocab size 一致；
- block size / max position 一致；
- Attention、MLP、Norm 结构一致；
- 权重 shape 一致。

### 11.2 Checkpoint 到推理权重的转换

不同框架的权重命名和组织方式不同，转换模块需要处理：

- key rename；
- tensor transpose；
- QKV split / merge；
- tied embedding；
- dtype 转换；
- safetensors 或 pt 格式保存。

### 11.3 Tokenizer 对齐

训练和推理必须使用同一套 tokenizer，否则模型输入 token id 将不一致，导致推理结果异常。

### 11.4 推理指标采集

TTFT、TPOT、吞吐、延迟分位数等指标需要在请求生命周期中准确记录。需要明确：

- 请求进入时间；
- Prefill 开始与结束时间；
- 第一个 token 输出时间；
- 每个 decode step 时间；
- 请求完成时间。

### 11.5 与后续动态调度系统衔接

平台采集的指标要为后续调度优化服务。因此指标设计不能只记录最终 latency，还要记录更细粒度的请求特征和资源状态。

---

## 12. 里程碑规划

### 阶段一：MVP 闭环

目标：打通最小可用链路。

任务：

- 接入 nanoGPT 训练脚本；
- 保存 checkpoint、config、tokenizer；
- 编写权重转换脚本；
- 在 nano-vLLM 中加载转换后的模型；
- 提供单请求文本生成能力；
- 输出基础 latency 和 tokens/s。

交付物：

```text
train.py
convert_nanogpt_to_nanovllm.py
serve.py
generate.py
README.md
```

---

### 阶段二：模型管理与任务管理

目标：让平台具备基本工程化能力。

任务：

- 设计模型元数据表；
- 设计任务状态表；
- 支持 TrainJob / ConvertJob / DeployJob；
- 支持模型状态查询；
- 支持日志记录；
- 支持 CLI 操作。

交付物：

```text
model_registry.py
job_manager.py
metadata.db
trainservex CLI
```

---

### 阶段三：推理评估与监控

目标：形成推理性能分析能力。

任务：

- 统计 TTFT；
- 统计 TPOT；
- 统计吞吐；
- 统计请求延迟分位数；
- 采集 GPU 显存；
- 编写 benchmark 脚本；
- 输出评估报告。

交付物：

```text
benchmark.py
metrics_collector.py
report.json
report.md
```

---

### 阶段四：推理优化实验

目标：探索不同推理参数对性能的影响。

任务：

- 调整 max_model_len；
- 调整 batch size；
- 调整并发请求数；
- 分析 Prefill 与 Decode 性能差异；
- 对比不同模型配置下的指标变化。

交付物：

```text
experiments/
performance_report.md
```

---

### 阶段五：动态调度扩展

目标：与后续 AI Infra 调度系统结合。

任务：

- 引入请求路由模块；
- 引入多实例部署；
- 支持运行时参数动态调节；
- 支持实例数量动态调节；
- 支持请求侧统计与 GPU 指标联动。

交付物：

```text
router.py
scheduler.py
autoscaler.py
runtime_tuner.py
```

---

## 13. 推荐项目目录结构

```text
trainservex/
├── README.md
├── configs/
│   ├── train_nanogpt.yaml
│   ├── convert.yaml
│   └── serve.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── tokenizer/
├── train/
│   ├── nanogpt_adapter.py
│   ├── train_runner.py
│   └── eval_runner.py
├── registry/
│   ├── model_registry.py
│   ├── metadata_store.py
│   └── schema.py
├── converter/
│   ├── convert_nanogpt.py
│   ├── key_mapping.py
│   └── validate.py
├── serving/
│   ├── nanovllm_adapter.py
│   ├── server.py
│   ├── generate.py
│   └── tokenizer.py
├── benchmark/
│   ├── benchmark.py
│   ├── workload.py
│   └── report.py
├── monitor/
│   ├── metrics_collector.py
│   ├── gpu_monitor.py
│   └── request_logger.py
├── scheduler/
│   ├── router.py
│   ├── runtime_tuner.py
│   └── autoscaler.py
├── cli/
│   └── main.py
└── tests/
    ├── test_converter.py
    ├── test_tokenizer.py
    └── test_inference.py
```

---

## 14. 第一阶段最小实现建议

第一阶段不要做得太大，建议只实现以下最小闭环：

```text
1. 使用 nanoGPT 训练一个 char-level 小模型
2. 保存 checkpoint.pt 与 meta.pkl
3. 实现 convert_nanogpt_to_nanovllm.py
4. 在 nano-vLLM 中新增 NanoGPTForCausalLM 加载逻辑
5. 实现单请求 generate
6. 记录 latency、tokens/s、显存占用
7. 输出 README 和流程图
```

这一步完成后，项目就已经具备“训推一体”的核心价值。

---

## 15. 简历表达建议

### 简历项目标题

NanoLLMOps：基于 nanoGPT 与 nano-vLLM 的轻量级训推一体平台

### 简历描述版本

- 设计并实现轻量级训推一体平台，打通 nanoGPT 模型训练、Checkpoint 管理、权重转换、nano-vLLM 推理部署与性能评估链路。
- 实现训练产物到推理框架的适配流程，处理模型配置对齐、Tokenizer 对齐、权重 Key 映射、QKV 拆分与 Linear 权重转换等问题。
- 分析 nano-vLLM 推理执行流程，重点梳理模型加载、KV Cache 管理、Prefill/Decode、请求调度与采样生成机制。
- 设计推理评估模块，采集 TTFT、TPOT、吞吐、请求延迟、GPU 显存等指标，为后续动态调度与运行时参数调优提供数据基础。

---

## 16. 项目价值总结

NanoLLMOps 的价值不在于实现一个完整工业级平台，而在于用轻量级方式打通大模型工程中最重要的一条链路：

```text
训练产物 → 模型管理 → 权重转换 → 推理部署 → 性能评估 → 优化反馈
```

它能够很好地体现 AI Infra 方向中的核心能力：

- 懂模型训练；
- 懂模型结构；
- 懂推理框架；
- 懂权重适配；
- 懂服务部署；
- 懂性能指标；
- 懂后续调度优化方向。

因此，该项目非常适合作为 AI Infra 简历中的代表项目，也可以作为后续深入 vLLM、SGLang、推理调度、KV Cache 优化和多实例协同服务的基础平台。

