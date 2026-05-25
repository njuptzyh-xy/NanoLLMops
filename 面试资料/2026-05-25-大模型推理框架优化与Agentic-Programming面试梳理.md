# 大模型推理框架优化与 Agentic Programming 面试梳理

更新时间：2026-05-25

## 1. 面试定位一句话版本

如果面试官问“你这个项目是做什么的”，建议先用这句话开场：

`NanoLLMOps` 是一个基于 `nanoGPT` 与 `nano-vllm` 思路构建的轻量级训推一体实验平台，目标是打通训练产物管理、权重转换、推理加载、服务部署与推理指标采集，并在这个闭环里持续做推理框架优化和 Agent 化编排探索。

## 2. 你可以怎么讲三个项目

### 2.1 nanoGPT

你可以把 `nanoGPT` 定位为训练侧参考系：

- 我把它当作训练与 checkpoint 产物的来源
- 重点理解的是 `train.py`、`model.py`、`sample.py` 以及 `shakespeare_char` 这条最小实验链路
- 它帮助我掌握了小型 GPT 的训练结构、checkpoint payload 组织方式、字符级 tokenizer 以及 tied embedding / lm_head 这类实现细节

面试里建议强调：

- 你不是只会“跑训练脚本”，而是理解训练侧产物如何进入推理侧
- 你知道 `model_args`、`state_dict`、`meta.pkl` 分别承担什么作用
- 你能从训练侧视角解释后续推理适配为什么困难

### 2.2 nano-vllm

你可以把 `nano-vllm` 定位为推理侧参考系：

- 我重点拆解了它的 `scheduler`、`model_runner`、`llm_engine`、attention 和 loader 相关实现
- 关注点不是复刻整个工程，而是提取推理框架最关键的运行时思想
- 核心包括：`Prefill / Decode` 分离、请求调度、KV Cache 管理、block 化缓存、吞吐优先的执行路径

面试里建议强调：

- 你理解推理框架和普通 `model.generate()` 的本质区别
- 你知道推理框架优化的核心不是“把模型跑起来”，而是“如何高吞吐、低延迟、可并发地跑起来”
- 你能自然过渡到 continuous batching、KV cache、调度策略这些考点

### 2.3 NanoLLMOps

你可以把 `NanoLLMOps` 定位为你自己的集成与验证平台：

- 目标不是替代 `nanoGPT` 和 `nano-vllm`
- 而是把训练侧和推理侧能力收敛到一条可运行、可验证、可解释的闭环里
- 当前已经实现了最小训练入口、artifact 打包、checkpoint 转换、离线推理、服务部署，以及一条 `vLLM-style runtime`

建议你对外表述为：

`NanoLLMOps` 是我把训练框架理解、推理框架理解和平台工程能力合在一起做验证的主项目。

## 3. 你的实际经办内容

这一段最重要，建议你尽量按“我做了什么”来讲。

### 3.1 训练入口与产物管理

我做了训练入口封装和训练产物标准化，核心文件包括：

- `scripts/train_nanogpt.py`
- `src/nanollmops/train/nanogpt_runner.py`
- `scripts/package_nanogpt_model.py`
- `src/nanollmops/artifacts.py`

你可以这样讲：

- 我先把 `nanoGPT` 的训练脚本包了一层统一入口，避免后续平台层直接耦合原始训练脚本
- 然后把 checkpoint、tokenizer 元信息、manifest、conversion plan 固化成标准 artifact 目录
- 这样后面做转换、部署和回归验证时，就不是随手拿一个 `ckpt.pt`，而是有结构化产物可追踪

面试价值：

- 体现你有工程化意识
- 体现你知道训练产物管理是训推衔接的第一步

### 3.2 checkpoint 到推理产物转换

我做了从 `nanoGPT ckpt.pt` 到推理侧 `converted/` 目录的转换与校验，核心文件包括：

- `scripts/convert_nanogpt_to_nanovllm.py`
- `src/nanollmops/converter/convert.py`
- `src/nanollmops/converter/planner.py`
- `src/nanollmops/converter/validate.py`
- `scripts/validate_converted_nanogpt.py`

这里你要重点讲清楚 4 个问题：

1. 配置对齐

- 把训练侧 `model_args` 转成推理侧可消费的 `config.json`
- 保留 `vocab_size`、`block_size`、`n_layer`、`n_head`、`n_embd` 等结构信息

2. tokenizer 对齐

- `nanoGPT` 的 `shakespeare_char` 使用字符级 tokenizer
- 推理侧如果不用同一套 `stoi/itos` 映射，就会出现 token id 不一致，导致结果异常
- 所以我把 `meta.pkl` 中的信息同步固化到转换产物中

3. 权重映射与 QKV 拆分

- `nanoGPT` 里 `attn.c_attn` 是 fused QKV
- 推理侧更适合拆成 `q_proj / k_proj / v_proj`
- 转换时要处理 key mapping，也要处理 tensor shape 的正确性

4. tied weights 与 safetensors 写盘问题

- `lm_head.weight` 和 embedding weight 在训练侧可能共享底层存储
- `safetensors` 不接受共享存储 tensor
- 所以我在转换写盘前对输出 tensor 做了独立存储归一化

面试价值：

- 这是非常典型的“训练框架到推理框架适配”问题
- 面试官一般会顺着问你：最难点是什么、你怎么验证 shape、你怎么保证 tokenizer 一致

### 3.3 最小推理链路与服务部署

我实现了两条推理路径。

第一条是最小直接推理路径，核心文件：

- `src/nanollmops/serving/nanogpt.py`
- `scripts/infer_nanogpt.py`
- `scripts/serve_nanogpt.py`

这条链路的特点：

- 直接加载 `nanoGPT` checkpoint
- 直接做字符级 encode / decode
- 提供离线生成和简单服务部署能力
- 输出 `latency_ms` 和 `tokens_per_second`

第二条是 `vLLM-style runtime` 路径，核心文件：

- `src/nanollmops/runtime/sequence.py`
- `src/nanollmops/runtime/block_manager.py`
- `src/nanollmops/runtime/scheduler.py`
- `src/nanollmops/runtime/nanogpt_vllm.py`
- `scripts/infer_nanogpt_vllm.py`
- `scripts/serve_nanogpt_vllm.py`

这条链路的特点：

- 不再只是直接调用 `model.generate()`
- 开始引入 `Sequence` 抽象
- 引入 `BlockManager` 做 KV Cache block 管理
- 引入 `Scheduler` 做请求调度
- 明确区分 `Prefill` 和 `Decode`

面试里建议强调：

- 第一条链路证明我能把模型闭环先跑通
- 第二条链路证明我开始真正进入推理框架的运行时设计

## 4. 结合岗位重点，你要突出哪些考点

这个岗位的关键词是“大模型推理框架优化与 Agentic Programming”，建议你把面试重点主动收敛到下面两组能力。

### 4.1 推理框架优化

你可以重点讲 6 个方向。

#### 1. Prefill / Decode 分离

你可以这样答：

- 推理场景里，prompt 首轮计算和后续逐 token decode 的计算模式完全不同
- Prefill 关注大段上下文一次性编码
- Decode 关注复用 KV Cache、降低单 token 迭代成本
- 所以在 runtime 层把两者拆开，是做调度、批处理和延迟优化的前提

项目映射：

- `src/nanollmops/runtime/nanogpt_vllm.py` 中已经显式拆成 `_run_prefill` 和 `_run_decode`

#### 2. KV Cache 与 block 化管理

你可以这样答：

- 大模型推理瓶颈之一是显存与 cache 管理
- 如果把请求上下文按 block 管理，就更容易做复用、回收和调度
- 这也是 vLLM 一类系统为什么强调 block table / paged cache 思路

项目映射：

- `BlockManager` 维护 `free_block_ids`、`used_block_ids`、`hash_to_block_id`
- 已经有 prefix hash 和 block 复用的雏形

#### 3. 调度器设计

你可以这样答：

- 推理框架的调度器不是普通任务队列
- 它需要同时考虑 `max_num_seqs`、`max_num_batched_tokens`、KV Cache 是否还能分配、是否要 preempt
- 本质是在吞吐、时延和显存之间做平衡

项目映射：

- `Scheduler` 里已经有 waiting/running 双队列
- 已实现 `schedule`、`preempt`、`postprocess`
- 已经体现出 runtime 调度思维，而不是单请求串行执行

#### 4. continuous batching 的理解

你可以这样答：

- continuous batching 的核心不是“多个请求拼一起”
- 而是让不同阶段的请求在运行时动态进出 batch，提高设备利用率
- 前提是请求状态、cache 状态和调度状态必须显式建模

这里要诚实：

- 我目前项目里实现的是轻量版、可解释版 runtime
- 已经具备 request/sequence/block/scheduler 这些核心抽象
- 但离工业级 continuous batching 还有距离，比如没有完整的异步执行、没有真实 GPU kernel 层优化、也没有复杂公平性策略

这样的回答通常比硬说“我完全实现了 vLLM”更稳。

#### 5. 性能指标体系

你可以主动提指标，不要等面试官问：

- TTFT
- TPOT
- 吞吐
- P95 / P99 latency
- 显存占用
- batch 利用率

项目映射：

- 当前项目已经输出 `latency_ms` 和 `tokens_per_second`
- 文档中已经把 TTFT、TPOT、吞吐和显存采集列为下一阶段重点

#### 6. 模型加载与结构适配

很多推理岗位也会看这一点。

你可以这样答：

- 推理优化不只是 runtime，还包括模型能否稳定加载
- 如果训练侧结构和推理侧结构不一致，后面所有调度和缓存优化都无从谈起
- 所以我先做了 config 对齐、权重 key mapping、QKV 拆分和转换后校验

### 4.2 Agentic Programming

这个部分你不建议讲得太空，最好从“面向推理平台的 Agent 化”切入。

#### 1. 你对 Agentic Programming 的理解

建议表述：

Agentic Programming 不是简单把大模型接几个工具，而是把任务规划、状态管理、工具调用、结果校验、失败恢复和可观测性组织成稳定闭环，让系统具备“能执行、能反馈、能重试、能演进”的能力。

#### 2. 为什么这个项目和 Agent 很相关

你可以这样讲：

- NanoLLMOps 本身就在管理多阶段链路：训练、打包、转换、验证、部署、评估
- 这些步骤天然适合抽象成 agent workflow
- 每一步都需要输入规范、状态记录、执行结果和失败回溯

这跟纯聊天机器人完全不是一个层次。

#### 3. 你项目里已经具备的 Agent 化基础

- 训练、转换、验证、推理、服务都有稳定脚本入口
- artifact 是结构化的
- 文档里持续维护阶段状态、验证结果和后续任务
- 这些都非常适合继续升级成 task planner + tool runner + state store 的 Agent 执行框架

#### 4. 如果面试官问“你会怎么把它做成 Agent 系统”

建议你按这条回答：

1. 把训练、转换、校验、部署、benchmark 抽象成工具
2. 设计任务状态机，至少包含 pending、running、succeeded、failed
3. 给每一步定义输入 schema、输出 schema 和失败重试策略
4. 把 artifact、日志、指标作为共享上下文
5. 增加 planner，让 Agent 能根据目标自动决定是先训练、先转换还是先验证
6. 最后再接入评估反馈，让 Agent 能根据 TTFT、TPOT、吞吐结果继续调参或切换部署参数

这个回答会让面试官觉得你对 Agentic Programming 的理解偏系统设计，而不是偏概念包装。

## 5. 建议你主动讲的难点与取舍

这一段很加分，因为它体现你会做技术判断。

### 5.1 我为什么没有一开始就硬接完整 nano-vllm

建议说法：

- 因为第一阶段最大的风险不是接口层，而是模型结构适配与闭环可验证性
- 如果训练产物、tokenizer、权重转换都还不稳定，直接上完整 runtime 会让问题定位非常困难
- 所以我先做最小闭环，再做 vLLM-style runtime，把复杂度逐步收敛

### 5.2 我为什么要先做 char-level 小模型

建议说法：

- 这是为了先把训推闭环和运行时机制验证清楚
- 小模型更容易做结构对齐、转换验证和调试
- 对于推理框架学习阶段，先验证调度与 cache 抽象，比一开始追求大模型规模更重要

### 5.3 当前项目的边界在哪里

建议你主动承认：

- 现在还是轻量实验平台，不是工业级推理引擎
- `vLLM-style runtime` 体现的是核心思想和可解释实现，不是完整复刻
- 真正的下一步是把 converted 产物稳定接到更原生的推理后端，并补齐 benchmark、监控和更真实的调度策略

## 6. 面试高频问答模板

### Q1：你这个项目最有技术含量的部分是什么？

建议答法：

我认为最有技术含量的是训练侧到推理侧的适配层，以及基于这个适配层进一步抽象出的 `vLLM-style runtime`。前者解决的是 config、tokenizer、权重结构、QKV 拆分和 safetensors 写盘问题，后者解决的是 Sequence、KV Cache block、Scheduler、Prefill/Decode 这些推理框架核心抽象。

### Q2：你对推理框架优化的理解是什么？

建议答法：

我理解推理框架优化主要分三层：第一层是模型结构和加载路径的稳定性，第二层是 runtime 里的 Prefill/Decode、KV Cache、continuous batching 和调度策略，第三层是服务级指标体系，比如 TTFT、TPOT、吞吐和尾延迟。我的项目目前已经把第一层做实，并进入第二层的轻量实现。

### Q3：如果让你继续优化，你下一步做什么？

建议答法：

我会先补齐标准 benchmark 与观测指标，再推进 converted 产物到更原生推理后端的接入，之后重点做三件事：一是更真实的 continuous batching，二是更细粒度的 KV Cache 回收与复用，三是围绕 TTFT / TPOT 做参数和调度策略对比实验。

### Q4：你理解的 Agentic Programming 和工作流编排有什么区别？

建议答法：

我觉得工作流编排更偏静态 DAG，而 Agentic Programming 更强调动态规划、基于上下文决定下一步动作、失败恢复和反馈闭环。对 NanoLLMOps 来说，静态流程可以完成训练到部署，但如果要让系统根据验证结果自动决定是否重训、重转、调参或切换部署参数，就需要 Agent 化能力。

### Q5：你在项目里最注意避免什么风险？

建议答法：

我最注意避免过早堆复杂度。比如一开始没有直接把所有精力都放在 Web 平台或复杂调度上，而是先保证训练产物能被稳定转换和加载。因为这一步不稳，后面所有优化都没有可靠基线。

## 7. 简历或自我介绍可直接复用的表述

### 简历版

- 设计并实现基于 `nanoGPT` 与 `nano-vllm` 思路的轻量级训推一体平台 `NanoLLMOps`，打通训练入口、artifact 管理、checkpoint 转换、离线推理、服务部署与推理指标采集链路。
- 实现 `nanoGPT` checkpoint 到推理产物的转换与校验，完成 config 对齐、tokenizer 对齐、权重 key mapping、QKV 拆分及 `safetensors` 写盘问题处理。
- 实现轻量级 `vLLM-style runtime`，抽象 `Sequence`、`BlockManager`、`Scheduler` 与 `Prefill/Decode` 执行路径，验证 KV Cache block 化管理与推理调度流程。
- 面向后续 Agent 化平台演进，沉淀训练、转换、验证、部署等标准化工具入口与结构化 artifact，为任务编排、状态管理与反馈闭环打基础。

### 自我介绍版

我最近主要做的是围绕 `nanoGPT`、`nano-vllm` 和 `NanoLLMOps` 这条线的项目。我的核心目标不是单纯训练一个模型，而是把训练产物如何进入推理系统这条链路真正打通。我在项目里做了训练入口封装、artifact 标准化、checkpoint 到推理产物的转换与校验，还实现了一条轻量级的 `vLLM-style runtime`，把 `Prefill/Decode`、KV Cache block 管理和调度器这些推理框架核心概念落成了可运行代码。我对这个项目最看重的是，它让我把训练框架、推理框架和后续 Agent 化平台能力串到了同一个闭环里。

## 8. 明天面试前最后复习建议

建议至少把下面这些点讲顺：

1. 为什么要做 `nanoGPT -> NanoLLMOps -> vLLM-style runtime` 这条链
2. `ckpt.pt`、`meta.pkl`、`config.json`、`model.safetensors` 分别解决什么问题
3. 为什么 fused QKV 需要拆分
4. 什么是 `Prefill / Decode`
5. 为什么 KV Cache 要 block 化管理
6. 调度器需要平衡哪些资源约束
7. Agentic Programming 在这个项目里最自然的落地方向是什么

如果只能临场记住一句主线，就记这句：

我做的不是一个单点脚本，而是一条从训练产物到推理运行时再到后续 Agent 化编排的可验证闭环。
