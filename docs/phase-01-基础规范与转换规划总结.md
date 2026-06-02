# Phase 0/1 总结：基础规范与转换规划

更新时间：2026-05-18

## 本阶段目标

这一阶段没有直接修改推理内核，而是先完成训推闭环的前置基础设施：

- 明确 `nanoGPT` 训练产物在 NanoLLMOps 中的统一描述方式
- 明确转换产物目录结构
- 明确从 `ckpt.pt` 到推理模型目录的规划入口
- 明确第一版权重映射规则与当前开放问题

这是 Phase 1 的第一步。没有这层契约，后面的训练接入、权重转换和推理加载都会变成临时脚本堆叠。

## 本阶段处理的内容

### 1. 建立训练产物与状态模型

新增统一数据结构，用来描述：

- 模型状态：`TRAINED`、`CONVERTED`、`DEPLOYED` 等
- `nanoGPT` 模型配置：`block_size`、`vocab_size`、`n_layer`、`n_head`、`n_embd`
- tokenizer 规格
- 训练产物 manifest
- 权重转换规则
- 转换计划对象
- 转换产物目录布局

这些结构解决的是“平台内部怎么理解一个训练好的模型”这个问题。

### 2. 建立转换规划器

新增 `converter/planner.py`，能力包括：

- 解析 `meta.pkl`
- 识别当前 char-level tokenizer 形态
- 解析 `nanoGPT` checkpoint payload 的关键字段
- 生成标准化训练产物描述
- 生成第一版转换计划
- 产出 JSON manifest 和 Markdown 摘要

这一步还没有真正执行 tensor 级转换，但已经把“转换前检查”和“转换输出目标”固定下来了。

### 3. 明确第一版权重映射规则

本阶段把最关键的映射先文档化、结构化，包括：

- `wte` -> `embed_tokens`
- `wpe` -> `position_embeddings`
- `c_attn` -> `q_proj/k_proj/v_proj`
- `c_proj` -> `o_proj`
- `mlp.c_fc/c_proj` -> 推理侧 MLP 线性层
- block 内 `LayerNorm`
- `lm_head.weight`

这一步的价值不是“已经转好了”，而是把后面真正写转换器时最容易出错的地方先压成明确规则。

### 4. 新增可执行脚本

新增 `scripts/plan_nanogpt_conversion.py`：

- 输入：`ckpt.pt`、可选 `meta.pkl`、模型名、版本号
- 输出：
  - `artifact_manifest.json`
  - `conversion_plan.json`
  - `conversion_plan.md`

这使得后续每一个训练好的模型都可以先做一次标准化检查，再进入真正转换。

### 5. 增加基础测试

新增 `tests/test_conversion_planner.py`，覆盖：

- char-level tokenizer 识别
- checkpoint payload 解析
- conversion plan 生成
- plan bundle 写出

测试目前用合成 payload 跑，不依赖本机必须先装好 `torch`。

## 修改文件概览

新增代码：

- `src/nanollmops/artifacts.py`
- `src/nanollmops/converter/__init__.py`
- `src/nanollmops/converter/planner.py`
- `scripts/plan_nanogpt_conversion.py`
- `tests/test_conversion_planner.py`

新增文档：

- `docs/phase-01-基础规范与转换规划总结.md`

同步更新：

- `docs/开发规划.md`

## 当前结果

目前仓库已经具备：

- 统一描述训练产物的内部模型
- 可复用的转换规划入口
- 明确的第一版权重映射规则
- 可沉淀到仓库内的 manifest/plan 文档能力

换句话说，NanoLLMOps 已经从“只有产品文档”进入到“有工程契约和执行入口”的状态。

## 当前未完成的部分

本阶段刻意没有做以下内容：

- 真正把 `ckpt.pt` 转成 `safetensors`
- 真正实现推理侧 `NanoGPTForCausalLM`
- 真正把模型挂到 `nano-vllm` 运行

这是有意控制范围。当前先把接口和规划固定，再进入 tensor 转换和推理模型接入。

## 下一步建议

下一阶段应直接进入真正的 Phase 1 核心实现：

1. 设计推理侧 `NanoGPTForCausalLM` 结构
2. 根据当前 mapping rules 编写真正的 checkpoint 转换器
3. 输出 HF-compatible `config.json` 与 `model.safetensors`
4. 验证模型能被推理侧加载
