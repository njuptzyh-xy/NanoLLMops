# Phase 1 总结：最小训推闭环

更新时间：2026-06-02

关联任务：`P1-001`、`P1-002`

## 本阶段目标

先用 `nanoGPT` 小模型跑通一条真实、可验证的最小链路：

```text
nanoGPT 训练 -> checkpoint/meta 打包 -> 权重转换 -> 产物校验
-> 本地推理 -> vLLM 风格运行 demo -> HTTP 服务
```

本阶段先解决“链路是否能跑通”。正式接入 `nano-vllm` 推理后端属于下一阶段任务。

## 当前结论

`Phase 1` 已完成。仓库已经具备：

- 统一训练入口
- 训练产物打包
- 原始 checkpoint 本地推理
- HTTP 服务入口
- 真实 `ckpt.pt -> converted/model.safetensors` 转换
- 转换产物完整性校验
- converted 目录直接加载推理
- converted 目录接入 vLLM 风格 runtime
- CPU 可运行的 vLLM 风格请求调度与 KV Cache demo

当前进入 `Phase 1 -> Phase 2` 过渡阶段。下一步重点是调研并接入正式 `nano-vllm` 原生推理路径。

## 核心解决思路

1. 先选用 `shakespeare_char` 小模型，降低训练和推理验证成本。
2. 将原始 `ckpt.pt`、`meta.pkl` 和转换结果放入统一 artifact 目录。
3. 转换后先检查权重名称和形状，再交给推理后端加载。
4. 先在本仓库实现简化版 vLLM 风格运行链路，验证调度和 KV Cache 管理逻辑。
5. 等模型产物加载稳定后，再继续接入正式 `nano-vllm` 后端。

## 核心代码位置

| 能力 | 核心代码或脚本 |
| --- | --- |
| 训练入口 | `src/nanollmops/train/nanogpt_runner.py`、`scripts/train_nanogpt.py` |
| 产物目录和状态描述 | `src/nanollmops/artifacts.py` |
| 产物打包 | `scripts/package_nanogpt_model.py` |
| 转换规划 | `src/nanollmops/converter/planner.py`、`scripts/plan_nanogpt_conversion.py` |
| 真实权重转换 | `src/nanollmops/converter/convert.py`、`scripts/convert_nanogpt_to_nanovllm.py` |
| 转换后校验 | `src/nanollmops/converter/validate.py`、`scripts/validate_converted_nanogpt.py` |
| 原始 checkpoint 推理 | `src/nanollmops/serving/nanogpt.py`、`scripts/infer_nanogpt.py` |
| converted 目录推理 | `src/nanollmops/serving/nanogpt.py`、`scripts/infer_converted_nanogpt.py` |
| converted 权重共享加载 | `src/nanollmops/converter/load.py` |
| HTTP 服务 | `src/nanollmops/serving/api.py`、`scripts/serve_nanogpt.py` |
| vLLM 风格运行链路 | `src/nanollmops/runtime/`、`scripts/infer_nanogpt_vllm.py`、`scripts/infer_converted_nanogpt_vllm.py` |

## 关键修复

真实转换时发现 `lm_head.weight` 和 `embed_tokens.weight` 共享底层存储，`safetensors` 不能直接保存。

处理方式：在写盘前复制转换后的 tensor，使每个输出权重拥有独立存储。实现位置：`src/nanollmops/converter/convert.py`。

## 测试脚本

| 测试脚本 | 覆盖内容 | 结果 |
| --- | --- | --- |
| `tests/test_conversion_planner.py` | checkpoint 解析、tokenizer 识别、转换计划输出 | 已通过 |
| `tests/test_converter.py` | 权重转换、配置文件输出、共享权重处理 | 已通过 |
| `tests/test_validate_converter.py` | 转换后权重名称和形状校验、错误提示 | 已通过 |
| `tests/test_converted_inference.py` | converted 配置、tokenizer、safetensors 加载与普通推理、vLLM 风格推理 | 已通过 |
| `tests/test_runtime.py` | Sequence、prefix cache、decode block 扩展、scheduler 资源释放 | 已通过 |

## 真实验证结果

- 使用 `/home/zyh-ub/PyRepos/nanoGPT/out-shakespeare-char/ckpt.pt` 完成真实转换
- 已生成 `artifacts/shakespeare-char-v1/converted/model.safetensors`
- 转换后校验结果：`tensor_count = 36`、`checked_keys = 36`
- 原始 checkpoint 本地推理成功
- converted 目录直接推理成功，真实烟测输出 `"To be gon"`
- converted 目录进入 vLLM 风格 runtime 成功，真实烟测输出 `"To be you"`
- vLLM 风格离线推理成功
- vLLM 风格服务 `GET /health` 和 `POST /generate` 调用成功

## 当前未完成

- 正式接入 `nano-vllm` 原生推理路径

## 下一阶段目标

1. 调研并接入正式 `nano-vllm` 原生推理路径。
2. 明确 NanoGPT 模型适配与原生后端之间的边界。
3. 保留当前 CPU demo 作为回归基线。
