# MVP 端到端运行说明

## 目标

先跑通以下最小链路：

```text
nanoGPT 训练 -> checkpoint/meta 产物 -> NanoLLMOps artifact 打包 -> 本地推理/服务部署
```

当前这条链路先使用本仓库的最小 NanoGPT 推理服务跑通，已经支持原始 checkpoint 和 `converted/` 目录两种加载方式，后续再接入 `nano-vllm`。

## 1. 训练

```bash
conda run -n nanollmops python scripts/train_nanogpt.py --config /home/zyh-ub/PyRepos/NanoLLMOps/configs/train_shakespeare_char.py -- --device=cpu --compile=False
```

说明：

- 底层实际调用的是 `/home/zyh-ub/PyRepos/nanoGPT/train.py`
- `--device=cpu --compile=False` 只是示例；如果本机 GPU 环境可用，可以改成 `--device=cuda`

## 2. 打包产物

```bash
conda run -n nanollmops python scripts/package_nanogpt_model.py \
  --checkpoint /home/zyh-ub/PyRepos/nanoGPT/out-shakespeare-char/ckpt.pt \
  --meta /home/zyh-ub/PyRepos/nanoGPT/data/shakespeare_char/meta.pkl \
  --model-name shakespeare-char \
  --version v1
```

输出目录示例：

```text
artifacts/shakespeare-char-v1/
├── artifact_manifest.json
├── conversion_plan.json
├── conversion_plan.md
└── source/
    ├── ckpt.pt
    └── meta.pkl
```

## 3. 本地推理

```bash
conda run -n nanollmops python scripts/infer_nanogpt.py \
  --checkpoint artifacts/shakespeare-char-v1/source/ckpt.pt \
  --meta artifacts/shakespeare-char-v1/source/meta.pkl \
  --prompt "To be or not to be" \
  --max-new-tokens 64
```

## 4. 启动服务

```bash
conda run -n nanollmops python scripts/serve_nanogpt.py \
  --checkpoint artifacts/shakespeare-char-v1/source/ckpt.pt \
  --meta artifacts/shakespeare-char-v1/source/meta.pkl \
  --host 127.0.0.1 \
  --port 8000
```

如果当前 `nanollmops` 环境里还没有 `torch`，可以先用标准库 HTTP 版本做 demo：

```bash
/home/zyh-ub/miniconda3/envs/nanoGPT/bin/python scripts/serve_nanogpt_simple.py \
  --checkpoint artifacts/shakespeare-char-v1/source/ckpt.pt \
  --meta artifacts/shakespeare-char-v1/source/meta.pkl \
  --host 127.0.0.1 \
  --port 8012 \
  --device cpu \
  --dtype float32
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "To be or not to be",
    "max_new_tokens": 64,
    "temperature": 0.8,
    "top_k": 200
  }'
```

标准库 HTTP demo 的真实烟测结果：

```text
GET  /health   -> 200 {"status": "ok", "model": "shakespeare-char-v1"}
POST /generate -> 200
```

示例返回：

```json
{
  "prompt": "To be or not to be",
  "output": "To be or not to bettlers thing so plow.\n\nD",
  "input_tokens": 18,
  "output_tokens": 24,
  "latency_ms": 67.855,
  "tokens_per_second": 353.695,
  "model": "shakespeare-char-v1"
}
```

## 5. 从 converted 目录直接推理

转换并校验后，可以不再读取 `source/ckpt.pt` 和 `source/meta.pkl`，直接加载 `converted/`：

```bash
conda run -n nanollmops python scripts/infer_converted_nanogpt.py \
  --model-dir artifacts/shakespeare-char-v1/converted \
  --prompt "To be" \
  --max-new-tokens 4 \
  --device cpu \
  --dtype float32
```

加载过程会先校验 `config.json` 和 `model.safetensors`，再读取 `tokenizer.json` 并重组 NanoGPT fused QKV 权重。

真实烟测结果：

```text
input_tokens  -> 5
output_tokens -> 4
output        -> "To be gon"
model         -> "shakespeare-char-v1"
```

## 当前范围

已打通：

- 训练入口封装
- checkpoint/meta 打包
- converted 目录直接加载推理
- 本地推理入口
- FastAPI 服务入口
- vLLM-style runtime：
  - `Sequence`
  - `BlockManager`
  - `Scheduler`
  - `KV Cache`
  - `Prefill / Decode`

暂未打通：

- converted 目录接入 vLLM-style runtime
- `nano-vllm` 原生 `flash-attn/triton` 路径

补充：

- 仓库现在已经有 `scripts/convert_nanogpt_to_nanovllm.py`
- 该脚本会稳定写出 `converted/config.json`、`generation_config.json`、`tokenizer.json`
- 若当前环境安装了 `torch + safetensors`，还会继续输出 `converted/model.safetensors`
- 若环境依赖不完整，可先用 `--metadata-only` 生成转换骨架
- 当前 `nanollmops` 环境已经补齐：
  - `torch`
  - `safetensors`
  - `numpy`
- 已完成真实 `ckpt.pt -> model.safetensors` 烟测
- 已新增 `scripts/validate_converted_nanogpt.py`，用于校验转换产物

转换命令：

```bash
conda run -n nanollmops python scripts/convert_nanogpt_to_nanovllm.py \
  --checkpoint /home/zyh-ub/PyRepos/nanoGPT/out-shakespeare-char/ckpt.pt \
  --meta /home/zyh-ub/PyRepos/nanoGPT/data/shakespeare_char/meta.pkl \
  --model-name shakespeare-char \
  --version v1
```

转换后校验：

```bash
conda run -n nanollmops python scripts/validate_converted_nanogpt.py \
  --model-dir artifacts/shakespeare-char-v1/converted
```

当前真实校验结果：

```text
validated -> artifacts/shakespeare-char-v1/converted
tensor_count -> 36
checked_keys -> 36
```

## 6. vLLM-style demo

离线运行：

```bash
conda run -n nanoGPT python scripts/infer_nanogpt_vllm.py \
  --checkpoint artifacts/shakespeare-char-v1/source/ckpt.pt \
  --meta artifacts/shakespeare-char-v1/source/meta.pkl \
  --prompt "To be or not to be" \
  --max-tokens 24 \
  --device cpu \
  --dtype float32 \
  --max-model-len 64 \
  --kvcache-block-size 16 \
  --num-kvcache-blocks 64
```

启动服务：

```bash
/home/zyh-ub/miniconda3/envs/nanoGPT/bin/python scripts/serve_nanogpt_vllm.py \
  --checkpoint artifacts/shakespeare-char-v1/source/ckpt.pt \
  --meta artifacts/shakespeare-char-v1/source/meta.pkl \
  --host 127.0.0.1 \
  --port 8013 \
  --device cpu \
  --dtype float32 \
  --max-model-len 64 \
  --kvcache-block-size 16 \
  --num-kvcache-blocks 64
```

健康检查：

```bash
curl http://127.0.0.1:8013/health
```

生成请求：

```bash
curl -X POST http://127.0.0.1:8013/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "To be or not to be",
    "max_tokens": 24,
    "temperature": 0.8
  }'
```

这些仍然属于下一阶段的重点，但“真实转换”和“转换后自检”已经不再停留在计划阶段。
