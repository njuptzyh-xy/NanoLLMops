# MVP 端到端运行说明

## 目标

先跑通以下最小链路：

```text
nanoGPT 训练 -> checkpoint/meta 产物 -> NanoLLMOps artifact 打包 -> 本地推理/服务部署
```

当前这条链路先使用本仓库的最小 NanoGPT 推理服务跑通，后续再接入 `nano-vllm`。

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

## 当前范围

已打通：

- 训练入口封装
- checkpoint/meta 打包
- 本地推理入口
- FastAPI 服务入口

暂未打通：

- `nanoGPT ckpt.pt -> safetensors`
- HF-compatible `config.json`
- `nano-vllm` 原生加载与调度

这些是下一阶段的重点。
