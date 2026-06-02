from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from torch.nn import functional as F

from nanollmops.runtime.config import RuntimeConfig
from nanollmops.runtime.sampling_params import SamplingParams
from nanollmops.runtime.scheduler import Scheduler
from nanollmops.runtime.sequence import Sequence


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]) -> None:
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def from_meta(cls, meta_path: str) -> "CharTokenizer":
        with open(meta_path, "rb") as handle:
            meta = pickle.load(handle)
        return cls(meta["stoi"], meta["itos"])

    @classmethod
    def from_tokenizer_json(cls, tokenizer_path: str) -> "CharTokenizer":
        with open(tokenizer_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        stoi = payload["model"]["vocab"]
        return cls(stoi, {token_id: token for token, token_id in stoi.items()})

    def encode(self, text: str) -> list[int]:
        unknown = [char for char in text if char not in self.stoi]
        if unknown:
            joined = "".join(sorted(set(unknown)))
            raise ValueError(f"Prompt contains tokens outside tokenizer vocabulary: {joined!r}")
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(tensor, self.weight.shape, self.weight, self.bias, 1e-5)


class CachedSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.cache_block_size = config.kvcache_block_size

    def _slot(self, seq: Sequence, position: int) -> tuple[int, int]:
        block_index = position // self.cache_block_size
        block_id = seq.block_table[block_index]
        offset = position % self.cache_block_size
        return block_id, offset

    def _store(self, cache_k: torch.Tensor, cache_v: torch.Tensor, seq: Sequence, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        block_id, offset = self._slot(seq, position)
        cache_k[block_id, offset] = k
        cache_v[block_id, offset] = v

    def _load_history(self, cache_k: torch.Tensor, cache_v: torch.Tensor, seq: Sequence, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        remaining = length
        for block_id in seq.block_table:
            take = min(self.cache_block_size, remaining)
            if take <= 0:
                break
            keys.append(cache_k[block_id, :take])
            values.append(cache_v[block_id, :take])
            remaining -= take
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def forward_token(
        self,
        hidden: torch.Tensor,
        seq: Sequence,
        position: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.c_attn(hidden)
        q, k, v = qkv.split(hidden.size(-1), dim=-1)
        q = q.view(self.n_head, self.head_dim)
        k = k.view(self.n_head, self.head_dim)
        v = v.view(self.n_head, self.head_dim)
        self._store(cache_k, cache_v, seq, position, k, v)
        keys, values = self._load_history(cache_k, cache_v, seq, position + 1)
        scores = (keys * q.unsqueeze(0)).sum(dim=-1).transpose(0, 1) * self.scale
        probs = F.softmax(scores, dim=-1)
        context = torch.einsum("ht,thd->hd", probs, values)
        context = context.reshape(1, -1)
        return self.c_proj(context)


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(tensor)))


class CachedBlock(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CachedSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward_token(
        self,
        hidden: torch.Tensor,
        seq: Sequence,
        position: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attn.forward_token(self.ln_1(hidden), seq, position, cache_k, cache_v)
        hidden = hidden + self.mlp(self.ln_2(hidden))
        return hidden


@dataclass(slots=True)
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False
    kvcache_block_size: int = 16


class CachedGPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([CachedBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward_token(
        self,
        token_id: int,
        position: int,
        seq: Sequence,
        layer_k_caches: list[torch.Tensor],
        layer_v_caches: list[torch.Tensor],
        device: str,
    ) -> torch.Tensor:
        idx = torch.tensor([token_id], dtype=torch.long, device=device)
        pos = torch.tensor([position], dtype=torch.long, device=device)
        hidden = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for layer_id, block in enumerate(self.transformer.h):
            hidden = block.forward_token(hidden, seq, position, layer_k_caches[layer_id], layer_v_caches[layer_id])
        hidden = self.transformer.ln_f(hidden)
        return self.lm_head(hidden)


class VLLMStyleNanoGPT:
    def __init__(
        self,
        checkpoint_path: str,
        meta_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.meta_path = meta_path
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.runtime_config = runtime_config or RuntimeConfig()

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_args = checkpoint["model_args"]
        model_args["kvcache_block_size"] = self.runtime_config.kvcache_block_size
        self.model_config = GPTConfig(**model_args)
        self.model = CachedGPTModel(self.model_config).to(device=device, dtype=self.dtype)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for key, _ in list(state_dict.items()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.tokenizer = CharTokenizer.from_meta(meta_path)
        self._initialize_runtime_state()
        checkpoint_file = Path(checkpoint_path)
        self.model_name = checkpoint_file.parent.parent.name if checkpoint_file.parent.name == "source" else checkpoint_file.parent.name

    @classmethod
    def from_converted(
        cls,
        model_dir: str,
        device: str = "cpu",
        dtype: str = "float32",
        runtime_config: RuntimeConfig | None = None,
    ) -> "VLLMStyleNanoGPT":
        from nanollmops.converter.load import load_converted_nanogpt_state_dict

        config, state_dict = load_converted_nanogpt_state_dict(model_dir, device=device)
        engine = cls.__new__(cls)
        engine.checkpoint_path = None
        engine.meta_path = None
        engine.device = device
        engine.dtype = getattr(torch, dtype)
        engine.runtime_config = runtime_config or RuntimeConfig()
        engine.model_config = GPTConfig(
            block_size=int(config["n_positions"]),
            vocab_size=int(config["vocab_size"]),
            n_layer=int(config["n_layer"]),
            n_head=int(config["n_head"]),
            n_embd=int(config["n_embd"]),
            dropout=float(config.get("resid_pdrop", 0.0)),
            bias=bool(config.get("bias", False)),
            kvcache_block_size=engine.runtime_config.kvcache_block_size,
        )
        engine.model = CachedGPTModel(engine.model_config).to(device=device, dtype=engine.dtype)
        engine.model.load_state_dict(state_dict)
        engine.model.eval()
        engine.tokenizer = CharTokenizer.from_tokenizer_json(str(Path(model_dir) / "tokenizer.json"))
        engine._initialize_runtime_state()
        engine.model_name = Path(model_dir).resolve().parent.name
        return engine

    def _initialize_runtime_state(self) -> None:
        self.scheduler = Scheduler(self.runtime_config)
        self.layer_k_caches = [
            torch.zeros(
                self.runtime_config.num_kvcache_blocks,
                self.runtime_config.kvcache_block_size,
                self.model_config.n_head,
                self.model_config.n_embd // self.model_config.n_head,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.model_config.n_layer)
        ]
        self.layer_v_caches = [
            torch.zeros_like(self.layer_k_caches[layer_id]) for layer_id in range(self.model_config.n_layer)
        ]

    def add_request(self, prompt: str, sampling_params: SamplingParams) -> Sequence:
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > self.runtime_config.max_model_len:
            raise ValueError("prompt exceeds max_model_len")
        seq = Sequence(token_ids=token_ids, sampling_params=sampling_params, block_size=self.runtime_config.kvcache_block_size)
        self.scheduler.add(seq)
        return seq

    @torch.inference_mode()
    def _run_prefill(self, seq: Sequence) -> torch.Tensor:
        logits = None
        for position in range(seq.num_cached_tokens, len(seq)):
            logits = self.model.forward_token(
                token_id=seq[position],
                position=position,
                seq=seq,
                layer_k_caches=self.layer_k_caches,
                layer_v_caches=self.layer_v_caches,
                device=self.device,
            )
        seq.num_cached_tokens = len(seq)
        if logits is None:
            raise RuntimeError("prefill produced no logits")
        return logits

    @torch.inference_mode()
    def _run_decode(self, seq: Sequence) -> torch.Tensor:
        position = len(seq) - 1
        logits = self.model.forward_token(
            token_id=seq.last_token,
            position=position,
            seq=seq,
            layer_k_caches=self.layer_k_caches,
            layer_v_caches=self.layer_v_caches,
            device=self.device,
        )
        seq.num_cached_tokens = len(seq)
        return logits

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float) -> int:
        scaled = logits[0] / temperature
        probs = F.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def step(self) -> list[tuple[int, list[int]]]:
        seqs, is_prefill = self.scheduler.schedule()
        next_token_ids = []
        for seq in seqs:
            logits = self._run_prefill(seq) if is_prefill else self._run_decode(seq)
            next_token_ids.append(self.sample(logits, seq.temperature))
        self.scheduler.postprocess(seqs, next_token_ids)
        return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

    def generate(self, prompt: str, sampling_params: SamplingParams) -> dict[str, object]:
        start = perf_counter()
        seq = self.add_request(prompt, sampling_params)
        outputs: dict[int, list[int]] = {}
        while not self.scheduler.is_finished():
            for seq_id, token_ids in self.step():
                outputs[seq_id] = token_ids
        end = perf_counter()
        completion_ids = outputs[seq.seq_id]
        output_ids = seq.token_ids
        decoded = self.tokenizer.decode(output_ids)
        latency_ms = (end - start) * 1000
        tps = len(completion_ids) / (end - start) if end > start else 0.0
        return {
            "prompt": prompt,
            "output": decoded,
            "input_tokens": len(seq.token_ids) - len(completion_ids),
            "output_tokens": len(completion_ids),
            "latency_ms": round(latency_ms, 3),
            "tokens_per_second": round(tps, 3),
            "model": self.model_name,
            "engine": "vllm-style-gpt",
            "kvcache_block_size": self.runtime_config.kvcache_block_size,
            "num_kvcache_blocks": self.runtime_config.num_kvcache_blocks,
        }
