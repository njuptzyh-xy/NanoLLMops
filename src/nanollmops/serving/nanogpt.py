from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(tensor, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = tensor.size()
        q, k, v = self.c_attn(tensor).split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.c_fc(tensor)
        tensor = self.gelu(tensor)
        tensor = self.c_proj(tensor)
        return self.dropout(tensor)


class Block(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor + self.attn(self.ln_1(tensor))
        tensor = tensor + self.mlp(self.ln_2(tensor))
        return tensor


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, Any]:
        device = idx.device
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}, block size is {self.config.block_size}"
            )
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        tensor = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            tensor = block(tensor)
        tensor = self.transformer.ln_f(tensor)
        if targets is not None:
            logits = self.lm_head(tensor)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(tensor[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(default=200, ge=1)


class GenerationResponse(BaseModel):
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tokens_per_second: float
    model: str


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


class NanoGPTDeployment:
    def __init__(
        self,
        checkpoint_path: str,
        meta_path: str,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.meta_path = meta_path
        self.device = device
        self.dtype = getattr(torch, dtype)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_args = checkpoint["model_args"]
        checkpoint_file = Path(checkpoint_path)
        self.model_name = checkpoint_file.parent.parent.name if checkpoint_file.parent.name == "source" else checkpoint_file.parent.name
        self.model_config = GPTConfig(**model_args)
        self.model = GPT(self.model_config)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for key, _ in list(state_dict.items()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device=device, dtype=self.dtype)
        self.tokenizer = CharTokenizer.from_meta(meta_path)

    @classmethod
    def from_converted(
        cls,
        model_dir: str,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> "NanoGPTDeployment":
        from nanollmops.converter.load import load_converted_nanogpt_state_dict

        config, state_dict = load_converted_nanogpt_state_dict(model_dir, device=device)
        deployment = cls.__new__(cls)
        deployment.checkpoint_path = None
        deployment.meta_path = None
        deployment.device = device
        deployment.dtype = getattr(torch, dtype)
        deployment.model_name = Path(model_dir).resolve().parent.name
        deployment.model_config = GPTConfig(
            block_size=int(config["n_positions"]),
            vocab_size=int(config["vocab_size"]),
            n_layer=int(config["n_layer"]),
            n_head=int(config["n_head"]),
            n_embd=int(config["n_embd"]),
            dropout=float(config.get("resid_pdrop", 0.0)),
            bias=bool(config.get("bias", False)),
        )
        deployment.model = GPT(deployment.model_config)
        deployment.model.load_state_dict(state_dict)
        deployment.model.eval().to(device=device, dtype=deployment.dtype)
        deployment.tokenizer = CharTokenizer.from_tokenizer_json(
            str(Path(model_dir) / "tokenizer.json")
        )
        return deployment

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = perf_counter()
        input_ids = self.tokenizer.encode(request.prompt)
        tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)[None, ...]
        output = self.model.generate(
            tensor,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )[0].tolist()
        end = perf_counter()
        decoded = self.tokenizer.decode(output)
        generated = output[len(input_ids) :]
        latency_ms = (end - start) * 1000
        tps = len(generated) / (end - start) if end > start else 0.0
        return GenerationResponse(
            prompt=request.prompt,
            output=decoded,
            input_tokens=len(input_ids),
            output_tokens=len(generated),
            latency_ms=round(latency_ms, 3),
            tokens_per_second=round(tps, 3),
            model=self.model_name,
        )
