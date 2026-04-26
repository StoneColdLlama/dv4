"""
dv4/model/dv4_transformer.py

DV4Transformer — Qwen-style transformer using DV4Linear layers.

Configs:
  tiny_config:  ~41M params  — debug/PoC
  poc_config:   ~120M params — small experiments
  medium_config: ~500M params — workstation PoC run (4 topics, paper result)
  large_config:  ~1B params  — full paper run

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .dv4_linear import DV4Linear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_dim = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class DV4Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = DV4Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = DV4Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = DV4Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = DV4Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        causal = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal)

    def forward(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        q = apply_rope(q, rope_cos[:T], rope_sin[:T])
        k = apply_rope(k, rope_cos[:T], rope_sin[:T])
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = attn + self.causal_mask[:T, :T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        return self.o_proj(out)


class DV4FFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = DV4Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj   = DV4Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = DV4Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DV4Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, max_seq_len, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = DV4Attention(hidden_dim, num_heads, max_seq_len, dropout)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = DV4FFN(hidden_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.dropout(self.attn(self.attn_norm(x), rope_cos, rope_sin))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DV4Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151665,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            DV4Block(hidden_dim, num_heads, ffn_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        head_dim = hidden_dim // num_heads
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_seq_len, rope_base)
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        print(f"DV4Transformer: {total/1e6:.1f}M total parameters")
        print(f"  hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads}, ffn={ffn_dim}")

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, DV4Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def count_dv4_layers(self) -> int:
        return sum(1 for m in self.modules() if isinstance(m, DV4Linear))

    @classmethod
    def tiny_config(cls, **kwargs) -> 'DV4Transformer':
        """~41M params — debug"""
        config = dict(vocab_size=151665, hidden_dim=256, num_layers=4,
                      num_heads=4, ffn_dim=512, max_seq_len=256, dropout=0.1)
        config.update(kwargs)
        return cls(**config)

    @classmethod
    def poc_config(cls, **kwargs) -> 'DV4Transformer':
        """~120M params — small experiments"""
        config = dict(vocab_size=151665, hidden_dim=768, num_layers=12,
                      num_heads=12, ffn_dim=2048, max_seq_len=512, dropout=0.1)
        config.update(kwargs)
        return cls(**config)

    @classmethod
    def medium_config(cls, **kwargs) -> 'DV4Transformer':
        """~500M params — workstation PoC, 4 topics, paper result target"""
        config = dict(vocab_size=151665, hidden_dim=1024, num_layers=24,
                      num_heads=16, ffn_dim=4096, max_seq_len=1024, dropout=0.1)
        config.update(kwargs)
        return cls(**config)

    @classmethod
    def large_config(cls, **kwargs) -> 'DV4Transformer':
        """~1B params — full paper run"""
        config = dict(vocab_size=151665, hidden_dim=2048, num_layers=24,
                      num_heads=16, ffn_dim=5632, max_seq_len=2048, dropout=0.05)
        config.update(kwargs)
        return cls(**config)
