"""
 Core building blocks for the decoder‑only Transformer used in the
 `scratch_transformer` project.

 Everything here is **framework‑agnostic** except for the obvious PyTorch
 dependencies, so you can copy‑paste these classes into any model script or
 notebook.  Each module is deliberately minimal—no external libraries beyond
 `torch`—to make the math crystal‑clear and easy to unit‑test.
 
 Shapes follow the GPT convention: **(batch, seq_len, d_model)**.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "causal_mask",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "FeedForward",
    "PositionalEncoding",
    "DecoderBlock",
]

# -----------------------------------------------------------------------------
# helper: causal mask
# -----------------------------------------------------------------------------

def causal_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return an additive causal attention mask of shape (1, 1, L, L).

    * 0 on or below the main diagonal (tokens can see themselves and left).
    * -inf above the diagonal (masked positions).

    The mask is designed to be broadcast across (batch, num_heads) dims.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask.triu_(1)  # upper‑triangular (*strict* upper part)
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)


# -----------------------------------------------------------------------------
# 1. Scaled dot‑product attention (single head)
# -----------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, H, L, d_k)
        k: torch.Tensor,  # (B, H, L, d_k)
        v: torch.Tensor,  # (B, H, L, d_v)
        mask: Optional[torch.Tensor] = None,  # (B, 1/L/H, L, L)
    ) -> torch.Tensor:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,L,L)
        if mask is not None:
            scores = scores + mask  # additive mask: -inf where masked
        attn = F.softmax(scores, dim=-1)  # (B,H,L,L)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)  # (B,H,L,d_v)


# -----------------------------------------------------------------------------
# 2. Multi‑head attention
# -----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention with casual masking support.

    Inputs/outputs use (B, L, d_model).  Heads are handled internally.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Shape (B, L, D) -> (B, H, L, d_k)."""
        B, L, _ = x.size()
        x = x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        return x  # (B, H, L, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Shape (B, H, L, d_k) -> (B, L, D)."""
        B, H, L, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * d_k)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, D)
        mask: Optional[torch.Tensor] = None,  # (1 or B, 1 or H, L, L)
    ) -> torch.Tensor:
        B, L, _ = x.size()
        qkv = self.qkv_proj(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(self._split_heads, (q, k, v))  # each (B,H,L,d_k)

        if mask is not None and mask.dim() == 3:
            # (L,L) -> (1,1,L,L) broadcastable
            mask = mask.unsqueeze(0).unsqueeze(0)

        y = self.attn(q, k, v, mask)  # (B,H,L,d_k)
        y = self._merge_heads(y)  # (B,L,D)
        y = self.out_proj(y)  # (B,L,D)
        y = self.dropout(y)
        return y


# -----------------------------------------------------------------------------
# 3. Position‑wise feed‑forward network
# -----------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two‑layer MLP with GELU and dropout (used inside each Transformer block)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,D)
        return self.net(x)


# -----------------------------------------------------------------------------
# 4. Positional encoding (sinusoidal, no params)
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings as in the original Transformer paper."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# -----------------------------------------------------------------------------
# 5. Decoder block (self‑attention + FFN + residual/LayerNorm)
# -----------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """A single block of a decoder‑only Transformer (causal self‑attention)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int | None = None,
        *,
        n_heads:    int | None = None,  # alias support
        d_ff:       int | None = None,  # now truly optional
        dropout:    float      = 0.1,
    ) -> None:
        super().__init__()

        # alias handling
        if num_heads is None and n_heads is not None:
            num_heads = n_heads
        if num_heads is None:
            raise ValueError("Must specify num_heads or n_heads")

        # provide default feed-forward size
        if d_ff is None:
            d_ff = 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B,L,D)
        mask: Optional[torch.Tensor] = None,  # (L,L) or broadcastable additive
    ) -> torch.Tensor:
        # pre‑norm
        y = self.self_attn(self.ln1(x), mask)
        x = x + y  # residual
        y = self.ffn(self.ln2(x))
        return x + y  # residual
