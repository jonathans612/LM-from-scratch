from __future__ import annotations

"""GPT-style decoder-only Transformer model.

This module wires together:
  • Embedding layer + sinusoidal positions
  • N x DecoderBlock (from .layers)
  • Final LayerNorm + Linear vocab projection

The forward pass accepts integer token IDs and returns logits suitable for
`nn.CrossEntropyLoss`.
"""

import torch
import torch.nn as nn

from .layers import DecoderBlock, PositionalEncoding, causal_mask

__all__ = ["TransformerLM"]


class TransformerLM(nn.Module):
    """Minimal language model (decoder-only Transformer)."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # --- input embedding + pos enc ---
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)

        # --- stacked decoder blocks ---
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # tie weights à la GPT‑2 for slight perplexity boost
        self.lm_head.weight = self.tok_embed.weight  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: torch.Tensor,  # (B, L) int64
        *,
        attn_mask: torch.Tensor | None = None,  # additive mask (1,1,L,L) or (L,L)
    ) -> torch.Tensor:  # logits (B, L, V)
        B, L = token_ids.shape
        x = self.tok_embed(token_ids)  # (B,L,D)
        x = self.pos_enc(x)

        if attn_mask is None:
            attn_mask = causal_mask(L, device=x.device)
        elif attn_mask.dim() == 2:  # (L,L)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        for blk in self.blocks:
            x = blk(x, mask=attn_mask)

        x = self.norm(x)
        return self.lm_head(x)
