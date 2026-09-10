"""Encoder-decoder Transformer for the Multi30k leg, with per-head attention-matrix access.

Mask convention: padding is passed as key_padding_mask (B, T) bool; causality as a
2D (T, T) bool attn_mask (broadcast to all heads/batches by MHA).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

from divprune.model_module import register_model


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class _EncoderLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

    def attention_matrix(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        _, w = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=True, average_attn_weights=False)
        return w  # (B, H, T, T)


class _DecoderLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        y: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad: Optional[torch.Tensor],
        src_pad: Optional[torch.Tensor],
        causal: Optional[torch.Tensor],
    ) -> torch.Tensor:
        a, _ = self.self_attn(
            y, y, y, key_padding_mask=tgt_pad, attn_mask=causal, need_weights=False
        )
        y = self.norm1(y + self.dropout(a))
        a, _ = self.cross_attn(y, memory, memory, key_padding_mask=src_pad, need_weights=False)
        y = self.norm2(y + self.dropout(a))
        y = self.norm3(y + self.dropout(self.ff(y)))
        return y

    def self_attention_matrix(
        self,
        y: torch.Tensor,
        tgt_pad: Optional[torch.Tensor],
        causal: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, w = self.self_attn(
            y,
            y,
            y,
            key_padding_mask=tgt_pad,
            attn_mask=causal,
            need_weights=True,
            average_attn_weights=False,
        )
        return w

    def cross_attention_matrix(
        self, y: torch.Tensor, memory: torch.Tensor, src_pad: Optional[torch.Tensor]
    ) -> torch.Tensor:
        _, w = self.cross_attn(y, memory, memory, key_padding_mask=src_pad, need_weights=True, average_attn_weights=False)
        return w


@register_model("transformer_mt")
class TransformerMT(nn.Module):
    """3-layer encoder-decoder Transformer (KSEM 5822 config: 8 heads, d=256, d_ff=512)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_idx: int = cfg.pad_idx
        d = cfg.d_model
        self.enc_emb = nn.Embedding(cfg.src_vocab, d, padding_idx=self.pad_idx)
        self.dec_emb = nn.Embedding(cfg.tgt_vocab, d, padding_idx=self.pad_idx)
        self.pos = PositionalEncoding(d, cfg.dropout)
        self.encoder = nn.ModuleList([_EncoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.decoder = nn.ModuleList([_DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.out = nn.Linear(d, cfg.tgt_vocab)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_pad: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.pos(self.enc_emb(src))
        for layer in self.encoder:
            x = layer(x, src_pad)
        return x

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_pad: Optional[torch.Tensor],
        src_pad: Optional[torch.Tensor],
        causal: Optional[torch.Tensor],
    ) -> torch.Tensor:
        y = self.pos(self.dec_emb(tgt))
        for layer in self.decoder:
            y = layer(y, memory, tgt_pad, src_pad, causal)
        return self.out(y)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad: Optional[torch.Tensor],
        tgt_pad: Optional[torch.Tensor],
        causal: Optional[torch.Tensor],
        collect_attn: bool = False,
    ) -> dict:
        """Teacher-forcing forward.

        Returns:
            dict with "logits" (B, T_t, vocab); with collect_attn=True also
            "attn_mats": list of per-layer attention matrices (B, H, T, T).
        """
        memory = self.encode(src, src_pad)
        logits = self.decode(memory, tgt, tgt_pad, src_pad, causal)
        out: dict = {"logits": logits}
        if collect_attn:
            mats: List[torch.Tensor] = []
            x = self.pos(self.enc_emb(src))
            for layer in self.encoder:
                x = layer(x, src_pad)
                mats.append(layer.attention_matrix(x, src_pad))
            y = self.pos(self.dec_emb(tgt))
            for layer in self.decoder:
                y = layer(y, memory, tgt_pad, src_pad, causal)
                mats.append(layer.self_attention_matrix(y, tgt_pad, causal))
                mats.append(layer.cross_attention_matrix(y, memory, src_pad))
            out["attn_mats"] = mats
        return out
