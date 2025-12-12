import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbSparseSelfAttention(nn.Module):
    """简化的概率稀疏自注意力：选取top-k查询参与注意力，其余置零。"""
    def __init__(self, d_model: int, nhead: int, top_query_frac: float = 0.2, dropout: float = 0.1):
        super().__init__()
        assert 0 < top_query_frac <= 1
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.top_query_frac = top_query_frac
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B, h, L, d)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # 计算查询向量范数，选取top-k
        q_norm = q.pow(2).sum(-1)  # (B, h, L)
        k_top = max(1, int(L * self.top_query_frac))
        topk_vals, topk_idx = torch.topk(q_norm, k=k_top, dim=-1)  # (B, h, k)
        mask = torch.zeros_like(q_norm, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_idx, value=True)  # True表示保留
        # 将未保留的查询置零（也可以直接不参与注意力计算）
        q = q * mask.unsqueeze(-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,h,L,L)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)  # (B,h,L,d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class DistillBlock(nn.Module):
    """自注意力蒸馏：全局关系+局部卷积+ReLU+MaxPool缩短序列。"""
    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        # x, attn_out: (B, L, D)
        fused = x + attn_out  # 简化融合
        y = fused.transpose(1, 2)  # (B,D,L)
        y = self.conv(y)
        y = self.act(y)
        y = self.pool(y)  # (B,D,L/2)
        y = y.transpose(1, 2)  # (B,L/2,D)
        return y


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, top_query_frac: float, dropout: float, distill: bool):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, nhead, top_query_frac, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.distill = distill
        if distill:
            self.distill_block = DistillBlock(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        if self.distill:
            x = self.distill_block(x, attn_out)
        return x


class InformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, layers: int, top_query_frac: float, dropout: float, distill: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead, ff_dim, top_query_frac, dropout, distill) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, top_query_frac: float, dropout: float):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, nhead, top_query_frac, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        self_attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class InformerDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, layers: int, top_query_frac: float, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerDecoderLayer(d_model, nhead, ff_dim, top_query_frac, dropout) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory)
        return x


class InformerFull(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        enc_layers: int,
        dec_layers: int,
        ff_dim: int,
        dropout: float,
        top_query_frac: float,
        distill: bool,
        out_dim: Optional[int] = None,
        return_latent: bool = False,
    ):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(10000, d_model)
        self.encoder = InformerEncoder(d_model, nhead, ff_dim, enc_layers, top_query_frac, dropout, distill)
        self.decoder = InformerDecoder(d_model, nhead, ff_dim, dec_layers, top_query_frac, dropout)
        self.out_proj = nn.Linear(d_model, out_dim) if (out_dim is not None) else None
        self.return_latent = return_latent

    def forward(self, src: torch.Tensor, pred_len: int) -> torch.Tensor:
        # src: (B, L, D_in)
        B, L, _ = src.shape
        pos_idx = torch.arange(L, device=src.device).unsqueeze(0).repeat(B, 1)
        src_emb = self.value_proj(src) + self.pos_emb(pos_idx)  # (B,L,d_model)
        memory = self.encoder(src_emb)  # (B, L', d_model)
        start_token = src_emb[:, -1:, :].repeat(1, pred_len, 1)
        dec_out = self.decoder(start_token, memory)  # (B, pred_len, d_model)
        if self.return_latent or self.out_proj is None:
            return dec_out
        return self.out_proj(dec_out)
