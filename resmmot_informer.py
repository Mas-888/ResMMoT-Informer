"""ResMMoT-Informer: Residual Multi-scale Mixture-of-Experts TCN + Informer with ProbSparse attention & distilling.

This version cleans previous corrupted definitions and integrates the upgraded Informer encoder-decoder.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from .modules.tcn import MultiScaleTCN
from .modules.informer_full import InformerFull
from .modules.prediction_head import PredictionHead


class ResMMoTInformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        tcn_out: int = 64,
        expert_cfgs: Optional[List[dict]] = None,
        d_model: Optional[int] = None,  # if None use 2*tcn_out after fusion
        dropout: float = 0.1,
        nhead: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 1,
        ff_dim: int = 256,
        top_query_frac: float = 0.2,
        distill: bool = True,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tcn_out = tcn_out
        if expert_cfgs is None:
            expert_cfgs = [
                {"kernel_size": 3, "num_levels": 2},
                {"kernel_size": 5, "num_levels": 2},
                {"kernel_size": 7, "num_levels": 2},
            ]

        # 1x1 CNN调整原始输入维度
        self.input_proj = nn.Conv1d(in_dim, tcn_out, kernel_size=1)

        # 多专家稀疏TCN (门控在内部)
        self.mmottcn = MultiScaleTCN(
            in_channels=in_dim,
            out_channels=tcn_out,
            expert_cfgs=expert_cfgs,
            dropout=dropout,
        )

        fused_dim = tcn_out * 2
        model_dim = d_model or fused_dim
        self.post_fuse_proj = nn.Linear(fused_dim, model_dim) if model_dim != fused_dim else nn.Identity()
        self.relu = nn.ReLU()

        # Informer (ProbSparse + Distilling)
        self.informer = InformerFull(
            input_dim=model_dim,
            d_model=model_dim,
            nhead=nhead,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            top_query_frac=top_query_frac,
            distill=distill,
            out_dim=None,  # return latent features
            return_latent=True,
        )
        self.pred_head = PredictionHead(
            in_dim=model_dim,
            out_dim=out_dim,
            conv_channels=[model_dim, model_dim],
            lstm_hidden=model_dim,
            lstm_layers=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, pred_len: int) -> torch.Tensor:
        # x: (B, L, Cin)
        x_c = x.transpose(1, 2)  # (B, Cin, L)
        base = self.input_proj(x_c)  # (B, tcn_out, L)
        expert = self.mmottcn(x_c)  # (B, tcn_out, L)
        fused = torch.cat([expert, base], dim=1)  # (B, 2*tcn_out, L)
        fused = self.relu(fused)
        fused = fused.transpose(1, 2)  # (B, L, 2*tcn_out)
        fused = self.post_fuse_proj(fused)  # (B, L, d_model)

        # Informer预测
        latent = self.informer(src=fused, pred_len=pred_len)  # (B, pred_len, d_model)
        y_hat = self.pred_head(latent)  # (B, pred_len, out_dim)
        return y_hat
