import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 因果卷积：padding = (kernel_size-1)*dilation，左侧pad，右侧不pad
        self.pad = nn.ConstantPad1d(((kernel_size-1)*dilation, 0), 0)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.pad2 = nn.ConstantPad1d(((kernel_size-1)*dilation, 0), 0)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pad(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers += [
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        return self.network(x)



# 稀疏多专家TCN，门控选择权重最大的2个专家输出
class MultiScaleTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expert_cfgs: List[dict],  # 每个dict: {'kernel_size': int, 'num_levels': int}
        dropout: float = 0.1,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            TemporalConvNet(
                num_inputs=in_channels,
                channels=[out_channels] * cfg['num_levels'],
                kernel_size=cfg['kernel_size'],
                dropout=dropout,
            ) for cfg in expert_cfgs
        ])
        self.num_experts = len(self.experts)
        self.gate = nn.Linear(out_channels, self.num_experts)  # 门控输入为每步特征
        self.out_proj = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        expert_outs = [expert(x) for expert in self.experts]  # list of (B, C, L)
        # 堆叠: (B, num_experts, C, L)
        expert_outs_stacked = torch.stack(expert_outs, dim=1)
        # 门控：对每个时间步，输入为各专家输出的均值特征
        # 先对专家输出做全局池化 (B, num_experts, C)
        pooled = expert_outs_stacked.mean(-1)  # (B, num_experts, C)
        # 对每个token，门控输入为C维，输出为num_experts
        # 这里简化为对每个样本整体做门控
        gate_in = pooled.mean(1)  # (B, C)
        gate_logits = self.gate(gate_in)  # (B, num_experts)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, num_experts)
        # 选最大2个专家
        top2 = torch.topk(gate_weights, k=2, dim=-1)
        mask = torch.zeros_like(gate_weights)
        mask.scatter_(1, top2.indices, 1.0)
        gate_weights = gate_weights * mask
        gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
        # 加权融合专家输出
        # (B, num_experts, C, L) * (B, num_experts, 1, 1)
        gate_weights_exp = gate_weights.unsqueeze(-1).unsqueeze(-1)
        fused = (expert_outs_stacked * gate_weights_exp).sum(1)  # (B, C, L)
        out = self.out_proj(fused)
        out = self.norm(out)
        out = self.act(out)
        return out
