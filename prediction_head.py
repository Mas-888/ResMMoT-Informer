from typing import List, Optional
import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    """多层1D CNN + LSTM + 全连接预测头.

    输入: (B, L, D_in)  —— 时间长度L，特征维D_in
    输出: (B, L, D_out) —— 逐时间步预测结果
    流程:
      1. (B, L, D_in) -> (B, D_in, L) 经过若干Conv1d层 (kernel_size=3, same padding)
      2. -> (B, L, C_last) 输入LSTM捕捉时间依赖
      3. -> 线性层逐步映射到输出维度
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        conv_channels: Optional[List[int]] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [in_dim, in_dim]
        layers = []
        prev = in_dim
        for c in conv_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(prev, c, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            prev = c
        self.convs = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=prev,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D_in)
        x_c = x.transpose(1, 2)  # (B, D_in, L)
        feat = self.convs(x_c)   # (B, C_last, L)
        feat = feat.transpose(1, 2)  # (B, L, C_last)
        lstm_out, _ = self.lstm(feat)  # (B, L, H)
        out = self.fc(lstm_out)  # (B, L, out_dim)
        return out
