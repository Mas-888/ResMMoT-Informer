"""干净版本的数据集与构建函数，实现六步流程前3步的预处理：
1. 小波去噪 (db4, 软阈值)
2. Min-Max 归一化 (仅在训练集上拟合)
3. 时间特征编码 (星期/月/日周期 sin/cos)
并生成滑动窗口 (lookback 默认 = 3*horizon)。
"""

from typing import List, Optional, Tuple as Tup
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pywt


class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        time_col: Optional[str] = None,
        lookback: Optional[int] = None,
        horizon: int = 24,
        train: bool = True,
        dropna: bool = True,
        scaler: Optional[Tup[MinMaxScaler, MinMaxScaler]] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_cols = target_cols or feature_cols
        self.horizon = horizon
        self.train = train

        df = pd.read_csv(csv_path)
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col)
        if dropna:
            df = df.dropna()

        X_raw = df[self.feature_cols].values.astype(np.float32)
        y_raw = df[self.target_cols].values.astype(np.float32)

        def wavelet_denoise(signal: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
            coeffs = pywt.wavedec(signal, wavelet)
            detail_coeffs = coeffs[-1]
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745 if detail_coeffs.size > 0 else 0.0
            uthresh = sigma * np.sqrt(2 * np.log(len(signal))) if sigma > 0 else 0.0
            denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
            rec = pywt.waverec(denoised_coeffs, wavelet)
            rec = rec[: len(signal)]
            if rec.shape[0] < len(signal):
                rec = np.pad(rec, (0, len(signal) - rec.shape[0]), mode='edge')
            return rec.astype(np.float32)

        # 波形去噪
        X = np.zeros_like(X_raw)
        for col in range(X_raw.shape[1]):
            X[:, col] = wavelet_denoise(X_raw[:, col])
        y = np.zeros_like(y_raw)
        for col in range(y_raw.shape[1]):
            y[:, col] = wavelet_denoise(y_raw[:, col])

        # 划分 85% 训练 / 15% 测试
        n = len(df)
        n_train = int(n * 0.85)
        if train:
            X = X[:n_train]
            y = y[:n_train]
        else:
            X = X[n_train:]
            y = y[n_train:]

        # 拟合或复用 scaler
        if scaler is None:
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            if train:
                self.scaler_x.fit(X)
                self.scaler_y.fit(y)
        else:
            self.scaler_x, self.scaler_y = scaler

        X = self.scaler_x.transform(X)
        y = self.scaler_y.transform(y)

        # 时间特征编码
        if time_col and time_col in df.columns:
            times = df[time_col].values
            dt_index = pd.to_datetime(times)
            if train:
                dt_index = dt_index[:n_train]
            else:
                dt_index = dt_index[n_train:]
            dow = dt_index.dayofweek
            dom = dt_index.day
            mon = dt_index.month
            dow_sin = np.sin(2 * np.pi * dow / 7.0)
            dow_cos = np.cos(2 * np.pi * dow / 7.0)
            dom_sin = np.sin(2 * np.pi * dom / 31.0)
            dom_cos = np.cos(2 * np.pi * dom / 31.0)
            mon_sin = np.sin(2 * np.pi * mon / 12.0)
            mon_cos = np.cos(2 * np.pi * mon / 12.0)
            time_feats = np.stack([dow_sin, dow_cos, dom_sin, dom_cos, mon_sin, mon_cos], axis=1).astype(np.float32)
            X = np.concatenate([X, time_feats], axis=1)

        self.X = X
        self.Y = y

        # 滑动窗口 lookback 默认 3*horizon
        if lookback is None:
            lookback = horizon * 3
        self.lookback = lookback
        self.indices = []
        total = len(X)
        for i in range(total - lookback - horizon + 1):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i : i + self.lookback]
        y = self.Y[i + self.lookback : i + self.lookback + self.horizon]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y


def build_datasets(
    csv_path: str,
    feature_cols: List[str],
    target_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    lookback: Optional[int] = None,
    horizon: int = 24,
):
    train_ds = TimeSeriesWindowDataset(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        time_col=time_col,
        lookback=lookback,
        horizon=horizon,
        train=True,
    )
    scaler = (train_ds.scaler_x, train_ds.scaler_y)
    test_ds = TimeSeriesWindowDataset(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        time_col=time_col,
        lookback=lookback,
        horizon=horizon,
        train=False,
        scaler=scaler,
    )
    return train_ds, None, test_ds, scaler
