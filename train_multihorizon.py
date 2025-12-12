import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

# 允许脚本直接运行 (添加 src 上级路径)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.resmmot_informer import ResMMoTInformer
from data.datasets import TimeSeriesWindowDataset


def compute_metrics(pred: torch.Tensor, true: torch.Tensor):
    # pred/true: (N, T, C)
    diff = pred - true
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5
    # flatten for R2 per feature average
    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    # Compute R2 per feature then average
    r2_list = []
    for c in range(true_np.shape[-1]):
        try:
            r2_list.append(r2_score(true_np[:, :, c].ravel(), pred_np[:, :, c].ravel()))
        except Exception:
            r2_list.append(float('nan'))
    r2 = float(torch.tensor([x for x in r2_list if x == x]).mean()) if len(r2_list) else float('nan')
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def build_dataset(csv_path: str, feature_cols, target_cols, time_col, lookback: int, horizon: int, train: bool, scaler=None):
    return TimeSeriesWindowDataset(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        time_col=time_col,
        lookback=lookback,
        horizon=horizon,
        train=train,
        scaler=scaler,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x, pred_len=y.size(1))
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x, pred_len=y.size(1))
            loss = loss_fn(y_hat, y)
            total += loss.item() * x.size(0)
            preds.append(y_hat.cpu())
            trues.append(y.cpu())
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    return total / len(loader.dataset), preds, trues


def multi_horizon_metrics(preds: torch.Tensor, trues: torch.Tensor, horizons):
    results = {}
    for h in horizons:
        p = preds[:, :h, :]
        t = trues[:, :h, :]
        results[h] = compute_metrics(p, t)
    return results


def auto_detect_columns(csv_path):
    import pandas as pd
    df_head = pd.read_csv(csv_path, nrows=5)
    numeric_cols = [c for c in df_head.columns if pd.api.types.is_numeric_dtype(df_head[c])]
    # Prefer 'Close' if exists
    if 'Close' in numeric_cols:
        return ['Close'], ['Close']
    # fallback: use first numeric column
    if numeric_cols:
        return [numeric_cols[0]], [numeric_cols[0]]
    raise ValueError("No numeric columns detected for features/targets.")


def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = cfg['data']['csv_path']
    feature_cols = cfg['data'].get('feature_cols')
    target_cols = cfg['data'].get('target_cols')
    time_col = cfg['data'].get('time_col')
    max_horizon = max(cfg['eval']['horizons'])
    lookback = cfg['data'].get('lookback') or max_horizon * 3

    if feature_cols is None or target_cols is None:
        feature_cols, target_cols = auto_detect_columns(csv_path)
        print(f"[Auto] Using feature_cols={feature_cols} target_cols={target_cols}")

    # Build train dataset (fit scalers)
    train_ds = build_dataset(csv_path, feature_cols, target_cols, time_col, lookback, max_horizon, train=True, scaler=None)
    scaler = (train_ds.scaler_x, train_ds.scaler_y)
    test_ds = build_dataset(csv_path, feature_cols, target_cols, time_col, lookback, max_horizon, train=False, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg['train']['batch_size'], shuffle=False)

    in_dim = len(feature_cols)
    out_dim = len(target_cols)

    model = ResMMoTInformer(
        in_dim=in_dim,
        out_dim=out_dim,
        tcn_out=cfg['model'].get('tcn_out', 64),
        expert_cfgs=cfg['model'].get('expert_cfgs'),
        d_model=cfg['model'].get('d_model'),
        dropout=cfg['model'].get('dropout', 0.1),
        nhead=cfg['model'].get('nhead', 4),
        enc_layers=cfg['model'].get('enc_layers', 2),
        dec_layers=cfg['model'].get('dec_layers', 1),
        ff_dim=cfg['model'].get('ff_dim', 256),
        top_query_frac=cfg['model'].get('top_query_frac', 0.2),
        distill=cfg['model'].get('distill', True),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train'].get('lr', 1e-3))

    epochs = cfg['train'].get('epochs', 5)
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {ep}/{epochs} train_loss={tr_loss:.4f}")

    test_loss, preds, trues = eval_model(model, test_loader, device)
    print(f"Test MSE loss (full horizon={max_horizon}): {test_loss:.4f}")

    horizons = cfg['eval']['horizons']
    results = multi_horizon_metrics(preds, trues, horizons)
    for h, mets in results.items():
        print(f"Horizon {h}: MAE={mets['MAE']:.4f} RMSE={mets['RMSE']:.4f} R2={mets['R2']:.4f}")

    save_dir = cfg['train'].get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(save_dir, 'mh_best.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config.multihorizon.yaml')
    args = parser.parse_args()
    main(args.config)
