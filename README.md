# ResMMoT-Informer (Reproduction)

This repository provides a runnable reproduction of a Residual Multi-Scale TCN Sparse Mixture-of-Experts + Informer-like model for long/short-term financial time series forecasting.

Note: The Informer module here is a simplified Transformer-based implementation (InformerLite) to keep dependencies minimal. You can later swap it with a full Informer (with ProbSparse attention and encoder distilling) if required.

## Structure

- `src/models/modules/tcn.py`: Multi-Scale TCN with residual blocks
- `src/models/modules/moe.py`: Sparse MoE with Top-K gating
- `src/models/modules/informer.py`: InformerLite (Transformer wrapper)
- `src/models/resmmot_informer.py`: Full model composition
- `src/data/datasets.py`: CSV loader, scaling, sliding windows
- `src/train.py`: Training and evaluation loop
- `src/config.yaml`: Default config (edit for your dataset)
- `tests/test_forward.py`: Minimal forward test
- `dataset/`: Provided financial datasets

## Quickstart

1) Install dependencies (macOS, zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Edit `src/config.yaml` to match your CSV headers. For example, if your CSV has columns `Date,Open,High,Low,Close,Volume`, set:

```yaml
data:
  csv_path: dataset/NASDAQ_100_Data_From_2010.csv
  feature_cols: ["Close"]
  target_cols: ["Close"]
  time_col: null  # or "Date" if present
  lookback: 96
  horizon: 24
```

3) Run a quick forward test:

```bash
python tests/test_forward.py
```

4) Train:

```bash
python -m src.train --config src/config.yaml
```

## Notes
- The code is designed to be modular. To use a full Informer, implement it under `src/models/modules/informer_full.py` and replace the InformerLite import in `resmmot_informer.py`.
- The Sparse MoE is simplified and does not include load balancing losses; you can extend it with auxiliary losses if needed.
- For multivariate targets, list multiple `target_cols` in the config.
