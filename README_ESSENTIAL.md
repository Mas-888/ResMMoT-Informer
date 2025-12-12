# ResMMoT-Informer 必需文件说明

本文件夹包含运行 ResMMoT-Informer 模型的所有必需文件（精简版）。

## 目录结构

```
ResMMoT-Informer-Essential/
├── requirements.txt              # Python 依赖
├── README.md                     # 本说明文档
├── .gitignore                    # Git 忽略配置
├── dataset/                      # 数据集目录（需自行添加数据）
├── src/
│   ├── __init__.py
│   ├── config.multihorizon.yaml  # 训练配置文件
│   ├── train_multihorizon.py     # 训练脚本
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py           # 数据加载器
│   └── models/
│       ├── __init__.py
│       ├── resmmot_informer.py   # 主模型
│       └── modules/
│           ├── __init__.py
│           ├── tcn.py            # TCN 模块
│           ├── informer_full.py  # Informer 模块
│           └── prediction_head.py # 预测头
└── tests/
    └── test_forward.py           # 前向测试脚本
```

## 快速开始

### 1. 环境准备
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 准备数据
将你的 CSV 数据文件放到 `dataset/` 目录下，例如：
```bash
cp /path/to/your/data.csv dataset/
```

### 3. 配置参数
编辑 `src/config.multihorizon.yaml`，设置：
- `csv_path`: 数据文件路径
- `feature_cols`: 特征列名
- `target_cols`: 目标列名
- `time_col`: 时间列名（可选）

### 4. 运行测试
```bash
python tests/test_forward.py
```

### 5. 训练模型
```bash
python src/train_multihorizon.py --config src/config.multihorizon.yaml
```

## 核心文件说明

### 模型相关
- **resmmot_informer.py**: ResMMoT-Informer 主模型类
- **tcn.py**: 多尺度时间卷积网络（MultiScaleTCN）
- **informer_full.py**: Informer 完整实现（ProbSparse 注意力 + 蒸馏）
- **prediction_head.py**: 预测头（1D CNN + LSTM + FC）

### 数据相关
- **datasets.py**: 时间序列数据集类（小波去噪、归一化、滑动窗口）

### 训练相关
- **train_multihorizon.py**: 多周期训练与评估脚本
- **config.multihorizon.yaml**: 配置文件

## 注意事项

1. **数据格式**: CSV 文件应包含时间序列数据，列名需在配置文件中明确指定
2. **CSV 分隔符**: NASDAQ 数据使用制表符（\t）分隔，需确保数据集加载器正确处理
3. **依赖**: 确保安装了所有必需的 Python 包（见 requirements.txt）

## 常见问题

- **ImportError**: 确保从项目根目录运行脚本
- **数据加载失败**: 检查 CSV 分隔符和列名是否正确
- **CUDA 错误**: 如无 GPU，模型会自动使用 CPU

## 原始项目

完整项目包含更多文件（基准模型、论文 PDF、额外配置等），本文件夹仅包含运行 ResMMoT-Informer 的最小必需文件集。
