import os
import sys
import torch

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models import ResMMoTInformer

def test_forward():
    B, L, C_in, pred_len = 4, 96, 3, 24
    x = torch.randn(B, L, C_in)
    model = ResMMoTInformer(in_dim=C_in, out_dim=C_in, d_model=64, nhead=4)
    y = model(x, pred_len)
    assert y.shape == (B, pred_len, C_in)

if __name__ == "__main__":
    test_forward()
    print("forward ok")
