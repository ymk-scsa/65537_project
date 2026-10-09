"""
search/cpp/export_torchscript.py
学習済みの重み (.pth) を、C++ 単体エンジン（LibTorch 版）で読める TorchScript (.ts) に変換する。

    python search/cpp/export_torchscript.py --mode appleS --weights weights/checkpoint.pth --out weights/appleS.ts

出力されるモデルは forward(x[N,447] float32, turn[N] int64) -> (policy[N,2187], value[N,1])
"""

from __future__ import annotations

import argparse
import os
import sys

import warnings

import torch

# torch.jit は新しい PyTorch で「非推奨」の警告が出るが、まだ動くので表示しない
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.model import create_model  # noqa: E402


class _Wrap(torch.nn.Module):
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m
        self.uses_turn = bool(getattr(m, "uses_turn", False))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if self.uses_turn:
            return self.m(x, t)
        return self.m(x)


def export(mode: str, weights: str, out: str, draw_black: float, draw_white: float) -> None:
    model = create_model(num_actions=2187, mode=mode)
    if weights and os.path.exists(weights):
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"loaded {weights}")
    else:
        print(f"[注意] 重み {weights} が見つからないので、学習前の重みで書き出します")
    if hasattr(model, "set_draw_values"):
        model.set_draw_values(draw_black, draw_white)
    model.eval()
    if hasattr(model, "fuse_for_inference"):
        model.fuse_for_inference()
    ex = (torch.zeros(8, 447), torch.zeros(8, dtype=torch.long))
    with torch.inference_mode():
        traced = torch.jit.trace(_Wrap(model).eval(), ex, check_trace=False)
        traced = torch.jit.freeze(traced)
        # 動作確認（バッチサイズを変えても動くか）
        for n in (1, 3, 32):
            p, v = traced(torch.zeros(n, 447), torch.ones(n, dtype=torch.long))
            assert p.shape == (n, 2187) and v.shape == (n, 1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    traced.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="appleS")
    ap.add_argument("--weights", default="weights/checkpoint.pth")
    ap.add_argument("--out", default="weights/model.ts")
    ap.add_argument("--draw_black", type=float, default=0.4)
    ap.add_argument("--draw_white", type=float, default=0.6)
    a = ap.parse_args()
    export(a.mode, a.weights, a.out, a.draw_black, a.draw_white)
