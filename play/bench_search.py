"""
play/bench_search.py
探索の速さ（1秒あたりに NN で評価できた局面数）を、Python 版と C++ 版で比べる。

    python play/bench_search.py --seconds 2 --threads 4
    python play/bench_search.py --only cpp_mcts:appleS --weights weights/checkpoint.pth

重みがなくても速さは測れる（学習前の重みで動かす）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import cshogi  # noqa: E402
import torch  # noqa: E402

# いろいろな局面（序盤〜中盤）
POSITIONS = [
    ("startpos", []),
    ("startpos", "7g7f 3c3d 2g2f 4c4d 3i4h 3a4b 5i6h 5a6b 6h7h 6b7b".split()),
    ("startpos", "2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f".split()),
    ("lnsgk2nl/1r4gs1/p1pppp1pp/1p4p2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b Bb 1", []),
]


def _count_python(player) -> list[int]:
    """Python 版プレイヤーの _evaluate を数える。"""
    counter = [0]
    orig = player._evaluate

    def wrapped(sfen):
        counter[0] += 1
        return orig(sfen)

    player._evaluate = wrapped
    return counter


def run_one(kind: str, mode: str, weights: str, seconds: float, batch: int) -> float:
    if kind.startswith("cpp_"):
        from search.cpp_player import CppSearchPlayer

        p = CppSearchPlayer(algo=kind[4:], blocks_config_mode=mode, modelfile=weights, batch_size=batch)
        p.isready()
        total_n, total_t = 0, 0.0
        for sfen, moves in POSITIONS:
            p.position(sfen, moves)
            res = p.searcher.go(seconds * 1000.0, 0)
            total_n += res["evaluated"]
            total_t += res["time_ms"] / 1000.0
        return total_n / max(total_t, 1e-9)

    if kind == "py_npls":
        from search.npls import NPLSPlayer as P
    elif kind == "py_mcts":
        from search.mcts import MCTSPlayer as P
    else:
        raise ValueError(kind)
    p = P(blocks_config_mode=mode, modelfile=weights)
    counter = _count_python(p)
    total_t = 0.0
    for sfen, moves in POSITIONS:
        p.position(sfen, moves)
        p.set_limits(think_time_ms=int(seconds * 1000))
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            p.go()
        total_t += time.time() - t0
    return counter[0] / max(total_t, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0, help="1局面あたりの探索時間")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weights", default="")
    ap.add_argument("--weights_mode", default="appleS", help="--weights を使うモデル（それ以外は学習前の重み）")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="kind:mode の形で指定（例 cpp_mcts:appleS py_npls:modelC）",
    )
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    targets = a.only or [
        "py_npls:modelC",
        "py_npls:fastA",
        "py_npls:appleS",
        "cpp_npls:appleS",
        "cpp_mcts:appleS",
        "cpp_mcts:appleXS",
        "cpp_mcts:appleM",
        "cpp_mcts:fastA",
    ]
    print(f"threads={torch.get_num_threads()} seconds/position={a.seconds} batch={a.batch}")
    print(f"{'探索:モデル':<22}{'局面/秒':>10}")
    for t in targets:
        kind, mode = t.split(":")
        try:
            w = a.weights if mode == a.weights_mode else ""
            nps = run_one(kind, mode, w, a.seconds, a.batch)
            print(f"{t:<22}{nps:>10.0f}", flush=True)
        except Exception as exc:
            print(f"{t:<22}  error: {exc}", flush=True)


if __name__ == "__main__":
    main()
