"""
play/bench_movegen.py
合法手生成の「正しさ」と「速さ」を確かめる（CPU の cshogi と、GPU 向けのビットボード版）。

    python play/bench_movegen.py                       # CPU だけ
    python play/bench_movegen.py --device cuda         # GPU で測る
    python play/bench_movegen.py --device cuda --compile   # torch.compile でカーネルをまとめた版も測る

  1. ランダムな対局の局面で、ビットボード版が cshogi の合法手と完全に一致するか確認する
  2. 1秒あたりに何局面ぶん合法手を作れるかを比べる
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import cshogi

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "search", "cpp"))

from game.board import bit447_to_numpy, make_move_label  # noqa: E402
from game.gpu_bitboard import legal_move_mask_bb, make_compiled_core_bb  # noqa: E402
from game.gpu_movegen import legal_move_mask  # noqa: E402


def random_positions(n_games: int, seed: int = 0) -> list[str]:
    """駒を取る手・王手を多めにしたランダム対局から局面を集める（王手・ピン・持ち駒が多い局面も出る）。"""
    rnd = random.Random(seed)
    out = []
    for _ in range(n_games):
        b = cshogi.Board()
        for _ply in range(rnd.randint(30, 220)):
            ms = list(b.legal_moves)
            if not ms:
                break
            out.append(b.sfen())
            caps = [m for m in ms if b.piece(cshogi.move_to(m)) != 0]
            drops = [m for m in ms if cshogi.move_is_drop(m)]
            r = rnd.random()
            if caps and r < 0.5:
                m = rnd.choice(caps)
            elif drops and r < 0.8:
                m = rnd.choice(drops)
            else:
                m = rnd.choice(ms)
            b.push(m)
        out.append(b.sfen())
    return out


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, x, device, reps=5):
    fn(x)
    sync(device)
    t = time.perf_counter()
    for _ in range(reps):
        fn(x)
    sync(device)
    return (time.perf_counter() - t) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--check", type=int, default=5000, help="正しさを確かめる局面数")
    ap.add_argument("--batches", nargs="*", type=int, default=[256, 4096, 32768, 131072])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="torch.compile 版も測る（GPU 向け）")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    if device.type == "cpu":
        a.batches = [b for b in a.batches if b <= 8192]

    sfens = random_positions(a.games)
    boards = [cshogi.Board(s) for s in sfens]
    feats = torch.tensor(np.stack([bit447_to_numpy(b) for b in boards]))
    print(f"局面数 {len(sfens)}  device={device}  CPUスレッド={torch.get_num_threads()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. 正しさ
    m = min(a.check, len(boards))
    mask = legal_move_mask_bb(feats[:m].to(device)).cpu()
    bad = sum(
        set(torch.nonzero(mask[i]).flatten().tolist()) != {make_move_label(mv, b.turn) for mv in b.legal_moves}
        for i, b in enumerate(boards[:m])
    )
    print(f"[正しさ] cshogi との不一致: {bad} / {m} 局面")

    # 2. 速さ（CPU 側の基準）
    t = time.perf_counter()
    for b in boards:
        _ = list(b.legal_moves)
    print(f"[CPU] cshogi（Python から1局面ずつ）          {len(boards) / (time.perf_counter() - t):>12,.0f} 局面/秒")
    t = time.perf_counter()
    for b in boards:
        _ = [make_move_label(mv, b.turn) for mv in b.legal_moves]
    print(f"[CPU] cshogi + 方策ラベル化（Python）         {len(boards) / (time.perf_counter() - t):>12,.0f} 局面/秒")
    print("      参考: cshogi の C++ を直接使うと 1 コアで約 90 万局面/秒（ラベル化込み、こちらで測定）")

    core = make_compiled_core_bb() if a.compile else None
    for bs in a.batches:
        reps = bs // len(sfens) + 1
        x = feats.repeat(reps, 1)[:bs].to(device)
        dt = bench(lambda z: legal_move_mask_bb(z), x, device)
        line = f"[{device.type.upper()}] ビットボード版 batch={bs:<7}           {bs / dt:>12,.0f} 局面/秒（{dt * 1000:.1f} ms）"
        if bs <= 8192:
            dt2 = bench(lambda z: legal_move_mask(z), x, device)
            line += f"  / マス単位版 {bs / dt2:,.0f}"
        print(line, flush=True)
        if core is not None:
            dt3 = bench(lambda z: legal_move_mask_bb(z, core=core), x, device)
            print(f"       ビットボード版 + torch.compile          {bs / dt3:>12,.0f} 局面/秒（{dt3 * 1000:.1f} ms）",
                  flush=True)


if __name__ == "__main__":
    main()
