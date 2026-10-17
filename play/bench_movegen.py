"""
play/bench_movegen.py
合法手生成の「正しさ」と「速さ」を確かめる。

    python play/bench_movegen.py                       # CPU だけ
    python play/bench_movegen.py --device cuda         # GPU で測る
    python play/bench_movegen.py --device cuda --compile   # torch.compile 版も測る

比べるもの:
  - cshogi（CPU、1 局面ずつ）
  - マス単位版（game/gpu_movegen.py。バッチが小さいとき）
  - ビットボード版（game/gpu_bitboard.py）
  - フラット版（game/gpu_bitboard_flat.py）
  - 駒リスト版（game/gpu_movelist.py。CUDA の Triton カーネル）

--compile のとき、ビットボード版とフラット版は torch.compile する。
駒リスト版は最初から Triton カーネルなので、compile の有無で変わらない。
大きいバッチでは mode=default を使う（max-autotune は MSVC が無いと CPU カーネル生成で落ちる）。
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
from game.gpu_bitboard_flat import legal_move_mask_flat, make_compiled_core_flat  # noqa: E402
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
    """1 回ならしてから、reps 回の平均秒数を返す。ならしにはコンパイル時間も入る。"""
    fn(x)
    sync(device)
    t = time.perf_counter()
    for _ in range(reps):
        fn(x)
    sync(device)
    return (time.perf_counter() - t) / reps


def n_mismatch(mask, boards) -> int:
    """マスクの合法手集合が、cshogi の LegalAll と違う局面の数。"""
    return sum(
        set(torch.nonzero(mask[i]).flatten().tolist()) != {make_move_label(mv, b.turn) for mv in b.legal_moves}
        for i, b in enumerate(boards)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--check", type=int, default=5000, help="正しさを確かめる局面数")
    ap.add_argument("--batches", nargs="*", type=int, default=[256, 4096, 32768, 131072])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="torch.compile 版も測る（GPU 向け）")
    ap.add_argument("--compile-mode", default="default",
                    help="torch.compile の mode。大きいバッチは default（max-autotune は cl が要ることがある）")
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

    # 1. 正しさ（cshogi の LegalAll。all_moves=True がこれと一致する）
    m = min(a.check, len(boards))
    sample = feats[:m].to(device)
    checked = boards[:m]
    print(f"[正しさ] ビットボード版 不一致 {n_mismatch(legal_move_mask_bb(sample).cpu(), checked)} / {m}")
    print(f"[正しさ] フラット版     不一致 {n_mismatch(legal_move_mask_flat(sample).cpu(), checked)} / {m}")
    if device.type == "cuda":
        from game.gpu_movelist import legal_move_mask_list
        print(f"[正しさ] 駒リスト版     不一致 {n_mismatch(legal_move_mask_list(sample).cpu(), checked)} / {m}")

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

    core_bb = make_compiled_core_bb(a.compile_mode) if a.compile else None
    core_flat = make_compiled_core_flat(a.compile_mode) if a.compile else None
    list_fn = None
    if device.type == "cuda":
        from game.gpu_movelist import legal_move_mask_list
        list_fn = legal_move_mask_list
    for bs in a.batches:
        reps = bs // len(sfens) + 1
        x = feats.repeat(reps, 1)[:bs].to(device)
        jobs = [("ビットボード版", lambda z: legal_move_mask_bb(z))]
        if core_bb is not None:
            jobs.append(("ビットボード版 + compile", lambda z, c=core_bb: legal_move_mask_bb(z, core=c)))
        jobs.append(("フラット版", lambda z: legal_move_mask_flat(z)))
        if core_flat is not None:
            jobs.append(("フラット版 + compile", lambda z, c=core_flat: legal_move_mask_flat(z, core=c)))
        if list_fn is not None:
            jobs.append(("駒リスト版", list_fn))
        if bs <= 8192:
            jobs.append(("マス単位版", lambda z: legal_move_mask(z)))
        print(f"[{device.type.upper()}] batch={bs}", flush=True)
        for name, fn in jobs:
            dt = bench(fn, x, device)
            print(f"       {name:<24} {bs / dt:>12,.0f} 局面/秒（{dt * 1000:.1f} ms）", flush=True)


if __name__ == "__main__":
    main()
