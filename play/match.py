"""
対局ランナー（学習とは独立）。

同一または異なる重み・探索設定のエンジン同士で対局し、
勝敗集計と棋譜（CSA）を出力する。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

import cshogi

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from search.mcts import MCTSPlayer
from search.mc_rzf import MCRZFPlayer
from search.npls import NPLSPlayer

Player = Union[MCTSPlayer, NPLSPlayer, MCRZFPlayer]


@dataclass
class SideConfig:
    name: str
    search_type: str
    mode: str
    weights: str


def build_player(cfg: SideConfig) -> Player:
    common = dict(
        blocks_config_mode=cfg.mode,
        modelfile=cfg.weights,
        name=cfg.name,
    )
    st = cfg.search_type.lower()
    if st in ("mcts_cpp", "npls_cpp"):
        from search.cpp_player import CppSearchPlayer

        return CppSearchPlayer(algo=st.split("_")[0], **common)
    if st == "npls":
        return NPLSPlayer(**common)
    if st == "mc_rzf":
        return MCRZFPlayer(**common)
    if st == "mcts":
        return MCTSPlayer(**common)
    raise ValueError(f"unknown search_type: {cfg.search_type}")


def _is_terminal_move(usi: str) -> bool:
    return usi in ("resign", "win", "resnet", "")


def play_one_game(
    black: Player,
    white: Player,
    *,
    think_time_ms: int,
    max_moves: int,
    verbose: bool = True,
) -> tuple[str, list[str]]:
    """
    1 局対局する。

    Returns
    -------
    result : "1-0" / "0-1" / "draw"
    moves  : USI 指し手リスト
    """
    board = cshogi.Board()
    moves: list[str] = []

    while not board.is_game_over() and len(moves) < max_moves:
        player = black if board.turn == cshogi.BLACK else white
        side = "BLACK" if board.turn == cshogi.BLACK else "WHITE"
        mark = "▲" if board.turn == cshogi.BLACK else "△"

        player.position("startpos", moves)
        player.set_limits(think_time_ms=think_time_ms)

        t0 = time.time()
        best_usi, _ = player.go()
        elapsed_ms = (time.time() - t0) * 1000.0

        if _is_terminal_move(best_usi):
            if verbose:
                print(f"  {side} resign/invalid: {best_usi!r}")
            return ("0-1" if board.turn == cshogi.BLACK else "1-0"), moves

        try:
            move = board.move_from_usi(best_usi)
        except Exception:
            if verbose:
                print(f"  {side} illegal USI {best_usi!r} -> resign")
            return ("0-1" if board.turn == cshogi.BLACK else "1-0"), moves

        if move not in board.legal_moves:
            if verbose:
                print(f"  {side} illegal move {best_usi} -> resign")
            return ("0-1" if board.turn == cshogi.BLACK else "1-0"), moves

        board.push(move)
        moves.append(best_usi)
        if verbose:
            print(f"  {len(moves):3d}. {mark} {best_usi:8s} ({elapsed_ms:.0f} ms)")

    if board.is_game_over():
        # 手番側に合法手がない = 手番側の負け（Board.result() は cshogi に存在しない）
        return ("0-1" if board.turn == cshogi.BLACK else "1-0"), moves
    return "draw", moves


def save_csa(
    path: str,
    moves: list[str],
    result: str,
    black_name: str,
    white_name: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    board = cshogi.Board()
    lines = [
        "V2.2",
        f"N+{black_name}",
        f"N-{white_name}",
        f"$START_TIME:{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",
        "PI",
        "+",
    ]
    turn = "+"
    for usi in moves:
        move = board.move_from_usi(usi)
        lines.append(f"{turn}{cshogi.move_to_csa(move)}")
        board.push(move)
        turn = "-" if turn == "+" else "+"

    if result == "1-0":
        # 後手投了
        lines.append("%TORYO")
    elif result == "0-1":
        lines.append("%TORYO")
    else:
        lines.append("%SENNICHITE")
    lines.append(f"'result {result}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_match(args: argparse.Namespace) -> None:
    black_cfg = SideConfig(
        name=args.black_name,
        search_type=args.black_search,
        mode=args.black_mode,
        weights=args.black_weights,
    )
    white_cfg = SideConfig(
        name=args.white_name,
        search_type=args.white_search,
        mode=args.white_mode,
        weights=args.white_weights,
    )

    print("=== Match setup ===")
    print(
        f"  BLACK: {black_cfg.name}  search={black_cfg.search_type}  "
        f"mode={black_cfg.mode}  weights={black_cfg.weights}"
    )
    print(
        f"  WHITE: {white_cfg.name}  search={white_cfg.search_type}  "
        f"mode={white_cfg.mode}  weights={white_cfg.weights}"
    )
    print(f"  games={args.games}  think_time_ms={args.think_time_ms}  max_moves={args.max_moves}")

    black = build_player(black_cfg)
    white = build_player(white_cfg)

    # 集計は black_cfg / white_cfg の論理側（入れ替え後も補正）
    score = {"1-0": 0, "0-1": 0, "draw": 0}
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for g in range(1, args.games + 1):
        swap = args.swap_colors and (g % 2 == 0)
        b_player, w_player = (white, black) if swap else (black, white)
        b_name = white_cfg.name if swap else black_cfg.name
        w_name = black_cfg.name if swap else white_cfg.name

        print(f"\n--- Game {g}/{args.games}  BLACK={b_name}  WHITE={w_name} ---")
        result, moves = play_one_game(
            b_player,
            w_player,
            think_time_ms=args.think_time_ms,
            max_moves=args.max_moves,
            verbose=not args.quiet,
        )

        if swap:
            if result == "1-0":
                mapped = "0-1"
            elif result == "0-1":
                mapped = "1-0"
            else:
                mapped = "draw"
        else:
            mapped = result

        score[mapped] += 1
        print(
            f"  board_result={result}  moves={len(moves)}  "
            f"mapped({black_cfg.name} vs {white_cfg.name})={mapped}"
        )

        if args.save_csa:
            csa_path = os.path.join(args.log_dir, f"match_{stamp}_g{g:03d}.csa")
            save_csa(csa_path, moves, result, b_name, w_name)
            print(f"  saved {csa_path}")

    b_wins = score["1-0"]
    w_wins = score["0-1"]
    draws = score["draw"]
    played = max(b_wins + w_wins + draws, 1)
    print("\n=== Summary ===")
    print(f"  {black_cfg.name} wins: {b_wins}")
    print(f"  {white_cfg.name} wins: {w_wins}")
    print(f"  draws: {draws}")
    print(f"  {black_cfg.name} score: {(b_wins + 0.5 * draws) / played:.3f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Engine vs Engine match (not for training)")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--think_time_ms", type=int, default=1000)
    p.add_argument("--max_moves", type=int, default=320)
    p.add_argument("--swap_colors", action="store_true", help="Swap colors every even game")
    p.add_argument("--save_csa", action="store_true", help="Save CSA under --log_dir")
    p.add_argument("--log_dir", default="data/log/match")
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--black_name", default="GNN-Black")
    p.add_argument("--black_search", default="npls", choices=["mcts", "npls", "mc_rzf", "mcts_cpp", "npls_cpp"])
    p.add_argument("--black_mode", default="modelC")
    p.add_argument("--black_weights", default="weights/checkpoint.pth")

    p.add_argument("--white_name", default="GNN-White")
    p.add_argument("--white_search", default="npls", choices=["mcts", "npls", "mc_rzf", "mcts_cpp", "npls_cpp"])
    p.add_argument("--white_mode", default="modelC")
    p.add_argument("--white_weights", default="weights/checkpoint.pth")

    return p.parse_args()


if __name__ == "__main__":
    run_match(_parse_args())
