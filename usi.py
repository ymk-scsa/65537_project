"""
usi.py
将棋所 / ShogiGUI / Shogidokoro 向け USI エンジン入口。

標準入力・標準出力のみで USI プロトコルを話す。
登録例: bat/usi.bat をエンジンとして指定する。
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search.mcts import MCTSPlayer
from search.mc_rzf import MCRZFPlayer
from search.npls import NPLSPlayer

Player = Union[MCTSPlayer, NPLSPlayer, MCRZFPlayer]
UsiResponse = tuple[str, Optional[str]]


class TournamentConfig:
    """大会・GUI から変える時間・引き分け関連パラメータ。"""

    def __init__(
        self,
        time_rate: float = 0.5,
        draw_value: int = 0,
        impasse_score: int = 28,
        impasse_value: int = 0,
        max_moves: int = 320,
        use_search: bool = True,
    ) -> None:
        self.time_rate = max(0.01, min(1.0, time_rate))
        self.draw_value = draw_value
        self.impasse_score = impasse_score
        self.impasse_value = impasse_value
        self.max_moves = max_moves
        self.use_search = use_search

    def calc_think_time_ms(
        self,
        remaining_ms: int,
        byoyomi_ms: int,
        moves_left_estimate: int = 60,
    ) -> int:
        if byoyomi_ms > 0:
            per_move = int(remaining_ms * self.time_rate / max(moves_left_estimate, 1))
            return max(per_move, byoyomi_ms)
        per_move = int(remaining_ms * self.time_rate / max(moves_left_estimate, 1))
        return max(per_move, 100)

    def moves_left_estimate(self, current_move: int) -> int:
        remaining = self.max_moves - current_move
        if remaining <= 30:
            return max(remaining, 1)
        return min(remaining, 60)


def _usi_print(line: str) -> None:
    """USI 応答は必ず flush する。"""
    print(line, flush=True)


def _normalize_bestmove(move: Optional[str]) -> str:
    if not move or move in ("resnet", "pass"):
        return "resign"
    return move


def _parse_setoption(rest: str) -> tuple[Optional[str], Optional[str]]:
    """
    setoption name <id> [value <x>]
    名前に空白が含まれる場合にも対応する。
    """
    parts = rest.split()
    if len(parts) < 2 or parts[0] != "name":
        return None, None
    if "value" in parts:
        vi = parts.index("value")
        name = " ".join(parts[1:vi])
        value = " ".join(parts[vi + 1 :])
        return name, value
    return " ".join(parts[1:]), ""


def _parse_position(rest: str) -> tuple[str, list[str]]:
    """
    position startpos [moves ...]
    position sfen <sfen...> [moves ...]
    を (board_spec, moves) に分解する。
    board_spec は "startpos" または SFEN 本体（"sfen " なし）。
    """
    parts = rest.split()
    if not parts:
        return "startpos", []

    moves: list[str] = []
    if "moves" in parts:
        mi = parts.index("moves")
        pos_tokens = parts[:mi]
        moves = parts[mi + 1 :]
    else:
        pos_tokens = parts

    if not pos_tokens:
        return "startpos", moves

    if pos_tokens[0] == "startpos":
        return "startpos", moves

    if pos_tokens[0] == "sfen":
        return " ".join(pos_tokens[1:]), moves

    # すでに SFEN 本体だけが来た場合
    return " ".join(pos_tokens), moves


def _parse_go(rest: str) -> dict:
    kwargs: dict = {}
    if not rest:
        return kwargs
    args = rest.split()
    if not args:
        return kwargs

    if args[0] == "infinite":
        kwargs["infinite"] = True
        return kwargs

    idx = 0
    if args[0] == "ponder":
        kwargs["ponder"] = True
        idx = 1
    if idx < len(args) and args[idx] == "mate":
        kwargs["mate"] = True
        idx += 1

    while idx < len(args):
        key = args[idx]
        if key in ("btime", "wtime", "byoyomi", "binc", "winc", "nodes", "movetime"):
            if idx + 1 >= len(args):
                break
            kwargs[key] = int(args[idx + 1])
            idx += 2
        else:
            idx += 1
    return kwargs


def _apply_option(config: TournamentConfig, name: str, value: str) -> None:
    try:
        if name == "TimeRate":
            config.time_rate = int(value) / 100.0
        elif name == "DrawValue":
            config.draw_value = int(value)
        elif name == "ImpasseScore":
            config.impasse_score = int(value)
        elif name == "ImpasseValue":
            config.impasse_value = int(value)
        elif name == "MaxMoves":
            config.max_moves = int(value)
        elif name == "UseSearch":
            config.use_search = value.lower() == "true"
        elif name == "USI_Ponder":
            # GUI 互換。探索本体は未対応でも受信はする。
            pass
    except ValueError:
        pass


def _inject_think_time(player: Player, config: TournamentConfig, kwargs: dict) -> None:
    import cshogi

    # GUI 側タイムアウトより先に bestmove を返すため余白を残す
    SAFETY_MS = 400

    if "movetime" in kwargs:
        mt = max(int(kwargs["movetime"]), 1)
        kwargs["think_time_ms"] = max(mt - SAFETY_MS, min(mt, 100))
        return
    if kwargs.get("infinite") or kwargs.get("mate"):
        kwargs["think_time_ms"] = 24 * 60 * 60 * 1000
        return

    current_move = 0
    try:
        current_move = player.board.move_number
    except AttributeError:
        pass

    try:
        turn = player.board.turn
    except AttributeError:
        turn = cshogi.BLACK

    remaining_ms = kwargs.get("btime" if turn == cshogi.BLACK else "wtime", 0)
    byoyomi_ms = kwargs.get("byoyomi", 0)
    remaining_ms += kwargs.get("binc" if turn == cshogi.BLACK else "winc", 0)

    moves_left = config.moves_left_estimate(current_move)
    think = config.calc_think_time_ms(remaining_ms, byoyomi_ms, moves_left)
    if byoyomi_ms > 0:
        # 秒読み対局では GUI が厳格に切るため、byoyomi 自体からも余白を取る
        think = min(think, max(byoyomi_ms - SAFETY_MS, 100))
    kwargs["think_time_ms"] = max(int(think), 50)


def _fallback_move(player: Player) -> str:
    """探索失敗時でも合法手を返す。"""
    try:
        moves = list(player.board.legal_moves)
        if moves:
            import cshogi

            return cshogi.move_to_usi(moves[0])
    except Exception:
        pass
    return "resign"


def run(player: Player, config: TournamentConfig) -> None:
    """USI メインループ。stdout にはプロトコル行のみを出す。"""
    import threading

    player.tournament_config = config

    executor = ThreadPoolExecutor(max_workers=1)
    future: Optional[Future] = None
    last_limits: dict = {}
    bestmove_lock = threading.Lock()
    bestmove_sent = False

    def _emit_bestmove(bestmove: str, ponder_move: Optional[str] = None) -> None:
        nonlocal bestmove_sent
        with bestmove_lock:
            if bestmove_sent:
                return
            reply = "bestmove " + _normalize_bestmove(bestmove)
            if ponder_move:
                reply += " ponder " + ponder_move
            _usi_print(reply)
            bestmove_sent = True

    def _cancel_search() -> None:
        nonlocal future, bestmove_sent
        if future is None:
            return
        player.stop()
        try:
            future.result(timeout=5.0)
        except Exception:
            pass
        future = None

    while True:
        try:
            cmd_line = sys.stdin.readline()
        except Exception:
            break
        if cmd_line == "":
            break
        cmd_line = cmd_line.strip()
        if not cmd_line:
            continue

        tokens = cmd_line.split(" ", 1)
        cmd = tokens[0]
        rest = tokens[1] if len(tokens) > 1 else ""

        if cmd == "usi":
            player.usi()
            _usi_print("option name TimeRate type spin default 50 min 1 max 100")
            _usi_print("option name DrawValue type spin default 0 min -1000 max 1000")
            _usi_print("option name ImpasseScore type spin default 28 min 1 max 38")
            _usi_print("option name ImpasseValue type spin default 0 min -1000 max 1000")
            _usi_print("option name MaxMoves type spin default 320 min 1 max 9999")
            _usi_print("option name UseSearch type check default true")
            _usi_print("option name USI_Ponder type check default false")
            _usi_print("usiok")

        elif cmd == "setoption":
            opt_name, opt_val = _parse_setoption(rest)
            if opt_name is not None and opt_val is not None:
                _apply_option(config, opt_name, opt_val)
            player.setoption(rest.split())

        elif cmd == "isready":
            _cancel_search()
            player.isready()
            _usi_print("readyok")

        elif cmd == "usinewgame":
            _cancel_search()

        elif cmd == "position":
            # 探索中の盤面更新は競合するため、先に止める
            _cancel_search()
            board_spec, usi_moves = _parse_position(rest)
            player.position(board_spec, usi_moves)

        elif cmd == "go":
            _cancel_search()
            kwargs = _parse_go(rest)
            _inject_think_time(player, config, kwargs)
            player.set_limits(**kwargs)
            last_limits = dict(kwargs)
            need_print_bestmove = "ponder" not in kwargs and "infinite" not in kwargs
            bestmove_sent = False

            def go_and_print() -> UsiResponse:
                try:
                    bestmove, ponder_move = player.go()
                    bestmove = _normalize_bestmove(bestmove)
                except Exception as exc:
                    _usi_print(f"info string search error: {exc}")
                    bestmove, ponder_move = _fallback_move(player), None
                if need_print_bestmove:
                    _emit_bestmove(bestmove, None)  # ponder は既定オフ（二重応答防止）
                return bestmove, ponder_move

            future = executor.submit(go_and_print)

        elif cmd == "stop":
            player.stop()
            if future is not None:
                try:
                    bestmove, ponder_move = future.result(timeout=5.0)
                except Exception as exc:
                    _usi_print(f"info string stop error: {exc}")
                    bestmove, ponder_move = _fallback_move(player), None
                _emit_bestmove(bestmove, None)
                future = None

        elif cmd == "ponderhit":
            last_limits.pop("ponder", None)
            player.ponderhit(last_limits)
            if future is not None:
                try:
                    bestmove, ponder_move = future.result(timeout=5.0)
                except Exception as exc:
                    _usi_print(f"info string ponderhit error: {exc}")
                    bestmove, ponder_move = _fallback_move(player), None
                _emit_bestmove(bestmove, None)
                future = None

        elif cmd == "gameover":
            _cancel_search()

        elif cmd == "quit":
            _cancel_search()
            player.quit()
            break


def build_player(args: argparse.Namespace) -> Player:
    common = dict(
        blocks_config_mode=args.mode,
        modelfile=args.weights,
    )
    if args.search_type == "npls":
        return NPLSPlayer(**common, name=args.name + " (NPLS)")
    if args.search_type == "mc_rzf":
        return MCRZFPlayer(**common, name=args.name + " (MC-RZF)")
    return MCTSPlayer(**common, name=args.name)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GNN-Shogi USI engine for ShogiGUI / Shogidokoro",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", type=str, default="modelC")
    p.add_argument("--weights", type=str, default="weights/checkpoint.pth")
    p.add_argument("--name", type=str, default="GNN-Shogi")
    p.add_argument("--search_type", type=str, default="npls", choices=["mcts", "npls", "mc_rzf"])
    p.add_argument("--time_rate", type=float, default=0.5)
    p.add_argument("--draw_value", type=int, default=0)
    p.add_argument("--impasse_score", type=int, default=28)
    p.add_argument("--impasse_value", type=int, default=0)
    p.add_argument("--max_moves", type=int, default=320)
    p.add_argument("--use_search", type=int, default=1, choices=[0, 1])
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    # GUI 連携では行バッファリングを避ける
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stdin.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    args = parse_args(argv)
    config = TournamentConfig(
        time_rate=args.time_rate,
        draw_value=args.draw_value,
        impasse_score=args.impasse_score,
        impasse_value=args.impasse_value,
        max_moves=args.max_moves,
        use_search=bool(args.use_search),
    )
    player = build_player(args)
    run(player, config)


if __name__ == "__main__":
    main()
