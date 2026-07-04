"""
agent/main.py
USI（将棋所/ShogiGUI/Shogidokoro）通信エンジンのエントリーポイント。

【大会パラメータ（batファイルから指定可能）】
  --time_rate         持ち時間使用率  (0.0〜1.0, default=0.5)
                      やねうら王の "ShortMover" 相当。
                      1.0 で持ち時間を最大限使い切る積極的運用。
  --draw_value        千日手評価値    (default=0)
                      正値=千日手を有利とみなす / 負値=不利とみなす。
  --impasse_score     持将棋局面の点数計算基準
                      (default=28, 大会規定に合わせる)
  --impasse_value     持将棋評価値    (default=0)
                      正値=持将棋を有利とみなす / 負値=不利とみなす。
  --max_moves         最大手数        (default=320)
                      大会規定に合わせて変更。勝勢でも最大手数内に決めきる
                      必要があり、大劣勢でも最大手数まで粘れば引き分けになる。
  --use_search        探索部を使用するか (1=yes, 0=no, default=1)
                      0 のとき評価関数の出力をそのまま指し手に使う（高速モード）。
  --search_type       探索アルゴリズム (mcts / npls / mc_rzf / ads_ab, default=mcts)
  --mode              モデルアーキテクチャモード (default=30blocks)
  --weights           重みファイルパス (default=weights/checkpoint.pth)
  --name              エンジン名       (default=GNN-Shogi)

【千日手・持将棋の扱い】
  cshogi は is_draw() で千日手・持将棋を検出する。
  - 千日手 (REPETITION_DRAW / REPETITION_WIN / REPETITION_LOSE):
      draw_value を返す。
  - 持将棋 (board.is_nyugyoku()):
      双方の点数が impasse_score を超えていれば入玉宣言できる局面として
      impasse_value を返す。
  - 最大手数到達 (board.move_number >= max_moves):
      引き分け扱いで 0 を返す（残り手数が少なくなると探索時間を伸ばす）。
"""

from __future__ import annotations #「型ヒント（型注釈）」をより便利に、エラーを起こさずに書くため

import sys 
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(__file__))

from search.mcts import MCTSPlayer
from search.npls import NPLSPlayer
from search.mc_rzf import MCRZFPlayer

UsiResponse = tuple[str, str]


# ─────────────────────────────────────────────────────────
# TournamentConfig  ―  大会パラメータをまとめて管理するクラス
# ─────────────────────────────────────────────────────────

class TournamentConfig:
    """
    大会ごとに変わるパラメータを一元管理する。
    Player 側は self.config.xxx で参照する。
    """

    def __init__(
        self,
        time_rate: float = 0.5,
        draw_value: int = 0,
        impasse_score: int = 28,
        impasse_value: int = 0,
        max_moves: int = 320,
        use_search: bool = True,
    ) -> None:
        # ── 持ち時間管理 ──────────────────────────────
        # go コマンドの btime/wtime からこのレートで今回の使用時間を算出する。
        # 0.5 なら残り時間の 50% を 1 手に充てる（やや保守的）。
        # 秒読み (byoyomi) がある場合は byoyomi をそのまま上限にする。
        self.time_rate: float = max(0.01, min(1.0, time_rate))

        # ── 千日手 ────────────────────────────────────
        # Player.go() が千日手を検出したとき返す評価値（cp 単位想定）。
        # 0  : 千日手は互角（標準）
        # +N : 千日手を歓迎する（入玉狙いなど）
        # -N : 千日手を避ける（攻め将棋向け）
        self.draw_value: int = draw_value

        # ── 持将棋 ────────────────────────────────────
        # 入玉宣言のための最低点数（通常 28 点）。大会ルールに合わせる。
        self.impasse_score: int = impasse_score
        # 持将棋局面の評価値（千日手と同様の使い方）。
        self.impasse_value: int = impasse_value

        # ── 最大手数 ──────────────────────────────────
        # 大会規定の最大手数。
        # - 残り手数が少なくなると探索時間を伸ばして確実に決める。
        # - 大劣勢でも最大手数まで粘れば引き分けになる。
        self.max_moves: int = max_moves

        # ── 探索部フラグ ──────────────────────────────
        # False のとき評価関数の policy 出力をそのまま指し手にする。
        # 軽量マシン環境・予選など速度優先時に使う。
        self.use_search: bool = use_search

    # ─── 時間計算ヘルパー ─────────────────────────────────

    def calc_think_time_ms(
        self,
        remaining_ms: int,
        byoyomi_ms: int,
        moves_left_estimate: int = 60,
    ) -> int:
        """
        1 手あたりの思考時間（ミリ秒）を計算して返す。

        Parameters
        ----------
        remaining_ms      : 残り持ち時間 (ms)
        byoyomi_ms        : 秒読み時間 (ms, 0 なら持ち時間のみ)
        moves_left_estimate : 残り想定手数（デフォルト 60 手）
        """
        if byoyomi_ms > 0:
            # 秒読みがある場合: 残り持ち時間を moves_left_estimate で割り
            # byoyomi の分だけ余裕を持たせる
            per_move = int(remaining_ms * self.time_rate / max(moves_left_estimate, 1))
            return max(per_move, byoyomi_ms)
        else:
            # 持ち時間のみ: 残り時間を均等配分してレートを掛ける
            per_move = int(remaining_ms * self.time_rate / max(moves_left_estimate, 1))
            return max(per_move, 100)  # 最低 100ms

    def moves_left_estimate(self, current_move: int) -> int:
        """
        現在手数から残り想定手数を返す。
        最大手数が近づくほど少なく見積もり、探索時間を増やす。
        """
        remaining = self.max_moves - current_move
        # 残り 30 手以内になったら全力で読む
        if remaining <= 30:
            return max(remaining, 1)
        return min(remaining, 60)


# ─────────────────────────────────────────────────────────
# USI 通信ループ
# ─────────────────────────────────────────────────────────

def run(
    player: Union[MCTSPlayer, NPLSPlayer, MCRZFPlayer],
    config: TournamentConfig,
) -> None:
    """
    USI 通信ループ。
    標準入力からコマンドを 1 行ずつ読み込み、対応するメソッドを呼び出す。
    config は player に渡して時間管理・千日手・持将棋・最大手数を制御させる。
    """
    # Player に TournamentConfig を注入する
    # （Player クラスが config 属性を参照できるようにする）
    player.tournament_config = config

    executor = ThreadPoolExecutor(max_workers=1)
    future: Optional[Future] = None
    last_limits: dict = {}

    while True:
        try:
            cmd_line = input().strip()
        except EOFError:
            break

        if not cmd_line:
            continue

        tokens = cmd_line.split(" ", 1)
        cmd = tokens[0]
        rest = tokens[1] if len(tokens) > 1 else ""

        # ── usi ─────────────────────────────────────
        if cmd == "usi":
            player.usi()
            # 大会パラメータを USI オプションとして通知する
            # （将棋所 / ShogiGUI の「エンジン設定」から上書き可能）
            print(f"option name TimeRate type spin default 50 min 1 max 100")
            print(f"option name DrawValue type spin default 0 min -1000 max 1000")
            print(f"option name ImpasseScore type spin default 28 min 1 max 38")
            print(f"option name ImpasseValue type spin default 0 min -1000 max 1000")
            print(f"option name MaxMoves type spin default 320 min 1 max 9999")
            print(f"option name UseSearch type check default true")
            print("usiok", flush=True)

        # ── setoption ───────────────────────────────
        elif cmd == "setoption":
            # setoption name <Name> value <Value>
            parts = rest.split()
            if len(parts) >= 4 and parts[0] == "name" and parts[2] == "value":
                opt_name = parts[1]
                opt_val = parts[3]
                _apply_option(config, opt_name, opt_val)
            player.setoption(rest.split())

        # ── isready ─────────────────────────────────
        elif cmd == "isready":
            player.isready()
            print("readyok", flush=True)

        # ── usinewgame ──────────────────────────────
        elif cmd == "usinewgame":
            pass  # 新局の初期化は isready 時に済んでいる

        # ── position ────────────────────────────────
        elif cmd == "position":
            args = rest.split("moves")
            sfen = args[0].strip()
            usi_moves = args[1].split() if len(args) > 1 else []
            player.position(sfen, usi_moves)

        # ── go ──────────────────────────────────────
        elif cmd == "go":
            kwargs: dict = {}
            if rest:
                args = rest.split()
                if args[0] == "infinite":
                    kwargs["infinite"] = True
                else:
                    idx = 0
                    if args[0] == "ponder":
                        kwargs["ponder"] = True
                        idx = 1
                    while idx < len(args) - 1:
                        key = args[idx]
                        val = args[idx + 1]
                        if key in ("btime", "wtime", "byoyomi", "binc", "winc", "nodes"):
                            kwargs[key] = int(val)
                        idx += 2

            # 持ち時間から今回の思考時間を計算して player に渡す
            _inject_think_time(player, config, kwargs)

            player.set_limits(**kwargs)
            last_limits = dict(kwargs)
            need_print_bestmove = "ponder" not in kwargs and "infinite" not in kwargs

            def go_and_print() -> UsiResponse:
                bestmove, ponder_move = player.go()
                if need_print_bestmove:
                    reply = "bestmove " + bestmove
                    if ponder_move:
                        reply += " ponder " + ponder_move
                    print(reply, flush=True)
                return bestmove, ponder_move

            future = executor.submit(go_and_print)

        # ── stop ────────────────────────────────────
        elif cmd == "stop":
            player.stop()
            if future is not None:
                bestmove, _ = future.result()
                print("bestmove " + bestmove, flush=True)
                future = None

        # ── ponderhit ───────────────────────────────
        elif cmd == "ponderhit":
            last_limits.pop("ponder", None)
            player.ponderhit(last_limits)
            if future is not None:
                bestmove, ponder_move = future.result()
                reply = "bestmove " + bestmove
                if ponder_move:
                    reply += " ponder " + ponder_move
                print(reply, flush=True)
                future = None

        # ── quit ────────────────────────────────────
        elif cmd == "quit":
            player.quit()
            if future is not None:
                future.result()
            break


# ─────────────────────────────────────────────────────────
# ヘルパー関数
# ─────────────────────────────────────────────────────────

def _apply_option(config: TournamentConfig, name: str, value: str) -> None:
    """setoption で受け取った値を TournamentConfig に反映する。"""
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
    except ValueError:
        pass  # 不正な値は無視


def _inject_think_time(
    player,
    config: TournamentConfig,
    kwargs: dict,
) -> None:
    """
    go コマンドの時間情報から今回の思考時間を計算して
    kwargs["think_time_ms"] に追記する。
    Player 側で set_limits(**kwargs) から参照できる。
    """
    import cshogi
    # player が board を持っている場合は手数を取得する
    current_move = 0
    try:
        current_move = player.board.move_number
    except AttributeError:
        pass

    # 手番に応じて残り時間を取得
    try:
        turn = player.board.turn
    except AttributeError:
        turn = cshogi.BLACK

    remaining_ms = kwargs.get("btime" if turn == cshogi.BLACK else "wtime", 0)
    byoyomi_ms   = kwargs.get("byoyomi", 0)
    moves_left   = config.moves_left_estimate(current_move)

    kwargs["think_time_ms"] = config.calc_think_time_ms(
        remaining_ms, byoyomi_ms, moves_left
    )


# ─────────────────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GNN将棋AIのUSIエンジン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── モデル・探索 ──────────────────────────────────
    p.add_argument("--mode",        type=str,   default="modelC",
                   help="モデルアーキテクチャモード")
    p.add_argument("--weights",     type=str,   default="weights/checkpoint.pth",
                   help="重みファイルのパス")
    p.add_argument("--name",        type=str,   default="GNN-Shogi",
                   help="USIエンジン名")
    p.add_argument("--search_type", type=str,   default="npls",
                   choices=["mcts", "npls", "mc_rzf"],
                   help="探索アルゴリズム")

    # ── 大会パラメータ ────────────────────────────────
    p.add_argument("--time_rate",     type=float, default=0.5,
                   help="持ち時間使用率 (0.01〜1.0)。"
                        "やねうら王の ShortMover 相当。")
    p.add_argument("--draw_value",    type=int,   default=0,
                   help="千日手の評価値 (cp単位)。"
                        "正=千日手OK / 負=千日手NG。")
    p.add_argument("--impasse_score", type=int,   default=28,
                   help="持将棋の入玉宣言に必要な最低点数。大会規定に合わせる。")
    p.add_argument("--impasse_value", type=int,   default=0,
                   help="持将棋局面の評価値 (cp単位)。")
    p.add_argument("--max_moves",     type=int,   default=320,
                   help="最大手数。大会規定に合わせる。")
    p.add_argument("--use_search",    type=int,   default=1,
                   choices=[0, 1],
                   help="探索部の有無。0=評価関数のみ / 1=探索あり。")

    return p.parse_args()


def _build_player(args: argparse.Namespace) -> Union[MCTSPlayer, NPLSPlayer, MCRZFPlayer]:
    """引数に応じた Player インスタンスを生成して返す。"""
    common = dict(
        blocks_config_mode=args.mode,
        modelfile=args.weights,
    )
    if args.search_type == "npls":
        return NPLSPlayer(**common, name=args.name + " (NPLS)")
    elif args.search_type == "mc_rzf":
        return MCRZFPlayer(**common, name=args.name + " (MC-RZF)")
    else:
        return MCTSPlayer(**common, name=args.name)


if __name__ == "__main__":
    args = _parse_args()

    config = TournamentConfig(
        time_rate     = args.time_rate,
        draw_value    = args.draw_value,
        impasse_score = args.impasse_score,
        impasse_value = args.impasse_value,
        max_moves     = args.max_moves,
        use_search    = bool(args.use_search),
    )

    player = _build_player(args)
    run(player, config)
