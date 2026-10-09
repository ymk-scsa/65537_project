"""
search/cpp_player.py
C++ 版の探索（search/cpp）を使うプレイヤー。

  - 木の探索・盤面・特徴量作成は C++（apple_search_cpp モジュール）
  - NN の推論は PyTorch で、葉の局面をまとめて（バッチで）評価する

Python 版（search/mcts.py, search/npls.py）と同じメソッドを持つので、usi.py からそのまま使える。
C++ モジュールは事前に `bash search/cpp/build.sh` でビルドしておく。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cshogi
import numpy as np
import warnings

import torch

# torch.jit は新しい PyTorch で「非推奨」の警告が出るが、まだ動くので表示しない
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_HERE, "cpp")
if _CPP_DIR not in sys.path:
    sys.path.insert(0, _CPP_DIR)

try:
    import apple_search_cpp as _cpp  # noqa: E402
except ImportError as exc:  # pragma: no cover - ビルドしていない場合
    _cpp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from model.model import create_model  # noqa: E402


def cpp_available() -> bool:
    return _cpp is not None


def _value_to_cp(v: float) -> int:
    """-1〜+1 の評価を、目安の評価値（cp）に直す（info 表示用）。"""
    p = min(max((v + 1.0) * 0.5, 1e-6), 1 - 1e-6)
    return int(round(-600.0 * np.log(1.0 / p - 1.0)))


class CppSearchPlayer:
    """C++ 探索 + PyTorch 推論のプレイヤー。algo は "mcts" か "npls"。"""

    def __init__(
        self,
        algo: str = "mcts",
        blocks_config_mode: str = "appleS",
        modelfile: Optional[str] = None,
        name: str = "Apple Engine",
        batch_size: int = 32,
        threads: int = 0,
        draw_value_black: float = 0.4,
        draw_value_white: float = 0.6,
        use_jit: bool = True,
    ) -> None:
        if _cpp is None:
            raise ImportError(
                "C++ 探索モジュールが見つかりません。`bash search/cpp/build.sh` でビルドしてください。"
                f"（{_IMPORT_ERROR}）"
            )
        self.algo = algo
        self.name = name
        self.board = cshogi.Board()  # usi.py が手番・手数を見るために使う
        self.think_time_ms = 1000
        self.node_limit = 0
        self.tournament_config = None
        self.draw_value_black = draw_value_black
        self.draw_value_white = draw_value_white

        if threads > 0:
            torch.set_num_threads(threads)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(num_actions=2187, mode=blocks_config_mode)
        if modelfile and os.path.exists(modelfile):
            try:
                state = torch.load(modelfile, map_location=self.device)
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                self.model.load_state_dict(state)
                print(f"info string Loaded weights from {modelfile}", flush=True)
            except Exception as exc:
                print(f"info string Failed to load weights: {exc}", flush=True)
        self.model.to(self.device).eval()
        if hasattr(self.model, "fuse_for_inference"):
            self.model.fuse_for_inference()  # BatchNorm を畳み込みに吸収（推論が少し速くなる）
        self.uses_turn = bool(getattr(self.model, "uses_turn", False))
        if self.uses_turn:
            self.model.set_draw_values(draw_value_black, draw_value_white)
        self._forward = self._build_forward(use_jit)

        self.searcher = _cpp.Searcher(self._evaluate, algo)
        cfg = self.searcher.config
        cfg.batch_size = batch_size
        cfg.draw_value_black = draw_value_black
        cfg.draw_value_white = draw_value_white

    # ── 推論 ────────────────────────────────────────────
    def _build_forward(self, use_jit: bool):
        model = self.model

        if self.uses_turn:
            def fwd(x, t):
                return model(x, turn=t)
        else:
            def fwd(x, t):
                return model(x, return_aux=False)

        if not use_jit or self.device.type != "cpu":
            return fwd
        # TorchScript にすると Python の呼び出しが減って少し速くなる（失敗したら通常版）
        try:
            ex_x = torch.zeros(4, 447)
            ex_t = torch.zeros(4, dtype=torch.long)

            class _Wrap(torch.nn.Module):
                def __init__(self, m, uses_turn):
                    super().__init__()
                    self.m = m
                    self.uses_turn = uses_turn

                def forward(self, x, t):
                    if self.uses_turn:
                        return self.m(x, t)
                    return self.m(x)

            with torch.inference_mode():
                traced = torch.jit.trace(_Wrap(model, self.uses_turn).eval(), (ex_x, ex_t), check_trace=False)
                traced = torch.jit.freeze(traced)
                traced(ex_x, ex_t)
            return traced
        except Exception as exc:  # pragma: no cover
            print(f"info string jit disabled: {exc}", flush=True)
            return fwd

    def _evaluate(self, feats: np.ndarray, turns: np.ndarray):
        n = feats.shape[0]
        # バッチの大きさを 2 のべき乗にそろえる（形が毎回変わると JIT の最適化がやり直しになるため）
        m = 1 << max(n - 1, 0).bit_length()
        if m != n:
            feats = np.concatenate([feats, np.zeros((m - n, feats.shape[1]), dtype=feats.dtype)])
            turns = np.concatenate([turns, np.zeros(m - n, dtype=turns.dtype)])
        with torch.inference_mode():
            x = torch.from_numpy(feats).to(self.device)
            t = torch.from_numpy(turns).to(self.device)
            policy, value = self._forward(x, t)
            return (
                policy[:n].float().cpu().numpy(),
                value.float().reshape(-1)[:n].cpu().numpy(),
            )

    # ── USI 用のメソッド（Python 版と同じ形） ────────────
    def usi(self):
        print(f"id name {self.name}", flush=True)
        print("id author 65537_project", flush=True)

    def setoption(self, option_args):
        pass

    def isready(self):
        # 最初の推論は遅いので、ここで一度動かしておく
        self._evaluate(np.zeros((1, 447), dtype=np.float32), np.zeros(1, dtype=np.int64))

    def position(self, sfen, moves):
        if sfen == "startpos":
            self.board = cshogi.Board()
            spec = "startpos"
        else:
            if sfen.startswith("sfen "):
                sfen = sfen[5:].strip()
            self.board.set_sfen(sfen)
            spec = sfen
        for m in moves:
            self.board.push(self.board.move_from_usi(m))
        if not self.searcher.set_position(spec, list(moves)):
            print("info string C++ position error", flush=True)

    def set_limits(self, **kwargs):
        self.think_time_ms = kwargs.get("think_time_ms", 1000)
        self.node_limit = int(kwargs.get("nodes", 0) or 0)

    def stop(self):
        self.searcher.stop()

    def ponderhit(self, limits):
        pass

    def quit(self):
        self.searcher.stop()

    def go(self):
        cfg = self.searcher.config
        tc = self.tournament_config
        if tc is not None:
            cfg.max_moves = int(getattr(tc, "max_moves", cfg.max_moves))
        res = self.searcher.go(float(max(self.think_time_ms, 1)), self.node_limit)
        t = max(res["time_ms"], 1e-3)
        nodes = res["evaluated"]
        pv = " ".join(res["pv"])
        if res["reason"] == "mate":
            print(f"info score mate + pv {pv}", flush=True)
        elif res["reason"] == "search":
            print(
                f"info nodes {nodes} nps {int(nodes * 1000 / t)} time {int(t)} "
                f"score cp {_value_to_cp(res['value'])} pv {pv}",
                flush=True,
            )
        print(
            f"info string {self.algo}_cpp {res['reason']} playouts={res['playouts']} "
            f"evaluated={nodes} collisions={res['collisions']} time={int(t)}ms",
            flush=True,
        )
        return res["bestmove"], (res["ponder"] or None)


class MCTSCppPlayer(CppSearchPlayer):
    def __init__(self, **kw):
        kw.setdefault("name", "Apple MCTS (C++)")
        super().__init__(algo="mcts", **kw)


class NPLSCppPlayer(CppSearchPlayer):
    def __init__(self, **kw):
        kw.setdefault("name", "Apple NPLS (C++)")
        super().__init__(algo="npls", **kw)
