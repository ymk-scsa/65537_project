import os
import time
import random
import torch
import numpy as np
import cshogi
from game.board import make_move_label, bit447_to_numpy
from model.model import create_model

class MCRZFPlayer:
    """
    MC-RZF (Monte Carlo ResNet Zero Find) 探索実験プレイヤー。
    各合法手からNNのPolicyに従って短いロールアウトを行い、平均Valueが最も高い指し手を選択する。
    """
    def __init__(self, blocks_config_mode="modelC", modelfile=None, name="MC-RZF Engine"):
        self.blocks_config_mode = blocks_config_mode
        self.modelfile = modelfile
        self.name = name
        self.board = cshogi.Board()

        self.rollout_depth = 10
        self.num_simulations = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(num_actions=2187, mode=self.blocks_config_mode)
        self.model.to(self.device)
        self.model.eval()

        if self.modelfile and os.path.exists(self.modelfile):
            try:
                self.model.load_state_dict(torch.load(self.modelfile, map_location=self.device))
                print(f"info string Loaded weights from {self.modelfile}", flush=True)
            except Exception as e:
                print(f"info string Failed to load weights: {e}", flush=True)

        self.think_time_ms = 1000
        self.stop_search = False

    def usi(self):
        print(f"id name {self.name}", flush=True)
        print("id author Antigravity", flush=True)

    def setoption(self, option_args):
        pass

    def isready(self):
        pass

    def position(self, sfen, moves):
        if sfen == "startpos":
            self.board = cshogi.Board()
        else:
            if sfen.startswith("sfen "):
                sfen = sfen[5:].strip()
            self.board.set_sfen(sfen)
        for m in moves:
            move = self.board.move_from_usi(m)
            self.board.push(move)

    def set_limits(self, **kwargs):
        self.think_time_ms = kwargs.get("think_time_ms", 1000)

    def stop(self):
        self.stop_search = True

    def ponderhit(self, limits):
        pass

    def quit(self):
        pass

    def _evaluate(self, board_sfen):
        b = cshogi.Board(sfen=board_sfen)
        feat = bit447_to_numpy(b)
        feat_tensor = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value_tensor = self.model(feat_tensor, return_aux=False)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            value = value_tensor.cpu().numpy()[0][0]

        return policy, value

    def _rollout(self, start_sfen):
        temp_board = cshogi.Board(sfen=start_sfen)
        for _ in range(self.rollout_depth):
            if temp_board.is_game_over():
                break
            # Policyに基づいて指し手を選ぶか、あるいはランダム
            legal_moves = list(temp_board.legal_moves)
            if not legal_moves:
                break
            # 簡易的にランダムに指し手を選ぶ
            move = random.choice(legal_moves)
            temp_board.push(move)

        # 最終局面のValueを取得
        _, val = self._evaluate(temp_board.sfen())
        # 手番が切り替わっている可能性を考慮して、元の手番から見た価値に補正する必要があるが、
        # ここでは簡易評価として、現在のターンに応じて符号を調整
        if temp_board.turn != self.board.turn:
            val = -val
        return val

    def go(self):
        self.stop_search = False
        start_time = time.time()
        time_limit = self.think_time_ms / 1000.0

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return "resign", None

        move_scores = {move: [] for move in legal_moves}

        # 探索ループ
        sim_count = 0
        while not self.stop_search:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                break

            for move in legal_moves:
                if self.stop_search or (time.time() - start_time) >= time_limit:
                    break
                self.board.push(move)
                val = self._rollout(self.board.sfen())
                self.board.pop()
                move_scores[move].append(val)

            sim_count += 1
            if sim_count >= self.num_simulations:
                break

        # 最も平均スコアの高い指し手を選択
        best_move_obj = None
        best_score = -float('inf')
        for move, scores in move_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                if avg > best_score:
                    best_score = avg
                    best_move_obj = move

        if best_move_obj is None:
            best_move_obj = random.choice(legal_moves)

        best_move = cshogi.move_to_usi(best_move_obj)
        print(f"info string MC-RZF simulations: {sim_count}", flush=True)
        return best_move, None
