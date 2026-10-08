import os
import time
import math
import torch
import numpy as np
import cshogi
from game.board import make_move_label, bit447_to_numpy
from model.model import create_model

class MCTSNode:
    def __init__(self, board_sfen, move=None, parent=None, p=0.0):
        self.sfen = board_sfen
        self.move = move
        self.parent = parent
        self.children = {}  # usi_move -> MCTSNode
        self.n = 0
        self.w = 0.0
        self.q = 0.0
        self.p = p

    def puct(self, c_puct=1.5):
        if not self.parent:
            return 0.0
        # PUCT式
        return self.q + c_puct * self.p * (math.sqrt(self.parent.n) / (1 + self.n))

class MCTSPlayer:
    def __init__(self, blocks_config_mode="modelC", modelfile=None, name="MCTS Engine"):
        self.blocks_config_mode = blocks_config_mode
        self.modelfile = modelfile
        self.name = name
        self.board = cshogi.Board()

        self.c_puct = 1.5

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

    def go(self):
        self.stop_search = False
        start_time = time.time()
        time_limit = self.think_time_ms / 1000.0

        root_sfen = self.board.sfen()
        root = MCTSNode(root_sfen)

        # 探索ループ
        nodes_searched = 0
        while not self.stop_search:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                break

            # 1. Selection
            curr = root
            temp_board = cshogi.Board(sfen=root_sfen)

            while curr.children:
                # PUCT値が最大のものを選ぶ
                best_move = None
                best_val = -float('inf')
                for move_usi, child in curr.children.items():
                    val = child.puct(self.c_puct)
                    if val > best_val:
                        best_val = val
                        best_move = move_usi

                curr = curr.children[best_move]
                temp_board.push(temp_board.move_from_usi(best_move))

            # 2. Expansion & Evaluation
            if temp_board.is_game_over():
                # cshogi の is_game_over() は「手番側に合法手がない」= 手番側の負け
                # （Board.result() は cshogi に存在しないため使わない）
                value = -1.0
            else:
                policy, value = self._evaluate(temp_board.sfen())
                # 子供を展開
                for move in temp_board.legal_moves:
                    usi_move = cshogi.move_to_usi(move)
                    move_label = make_move_label(move, temp_board.turn)
                    p_val = policy[move_label]

                    temp_board.push(move)
                    child_sfen = temp_board.sfen()
                    temp_board.pop()

                    curr.children[usi_move] = MCTSNode(child_sfen, move=usi_move, parent=curr, p=p_val)

            # 3. Backpropagation
            v = value
            node = curr
            while node is not None:
                node.n += 1
                node.w += v
                node.q = node.w / node.n
                v = -v  # 手番が交代するため価値を反転
                node = node.parent

            nodes_searched += 1

        best_move = "resign"
        max_n = -1
        best_child = None

        print(f"info string Searched {nodes_searched} nodes", flush=True)

        for usi_move, child in root.children.items():
            if child.n > max_n:
                max_n = child.n
                best_move = usi_move
                best_child = child

        ponder_move = None
        if best_child and best_child.children:
            max_child_n = -1
            for opp_move, opp_child in best_child.children.items():
                if opp_child.n > max_child_n:
                    max_child_n = opp_child.n
                    ponder_move = opp_move

        return best_move, ponder_move
