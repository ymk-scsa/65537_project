import os
import time
import heapq
import torch
import numpy as np
import cshogi
from game.board import make_move_label, bit447_to_numpy
from model.model import create_model

class NPLSNode:
    def __init__(self, board_sfen, move=None, parent=None, p=0.0):
        self.sfen = board_sfen
        self.move = move  # このノードに至った指し手 (usi文字列)
        self.parent = parent
        self.children = {}  # usi_move -> NPLSNode
        self.n = 0  # 訪問回数
        self.v = 0.0  # NNのValue
        self.p = p  # NNのPolicy確率
        self.priority = 0.0  # heapqでの優先度

    def __lt__(self, other):
        # heapq用。優先度が小さい（＝最有望）方が先に出る
        return self.priority < other.priority

class NPLSPlayer:
    def __init__(self, blocks_config_mode="modelC", modelfile=None, name="NPLS Engine"):
        self.blocks_config_mode = blocks_config_mode
        self.modelfile = modelfile
        self.name = name
        self.board = cshogi.Board()

        # ハイパーパラメータ (SPSAで最適化される)
        self.w1 = 1.0  # Policyの重み
        self.w2 = 0.5  # Valueの重み
        self.w3 = 0.1  # 不確実性(1/(N(p)+1))の重み

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
        # オプション設定
        pass

    def isready(self):
        pass

    def position(self, sfen, moves):
        if sfen == "startpos":
            self.board = cshogi.Board()
        else:
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
        feat = bit447_to_numpy(b)  # shape (447,)
        feat_tensor = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 447)

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
        root = NPLSNode(root_sfen)

        # ルートのNN評価
        policy, value = self._evaluate(root_sfen)
        root.v = value

        # キューの初期化
        queue = []

        # ルートの子ノードを作成してキューへ
        temp_board = cshogi.Board(sfen=root_sfen)
        for move in temp_board.legal_moves:
            usi_move = cshogi.move_to_usi(move)
            move_label = make_move_label(move, temp_board.turn)
            p_val = policy[move_label]

            # 子局面の評価 (初期状態では親のValueを推定として使う)
            temp_board.push(move)
            child_sfen = temp_board.sfen()
            temp_board.pop()

            child = NPLSNode(child_sfen, move=usi_move, parent=root, p=p_val)
            # 初期優先度の計算
            child.priority = - (self.w1 * p_val + self.w2 * 0.0 + self.w3 * 1.0)
            root.children[usi_move] = child
            heapq.heappush(queue, child)

        root.n = 1

        # 探索ループ
        nodes_searched = 0
        while queue and not self.stop_search:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                break

            node = heapq.heappop(queue)
            node_board = cshogi.Board(sfen=node.sfen)
            if node_board.is_game_over():
                res = node_board.result()
                if res == "1-0":
                    node.v = 1.0 if node_board.turn == cshogi.BLACK else -1.0
                elif res == "0-1":
                    node.v = -1.0 if node_board.turn == cshogi.BLACK else 1.0
                else:
                    node.v = 0.0
            else:
                p_arr, val = self._evaluate(node.sfen)
                node.v = val

                for move in node_board.legal_moves:
                    usi_move = cshogi.move_to_usi(move)
                    move_label = make_move_label(move, node_board.turn)
                    p_val = p_arr[move_label]

                    node_board.push(move)
                    child_sfen = node_board.sfen()
                    node_board.pop()

                    child = NPLSNode(child_sfen, move=usi_move, parent=node, p=p_val)
                    # 優先度計算
                    child.priority = - (self.w1 * p_val + self.w2 * val + self.w3 * (1.0 / (node.n + 1)))
                    node.children[usi_move] = child
                    heapq.heappush(queue, child)

            # 逆伝搬の排除: 訪問回数のみ伝搬
            curr = node
            while curr is not None:
                curr.n += 1
                curr = curr.parent

            nodes_searched += 1

        best_move = "resnet"
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
