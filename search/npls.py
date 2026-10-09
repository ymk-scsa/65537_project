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
        feat = bit447_to_numpy(b)  # shape (447,)
        feat_tensor = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 447)

        with torch.no_grad():
            if getattr(self.model, "uses_turn", False):
                # apple 系モデル: 手番別ヘッドを使うので手番を渡す
                turn_t = torch.tensor([b.turn], dtype=torch.long, device=self.device)
                policy_logits, value_tensor = self.model(feat_tensor, turn=turn_t)
            else:
                policy_logits, value_tensor = self.model(feat_tensor, return_aux=False)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            value = value_tensor.cpu().numpy()[0][0]

        return policy, value

    def go(self):
        self.stop_search = False
        start_time = time.time()
        # わずかに短く切り、GUI 秒読み切れより先に返す
        time_limit = max(self.think_time_ms / 1000.0, 0.05)

        root_sfen = self.board.sfen()
        root = NPLSNode(root_sfen)

        legal = list(self.board.legal_moves)
        if not legal:
            return "resign", None

        try:
            policy, value = self._evaluate(root_sfen)
        except Exception as exc:
            print(f"info string eval error: {exc}", flush=True)
            return cshogi.move_to_usi(legal[0]), None
        root.v = value

        queue = []
        temp_board = cshogi.Board(sfen=root_sfen)
        for move in legal:
            usi_move = cshogi.move_to_usi(move)
            move_label = make_move_label(move, temp_board.turn)
            p_val = float(policy[move_label]) if 0 <= move_label < len(policy) else 0.0

            temp_board.push(move)
            child_sfen = temp_board.sfen()
            temp_board.pop()

            child = NPLSNode(child_sfen, move=usi_move, parent=root, p=p_val)
            child.priority = -(self.w1 * p_val + self.w2 * 0.0 + self.w3 * 1.0)
            root.children[usi_move] = child
            heapq.heappush(queue, child)

        root.n = 1

        nodes_searched = 0
        while queue and not self.stop_search:
            if (time.time() - start_time) >= time_limit:
                break

            node = heapq.heappop(queue)
            node_board = cshogi.Board(sfen=node.sfen)
            if node_board.is_game_over():
                # cshogi の is_game_over() は「手番側に合法手がない」= 手番側の負け
                # （Board.result() は cshogi に存在しないため使わない）
                node.v = -1.0
            else:
                if (time.time() - start_time) >= time_limit or self.stop_search:
                    break
                try:
                    p_arr, val = self._evaluate(node.sfen)
                except Exception as exc:
                    print(f"info string eval error: {exc}", flush=True)
                    break
                node.v = val

                for move in node_board.legal_moves:
                    if self.stop_search or (time.time() - start_time) >= time_limit:
                        break
                    usi_move = cshogi.move_to_usi(move)
                    move_label = make_move_label(move, node_board.turn)
                    p_val = float(p_arr[move_label]) if 0 <= move_label < len(p_arr) else 0.0

                    node_board.push(move)
                    child_sfen = node_board.sfen()
                    node_board.pop()

                    child = NPLSNode(child_sfen, move=usi_move, parent=node, p=p_val)
                    child.priority = -(
                        self.w1 * p_val + self.w2 * val + self.w3 * (1.0 / (node.n + 1))
                    )
                    node.children[usi_move] = child
                    heapq.heappush(queue, child)

            curr = node
            while curr is not None:
                curr.n += 1
                curr = curr.parent

            nodes_searched += 1

        # 訪問が無いときは policy 最大手を選ぶ（resign 事故防止）
        best_move = None
        best_child = None
        max_n = -1
        max_p = -1.0
        for usi_move, child in root.children.items():
            if child.n > max_n or (child.n == max_n and child.p > max_p):
                max_n = child.n
                max_p = child.p
                best_move = usi_move
                best_child = child
        if best_move is None:
            best_move = cshogi.move_to_usi(legal[0])

        print(f"info string Searched {nodes_searched} nodes", flush=True)

        ponder_move = None
        if best_child and best_child.children:
            max_child_n = -1
            for opp_move, opp_child in best_child.children.items():
                if opp_child.n > max_child_n:
                    max_child_n = opp_child.n
                    ponder_move = opp_move

        return best_move, ponder_move
