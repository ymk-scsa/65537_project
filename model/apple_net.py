"""
model/apple_net.py
RayGraph ハイブリッドネット（apple 系モデル）

【ねらい】
これまでのモデルは「CNN / GNN / Transformer を順番に積む」形だったが、
ここでは将棋の駒の動き方そのものを、ネットワークの構造として組み込む。

  1. Local    : 3x3 の深さ方向畳み込み（CNN）           … 金・銀・玉のような「近くの形」
  2. RayCast  : 8方向の「光線」に沿った情報の伝達       … 飛・角・香のような「走り駒の利き」
                マスごとに「透明度」を学習し、駒があると光がさえぎられる
                （=駒にぶつかると利きが止まる）ことを微分可能な形で表現する
  3. MoveGraph: 駒の種類に応じてゲートされるグラフ伝達 … 桂馬の跳び・玉の周り
                送り元のマスの駒が「その方向に利きを持つか」をゲートで学習する（利きグラフGNN）
  4. Global   : 81マスの自己注意（Transformer）を最後に1層だけ … 盤全体のつながり

  この 1→2→3 を1ブロックとして繰り返す。

【先手・後手】
  - 胴体（ブロック部分）は共通。手番の埋め込みを入力に足す。
  - 方策ヘッド: 共通部分 + 手番別の補正
  - 価値ヘッド: 手番別に 勝ち/引き分け/負け（WDL）の3つを出す
  - 探索に渡す価値は「勝ち点の期待値」から作る。
      期待勝ち点 E = P(勝) + d × P(引き分け)
      d は手番で変わる（電竜戦: 先手 0.4 / 後手 0.6）
      value = 2E - 1  （-1〜+1、手番側から見た値）

【入力】
  - (B, 447) の 447bit 特徴量（game/board.py と同じ）
  - turn: (B,) の long テンソル（0=先手番, 1=後手番）
    省略時は 448 列目を手番として読む。どちらもなければ先手番扱い。

CPU で速く動くことを優先し、チャンネル数は小さめ（appleS=64, appleM=96）。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SQ = 81
PAD = 81  # 盤外（パディング）用のインデックス


# ─────────────────────────────────────────────────────────
# 盤の幾何（マス番号 sq = file*9 + rank、view(9,9) で [file][rank]）
#   rank が小さいほど「上」（自分から見て相手陣の方向）
# ─────────────────────────────────────────────────────────

def _build_lines():
    """4 系統（段・筋・斜め・逆斜め）の線を作り、長さ 9 にパディングして返す。"""
    families = []
    # 筋（file 固定・rank 方向）: 上下
    families.append([[x * 9 + y for y in range(9)] for x in range(9)])
    # 段（rank 固定・file 方向）: 左右
    families.append([[x * 9 + y for x in range(9)] for y in range(9)])
    # 斜め（x + y 一定）
    diag = []
    for s in range(17):
        diag.append([x * 9 + (s - x) for x in range(9) if 0 <= s - x < 9])
    families.append(diag)
    # 逆斜め（x - y 一定）
    anti = []
    for d in range(-8, 9):
        anti.append([x * 9 + (x - d) for x in range(9) if 0 <= x - d < 9])
    families.append(anti)

    out = []
    for lines in families:
        idx = torch.full((len(lines), 9), PAD, dtype=torch.long)
        inv = torch.zeros(NUM_SQ, dtype=torch.long)
        for li, line in enumerate(lines):
            for pi, sq in enumerate(line):
                idx[li, pi] = sq
                inv[sq] = li * 9 + pi
        out.append((idx, inv))
    return out


# 桂馬の跳び（自分の桂は上へ、相手の桂は下へ）＋ 玉の8近傍
_MOVE_OFFSETS = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
    (-1, -2), (1, -2),          # 自分の桂（上に2つ）
    (-1, 2),  (1, 2),           # 相手の桂（下に2つ）
]


def _shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """(B,C,9,9) の内容を (file+dx, rank+dy) へずらす。はみ出た分は 0。"""
    out = torch.zeros_like(x)
    xs_src = slice(max(0, -dx), 9 - max(0, dx))
    xs_dst = slice(max(0, dx), 9 - max(0, -dx))
    ys_src = slice(max(0, -dy), 9 - max(0, dy))
    ys_dst = slice(max(0, dy), 9 - max(0, -dy))
    out[:, :, xs_dst, ys_dst] = x[:, :, xs_src, ys_src]
    return out


# ─────────────────────────────────────────────────────────
# ブロック
# ─────────────────────────────────────────────────────────

def _fold_bn_into_convs(bn: nn.BatchNorm2d, convs: list) -> None:
    """y = BN(h) のあとに 1x1 畳み込みが続くとき、BN を畳み込みの重みに吸収させる（推論用）。"""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    t = bn.bias - bn.running_mean * s
    for conv in convs:
        w = conv.weight.data                                        # (out, in, 1, 1)
        b = conv.bias.data if conv.bias is not None else torch.zeros(w.shape[0], device=w.device)
        new_b = b + (w[:, :, 0, 0] * t.view(1, -1)).sum(1)
        conv.weight.data = w * s.view(1, -1, 1, 1)
        if conv.bias is None:
            conv.bias = nn.Parameter(new_b)
        else:
            conv.bias.data = new_b


class LocalConv(nn.Module):
    """深さ方向 3x3 畳み込み + 点ごとの MLP（ConvNeXt を軽くしたもの）。"""

    def __init__(self, c: int, expand: int = 2):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, c * expand, 1)
        self.pw2 = nn.Conv2d(c * expand, c, 1)

    def forward(self, h):
        y = self.bn(self.dw(h))
        y = self.pw2(F.gelu(self.pw1(y)))
        return h + y


class RayCastMixer(nn.Module):
    """
    8方向の光線に沿って情報を集める（走り駒の利きの近似）。

    マス k の透明度 τ_k ∈ (0,1) を学習し、
      マス j がマス i から受け取る重み = Π_{j と i の間のマス k} (γ・τ_k)
    とする。駒がある（τ が小さい）と、その先は見えにくくなる。γ は距離による減衰。
    ヘッドごとに別の透明度を持つので「駒を透かして見る」ヘッドも作れる。

    計算は線に沿った漸化式  m(j) = v(j+1) + γ・τ(j+1)・m(j+1)  で行う（8ステップ）。
    4系統（筋・段・斜め・逆斜め）× 前後2方向 = 104 本の線をまとめて計算する
    （後ろ向きは、線を逆順に並べて前向きと同じ計算にする）。
    CPU で速くするため、並べ替えは行の取り出し、最後の変換は行列積で行う。
    """

    def __init__(self, c: int, heads: int = 4, vdim: Optional[int] = None):
        super().__init__()
        vdim = vdim or max(heads, c // 4)
        assert vdim % heads == 0
        self.heads = heads
        self.vdim = vdim
        self.dh = vdim // heads
        self.norm = nn.BatchNorm2d(c)
        self.to_tv = nn.Conv2d(c, heads + vdim, 1)    # 透明度(heads) と 値(vdim) をまとめて作る
        # 前後(2)×ヘッドごとの距離減衰 γ（sigmoid をかけて 0〜1 にする。初期値 ≈ 0.88）
        self.decay_logit = nn.Parameter(torch.full((2, heads), 2.0))
        self.proj = nn.Linear(vdim * 8, c)            # (系統4 × 方向2 × 値) → c

        lines = _build_lines()
        idx = torch.cat([i for i, _ in lines], dim=0)                   # (52,9) 盤外=81 は後ろ
        L = idx.shape[0]
        idx2 = torch.cat([idx, idx.flip(1)], dim=0)                     # (104,9) 後半は逆順の線
        self.n_lines = 2 * L
        self.register_buffer("gather_idx", idx2.t().reshape(-1), persistent=False)  # 行 = 位置*104 + 線
        scatter = torch.zeros(2, 4, NUM_SQ, dtype=torch.long)
        line_off = 0
        for f, (fi, inv) in enumerate(lines):
            for sq in range(NUM_SQ):
                li = line_off + int(inv[sq]) // 9
                pi = int(inv[sq]) % 9
                scatter[0, f, sq] = pi * (2 * L) + li                     # 前向き
                scatter[1, f, sq] = (8 - pi) * (2 * L) + (L + li)         # 後ろ向き（逆順の線）
            line_off += fi.shape[0]
        self.register_buffer("scatter_idx", scatter.reshape(-1), persistent=False)
        self.register_buffer("dir_of_line", torch.cat([torch.zeros(L), torch.ones(L)]).long(), persistent=False)
        pad_row = torch.cat([torch.ones(heads), torch.zeros(vdim)])     # 盤外: 透明(τ=1)・値0
        self.register_buffer("pad_row", pad_row, persistent=False)

    def forward(self, h):
        b, c = h.shape[0], h.shape[1]
        H, L2, dh, V = self.heads, self.n_lines, self.dh, self.vdim
        z = self.to_tv(self.norm(h)).flatten(2).permute(2, 0, 1)          # (81,B,H+V)
        z = torch.cat([torch.sigmoid(z[..., :H]), z[..., H:]], dim=2)
        z = torch.cat([z, self.pad_row.view(1, 1, H + V).expand(1, b, H + V)], dim=0)  # (82,B,H+V)
        g = z.index_select(0, self.gather_idx).view(9, L2, b, H + V)
        gamma = torch.sigmoid(self.decay_logit)[self.dir_of_line].view(1, L2, 1, H, 1)
        a = (g[..., :H].unsqueeze(-1) * gamma).expand(9, L2, b, H, dh).contiguous()
        v = g[..., H:].reshape(9, L2, b, H, dh)
        zero = v.new_zeros(L2, b, H, dh)
        outs = [zero] * 9
        cur = zero
        for j in range(7, -1, -1):                                      # 線の後ろ側から集める
            cur = torch.addcmul(v[j + 1], a[j + 1], cur)
            outs[j] = cur
        m = torch.stack(outs).view(9 * L2, b * V).index_select(0, self.scatter_idx)  # (2*4*81, B*V)
        m = m.view(2, 4, NUM_SQ, b, V).permute(3, 2, 1, 0, 4).reshape(b * NUM_SQ, 8 * V)  # 特徴 = (系統,方向,値)
        o = self.proj(m).view(b, NUM_SQ, c).transpose(1, 2).reshape(b, c, 9, 9)
        return h + o


class MoveGraphConv(nn.Module):
    """
    駒の動きでつながるグラフ（玉の8近傍＋桂の跳び）上でのメッセージ伝達。
    送り元マスの特徴から「その方向に利きがあるか」をゲートとして学習する（利きグラフ GNN）。
    """

    def __init__(self, c: int):
        super().__init__()
        n = len(_MOVE_OFFSETS)
        self.n = n
        self.norm = nn.BatchNorm2d(c)
        self.gate = nn.Conv2d(c, n, 1)
        self.val = nn.Conv2d(c, c, 1, bias=False)
        self.scale = nn.Parameter(torch.full((n, 1, c, 1, 1), 1.0 / n))
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, h):
        y = self.norm(h)
        g = F.pad(torch.sigmoid(self.gate(y)), (2, 2, 2, 2))            # (B,12,13,13)
        v = F.pad(self.val(y), (2, 2, 2, 2))                            # (B,C,13,13)
        m = 0
        for o, (dx, dy) in enumerate(_MOVE_OFFSETS):
            # 送り元 j → 受け手 j+o なので、受け手 t は t-o を読む
            xs, ys = slice(2 - dx, 11 - dx), slice(2 - dy, 11 - dy)
            m = m + self.scale[o] * (v[:, :, xs, ys] * g[:, o : o + 1, xs, ys])
        return h + self.proj(F.gelu(m))


class GlobalAttention(nn.Module):
    """81マスの自己注意（Transformer 1層）。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(nn.Linear(c, c * 2), nn.GELU(), nn.Linear(c * 2, c))
        self.pos = nn.Parameter(torch.zeros(1, NUM_SQ, c))

    def forward(self, h):
        b, c = h.shape[0], h.shape[1]
        t = h.flatten(2).transpose(1, 2) + self.pos              # (B,81,C)
        y = self.norm1(t)
        t = t + self.attn(y, y, y, need_weights=False)[0]
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(b, c, 9, 9)


class RayGraphBlock(nn.Module):
    """Local → RayCast → MoveGraph を1ブロックとする。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.local = LocalConv(c)
        self.ray = RayCastMixer(c, heads)
        self.graph = MoveGraphConv(c)

    def forward(self, h):
        return self.graph(self.ray(self.local(h)))


# ─────────────────────────────────────────────────────────
# 本体
# ─────────────────────────────────────────────────────────

class AppleNet(nn.Module):
    """RayGraph ハイブリッド + 手番別ヘッド。"""

    uses_turn = True  # 探索側が turn を渡すかどうかの目印

    def __init__(
        self,
        num_actions: int = 2187,
        channels: int = 64,
        blocks: int = 4,
        heads: int = 4,
        use_global: bool = True,
        draw_value_black: float = 0.4,
        draw_value_white: float = 0.6,
    ):
        super().__init__()
        assert num_actions == 27 * 81
        c = channels
        self.board_embedding = nn.Embedding(32, c)
        self.hand_mlp = nn.Sequential(nn.Linear(42, c), nn.GELU(), nn.Linear(c, c))
        self.turn_embedding = nn.Embedding(2, c)
        self.pos_embedding = nn.Parameter(torch.zeros(1, c, 9, 9))
        self.blocks = nn.ModuleList([RayGraphBlock(c, heads) for _ in range(blocks)])
        self.global_attn = GlobalAttention(c, heads) if use_global else None
        self.final_norm = nn.BatchNorm2d(c)

        # 方策: 共通 + 手番別の補正（1x1 畳み込みで 27 方向プレーン → 2187）
        self.policy_trunk = nn.Conv2d(c, c, 1)
        self.policy_shared = nn.Conv2d(c, 27, 1)
        self.policy_turn = nn.Conv2d(c, 27 * 2, 1)
        # 価値: 平均＋最大プーリング → 共通MLP → 手番別 WDL
        self.value_mlp = nn.Sequential(nn.Linear(2 * c, 128), nn.GELU())
        self.value_wdl = nn.Linear(128, 3 * 2)

        self.register_buffer(
            "draw_values", torch.tensor([draw_value_black, draw_value_white]), persistent=False
        )
        self.register_buffer(
            "bit_weights", torch.tensor([16.0, 8.0, 4.0, 2.0, 1.0]), persistent=False
        )
        nn.init.normal_(self.pos_embedding, std=0.02)

    @torch.no_grad()
    def fuse_for_inference(self) -> "AppleNet":
        """
        推論を少し速くする: 各ブロックの前にある BatchNorm を後ろの 1x1 畳み込みに吸収させる。
        学習には使わないこと（eval() の状態で呼ぶ）。
        """
        for blk in self.blocks:
            if isinstance(blk.ray.norm, nn.BatchNorm2d):
                _fold_bn_into_convs(blk.ray.norm, [blk.ray.to_tv])
                blk.ray.norm = nn.Identity()
            if isinstance(blk.graph.norm, nn.BatchNorm2d):
                _fold_bn_into_convs(blk.graph.norm, [blk.graph.gate, blk.graph.val])
                blk.graph.norm = nn.Identity()
        return self

    def set_draw_values(self, black: float, white: float) -> None:
        self.draw_values.copy_(torch.tensor([black, white], device=self.draw_values.device))

    # ── 内部 ─────────────────────────────────────────────
    def _split(self, x, turn):
        if turn is None:
            if x.shape[1] >= 448:
                turn = x[:, 447].long()
            else:
                turn = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return x[:, :447], turn.long().view(-1)

    def trunk(self, x, turn):
        b = x.shape[0]
        board_idx = (x[:, :405].view(b, 81, 5) * self.bit_weights).sum(-1).long()
        h = self.board_embedding(board_idx).transpose(1, 2).reshape(b, -1, 9, 9)
        h = h + self.pos_embedding
        g = self.hand_mlp(x[:, 405:447]) + self.turn_embedding(turn)
        h = h + g.view(b, -1, 1, 1)
        for blk in self.blocks:
            h = blk(h)
        if self.global_attn is not None:
            h = self.global_attn(h)
        return F.gelu(self.final_norm(h))

    def heads(self, h, turn):
        b = h.shape[0]
        p = F.gelu(self.policy_trunk(h))
        pt = self.policy_turn(p).view(b, 2, 27, 9, 9)
        pt = pt[torch.arange(b, device=h.device), turn]
        policy = (self.policy_shared(p) + pt).flatten(1)          # (B,2187) = dir*81 + sq
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        wdl = self.value_wdl(self.value_mlp(pooled)).view(b, 2, 3)
        wdl = wdl[torch.arange(b, device=h.device), turn]          # (B,3) 勝ち/引き分け/負け
        return policy, wdl

    # ── 公開 API ────────────────────────────────────────
    def forward_wdl(self, x, turn=None):
        """学習用: (policy_logits, wdl_logits) を返す。"""
        x, turn = self._split(x, turn)
        return self.heads(self.trunk(x, turn), turn)

    def wdl_to_value(self, wdl_logits, turn):
        p = torch.softmax(wdl_logits, dim=-1)
        d = self.draw_values[turn]
        expected = p[:, 0] + d * p[:, 1]
        return (2.0 * expected - 1.0).unsqueeze(1)

    def forward(self, x, turn=None, return_aux: bool = False):
        """
        探索用: (policy_logits (B,2187), value (B,1)) を返す。
        value は手番側から見た「勝ち点の期待値」を -1〜+1 にしたもの。
        """
        x, turn = self._split(x, turn)
        policy, wdl = self.heads(self.trunk(x, turn), turn)
        value = self.wdl_to_value(wdl, turn)
        if return_aux:
            return policy, value, {"wdl": wdl}
        return policy, value


APPLE_CONFIGS = {
    # CPU 対局向け（いちばん軽い）
    "appleS": dict(channels=64, blocks=4, heads=4, use_global=True),
    # 少し重い。後手（持ち時間10分）向け
    "appleM": dict(channels=96, blocks=6, heads=4, use_global=True),
    # 注意層なし（さらに速い）
    "appleXS": dict(channels=48, blocks=3, heads=4, use_global=False),
}


def create_apple_model(mode: str = "appleS", num_actions: int = 2187, **kw) -> AppleNet:
    cfg = dict(APPLE_CONFIGS[mode])
    cfg.update(kw)
    return AppleNet(num_actions=num_actions, **cfg)
