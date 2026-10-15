"""
game/gpu_bitboard.py
GPU でまとめて合法手を作る（ビットボード版・本命）

81 マスを 2 つの 64bit 整数（lo: マス 0〜62、hi: マス 63〜80）に詰めた「ビットボード」で表し、
シフト・AND・OR だけで N 局面分の合法手を一度に作る。すべて PyTorch の整数演算なので、
CPU でも GPU（CUDA）でも動く。

  - 8 方向 + 桂の 2 方向 = 10 方向を「方向の次元」としてまとめて計算する
    （方向ごとにシフト量が違うので、要素ごとにシフト量を変えられる torch のシフト演算を使う）
  - 走り駒の利きは、空きマスを通して 8 回広げる（fill）
  - 王手・ピンは、自玉から 8 方向に光線を伸ばして調べる
  - 結果は方策ラベル (N, 2187) の bool マスク（game/gpu_movegen.py と同じ）

正しさの基準は game/gpu_movegen.py（マス単位の分かりやすい版）と cshogi。
盤のマス番号は sq = x*9 + y（y=0 が敵陣の奥）。手番側から見た盤（後手番は 180 度回転済み）。
"""

from __future__ import annotations

import torch

from game.gpu_movegen import (
    B, DIRS8, DRAGON, G, HORSE, K, KNIGHT_DIRS, L, N_, P, R, S,
    _axis, _category, _slide_pieces, _step_pieces, decode_features,
)

LO_MASK = (1 << 63) - 1
HI_MASK = (1 << 18) - 1
OWN_DIRS = DIRS8 + KNIGHT_DIRS          # 自分の駒の動き（上向き）10 方向
D = len(OWN_DIRS)


# ─────────────────────────────────────────────────────────
# Python 側でのビットボード作成（テーブル用）
# ─────────────────────────────────────────────────────────
def _bb_of(squares) -> tuple:
    lo = hi = 0
    for sq in squares:
        if sq < 63:
            lo |= 1 << sq
        else:
            hi |= 1 << (sq - 63)
    return lo, hi


def _src_ok(dx: int, dy: int) -> tuple:
    """(dx,dy) だけ動いても盤内に残るマスの集合（シフトで隣の筋に回り込まないようにするマスク）。"""
    return _bb_of(x * 9 + y for x in range(9) for y in range(9) if 0 <= x + dx < 9 and 0 <= y + dy < 9)


def _build_tables():
    t = {}
    t["k_own"] = [dx * 9 + dy for dx, dy in OWN_DIRS]
    t["ok_own"] = [_src_ok(dx, dy) for dx, dy in OWN_DIRS]
    t["k_opp"] = [dx * 9 - dy for dx, dy in OWN_DIRS]                 # 相手は下向き
    t["ok_opp"] = [_src_ok(dx, -dy) for dx, dy in OWN_DIRS]
    # 走り駒の fill（Kogge-Stone 法）で使う、2倍・4倍の距離のシフト
    for mul in (2, 4):
        t[f"ok_own{mul}"] = [_src_ok(mul * dx, mul * dy) for dx, dy in OWN_DIRS]
        t[f"ok_opp{mul}"] = [_src_ok(mul * dx, -mul * dy) for dx, dy in OWN_DIRS]
    step = [[False] * 15 for _ in range(D)]
    slide = [[False] * 15 for _ in range(D)]
    for d, (dx, dy) in enumerate(DIRS8):
        for ty in _step_pieces(dx, dy):
            step[d][ty] = True
        for ty in _slide_pieces(dx, dy):
            slide[d][ty] = True
    step[8][N_] = step[9][N_] = True
    t["step"] = step
    t["slide"] = slide
    # 自玉から方向 d に伸ばした光線の先にいる相手の駒が、玉の方向に利くか
    # （相手の (-dx,-dy) は自分向きに直すと (-dx,+dy)）
    chk_step = [[False] * 15 for _ in range(8)]
    chk_slide = [[False] * 15 for _ in range(8)]
    for d, (dx, dy) in enumerate(DIRS8):
        for ty in _step_pieces(-dx, dy):
            chk_step[d][ty] = True
        for ty in _slide_pieces(-dx, dy):
            chk_slide[d][ty] = True
    t["chk_step"] = chk_step
    t["chk_slide"] = chk_slide
    t["axis"] = [_axis(dx, dy) for dx, dy in DIRS8] + [-1, -1]
    t["cat"] = [_category(dx, dy) for dx, dy in OWN_DIRS]
    t["zone"] = _bb_of(x * 9 + y for x in range(9) for y in range(3))
    t["full"] = (LO_MASK, HI_MASK)
    t["files"] = [_bb_of(x * 9 + y for y in range(9)) for x in range(9)]
    rank = lambda ys: _bb_of(x * 9 + y for x in range(9) for y in ys)  # noqa: E731
    t["rank0"] = rank([0])
    t["rank01"] = rank([0, 1])
    t["rank1"] = rank([1])
    t["rank2"] = rank([2])
    t["not_rank0"] = rank(range(1, 9))
    t["not_rank01"] = rank(range(2, 9))
    return t


# 成り・不成の扱いが同じ駒のグループ
_GROUPS = [
    ("P", (P,), True, "rank0"),
    ("N", (N_,), True, "rank01"),
    ("S", (S,), True, None),
    ("L", (L,), True, "rank0"),
    ("B", (B,), True, None),
    ("R", (R,), True, None),
    ("X", (G, 9, 10, 11, 12, HORSE, DRAGON), False, None),  # 金・成駒・馬・龍（成れない駒）
]
_GROUP_INDEX = {g[0]: i for i, g in enumerate(_GROUPS)}
_SLIDE_G = [_GROUP_INDEX[x] for x in ("L", "B", "R", "X")]       # 走る駒を含むグループ
_SINGLE_TYPES = [P, N_, S, L, B, R]                                 # _GROUPS の最初の 6 つと同じ順
_X_TYPES = list(_GROUPS[-1][1])


def _group_tables(t):
    gstep = [[[False] * 15 for _ in _GROUPS] for _ in range(D)]
    gslide = [[[False] * 15 for _ in _SLIDE_G] for _ in range(8)]
    for d in range(D):
        for gi, (_, types, _, _) in enumerate(_GROUPS):
            for ty in types:
                gstep[d][gi][ty] = t["step"][d][ty]
    for d in range(8):
        for j, gi in enumerate(_SLIDE_G):
            for ty in _GROUPS[gi][1]:
                gslide[d][j][ty] = t["slide"][d][ty]
    t["gstep"] = gstep
    t["gslide"] = gslide


_PY = _build_tables()
_group_tables(_PY)
_DEV_CACHE: dict = {}


def _T(device):
    key = str(device)
    if key in _DEV_CACHE:
        return _DEV_CACHE[key]
    p = _PY

    def bbt(pair_list):
        return torch.tensor(pair_list, dtype=torch.int64, device=device)

    t = dict(
        k_own=torch.tensor(p["k_own"], dtype=torch.int64, device=device),
        ok_own=bbt(p["ok_own"]),
        k_opp=torch.tensor(p["k_opp"], dtype=torch.int64, device=device),
        ok_opp=bbt(p["ok_opp"]),
        ok_own2=bbt(p["ok_own2"]),
        ok_own4=bbt(p["ok_own4"]),
        ok_opp2=bbt(p["ok_opp2"]),
        ok_opp4=bbt(p["ok_opp4"]),
        step=torch.tensor(p["step"], dtype=torch.bool, device=device),
        slide=torch.tensor(p["slide"], dtype=torch.bool, device=device),
        chk_step=torch.tensor(p["chk_step"], dtype=torch.bool, device=device),
        chk_slide=torch.tensor(p["chk_slide"], dtype=torch.bool, device=device),
        axis=torch.tensor(p["axis"], dtype=torch.int64, device=device),
        zone=bbt(p["zone"]),
        full=bbt(p["full"]),
        files=bbt(p["files"]),
        rank0=bbt(p["rank0"]),
        rank01=bbt(p["rank01"]),
        rank1=bbt(p["rank1"]),
        rank2=bbt(p["rank2"]),
        not_rank0=bbt(p["not_rank0"]),
        not_rank01=bbt(p["not_rank01"]),
        gstep_single=torch.tensor([[p["step"][d][ty] for ty in _SINGLE_TYPES] for d in range(D)],
                                  dtype=torch.int64, device=device),
        gstep_x=torch.tensor([[p["step"][d][ty] for ty in _X_TYPES] for d in range(D)],
                             dtype=torch.int64, device=device),
        gslide_single=torch.tensor([[p["slide"][d][ty] for ty in (L, B, R)] for d in range(8)],
                                   dtype=torch.int64, device=device),
        gslide_x=torch.tensor([[p["slide"][d][ty] for ty in _X_TYPES] for d in range(8)],
                              dtype=torch.int64, device=device),
        pow_lo=(2 ** torch.arange(63, dtype=torch.int64)).to(device),
        pow_hi=(2 ** torch.arange(18, dtype=torch.int64)).to(device),
        ar_lo=torch.arange(63, dtype=torch.int64, device=device),
        ar_hi=torch.arange(18, dtype=torch.int64, device=device),
    )
    full = t["full"]
    zero = torch.zeros(2, dtype=torch.int64, device=device)
    dead, useless, promotable = [], [], []
    for name, _types, prom, dz in _GROUPS:
        dead.append(t[dz] if dz else zero)
        promotable.append(-1 if prom else 0)  # -1 = すべてのビットが 1
        if name in ("P", "B", "R"):
            useless.append(torch.stack([full, t["zone"]]))   # 元が敵陣: 不成は全部省く / 外: 敵陣に入る不成を省く
        elif name == "L":
            useless.append(torch.stack([t["rank1"], t["rank1"]]))
        else:
            useless.append(torch.stack([zero, zero]))
    t["gdead"] = torch.stack(dead)                                   # (7,2)
    t["guseless"] = torch.stack(useless)                             # (7,2,2)
    t["gpromotable"] = torch.tensor(promotable, dtype=torch.int64, device=device)
    _DEV_CACHE[key] = t
    return t


# ─────────────────────────────────────────────────────────
# ビットボードの基本演算（最後の次元が [lo, hi]）
# ─────────────────────────────────────────────────────────
def pack(mask: torch.Tensor, T) -> torch.Tensor:
    """(..., 81) bool → (..., 2) int64"""
    m = mask.to(torch.int64)
    lo = (m[..., :63] * T["pow_lo"]).sum(-1)
    hi = (m[..., 63:] * T["pow_hi"]).sum(-1)
    return torch.stack([lo, hi], -1)


def pack_by_type(types: torch.Tensor, T) -> torch.Tensor:
    """(N,81) 駒の種類 → (N,15,2) 種類ごとのビットボード（0 番 = 空きマスは 0 にする）。"""
    n = types.shape[0]
    lo = torch.zeros(n, 15, dtype=torch.int64, device=types.device)
    hi = torch.zeros(n, 15, dtype=torch.int64, device=types.device)
    lo.scatter_add_(1, types[:, :63], T["pow_lo"].expand(n, 63))
    hi.scatter_add_(1, types[:, 63:], T["pow_hi"].expand(n, 18))
    out = torch.stack([lo, hi], -1)
    out[:, 0] = 0
    return out


def unpack(bb: torch.Tensor, T) -> torch.Tensor:
    """(..., 2) int64 → (..., 81) bool"""
    lo = (bb[..., 0:1] >> T["ar_lo"]) & 1
    hi = (bb[..., 1:2] >> T["ar_hi"]) & 1
    return torch.cat([lo, hi], -1).bool()


def nz(bb: torch.Tensor) -> torch.Tensor:
    return (bb[..., 0] | bb[..., 1]) != 0


def shift_k(bb: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """マス番号を k だけずらす（k は bb[...,0] の形にブロードキャストできるテンソル。正負どちらも可）。"""
    lo, hi = bb[..., 0], bb[..., 1]
    kp = k.clamp(min=0)
    kn = (-k).clamp(min=0)
    left_lo = (lo << kp) & LO_MASK
    left_hi = ((hi << kp) | (lo >> (63 - kp))) & HI_MASK
    right_lo = (lo >> kn) | ((hi << (63 - kn)) & LO_MASK)
    right_hi = hi >> kn
    pos = k > 0
    return torch.stack([torch.where(pos, left_lo, right_lo), torch.where(pos, left_hi, right_hi)], -1)


def move(bb, k, ok):
    """盤内に残るマスだけを (dx,dy) 方向に 1 歩動かす。ok は「動かしてよいマス」のマスク。"""
    return shift_k(bb & ok, k)


def fill(S, E, k, ok, ok2, ok4):
    """
    S から方向 k に、空きマス E を通ってたどり着けるマス（S を含む）。走り駒の利きの元になる。
    Kogge-Stone 法: 1・2・4 マスずつのシフト 3 回で、最大 7 マス先まで広げる。
    ok/ok2/ok4 はそれぞれ 1・2・4 マス動かすときに盤から出ないマスのマスク。
    """
    g = S
    p = E
    g = g | (p & move(g, k, ok))
    p = p & move(p, k, ok)
    g = g | (p & move(g, 2 * k, ok2))
    p = p & move(p, 2 * k, ok2)
    g = g | (p & move(g, 4 * k, ok4))
    return g


def or_reduce(bb: torch.Tensor, dim: int) -> torch.Tensor:
    """ビットごとの OR をある次元についてまとめる（重なりがあってもよい）。"""
    dim = dim % (bb.dim())
    out = bb.select(dim, 0)
    for i in range(1, bb.shape[dim]):
        out = out | bb.select(dim, i)
    return out


# ─────────────────────────────────────────────────────────
# 合法手生成の本体
# ─────────────────────────────────────────────────────────
def _core_bb(own, opp, own_hand, all_moves: bool = True):
    """
    own, opp: (N,81) 駒の種類（0=なし）。own_hand: (N,7)。
    返り値: (N,27,2) 方策ラベルごとの「行き先マス」のビットボード
    """
    device = own.device
    n = own.shape[0]
    T = _T(device)
    OWN = pack_by_type(own, T)                                           # (N,15,2) 種類ごとの自分の駒
    OPP = pack_by_type(opp, T)
    own_occ = OWN.sum(1)                                                 # 種類ごとのマスは重ならないので和 = OR
    opp_occ = OPP.sum(1)
    occ = own_occ | opp_occ
    full = T["full"]
    empty = ~occ & full
    Kbb = OWN[:, K]                                                      # (N,2)
    has_king = nz(Kbb)

    k_own = T["k_own"]                                                   # (10,)
    ok_own = T["ok_own"]                                                 # (10,2)
    k_opp = T["k_opp"]
    ok_opp = T["ok_opp"]

    # ── 相手の利き（自玉を取り除いた盤で。玉の逃げ先の判定用） ──
    e_noking = empty | Kbb
    opp_step = (OPP.unsqueeze(1) * T["step"].view(1, D, 15, 1)).sum(2)  # (N,10,2)
    opp_slide = (OPP.unsqueeze(1) * T["slide"].view(1, D, 15, 1)).sum(2)
    kv = k_opp.view(1, D)
    okv = ok_opp.view(1, D, 2)
    ok2v, ok4v = T["ok_opp2"].view(1, D, 2), T["ok_opp4"].view(1, D, 2)
    att = move(opp_step, kv, okv) | move(fill(opp_slide, e_noking.unsqueeze(1), kv, okv, ok2v, ok4v), kv, okv)
    opp_att = or_reduce(att, 1)                                          # (N,2)

    # ── 王手とピン（自玉から 8 方向） ───────────────────────
    k8 = k_own[:8].view(1, 8)
    ok8 = ok_own[:8].view(1, 8, 2)
    ok8_2, ok8_4 = T["ok_own2"][:8].view(1, 8, 2), T["ok_own4"][:8].view(1, 8, 2)
    Kb = Kbb.unsqueeze(1).expand(n, 8, 2)
    ray1 = move(fill(Kb, empty.unsqueeze(1), k8, ok8, ok8_2, ok8_4), k8, ok8)   # 最初の駒まで（駒を含む）
    blk1 = ray1 & occ.unsqueeze(1)
    adj = move(Kb, k8, ok8)
    chk_slide = (OPP.unsqueeze(1) * T["chk_slide"].view(1, 8, 15, 1)).sum(2)   # (N,8,2)
    chk_step = (OPP.unsqueeze(1) * T["chk_step"].view(1, 8, 15, 1)).sum(2)
    checker_d = blk1 & (chk_slide | (chk_step & adj))                    # (N,8,2)
    is_chk = nz(checker_d)                                               # (N,8)
    between_d = torch.where(is_chk.unsqueeze(-1), ray1 & empty.unsqueeze(1), torch.zeros_like(ray1))
    # ピン: 最初の駒が自分の駒（玉以外）で、その先の最初の駒が相手の走り駒
    own_blk = blk1 & (own_occ & ~Kbb).unsqueeze(1)
    ray2 = move(fill(own_blk, empty.unsqueeze(1), k8, ok8, ok8_2, ok8_4), k8, ok8)
    blk2 = ray2 & occ.unsqueeze(1)
    pinned_d = torch.where(nz(blk2 & chk_slide).unsqueeze(-1), own_blk, torch.zeros_like(own_blk))  # (N,8,2)
    # 桂の王手
    kn = move(Kbb.unsqueeze(1).expand(n, 2, 2), k_own[8:].view(1, 2), ok_own[8:].view(1, 2, 2)) & OPP[:, N_].unsqueeze(1)
    is_chk_n = nz(kn)                                                    # (N,2)
    n_checks = is_chk.sum(1) + is_chk_n.sum(1)
    n_checks = torch.where(has_king, n_checks, torch.zeros_like(n_checks))
    checkers = or_reduce(checker_d, 1) | or_reduce(kn, 1)
    between = or_reduce(between_d, 1)
    zero = torch.zeros_like(Kbb)
    one_chk = (n_checks == 1).unsqueeze(-1)
    no_chk = (n_checks == 0).unsqueeze(-1)
    target = torch.where(no_chk, full.expand(n, 2), torch.where(one_chk, checkers | between, zero))
    drop_target = torch.where(no_chk, full.expand(n, 2), torch.where(one_chk, between, zero))

    # ピンされた駒が動ける方向: その方向の線（axis）とピンの線が同じときだけ
    pinned_all = or_reduce(pinned_d, 1)
    pinned_all = torch.where(has_king.unsqueeze(-1), pinned_all, zero)
    pin_axis_bb = []                                                     # 線の種類 1〜4 ごとのピンされた駒
    axes8 = _PY["axis"][:8]
    for a in (1, 2, 3, 4):
        sel = [d for d in range(8) if axes8[d] == a]
        pin_axis_bb.append(or_reduce(pinned_d[:, sel], 1))
    pin_axis_bb = torch.stack(pin_axis_bb, 1)                           # (N,4,2)
    forbid = []
    for d in range(D):
        a = _PY["axis"][d]
        ok_pinned = pin_axis_bb[:, a - 1] if a > 0 else zero
        forbid.append(pinned_all & ~ok_pinned)
    forbid = torch.stack(forbid, 1)                                      # (N,10,2) その方向に動けない駒

    # ── 盤上の駒の手（玉以外） ──────────────────────────────
    # 成りの扱いが同じ駒をまとめた 7 グループ × 元のマスが敵陣か(2) で計算する
    zone = T["zone"]
    zsplit = torch.stack([zone, ~zone & full], 0)                       # (2,2) [敵陣にいる, いない]
    G_ = len(_GROUPS)
    # グループ X（金・成駒・馬・龍）は方向によって動ける駒が違うので種類ごとに、他は 1 種類なので旗で選ぶ
    single = OWN[:, _SINGLE_TYPES]                                       # (N,6,2) 歩 桂 銀 香 角 飛
    xs = OWN[:, _X_TYPES]                                                # (N,7,2)
    s_single = single.unsqueeze(1) * T["gstep_single"].view(1, D, 6, 1)  # (N,D,6,2)
    s_x = (xs.unsqueeze(1) * T["gstep_x"].view(1, D, 7, 1)).sum(2, keepdim=True)  # (N,D,1,2)
    s_step = torch.cat([s_single, s_x], 2)                               # (N,D,7,2)
    sl_single = single[:, [3, 4, 5]].unsqueeze(1) * T["gslide_single"].view(1, 8, 3, 1)  # 香 角 飛
    sl_x = (xs.unsqueeze(1) * T["gslide_x"].view(1, 8, 7, 1)).sum(2, keepdim=True)
    s_slide = torch.cat([sl_single, sl_x], 2)                            # (N,8,4,2)
    keep = ~forbid
    s_step = s_step & keep.view(n, D, 1, 2)
    s_slide = s_slide & keep[:, :8].view(n, 8, 1, 2)
    s_step = s_step.unsqueeze(3) & zsplit.view(1, 1, 1, 2, 2)           # (N,D,7,2,2)
    s_slide = s_slide.unsqueeze(3) & zsplit.view(1, 1, 1, 2, 2)         # (N,8,4,2,2)
    kv = k_own.view(1, D, 1, 1)
    okv = ok_own.view(1, D, 1, 1, 2)
    to = move(s_step, kv, okv)
    k8v = k_own[:8].view(1, 8, 1, 1)
    ok8v = ok_own[:8].view(1, 8, 1, 1, 2)
    ok8v2 = T["ok_own2"][:8].view(1, 8, 1, 1, 2)
    ok8v4 = T["ok_own4"][:8].view(1, 8, 1, 1, 2)
    to_sl = move(fill(s_slide, empty.view(n, 1, 1, 1, 2), k8v, ok8v, ok8v2, ok8v4), k8v, ok8v)
    sl_idx = torch.tensor(_SLIDE_G, device=device)
    to_8 = to[:, :8].index_add(2, sl_idx, to_sl)                        # 1マス動きと走りは重ならないので和で合わせる
    to = torch.cat([to_8, to[:, 8:]], 1)
    to = to & (~own_occ & target).view(n, 1, 1, 1, 2)                   # (N,D,7,2,2)

    # 成り・不成
    nonpromo = to & ~T["gdead"].view(1, 1, G_, 1, 2)
    if not all_moves:
        nonpromo = nonpromo & ~T["guseless"].view(1, 1, G_, 2, 2)
        lance3 = T["rank2"] & ~opp_occ                                  # 香の 3 段目への駒を取らない不成
        gl = _GROUP_INDEX["L"]
        nonpromo[:, :, gl] = nonpromo[:, :, gl] & ~lance3.view(n, 1, 1, 2)
    promo_mask = torch.stack([full, zone], 0)                            # 元が敵陣なら全部、そうでなければ行き先が敵陣
    promo = to & promo_mask.view(1, 1, 1, 2, 2) & T["gpromotable"].view(1, 1, G_, 1, 1)
    nonpromo_d = nonpromo.sum(dim=(2, 3))                                # (N,D,2) 重ならないので和 = OR
    promo_d = promo.sum(dim=(2, 3))

    # 玉の手
    kmove = move(Kbb.unsqueeze(1).expand(n, 8, 2), k_own[:8].view(1, 8), ok_own[:8].view(1, 8, 2))
    kmove = kmove & (~own_occ & ~opp_att & full).unsqueeze(1)
    nonpromo_d = torch.cat([nonpromo_d[:, :8] | kmove, nonpromo_d[:, 8:]], 1)

    # ── 持ち駒を打つ手 ───────────────────────────────────
    can = empty & drop_target                                            # (N,2)
    pawn_files = (T["files"].view(1, 9, 2) * nz(OWN[:, P].unsqueeze(1) & T["files"].view(1, 9, 2)).unsqueeze(-1)).sum(1)
    drops = []
    for hp in range(7):
        m = can
        if hp in (0, 1):
            m = m & T["not_rank0"]
        if hp == 2:
            m = m & T["not_rank01"]
        if hp == 0:
            m = m & ~pawn_files
        drops.append(torch.where((own_hand[:, hp] > 0).unsqueeze(-1), m, zero))
    drops = torch.stack(drops, 1)                                        # (N,7,2)

    # ラベルの方向順に並べる（10 方向はそれぞれ別の番号 0〜9）
    order = torch.tensor(_PY["cat"], device=device)
    lab = torch.zeros(n, 27, 2, dtype=torch.int64, device=device)
    lab[:, order] = nonpromo_d
    lab[:, order + 10] = promo_d
    lab[:, 20:27] = drops
    return lab


def legal_move_mask_bb(features: torch.Tensor, all_moves: bool = True, check_uchifuzume: bool = True,
                       core=None) -> torch.Tensor:
    """(N,447) → (N,2187) の合法手マスク（ビットボード版）。"""
    own, opp, own_hand, opp_hand = decode_features(features)
    n = own.shape[0]
    own = own.reshape(n, 81).long()
    opp = opp.reshape(n, 81).long()
    run = core or _core_bb
    lab = run(own, opp, own_hand, all_moves)
    T = _T(own.device)
    out = unpack(lab, T)                                                 # (N,27,81)
    if check_uchifuzume:
        out = _uchifuzume_bb(out, own, opp, own_hand, opp_hand, run, T)
    return out.reshape(n, 27 * 81)


def _uchifuzume_bb(out, own, opp, own_hand, opp_hand, run, T):
    """打ち歩詰めの手を取り除く（gpu_movegen.py と同じ考え方）。"""
    n = own.shape[0]
    okmask = opp == K
    oksq = okmask.float().argmax(dim=1)
    ox, oy = oksq // 9, oksq % 9
    dsq = ox * 9 + (oy + 1).clamp(max=8)
    cand = okmask.any(1) & (oy < 8) & out[:, 20].gather(1, dsq.unsqueeze(1)).squeeze(1)
    idx = cand.nonzero(as_tuple=True)[0]
    m = idx.numel()
    if m == 0:
        return out
    mm = 1 << (m - 1).bit_length()
    idx_p = torch.cat([idx, idx[:1].expand(mm - m)]) if mm != m else idx
    d = dsq[idx_p]
    own_s = own[idx_p].clone()
    own_s.scatter_(1, d.unsqueeze(1), torch.full_like(d, P).unsqueeze(1))
    hand_s = own_hand[idx_p].clone()
    hand_s[:, 0] -= 1
    new_own = opp[idx_p].flip(1)
    new_opp = own_s.flip(1)
    reply = run(new_own, new_opp, opp_hand[idx_p], True)
    mated = ~nz(reply).any(1)[:m]
    pos = dsq[idx]
    out[idx, 20, pos] = out[idx, 20, pos] & ~mated
    return out


def make_compiled_core_bb(mode: str = "max-autotune-no-cudagraphs"):
    """torch.compile で本体をカーネル融合する（GPU 向け）。"""

    def fn(own, opp, own_hand, all_moves=True):
        return _core_bb(own, opp, own_hand, all_moves)

    return torch.compile(fn, mode=mode, dynamic=False)
