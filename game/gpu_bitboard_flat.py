"""
game/gpu_bitboard_flat.py
ビットボード合法手の実験版。走りを広げる前に「駒グループ × 敵陣の内外」へ分けない。

game/gpu_bitboard.py は、成りの判定をビット演算だけで書くために、
方向ごとに駒を 7 グループ × 敵陣かどうか 2 へ展開してから Kogge-Stone で広げている。
ほとんどの面は全ビット 0 でも、シフトのたびに読み書きされる。

この版は行き先を「方向 × 実際にその向きへ動く駒種」だけにする。
成りは、元のマスが敵陣であるビットを駒と一緒にずらして、広げたあとで足す。
  成れる = そのビットが立っている、または行き先が敵陣
王手・ピン・打ち・打ち歩詰めは gpu_bitboard.py と同じ。

出力は (N, 2187) の bool。legal_move_mask_bb と比較できる。
"""

from __future__ import annotations

import torch

from game.gpu_bitboard import (
    _PY,
    _T,
    _uchifuzume_bb,
    fill,
    move,
    nz,
    or_reduce,
    pack_by_type,
    unpack,
)
from game.gpu_movegen import (
    B,
    DIRS8,
    K,
    KNIGHT_DIRS,
    L,
    N_,
    P,
    R,
    S,
    _slide_pieces,
    _step_pieces,
    decode_features,
)

# 成れる駒。金・成駒・馬・龍は成れない
_PROMOTABLE = (P, L, N_, S, B, R)
# all_moves=False のとき、成れるなら不成を残さない駒
_MUST_PROMOTE = (P, B, R)


def _build_channels():
    """(方向, 駒種, 成れるか) の一覧。方向は OWN_DIRS（8 方向 + 桂 2 方向）の番号。"""
    steps = []
    slides = []
    for d, (dx, dy) in enumerate(DIRS8):
        for ty in _slide_pieces(dx, dy):
            slides.append((d, ty, ty in _PROMOTABLE))
        for ty in _step_pieces(dx, dy):
            if ty == K:
                continue
            steps.append((d, ty, ty in _PROMOTABLE))
    for j, _dir in enumerate(KNIGHT_DIRS):
        steps.append((8 + j, N_, True))
    return steps, slides


_STEPS, _SLIDES = _build_channels()
_SLIDE_PROM_IDX = [i for i, c in enumerate(_SLIDES) if c[2]]


def _ids_by_dir(channels, n_dir=10):
    out = [[] for _ in range(n_dir)]
    for i, (d, _ty, _prom) in enumerate(channels):
        out[d].append(i)
    return out


_STEP_IDS = _ids_by_dir(_STEPS)
_SLIDE_IDS = _ids_by_dir(_SLIDES)
_CHAN_CACHE: dict = {}


def _chan(device):
    """方向番号・駒種・成りのフラグを、そのデバイス上のテンソルにしておく。"""
    key = str(device)
    if key in _CHAN_CACHE:
        return _CHAN_CACHE[key]
    T = _T(device)

    def pack(channels):
        dirs = torch.tensor([c[0] for c in channels], dtype=torch.long, device=device)
        types = torch.tensor([c[1] for c in channels], dtype=torch.long, device=device)
        # -1 = 成れる（全ビット 1）。0 だと AND したとき行き先が消える
        prom = torch.tensor([-1 if c[2] else 0 for c in channels], dtype=torch.int64, device=device)
        dead = torch.zeros(len(channels), 2, dtype=torch.int64, device=device)
        for i, (_d, ty, _p) in enumerate(channels):
            if ty in (P, L):
                dead[i] = T["rank0"]
            elif ty == N_:
                dead[i] = T["rank01"]
        must = torch.tensor([ty in _MUST_PROMOTE for _d, ty, _p in channels], dtype=torch.bool, device=device)
        lance = torch.tensor([ty == L for _d, ty, _p in channels], dtype=torch.bool, device=device)
        return dirs, types, prom, dead, must, lance

    t = dict(
        step=pack(_STEPS),
        slide=pack(_SLIDES),
        slide_prom_idx=torch.tensor(_SLIDE_PROM_IDX, dtype=torch.long, device=device),
    )
    _CHAN_CACHE[key] = t
    return t


def _src(own_bb, forbid, types, dirs):
    """その方向に動ける駒種のビットだけを集める。ピンで禁じられた駒は落とす。"""
    return own_bb[:, types] & ~forbid[:, dirs]


def _or_into(dst, val, ids_by_dir):
    """val (N,C,2) を、チャネルごとの方向へ OR で戻す。同じ方向の行き先は重ならない。"""
    for d, ids in enumerate(ids_by_dir):
        if not ids:
            continue
        acc = val[:, ids[0]]
        for i in ids[1:]:
            acc = acc | val[:, i]
        dst[:, d] |= acc


def _core_flat(own, opp, own_hand, all_moves: bool = True):
    """
    own, opp: (N,81) 駒の種類。own_hand: (N,7)。
    返り値: (N,27,2)。中身は gpu_bitboard._core_bb と同じ並び。
    """
    device = own.device
    n = own.shape[0]
    T = _T(device)
    ch = _chan(device)
    own_bb = pack_by_type(own, T)
    opp_bb = pack_by_type(opp, T)
    own_occ = own_bb.sum(1)
    opp_occ = opp_bb.sum(1)
    occ = own_occ | opp_occ
    full = T["full"]
    empty = ~occ & full
    king = own_bb[:, K]
    has_king = nz(king)
    D = 10

    k_own = T["k_own"]
    ok_own = T["ok_own"]
    k_opp = T["k_opp"]
    ok_opp = T["ok_opp"]

    # ── 相手の利き（自玉を取り除いた盤） ──
    e_noking = empty | king
    opp_step = (opp_bb.unsqueeze(1) * T["step"].view(1, D, 15, 1)).sum(2)
    opp_slide = (opp_bb.unsqueeze(1) * T["slide"].view(1, D, 15, 1)).sum(2)
    kv = k_opp.view(1, D)
    okv = ok_opp.view(1, D, 2)
    ok2v, ok4v = T["ok_opp2"].view(1, D, 2), T["ok_opp4"].view(1, D, 2)
    att = move(opp_step, kv, okv) | move(fill(opp_slide, e_noking.unsqueeze(1), kv, okv, ok2v, ok4v), kv, okv)
    opp_att = or_reduce(att, 1)

    # ── 王手とピン ──
    k8 = k_own[:8].view(1, 8)
    ok8 = ok_own[:8].view(1, 8, 2)
    ok8_2, ok8_4 = T["ok_own2"][:8].view(1, 8, 2), T["ok_own4"][:8].view(1, 8, 2)
    kb = king.unsqueeze(1).expand(n, 8, 2)
    ray1 = move(fill(kb, empty.unsqueeze(1), k8, ok8, ok8_2, ok8_4), k8, ok8)
    blk1 = ray1 & occ.unsqueeze(1)
    adj = move(kb, k8, ok8)
    chk_slide = (opp_bb.unsqueeze(1) * T["chk_slide"].view(1, 8, 15, 1)).sum(2)
    chk_step = (opp_bb.unsqueeze(1) * T["chk_step"].view(1, 8, 15, 1)).sum(2)
    checker_d = blk1 & (chk_slide | (chk_step & adj))
    is_chk = nz(checker_d)
    between_d = torch.where(is_chk.unsqueeze(-1), ray1 & empty.unsqueeze(1), torch.zeros_like(ray1))
    own_blk = blk1 & (own_occ & ~king).unsqueeze(1)
    ray2 = move(fill(own_blk, empty.unsqueeze(1), k8, ok8, ok8_2, ok8_4), k8, ok8)
    blk2 = ray2 & occ.unsqueeze(1)
    pinned_d = torch.where(nz(blk2 & chk_slide).unsqueeze(-1), own_blk, torch.zeros_like(own_blk))
    kn = move(king.unsqueeze(1).expand(n, 2, 2), k_own[8:].view(1, 2), ok_own[8:].view(1, 2, 2)) & opp_bb[:, N_].unsqueeze(1)
    n_checks = is_chk.sum(1) + nz(kn).sum(1)
    n_checks = torch.where(has_king, n_checks, torch.zeros_like(n_checks))
    checkers = or_reduce(checker_d, 1) | or_reduce(kn, 1)
    between = or_reduce(between_d, 1)
    zero = torch.zeros_like(king)
    one_chk = (n_checks == 1).unsqueeze(-1)
    no_chk = (n_checks == 0).unsqueeze(-1)
    target = torch.where(no_chk, full.expand(n, 2), torch.where(one_chk, checkers | between, zero))
    drop_target = torch.where(no_chk, full.expand(n, 2), torch.where(one_chk, between, zero))

    pinned_all = or_reduce(pinned_d, 1)
    pinned_all = torch.where(has_king.unsqueeze(-1), pinned_all, zero)
    axes8 = _PY["axis"][:8]
    pin_axis_bb = []
    for a in (1, 2, 3, 4):
        sel = [d for d in range(8) if axes8[d] == a]
        pin_axis_bb.append(or_reduce(pinned_d[:, sel], 1))
    pin_axis_bb = torch.stack(pin_axis_bb, 1)
    forbid = []
    for d in range(D):
        a = _PY["axis"][d]
        ok_pinned = pin_axis_bb[:, a - 1] if a > 0 else zero
        forbid.append(pinned_all & ~ok_pinned)
    forbid = torch.stack(forbid, 1)

    # ── 盤上の駒。グループにも敵陣にも分けず、方向 × 駒種でずらす ──
    zone = T["zone"]
    land = (~own_occ & target).unsqueeze(1)                          # (N,1,2)
    nonpromo_d = torch.zeros(n, D, 2, dtype=torch.int64, device=device)
    promo_d = torch.zeros(n, D, 2, dtype=torch.int64, device=device)

    def emit(channels, ids_by_dir, packed, reached, reached_from_zone):
        _dirs, _types, prom, dead, must, lance = packed
        legal = reached & land
        from_zone = reached_from_zone & land
        can = from_zone | (legal & zone)
        nonpromo = legal & ~dead.unsqueeze(0)
        promo = can & prom.view(1, -1, 1)
        if not all_moves:
            # 歩・角・飛は成れるなら不成を消す。香は 2 段目と、3 段目で取らない不成を消す
            nonpromo = nonpromo & ~torch.where(must.view(1, -1, 1), can, torch.zeros_like(can))
            lance_kill = T["rank1"] | (T["rank2"] & ~opp_occ)
            nonpromo = nonpromo & ~torch.where(lance.view(1, -1, 1), lance_kill.unsqueeze(1), torch.zeros_like(can))
        _or_into(nonpromo_d, nonpromo, ids_by_dir)
        _or_into(promo_d, promo, ids_by_dir)

    s_dirs, s_types, s_prom, _sd, _sm, _sl = ch["step"]
    step_src = _src(own_bb, forbid, s_types, s_dirs)
    sk = k_own[s_dirs].view(1, -1)
    sok = ok_own[s_dirs].view(1, -1, 2)
    step_to = move(step_src, sk, sok)
    # 成れる駒だけ、元が敵陣のビットを 1 マスずらす（1 歩駒に fill は要らない）
    step_zone_src = step_src & torch.where((s_prom != 0).view(1, -1, 1), zone, torch.zeros_like(zone))
    step_from_zone = move(step_zone_src, sk, sok)
    emit(_STEPS, _STEP_IDS, ch["step"], step_to, step_from_zone)

    d_dirs, d_types, _dprom, _dd, _dm, _dl = ch["slide"]
    slide_src = _src(own_bb, forbid, d_types, d_dirs)
    dk = k_own[d_dirs].view(1, -1)
    dok = ok_own[d_dirs].view(1, -1, 2)
    dok2 = T["ok_own2"][d_dirs].view(1, -1, 2)
    dok4 = T["ok_own4"][d_dirs].view(1, -1, 2)
    empty_c = empty.unsqueeze(1)
    slide_to = move(fill(slide_src, empty_c, dk, dok, dok2, dok4), dk, dok)
    # 成れる走り駒だけ、元が敵陣のビットを同じ経路で広げる
    prom_idx = ch["slide_prom_idx"]
    slide_from_zone = torch.zeros_like(slide_to)
    zsrc = slide_src[:, prom_idx] & zone
    zk, zok = dk[:, prom_idx], dok[:, prom_idx]
    zok2, zok4 = dok2[:, prom_idx], dok4[:, prom_idx]
    slide_from_zone[:, prom_idx] = move(fill(zsrc, empty_c, zk, zok, zok2, zok4), zk, zok)
    emit(_SLIDES, _SLIDE_IDS, ch["slide"], slide_to, slide_from_zone)

    # 玉の手（8 方向。成りはない）
    kmove = move(king.unsqueeze(1).expand(n, 8, 2), k_own[:8].view(1, 8), ok_own[:8].view(1, 8, 2))
    kmove = kmove & (~own_occ & ~opp_att & full).unsqueeze(1)
    nonpromo_d[:, :8] |= kmove

    # ── 持ち駒 ──
    can_drop = empty & drop_target
    pawn_files = (T["files"].view(1, 9, 2) * nz(own_bb[:, P].unsqueeze(1) & T["files"].view(1, 9, 2)).unsqueeze(-1)).sum(1)
    drops = []
    for hp in range(7):
        m = can_drop
        if hp in (0, 1):
            m = m & T["not_rank0"]
        if hp == 2:
            m = m & T["not_rank01"]
        if hp == 0:
            m = m & ~pawn_files
        drops.append(torch.where((own_hand[:, hp] > 0).unsqueeze(-1), m, zero))
    drops = torch.stack(drops, 1)

    order = torch.tensor(_PY["cat"], device=device)
    lab = torch.zeros(n, 27, 2, dtype=torch.int64, device=device)
    lab[:, order] = nonpromo_d
    lab[:, order + 10] = promo_d
    lab[:, 20:27] = drops
    return lab


def legal_move_mask_flat(features: torch.Tensor, all_moves: bool = True, check_uchifuzume: bool = True,
                         core=None) -> torch.Tensor:
    """(N,447) → (N,2187)。gpu_bitboard.legal_move_mask_bb と同じ意味。"""
    own, opp, own_hand, opp_hand = decode_features(features)
    n = own.shape[0]
    own = own.reshape(n, 81).long()
    opp = opp.reshape(n, 81).long()
    run = core or _core_flat
    lab = run(own, opp, own_hand, all_moves)
    T = _T(own.device)
    out = unpack(lab, T)
    if check_uchifuzume:
        out = _uchifuzume_bb(out, own, opp, own_hand, opp_hand, run, T)
    return out.reshape(n, 27 * 81)


def make_compiled_core_flat(mode: str = "max-autotune-no-cudagraphs"):
    """torch.compile で本体をカーネル融合する。"""

    def fn(own, opp, own_hand, all_moves=True):
        return _core_flat(own, opp, own_hand, all_moves)

    return torch.compile(fn, mode=mode, dynamic=False)
