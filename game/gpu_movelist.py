"""
game/gpu_movelist.py
駒リストで合法手を作る実験版。

1 ワープの各レーンが 1 局面を担当する。81 マスを一度走査して、
駒を種類ごとの固定長リストへ詰め、その後はリストの添字だけを進む。
ループはワープ内の最大枚数で打ち切る。ある種類が隣の局面に 1 枚でもあれば
その枚数までは全員が同じ命令を走るので、種類ごとにリストを分けてある。

盤はレーンのレジスタに全部は載らない。マスの有無は 27bit × 3 のビットセットで持ち、
駒種が要るときだけグローバルメモリの盤を読む。
出力は (N, 2187) の bool。legal_move_mask_bb と同じ意味。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from game.gpu_movegen import K, decode_features

BLOCK = 32
# 種類ごとの上限。歩 18、金相当（金・と・成香・成桂・成銀）34、その他は実戦の上限に少し余裕
_CAPS = (18, 4, 4, 4, 34, 4, 4, 4, 4)  # 歩香桂銀 金 角飛馬龍
_ONE = sum(_CAPS)          # 80
STRIDE_N = _ONE * 2        # 自分のリスト + 相手のリスト
STRIDE = tl.constexpr(STRIDE_N)


@triton.jit
def _bit_test(lo, mid, hi, sq):
    """sq のビットが立っているか。sq は 0..80 に丸めてから見る。"""
    sqc = tl.where((sq >= 0) & (sq < 81), sq, 0)
    in_lo = sqc < 27
    in_mid = (sqc >= 27) & (sqc < 54)
    sh_lo = tl.where(in_lo, sqc, 0)
    sh_mid = tl.where(in_mid, sqc - 27, 0)
    sh_hi = tl.where(sqc >= 54, sqc - 54, 0)
    b_lo = (lo >> sh_lo) & 1
    b_mid = (mid >> sh_mid) & 1
    b_hi = (hi >> sh_hi) & 1
    return tl.where(in_lo, b_lo, tl.where(in_mid, b_mid, b_hi))


@triton.jit
def _bit_on(lo, mid, hi, sq, cond):
    sqc = tl.where(cond, sq, 0)
    in_lo = sqc < 27
    in_mid = (sqc >= 27) & (sqc < 54)
    in_hi = sqc >= 54
    sh_lo = tl.where(in_lo, sqc, 0)
    sh_mid = tl.where(in_mid, sqc - 27, 0)
    sh_hi = tl.where(in_hi, sqc - 54, 0)
    one = tl.full(lo.shape, 1, tl.int32)
    lo = tl.where(cond & in_lo, lo | (one << sh_lo), lo)
    mid = tl.where(cond & in_mid, mid | (one << sh_mid), mid)
    hi = tl.where(cond & in_hi, hi | (one << sh_hi), hi)
    return lo, mid, hi


@triton.jit
def _bit_off(lo, mid, hi, sq, cond):
    sqc = tl.where(cond, sq, 0)
    in_lo = sqc < 27
    in_mid = (sqc >= 27) & (sqc < 54)
    in_hi = sqc >= 54
    sh_lo = tl.where(in_lo, sqc, 0)
    sh_mid = tl.where(in_mid, sqc - 27, 0)
    sh_hi = tl.where(in_hi, sqc - 54, 0)
    one = tl.full(lo.shape, 1, tl.int32)
    lo = tl.where(cond & in_lo, lo & ~(one << sh_lo), lo)
    mid = tl.where(cond & in_mid, mid & ~(one << sh_mid), mid)
    hi = tl.where(cond & in_hi, hi & ~(one << sh_hi), hi)
    return lo, mid, hi


@triton.jit
def _push(ptr, offs, count, sq, take, lane, BASE: tl.constexpr, CAP: tl.constexpr, STRIDE: tl.constexpr):
    ok = take & lane & (count < CAP)
    tl.store(ptr + offs * STRIDE + BASE + count, sq, mask=ok)
    return tl.where(ok, count + 1, count)


@triton.jit
def _emit(
    ok, from_y, to, to_y, pax,
    own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
    att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
    n_checks, out_ptr, offs, lane,
    CAT: tl.constexpr, AXIS: tl.constexpr,
    PROM: tl.constexpr, DEAD: tl.constexpr, MUST: tl.constexpr, LANCE: tl.constexpr, IS_KING: tl.constexpr,
    ALL_MOVES: tl.constexpr,
):
    """ok のマスへ、方向 CAT の手を 1 つ書く。"""
    to_c = tl.where(ok, to, 0)
    ok = ok & (_bit_test(own_lo, own_mid, own_hi, to_c) == 0)
    if IS_KING:
        ok = ok & (_bit_test(att_lo, att_mid, att_hi, to_c) == 0)
    else:
        pin_ok = (pax == 0) | (pax == AXIS)
        on_target = (n_checks == 0) | (
            (n_checks == 1) & ((_bit_test(chk_lo, chk_mid, chk_hi, to_c) | _bit_test(btw_lo, btw_mid, btw_hi, to_c)) != 0)
        )
        ok = ok & pin_ok & on_target
    if PROM:
        can_p = ok & ((from_y <= 2) | (to_y <= 2))
    else:
        can_p = tl.zeros(ok.shape, tl.int1)
    nonpromo = ok
    if DEAD == 1:
        nonpromo = nonpromo & (to_y != 0)
    elif DEAD == 2:
        nonpromo = nonpromo & (to_y > 1)
    if not ALL_MOVES:
        if MUST:
            nonpromo = nonpromo & ~can_p
        if LANCE:
            capture = _bit_test(opp_lo, opp_mid, opp_hi, to_c) != 0
            useless = (to_y == 1) | ((to_y == 2) & ~capture)
            nonpromo = nonpromo & ~useless
    one = tl.full(offs.shape, 1, tl.uint8)
    tl.store(out_ptr + offs * 2187 + CAT * 81 + to_c, one, mask=nonpromo & lane)
    if PROM:
        tl.store(out_ptr + offs * 2187 + (CAT + 10) * 81 + to_c, one, mask=can_p & lane)


@triton.jit
def _mv_slide(
    alive, y, pax, x_from,
    own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
    att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
    n_checks, out_ptr, offs, lane,
    DX: tl.constexpr, DY: tl.constexpr, CAT: tl.constexpr, AXIS: tl.constexpr,
    PROM: tl.constexpr, DEAD: tl.constexpr, MUST: tl.constexpr, LANCE: tl.constexpr,
    ALL_MOVES: tl.constexpr,
):
    going = alive
    for k in tl.range(1, 9, loop_unroll_factor=1):
        tx = x_from + DX * k
        tyy = y + DY * k
        on = going & (tx >= 0) & (tx < 9) & (tyy >= 0) & (tyy < 9)
        to = tl.where(on, tx * 9 + tyy, 0)
        _emit(
            on, y, to, tyy, pax,
            own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
            att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
            n_checks, out_ptr, offs, lane,
            CAT=CAT, AXIS=AXIS, PROM=PROM, DEAD=DEAD, MUST=MUST, LANCE=LANCE, IS_KING=False,
            ALL_MOVES=ALL_MOVES,
        )
        occ = (_bit_test(own_lo, own_mid, own_hi, to) | _bit_test(opp_lo, opp_mid, opp_hi, to)) != 0
        going = on & (occ == 0)


@triton.jit
def _mv_step(
    alive, x, y, pax,
    own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
    att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
    n_checks, out_ptr, offs, lane,
    DX: tl.constexpr, DY: tl.constexpr, CAT: tl.constexpr, AXIS: tl.constexpr,
    PROM: tl.constexpr, DEAD: tl.constexpr, MUST: tl.constexpr, LANCE: tl.constexpr, IS_KING: tl.constexpr,
    ALL_MOVES: tl.constexpr,
):
    tx = x + DX
    tyy = y + DY
    on = alive & (tx >= 0) & (tx < 9) & (tyy >= 0) & (tyy < 9)
    to = tl.where(on, tx * 9 + tyy, 0)
    _emit(
        on, y, to, tyy, pax,
        own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
        n_checks, out_ptr, offs, lane,
        CAT=CAT, AXIS=AXIS, PROM=PROM, DEAD=DEAD, MUST=MUST, LANCE=LANCE, IS_KING=IS_KING,
        ALL_MOVES=ALL_MOVES,
    )


@triton.jit
def _att_step(alive, x, y, alo, amid, ahi, DX: tl.constexpr, DY: tl.constexpr):
    tx = x + DX
    tyy = y + DY
    on = alive & (tx >= 0) & (tx < 9) & (tyy >= 0) & (tyy < 9)
    to = tl.where(on, tx * 9 + tyy, 0)
    return _bit_on(alo, amid, ahi, to, on)


@triton.jit
def _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX: tl.constexpr, DY: tl.constexpr):
    going = alive
    for k in tl.range(1, 9, loop_unroll_factor=1):
        tx = x + DX * k
        tyy = y + DY * k
        on = going & (tx >= 0) & (tx < 9) & (tyy >= 0) & (tyy < 9)
        to = tl.where(on, tx * 9 + tyy, 0)
        alo, amid, ahi = _bit_on(alo, amid, ahi, to, on)
        blocked = _bit_test(blo, bmid, bhi, to) != 0
        going = on & (blocked == 0)
    return alo, amid, ahi


@triton.jit
def _ray(
    has_k, kx, ky,
    own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
    own_ptr, opp_ptr, pin_ptr, offs, lane,
    n_checks, clo, cmid, chi, blo, bmid, bhi,
    DX: tl.constexpr, DY: tl.constexpr, AXIS: tl.constexpr, STEP: tl.constexpr, SLIDE: tl.constexpr,
):
    """自玉から 1 方向。王手・合い駒・ピンを足す。"""
    found = tl.zeros(offs.shape, tl.int32)
    first_sq = tl.zeros(offs.shape, tl.int32)
    first_k = tl.full(offs.shape, 99, tl.int32)
    for k in tl.range(1, 9, loop_unroll_factor=1):
        sx = kx + DX * k
        sy = ky + DY * k
        on = (has_k != 0) & (found == 0) & (sx >= 0) & (sx < 9) & (sy >= 0) & (sy < 9)
        sq = tl.where(on, sx * 9 + sy, 0)
        occ = (_bit_test(own_lo, own_mid, own_hi, sq) | _bit_test(opp_lo, opp_mid, opp_hi, sq)) != 0
        hit = on & occ
        first_sq = tl.where(hit, sq, first_sq)
        first_k = tl.where(hit, k, first_k)
        found = tl.where(hit, 1, found)
    own_ty = tl.load(own_ptr + offs * 81 + first_sq, mask=(found != 0) & lane, other=0).to(tl.int32)
    opp_ty = tl.load(opp_ptr + offs * 81 + first_sq, mask=(found != 0) & lane, other=0).to(tl.int32)
    is_opp = (found != 0) & (opp_ty != 0)
    sh = tl.where(is_opp, opp_ty, 0)
    bit = tl.full(offs.shape, 1, tl.int32) << sh
    is_check = is_opp & (((bit & SLIDE) != 0) | ((first_k == 1) & ((bit & STEP) != 0)))
    n_checks = n_checks + is_check.to(tl.int32)
    clo, cmid, chi = _bit_on(clo, cmid, chi, first_sq, is_check)
    for k in tl.range(1, 9, loop_unroll_factor=1):
        sx = kx + DX * k
        sy = ky + DY * k
        on = is_check & (k < first_k) & (sx >= 0) & (sx < 9) & (sy >= 0) & (sy < 9)
        sq = tl.where(on, sx * 9 + sy, 0)
        blo, bmid, bhi = _bit_on(blo, bmid, bhi, sq, on)
    found2 = tl.zeros(offs.shape, tl.int32)
    second_sq = tl.zeros(offs.shape, tl.int32)
    for k in tl.range(1, 9, loop_unroll_factor=1):
        sx = kx + DX * k
        sy = ky + DY * k
        on = (found != 0) & (found2 == 0) & (k > first_k) & (sx >= 0) & (sx < 9) & (sy >= 0) & (sy < 9)
        sq = tl.where(on, sx * 9 + sy, 0)
        occ = (_bit_test(own_lo, own_mid, own_hi, sq) | _bit_test(opp_lo, opp_mid, opp_hi, sq)) != 0
        hit = on & occ
        second_sq = tl.where(hit, sq, second_sq)
        found2 = tl.where(hit, 1, found2)
    opp2 = tl.load(opp_ptr + offs * 81 + second_sq, mask=(found2 != 0) & lane, other=0).to(tl.int32)
    is_opp2 = (found2 != 0) & (opp2 != 0)
    sh2 = tl.where(is_opp2, opp2, 0)
    bit2 = tl.full(offs.shape, 1, tl.int32) << sh2
    is_own_first = (found != 0) & (own_ty != 0) & (own_ty != 8)
    is_pin = is_own_first & is_opp2 & ((bit2 & SLIDE) != 0)
    tl.store(pin_ptr + offs * 81 + first_sq, tl.full(offs.shape, AXIS, tl.int8), mask=is_pin & lane)
    return n_checks, clo, cmid, chi, blo, bmid, bhi


@triton.jit
def _own_kind(
    count, scratch, pin_ptr, out_ptr, offs, lane,
    own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
    att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
    n_checks,
    BASE: tl.constexpr, CAP: tl.constexpr, KIND: tl.constexpr, STRIDE: tl.constexpr, ALL_MOVES: tl.constexpr,
):
    """KIND: 0歩 1香 2桂 3銀 4金相当 5角 6飛 7馬 8龍。"""
    m = tl.max(count, axis=0)
    for slot in tl.range(m, loop_unroll_factor=1):
        alive = (slot < count) & lane
        sq = tl.load(scratch + offs * STRIDE + BASE + slot, mask=alive, other=0)
        x = sq // 9
        y = sq % 9
        pax = tl.load(pin_ptr + offs * 81 + sq, mask=alive, other=0).to(tl.int32)
        if KIND == 0:
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
                     PROM=True, DEAD=1, MUST=True, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
        elif KIND == 1:
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
                      PROM=True, DEAD=1, MUST=False, LANCE=True, ALL_MOVES=ALL_MOVES)
        elif KIND == 2:
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=-2, CAT=8, AXIS=-1,
                     PROM=True, DEAD=2, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=-2, CAT=9, AXIS=-1,
                     PROM=True, DEAD=2, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
        elif KIND == 3:
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
                     PROM=True, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=-1, CAT=1, AXIS=3,
                     PROM=True, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=-1, CAT=2, AXIS=4,
                     PROM=True, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=1, CAT=6, AXIS=4,
                     PROM=True, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=1, CAT=7, AXIS=3,
                     PROM=True, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
        elif KIND == 4:
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=-1, CAT=1, AXIS=3,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=-1, CAT=2, AXIS=4,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=0, CAT=3, AXIS=2,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=0, CAT=4, AXIS=2,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=0, DY=1, CAT=5, AXIS=1,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
        elif KIND == 5:
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=-1, DY=-1, CAT=1, AXIS=3,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=1, DY=-1, CAT=2, AXIS=4,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=-1, DY=1, CAT=6, AXIS=4,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=1, DY=1, CAT=7, AXIS=3,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
        elif KIND == 6:
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=0, DY=1, CAT=5, AXIS=1,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=-1, DY=0, CAT=3, AXIS=2,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=1, DY=0, CAT=4, AXIS=2,
                      PROM=True, DEAD=0, MUST=True, LANCE=False, ALL_MOVES=ALL_MOVES)
        elif KIND == 7:
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=-1, DY=-1, CAT=1, AXIS=3,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=1, DY=-1, CAT=2, AXIS=4,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=-1, DY=1, CAT=6, AXIS=4,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=1, DY=1, CAT=7, AXIS=3,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=0, DY=1, CAT=5, AXIS=1,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=0, CAT=3, AXIS=2,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=0, CAT=4, AXIS=2,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
        elif KIND == 8:
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=0, DY=1, CAT=5, AXIS=1,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=-1, DY=0, CAT=3, AXIS=2,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_slide(alive, y, pax, x, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                      att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                      n_checks, out_ptr, offs, lane, DX=1, DY=0, CAT=4, AXIS=2,
                      PROM=False, DEAD=0, MUST=False, LANCE=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=-1, CAT=1, AXIS=3,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=-1, CAT=2, AXIS=4,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=-1, DY=1, CAT=6, AXIS=4,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)
            _mv_step(alive, x, y, pax, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
                     att_lo, att_mid, att_hi, chk_lo, chk_mid, chk_hi, btw_lo, btw_mid, btw_hi,
                     n_checks, out_ptr, offs, lane, DX=1, DY=1, CAT=7, AXIS=3,
                     PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=False, ALL_MOVES=ALL_MOVES)


@triton.jit
def _opp_kind(
    count, scratch, offs, lane, alo, amid, ahi, blo, bmid, bhi,
    BASE: tl.constexpr, CAP: tl.constexpr, KIND: tl.constexpr, STRIDE: tl.constexpr,
):
    """相手の駒リストから利きビットを足す。KIND は _own_kind と同じ番号。dy は下向き。"""
    m = tl.max(count, axis=0)
    for slot in tl.range(m, loop_unroll_factor=1):
        alive = (slot < count) & lane
        sq = tl.load(scratch + offs * STRIDE + BASE + slot, mask=alive, other=0)
        x = sq // 9
        y = sq % 9
        if KIND == 0:
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=0, DY=1)
        elif KIND == 1:
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=0, DY=1)
        elif KIND == 2:
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=2)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=2)
        elif KIND == 3:
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=0, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=-1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=-1)
        elif KIND == 4:
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=0, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=0)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=0)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=0, DY=-1)
        elif KIND == 5:
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=-1, DY=-1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=1, DY=-1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=-1, DY=1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=1, DY=1)
        elif KIND == 6:
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=0, DY=-1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=0, DY=1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=-1, DY=0)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=1, DY=0)
        elif KIND == 7:
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=-1, DY=-1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=1, DY=-1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=-1, DY=1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=1, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=0, DY=-1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=0, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=0)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=0)
        elif KIND == 8:
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=0, DY=-1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=0, DY=1)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=-1, DY=0)
            alo, amid, ahi = _att_slide(alive, x, y, alo, amid, ahi, blo, bmid, bhi, DX=1, DY=0)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=-1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=-1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=-1, DY=1)
            alo, amid, ahi = _att_step(alive, x, y, alo, amid, ahi, DX=1, DY=1)
    return alo, amid, ahi


@triton.jit
def _kernel(own_ptr, opp_ptr, hand_ptr, scratch_ptr, pin_ptr, out_ptr, n,
            ALL_MOVES: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = offs < n
    z = tl.zeros((BLOCK,), tl.int32)
    own_lo, own_mid, own_hi = z, z, z
    opp_lo, opp_mid, opp_hi = z, z, z
    c_p = z
    c_l = z
    c_n = z
    c_s = z
    c_g = z
    c_b = z
    c_r = z
    c_h = z
    c_d = z
    o_p = z
    o_l = z
    o_n = z
    o_s = z
    o_g = z
    o_b = z
    o_r = z
    o_h = z
    o_d = z
    file_bits = z
    ksq = z
    has_k = z
    oksq = z
    has_ok = z
    # 自分 0..79、相手 80..159。歩18 香4 桂4 銀4 金34 角4 飛4 馬4 龍4
    for sq in tl.range(81, loop_unroll_factor=1):
        ty = tl.load(own_ptr + offs * 81 + sq, mask=lane, other=0).to(tl.int32)
        oty = tl.load(opp_ptr + offs * 81 + sq, mask=lane, other=0).to(tl.int32)
        sqv = tl.full(offs.shape, sq, tl.int32)
        own_lo, own_mid, own_hi = _bit_on(own_lo, own_mid, own_hi, sqv, ty != 0)
        opp_lo, opp_mid, opp_hi = _bit_on(opp_lo, opp_mid, opp_hi, sqv, oty != 0)
        c_p = _push(scratch_ptr, offs, c_p, sqv, ty == 1, lane, BASE=0, CAP=18, STRIDE=STRIDE)
        c_l = _push(scratch_ptr, offs, c_l, sqv, ty == 2, lane, BASE=18, CAP=4, STRIDE=STRIDE)
        c_n = _push(scratch_ptr, offs, c_n, sqv, ty == 3, lane, BASE=22, CAP=4, STRIDE=STRIDE)
        c_s = _push(scratch_ptr, offs, c_s, sqv, ty == 4, lane, BASE=26, CAP=4, STRIDE=STRIDE)
        gold = (ty == 7) | (ty == 9) | (ty == 10) | (ty == 11) | (ty == 12)
        c_g = _push(scratch_ptr, offs, c_g, sqv, gold, lane, BASE=30, CAP=34, STRIDE=STRIDE)
        c_b = _push(scratch_ptr, offs, c_b, sqv, ty == 5, lane, BASE=64, CAP=4, STRIDE=STRIDE)
        c_r = _push(scratch_ptr, offs, c_r, sqv, ty == 6, lane, BASE=68, CAP=4, STRIDE=STRIDE)
        c_h = _push(scratch_ptr, offs, c_h, sqv, ty == 13, lane, BASE=72, CAP=4, STRIDE=STRIDE)
        c_d = _push(scratch_ptr, offs, c_d, sqv, ty == 14, lane, BASE=76, CAP=4, STRIDE=STRIDE)
        o_p = _push(scratch_ptr, offs, o_p, sqv, oty == 1, lane, BASE=80, CAP=18, STRIDE=STRIDE)
        o_l = _push(scratch_ptr, offs, o_l, sqv, oty == 2, lane, BASE=98, CAP=4, STRIDE=STRIDE)
        o_n = _push(scratch_ptr, offs, o_n, sqv, oty == 3, lane, BASE=102, CAP=4, STRIDE=STRIDE)
        o_s = _push(scratch_ptr, offs, o_s, sqv, oty == 4, lane, BASE=106, CAP=4, STRIDE=STRIDE)
        ogold = (oty == 7) | (oty == 9) | (oty == 10) | (oty == 11) | (oty == 12)
        o_g = _push(scratch_ptr, offs, o_g, sqv, ogold, lane, BASE=110, CAP=34, STRIDE=STRIDE)
        o_b = _push(scratch_ptr, offs, o_b, sqv, oty == 5, lane, BASE=144, CAP=4, STRIDE=STRIDE)
        o_r = _push(scratch_ptr, offs, o_r, sqv, oty == 6, lane, BASE=148, CAP=4, STRIDE=STRIDE)
        o_h = _push(scratch_ptr, offs, o_h, sqv, oty == 13, lane, BASE=152, CAP=4, STRIDE=STRIDE)
        o_d = _push(scratch_ptr, offs, o_d, sqv, oty == 14, lane, BASE=156, CAP=4, STRIDE=STRIDE)
        file_bits = tl.where(ty == 1, file_bits | (tl.full(offs.shape, 1, tl.int32) << (sq // 9)), file_bits)
        take_k = (ty == 8) & (has_k == 0)
        ksq = tl.where(take_k, sqv, ksq)
        has_k = tl.where(ty == 8, 1, has_k)
        take_ok = (oty == 8) & (has_ok == 0)
        oksq = tl.where(take_ok, sqv, oksq)
        has_ok = tl.where(oty == 8, 1, has_ok)

    # 相手の利き。自玉のマスは塞がない（逃げ道の先まで通す）
    blk_lo = own_lo | opp_lo
    blk_mid = own_mid | opp_mid
    blk_hi = own_hi | opp_hi
    blk_lo, blk_mid, blk_hi = _bit_off(blk_lo, blk_mid, blk_hi, ksq, has_k != 0)
    alo, amid, ahi = z, z, z
    # 相手玉
    okx = oksq // 9
    oky = oksq % 9
    ak = (has_ok != 0) & lane
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=0, DY=-1)
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=-1, DY=-1)
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=1, DY=-1)
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=-1, DY=0)
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=1, DY=0)
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=0, DY=1)
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=-1, DY=1)
    alo, amid, ahi = _att_step(ak, okx, oky, alo, amid, ahi, DX=1, DY=1)
    alo, amid, ahi = _opp_kind(o_p, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=80, CAP=18, KIND=0, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_l, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=98, CAP=4, KIND=1, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_n, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=102, CAP=4, KIND=2, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_s, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=106, CAP=4, KIND=3, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_g, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=110, CAP=34, KIND=4, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_b, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=144, CAP=4, KIND=5, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_r, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=148, CAP=4, KIND=6, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_h, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=152, CAP=4, KIND=7, STRIDE=STRIDE)
    alo, amid, ahi = _opp_kind(o_d, scratch_ptr, offs, lane, alo, amid, ahi, blk_lo, blk_mid, blk_hi, BASE=156, CAP=4, KIND=8, STRIDE=STRIDE)

    kx = ksq // 9
    ky = ksq % 9
    n_checks = z
    clo, cmid, chi = z, z, z
    btw_lo, btw_mid, btw_hi = z, z, z
    # 玉から見た 8 方向。STEP/SLIDE は「相手がその向きへ利く駒種」のビット
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=0, DY=-1, AXIS=1, STEP=16274, SLIDE=16452)
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=-1, DY=-1, AXIS=3, STEP=24464, SLIDE=8224)
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=1, DY=-1, AXIS=4, STEP=24464, SLIDE=8224)
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=-1, DY=0, AXIS=2, STEP=16256, SLIDE=16448)
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=1, DY=0, AXIS=2, STEP=16256, SLIDE=16448)
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=0, DY=1, AXIS=1, STEP=16256, SLIDE=16448)
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=-1, DY=1, AXIS=4, STEP=16656, SLIDE=8224)
    n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi = _ray(
        has_k, kx, ky, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
        own_ptr, opp_ptr, pin_ptr, offs, lane, n_checks, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
        DX=1, DY=1, AXIS=3, STEP=16656, SLIDE=8224)
    for ndx in tl.static_range(-1, 2, 2):
        sx = kx + ndx
        sy = ky - 2
        on = (has_k != 0) & (sx >= 0) & (sx < 9) & (sy >= 0)
        sqn = tl.where(on, sx * 9 + sy, 0)
        oty = tl.load(opp_ptr + offs * 81 + sqn, mask=on & lane, other=0).to(tl.int32)
        is_c = on & (oty == 3)
        n_checks = n_checks + is_c.to(tl.int32)
        clo, cmid, chi = _bit_on(clo, cmid, chi, sqn, is_c)
    n_checks = tl.where(has_k != 0, n_checks, z)

    _own_kind(c_p, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=0, CAP=18, KIND=0, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_l, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=18, CAP=4, KIND=1, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_n, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=22, CAP=4, KIND=2, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_s, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=26, CAP=4, KIND=3, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_g, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=30, CAP=34, KIND=4, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_b, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=64, CAP=4, KIND=5, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_r, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=68, CAP=4, KIND=6, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_h, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=72, CAP=4, KIND=7, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)
    _own_kind(c_d, scratch_ptr, pin_ptr, out_ptr, offs, lane, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
              alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi, n_checks,
              BASE=76, CAP=4, KIND=8, STRIDE=STRIDE, ALL_MOVES=ALL_MOVES)

    pax0 = z
    aking = (has_k != 0) & lane
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=0, DY=-1, CAT=0, AXIS=1,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=-1, DY=-1, CAT=1, AXIS=3,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=1, DY=-1, CAT=2, AXIS=4,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=-1, DY=0, CAT=3, AXIS=2,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=1, DY=0, CAT=4, AXIS=2,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=0, DY=1, CAT=5, AXIS=1,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=-1, DY=1, CAT=6, AXIS=4,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)
    _mv_step(aking, kx, ky, pax0, own_lo, own_mid, own_hi, opp_lo, opp_mid, opp_hi,
             alo, amid, ahi, clo, cmid, chi, btw_lo, btw_mid, btw_hi,
             n_checks, out_ptr, offs, lane, DX=1, DY=1, CAT=7, AXIS=3,
             PROM=False, DEAD=0, MUST=False, LANCE=False, IS_KING=True, ALL_MOVES=ALL_MOVES)

    h0 = tl.load(hand_ptr + offs * 7 + 0, mask=lane, other=0).to(tl.int32)
    h1 = tl.load(hand_ptr + offs * 7 + 1, mask=lane, other=0).to(tl.int32)
    h2 = tl.load(hand_ptr + offs * 7 + 2, mask=lane, other=0).to(tl.int32)
    h3 = tl.load(hand_ptr + offs * 7 + 3, mask=lane, other=0).to(tl.int32)
    h4 = tl.load(hand_ptr + offs * 7 + 4, mask=lane, other=0).to(tl.int32)
    h5 = tl.load(hand_ptr + offs * 7 + 5, mask=lane, other=0).to(tl.int32)
    h6 = tl.load(hand_ptr + offs * 7 + 6, mask=lane, other=0).to(tl.int32)
    one8 = tl.full(offs.shape, 1, tl.uint8)
    for sq in tl.range(81, loop_unroll_factor=1):
        sqv = tl.full(offs.shape, sq, tl.int32)
        empty = (_bit_test(own_lo, own_mid, own_hi, sqv) | _bit_test(opp_lo, opp_mid, opp_hi, sqv)) == 0
        btw = _bit_test(btw_lo, btw_mid, btw_hi, sqv) != 0
        can = empty & lane & ((n_checks == 0) | ((n_checks == 1) & btw))
        y = sq % 9
        pawn_free = ((file_bits >> (sq // 9)) & 1) == 0
        tl.store(out_ptr + offs * 2187 + 20 * 81 + sq, one8, mask=can & (h0 > 0) & (y > 0) & pawn_free)
        tl.store(out_ptr + offs * 2187 + 21 * 81 + sq, one8, mask=can & (h1 > 0) & (y > 0))
        tl.store(out_ptr + offs * 2187 + 22 * 81 + sq, one8, mask=can & (h2 > 0) & (y > 1))
        tl.store(out_ptr + offs * 2187 + 23 * 81 + sq, one8, mask=can & (h3 > 0))
        tl.store(out_ptr + offs * 2187 + 24 * 81 + sq, one8, mask=can & (h4 > 0))
        tl.store(out_ptr + offs * 2187 + 25 * 81 + sq, one8, mask=can & (h5 > 0))
        tl.store(out_ptr + offs * 2187 + 26 * 81 + sq, one8, mask=can & (h6 > 0))


def _generate(own: torch.Tensor, opp: torch.Tensor, own_hand: torch.Tensor, all_moves: bool) -> torch.Tensor:
    """own, opp: (N,81) int8。own_hand: (N,7)。返り値は (N,2187) bool。打ち歩詰めは見ない。"""
    n = own.shape[0]
    if n == 0:
        return torch.zeros(0, 2187, dtype=torch.bool, device=own.device)
    n_pad = (n + BLOCK - 1) // BLOCK * BLOCK
    scratch = torch.empty(n_pad, STRIDE_N, dtype=torch.int32, device=own.device)
    pin = torch.zeros(n_pad, 81, dtype=torch.int8, device=own.device)
    out = torch.zeros(n, 2187, dtype=torch.uint8, device=own.device)
    _kernel[(n_pad // BLOCK,)](
        own, opp, own_hand, scratch, pin, out, n,
        ALL_MOVES=all_moves, BLOCK=BLOCK, num_warps=1,
    )
    return out.bool()


def _remove_uchifuzume(out, own, opp, own_hand, opp_hand):
    n = own.shape[0]
    okmask = opp == K
    oksq = okmask.float().argmax(dim=1)
    ox, oy = oksq // 9, oksq % 9
    dsq = ox * 9 + (oy + 1).clamp(max=8)
    view = out.view(n, 27, 81)
    cand = okmask.any(1) & (oy < 8) & view[:, 20].gather(1, dsq.unsqueeze(1)).squeeze(1)
    idx = cand.nonzero(as_tuple=True)[0]
    m = int(idx.numel())
    if m == 0:
        return out
    d = dsq[idx]
    own_s = own[idx].clone()
    own_s.scatter_(1, d.unsqueeze(1), torch.ones(m, 1, dtype=own.dtype, device=own.device))
    reply = _generate(opp[idx].flip(1).contiguous(), own_s.flip(1).contiguous(), opp_hand[idx].contiguous(), True)
    mated = ~reply.view(m, -1).any(1)
    view[idx, 20, d] = view[idx, 20, d] & ~mated
    return view.reshape(n, 27 * 81)


def legal_move_mask_list(features: torch.Tensor, all_moves: bool = True, check_uchifuzume: bool = True) -> torch.Tensor:
    """(N,447) → (N,2187)。CUDA 上の駒リストカーネル。"""
    if features.device.type != "cuda":
        raise RuntimeError("legal_move_mask_list は CUDA が必要です")
    own, opp, own_hand, opp_hand = decode_features(features)
    n = own.shape[0]
    own = own.reshape(n, 81).to(torch.int8).contiguous()
    opp = opp.reshape(n, 81).to(torch.int8).contiguous()
    own_hand = own_hand.contiguous()
    opp_hand = opp_hand.contiguous()
    out = _generate(own, opp, own_hand, all_moves)
    if check_uchifuzume:
        out = _remove_uchifuzume(out, own, opp, own_hand, opp_hand)
    return out
