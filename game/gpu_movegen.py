"""
game/gpu_movegen.py
GPU でまとめて合法手を作る（ビットボード演算版）

【ねらい】
NN の入力（447bit 特徴量）は GPU 上にあるので、合法手も GPU 上で作れれば
「特徴量 → 合法手マスク → 方策の softmax」を CPU に戻さずに済む。
ここでは N 局面分の合法手を、マス集合（ビットボード）の論理演算とシフトだけで一度に作る。
すべて PyTorch のテンソル演算なので、CPU でも GPU（CUDA）でもそのまま動く。

【入力】 features: (N, 447) の特徴量（game/board.py の make_bit447 と同じ）
   - 手番側から見た盤（後手番は 180 度回転済み）なので、「自分の駒は上に進む」として計算できる
【出力】 (N, 2187) の bool テンソル。方策ラベル（dir*81 + to）ごとに合法なら True
   - 合法手と方策ラベルは 1 対 1 に対応する（走り駒は途中で止まるので、同じラベルの手は 1 つだけ）

【扱うルール】
   - 駒の動き（走り駒は駒にぶつかると止まる）、成り・不成（行き所のない駒は不成不可）
   - 自玉への王手放置の禁止: 王手している駒の数、合い駒・取る手、ピン（釘付け）、玉の逃げ先
   - 打つ手: 二歩、行き所のない駒を打てない、打ち歩詰め
   - all_moves=True で cshogi の legal_moves（LegalAll）と完全に同じ
   - all_moves=False で「ほぼ指す意味のない不成」を省く（探索の幅を減らす用）:
     成れるときの歩・角・飛の不成、香の2段目への不成、香の3段目への駒を取らない不成
     ※ cshogi の C++ にある Legal とは一致しない（あちらは不成をほとんど残す）

盤のマス番号は sq = x*9 + y（x: 筋側の番号、y: 段。y=0 が敵陣の奥）。
"""

from __future__ import annotations

from typing import Optional

import torch

# ── 駒の種類（Apery / cshogi と同じ番号） ───────────────────────
P, L, N_, S, B, R, G, K = 1, 2, 3, 4, 5, 6, 7, 8
PP, PL, PN, PS, HORSE, DRAGON = 9, 10, 11, 12, 13, 14

# 方策ラベルの方向（game/board.py と同じ）
UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT = range(10)

# 8 方向 (dx, dy)。y が小さい方が「上」
DIRS8 = [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)]
KNIGHT_DIRS = [(-1, -2), (1, -2)]

_GOLDS = (G, PP, PL, PN, PS)


def _category(dx: int, dy: int) -> int:
    """(dx, dy) の向きから方策ラベルの方向番号を決める（game/board.py の make_move_label と同じ規則）。"""
    if dy < 0:
        if dx == 0:
            return UP
        if dy == -2 and dx == 1:
            return UP2_RIGHT
        if dy == -2 and dx == -1:
            return UP2_LEFT
        return UP_RIGHT if dx > 0 else UP_LEFT
    if dy == 0:
        return RIGHT if dx > 0 else LEFT
    if dx == 0:
        return DOWN
    return DOWN_RIGHT if dx > 0 else DOWN_LEFT


def _axis(dx: int, dy: int) -> int:
    """方向の「線の種類」: 1=筋(縦), 2=段(横), 3=斜め(\\), 4=斜め(/)。"""
    if dx == 0:
        return 1
    if dy == 0:
        return 2
    return 3 if dx == dy else 4


def _step_pieces(dx: int, dy: int) -> tuple:
    """自分の駒（上向き）で、(dx,dy) へ 1 マス動ける駒の種類。"""
    s = [K]
    if (dx, dy) == (0, -1):
        s += [P, S, *_GOLDS, HORSE]
    elif (dx, dy) in ((-1, -1), (1, -1)):
        s += [S, *_GOLDS, DRAGON]
    elif (dx, dy) in ((-1, 0), (1, 0)):
        s += [*_GOLDS, HORSE]
    elif (dx, dy) == (0, 1):
        s += [*_GOLDS, HORSE]
    elif (dx, dy) in ((-1, 1), (1, 1)):
        s += [S, DRAGON]
    return tuple(s)


def _slide_pieces(dx: int, dy: int) -> tuple:
    """自分の駒（上向き）で、(dx,dy) の方向に走れる駒の種類。"""
    if dx == 0 or dy == 0:
        return (R, DRAGON, L) if (dx, dy) == (0, -1) else (R, DRAGON)
    return (B, HORSE)


def _in_set(t: torch.Tensor, types: tuple) -> torch.Tensor:
    m = torch.zeros_like(t, dtype=torch.bool)
    for v in types:
        m |= t == v
    return m


def shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """(N,9,9) の盤を (dx,dy) だけずらす（盤外に出た分は捨て、空いた所は 0）。ビットボードのシフトに当たる。"""
    out = torch.zeros_like(x)
    xs = slice(max(0, -dx), 9 - max(0, dx))
    xd = slice(max(0, dx), 9 - max(0, -dx))
    ys = slice(max(0, -dy), 9 - max(0, dy))
    yd = slice(max(0, dy), 9 - max(0, -dy))
    out[:, xd, yd] = x[:, xs, ys]
    return out


# ── 盤の幾何テーブル（ピン・王手の判定用） ───────────────────────
def _build_rays() -> torch.Tensor:
    """rays[sq, d, k]: マス sq から方向 d に k+1 マス進んだマス（盤外は 81）。"""
    t = torch.full((81, 8, 8), 81, dtype=torch.long)
    for sq in range(81):
        x0, y0 = divmod(sq, 9)
        for d, (dx, dy) in enumerate(DIRS8):
            x, y = x0, y0
            for k in range(8):
                x += dx
                y += dy
                if not (0 <= x < 9 and 0 <= y < 9):
                    break
                t[sq, d, k] = x * 9 + y
    return t


_RAYS_CPU = _build_rays()
_TABLE_CACHE: dict = {}


def _tables(device):
    key = str(device)
    if key not in _TABLE_CACHE:
        sq = torch.arange(81, device=device)
        _TABLE_CACHE[key] = dict(
            rays=_RAYS_CPU.to(device),
            sq_grid=sq.view(1, 9, 9).to(torch.int16),
            y_grid=(sq % 9).view(1, 9, 9),
            x_of=(sq // 9),
        )
    return _TABLE_CACHE[key]


# ── 特徴量 → 盤 ──────────────────────────────────────────────
_HAND_WIDTHS = (5, 3, 3, 3, 3, 2, 2)  # 歩 香 桂 銀 金 角 飛


def decode_features(features: torch.Tensor):
    """
    (N,447) → own(N,9,9), opp(N,9,9), own_hand(N,7), opp_hand(N,7)
    own/opp は駒の種類（0=なし）。持ち駒の順は 歩 香 桂 銀 金 角 飛。
    """
    n = features.shape[0]
    f = features[:, :447].round().to(torch.int16)
    bits = f[:, :405].view(n, 81, 5)
    pt = bits[:, :, 1] * 8 + bits[:, :, 2] * 4 + bits[:, :, 3] * 2 + bits[:, :, 4]
    is_opp = bits[:, :, 0] == 1
    own = torch.where(is_opp, torch.zeros_like(pt), pt).view(n, 9, 9)
    opp = torch.where(is_opp, pt, torch.zeros_like(pt)).view(n, 9, 9)
    hands = []
    k = 405
    for _side in range(2):
        cols = []
        for w in _HAND_WIDTHS:
            v = torch.zeros(n, dtype=torch.int16, device=features.device)
            for i in range(w):
                v = v * 2 + f[:, k + i]
            cols.append(v)
            k += w
        hands.append(torch.stack(cols, dim=1))
    return own, opp, hands[0], hands[1]


# ── 利き（attack）の計算 ─────────────────────────────────────
def attacks(pieces: torch.Tensor, occ: torch.Tensor, upward: bool, early_exit: bool = False) -> torch.Tensor:
    """
    pieces: (N,9,9) 駒の種類（0=なし）、occ: (N,9,9) bool 駒があるマス
    upward=True なら上に進む側（手番側）、False なら下に進む側（相手）。
    返り値: (N,9,9) bool その駒たちが利いているマス（味方・敵の駒があるマスも含む）
    """
    sgn = 1 if upward else -1
    att = torch.zeros_like(occ)
    empty = ~occ
    for dx, dy in DIRS8:
        ddy = dy * sgn
        st = _in_set(pieces, _step_pieces(dx, dy))
        att |= shift(st, dx, ddy)
        sl = _in_set(pieces, _slide_pieces(dx, dy))
        cur = shift(sl, dx, ddy)
        for _ in range(8):
            if early_exit and not bool(cur.any()):
                break
            att |= cur
            cur = shift(cur & empty, dx, ddy)
    for dx, dy in KNIGHT_DIRS:
        att |= shift(pieces == N_, dx, dy * sgn)
    return att


# ── 合法手生成 ──────────────────────────────────────────────
def legal_move_mask(
    features: torch.Tensor,
    all_moves: bool = True,
    check_uchifuzume: bool = True,
    core=None,
) -> torch.Tensor:
    """
    (N,447) の特徴量から (N,2187) の合法手マスクを作る。
    core に make_compiled_core() の結果を渡すと、torch.compile で速くした版を使う。
    """
    own, opp, own_hand, opp_hand = decode_features(features)
    return legal_move_mask_from_board(own, opp, own_hand, opp_hand, all_moves, check_uchifuzume, core)


def legal_move_mask_from_board(own, opp, own_hand, opp_hand, all_moves=True, check_uchifuzume=True, core=None):
    if core is None:
        early = own.device.type == "cpu"  # CPU では駒がないとき途中で打ち切る（GPU では同期が遅いので打ち切らない）
        out, cand, dsq = _core(own, opp, own_hand, opp_hand, all_moves, early)
    else:
        out, cand, dsq = core(own, opp, own_hand, opp_hand, all_moves)
    if check_uchifuzume:
        out = _remove_uchifuzume(out, cand, dsq, own, opp, own_hand, opp_hand, core)
    return out.view(own.shape[0], 27 * 81)


def _core(own, opp, own_hand, opp_hand, all_moves: bool = True, early_exit: bool = False):
    """
    合法手生成の本体（打ち歩詰め以外）。データに応じた分岐がないので torch.compile しやすい。
    返り値: out (N,27,81) / 打ち歩詰めの候補か (N,) / 候補のマス (N,)
    """
    device = own.device
    n = own.shape[0]
    T = _tables(device)
    own_occ = own > 0
    opp_occ = opp > 0
    occ = own_occ | opp_occ
    empty = ~occ
    y = T["y_grid"]
    out = torch.zeros(n, 27, 9, 9, dtype=torch.bool, device=device)

    # ── 自玉の位置・王手・ピン ─────────────────────────────
    king_mask = (own == K).view(n, 81)
    has_king = king_mask.any(dim=1)
    ksq = king_mask.float().argmax(dim=1)                          # (N,)
    pad_occ = torch.cat([occ.view(n, 81), torch.zeros(n, 1, dtype=torch.bool, device=device)], 1)
    pad_own = torch.cat([own.view(n, 81), torch.zeros(n, 1, dtype=own.dtype, device=device)], 1)
    pad_opp = torch.cat([opp.view(n, 81), torch.zeros(n, 1, dtype=opp.dtype, device=device)], 1)
    rays = T["rays"][ksq]                                          # (N,8,8) 玉から各方向のマス
    ray_occ = torch.gather(pad_occ, 1, rays.view(n, 64)).view(n, 8, 8)
    ray_own = torch.gather(pad_own, 1, rays.view(n, 64)).view(n, 8, 8)
    ray_opp = torch.gather(pad_opp, 1, rays.view(n, 64)).view(n, 8, 8)
    ar8 = torch.arange(8, device=device).view(1, 1, 8)
    big = torch.full_like(ray_occ, 99, dtype=torch.long)
    occ_idx = torch.where(ray_occ, ar8.expand_as(ray_occ), big)
    first = occ_idx.min(dim=2).values                               # (N,8) 最初の駒の位置（99=なし）
    occ_idx2 = torch.where(occ_idx > first.unsqueeze(2), occ_idx, big)
    second = occ_idx2.min(dim=2).values                             # 2番目の駒の位置

    def at(t, idx):
        return torch.gather(t, 2, idx.clamp(max=7).unsqueeze(2)).squeeze(2)

    first_opp = at(ray_opp, first)
    first_own = at(ray_own, first)
    second_opp = at(ray_opp, second)
    checker_sq = torch.zeros(n, 82, dtype=torch.bool, device=device)  # 81 列目は捨て場
    between_sq = torch.zeros(n, 81, dtype=torch.bool, device=device)
    n_checks = torch.zeros(n, dtype=torch.long, device=device)
    pin_axis = torch.zeros(n, 82, dtype=torch.long, device=device)
    for d, (dx, dy) in enumerate(DIRS8):
        # 相手の駒（下向き）がマス s から玉の方へ (-dx,-dy) 向きに利くか。
        # 相手の (-dx,-dy) は、自分（上向き）に直すと (-dx, +dy)
        sdx, sdy = -dx, dy
        step_t = _step_pieces(sdx, sdy)
        slide_t = _slide_pieces(sdx, sdy)
        fo = first_opp[:, d]
        has_first = first[:, d] < 99
        is_check = has_first & (
            _in_set(fo, slide_t) | ((first[:, d] == 0) & _in_set(fo, step_t))
        )
        n_checks += is_check.long()
        idx = rays[:, d, :]                                         # (N,8)
        csq = torch.gather(idx, 1, first[:, d].clamp(max=7).unsqueeze(1)).squeeze(1)
        checker_sq.scatter_(1, torch.where(is_check, csq, torch.full_like(csq, 81)).unsqueeze(1),
                            is_check.unsqueeze(1))
        btw = (ar8.view(1, 8) < first[:, d].unsqueeze(1)) & is_check.unsqueeze(1)
        bsq = torch.where(btw, idx, torch.full_like(idx, 81))
        tmp = torch.zeros(n, 82, dtype=torch.bool, device=device)
        tmp.scatter_(1, bsq, btw)
        between_sq |= tmp[:, :81]
        # ピン: 最初が自分の駒（玉以外）、2番目が相手の走り駒で、こちらに利く
        so = second_opp[:, d]
        pinned = has_first & (first_own[:, d] > 0) & (second[:, d] < 99) & _in_set(so, slide_t)
        psq = torch.where(pinned, csq, torch.full_like(csq, 81))
        pin_axis.scatter_(1, psq.unsqueeze(1), torch.full_like(psq, _axis(dx, dy)).unsqueeze(1))
    # 桂馬の王手（相手の桂は下に跳ぶので、玉から見て (∓1,-2) の位置）
    kx, ky = ksq // 9, ksq % 9
    for dx in (-1, 1):
        sx, sy = kx + dx, ky - 2
        ok = (sx >= 0) & (sx < 9) & (sy >= 0)
        s = torch.where(ok, sx * 9 + sy, torch.zeros_like(sx))
        is_check = ok & (torch.gather(pad_opp, 1, s.unsqueeze(1)).squeeze(1) == N_)
        n_checks += is_check.long()
        checker_sq.scatter_(1, torch.where(is_check, s, torch.full_like(s, 81)).unsqueeze(1),
                            is_check.unsqueeze(1))
    checker_sq = checker_sq[:, :81]
    # 自玉がない局面（詰将棋など）では王手もピンもない
    pin_axis = torch.where(has_king.view(n, 1), pin_axis[:, :81], torch.zeros_like(pin_axis[:, :81])).view(n, 9, 9)
    n_checks = torch.where(has_king, n_checks, torch.zeros_like(n_checks))

    # 王手されているときの「玉以外の手」の行き先: 王手駒を取る or 間に入る
    target = torch.where(
        (n_checks == 0).view(n, 1),
        torch.ones(n, 81, dtype=torch.bool, device=device),
        (checker_sq | between_sq) & (n_checks == 1).view(n, 1),
    ).view(n, 9, 9)
    drop_target = torch.where(
        (n_checks == 0).view(n, 1),
        torch.ones(n, 81, dtype=torch.bool, device=device),
        between_sq & (n_checks == 1).view(n, 1),
    ).view(n, 9, 9)

    # 玉の逃げ先チェック用: 自玉を取り除いた盤での相手の利き
    occ_no_king = occ & ~(own == K)
    opp_att = attacks(opp, occ_no_king, upward=False, early_exit=early_exit)

    # ── 盤上の駒を動かす手 ─────────────────────────────────
    sq_grid = T["sq_grid"].expand(n, 9, 9)
    not_own = ~own_occ
    is_king_piece = own == K

    def emit(cat, mover_t, mover_from, dx, dy, axis=None):
        """方向 cat への手（to ごとに動かした駒の種類と元のマスが分かっている）を出力に足す。"""
        valid = (mover_t > 0) & not_own
        is_k = mover_t == K
        # 玉の手: 相手の利きがないマスだけ
        ok_k = is_k & ~opp_att
        # 玉以外: 王手への対応 + ピン
        f = mover_from.long().clamp(0, 80)
        pax = torch.gather(pin_axis.view(n, 81), 1, f.view(n, 81)).view(n, 9, 9)
        ax = _axis(dx, dy) if axis is None else axis
        ok_o = ~is_k & target & ((pax == 0) | (pax == ax))
        legal = valid & (ok_k | ok_o)
        from_y = (f % 9)
        zone = (y <= 2) | (from_y <= 2)
        t = mover_t
        promotable = (t >= P) & (t <= R)
        can_promote = legal & promotable & zone
        # 不成: 行き所のない駒はだめ
        dead = (((t == P) | (t == L)) & (y == 0)) | ((t == N_) & (y <= 1))
        nonpromo = legal & ~dead
        if not all_moves:
            # 意味の薄い不成を省く: 歩・角・飛は成れるなら成る。香の2段目の不成、3段目への駒を取らない不成も省く
            useless = promotable & zone & ((t == P) | (t == B) | (t == R))
            useless |= (t == L) & ((y == 1) | ((y == 2) & ~opp_occ))
            nonpromo &= ~useless
        out[:, cat] |= nonpromo
        out[:, cat + 10] |= can_promote

    for dx, dy in DIRS8:
        cat = _category(dx, dy)
        step_t = _step_pieces(dx, dy)
        slide_t = _slide_pieces(dx, dy)
        st_mask = _in_set(own, step_t)
        mt = shift(torch.where(st_mask, own, torch.zeros_like(own)), dx, dy)
        mf = shift(torch.where(st_mask, sq_grid, torch.zeros_like(sq_grid)), dx, dy)
        sl_mask = _in_set(own, slide_t)
        if not early_exit or bool(sl_mask.any()):
            ct = shift(torch.where(sl_mask, own, torch.zeros_like(own)), dx, dy)
            cf = shift(torch.where(sl_mask, sq_grid, torch.zeros_like(sq_grid)), dx, dy)
            for _ in range(8):
                live = ct > 0
                if early_exit and not bool(live.any()):
                    break
                # 走り駒が届くマスには、ほかの駒が同じ向きで来ることはない（途中のマスが空なので）
                mt = torch.where(live, ct, mt)
                mf = torch.where(live, cf, mf)
                go = live & empty
                ct = shift(torch.where(go, ct, torch.zeros_like(ct)), dx, dy)
                cf = shift(torch.where(go, cf, torch.zeros_like(cf)), dx, dy)
        emit(cat, mt, mf, dx, dy)

    for dx, dy in KNIGHT_DIRS:
        cat = _category(dx, dy)
        nm = own == N_
        mt = shift(torch.where(nm, own, torch.zeros_like(own)), dx, dy)
        mf = shift(torch.where(nm, sq_grid, torch.zeros_like(sq_grid)), dx, dy)
        emit(cat, mt, mf, dx, dy, axis=-1)  # 桂はピンされていたら動けない

    # ── 持ち駒を打つ手 ─────────────────────────────────────
    has_hand = own_hand > 0                                         # (N,7)
    can_drop_sq = empty & drop_target
    pawn_files = (own == P).any(dim=2, keepdim=True)                # (N,9,1) 二歩
    for hp in range(7):
        ok = can_drop_sq & has_hand[:, hp].view(n, 1, 1)
        if hp in (0, 1):  # 歩・香は 1段目に打てない
            ok = ok & (y > 0)
        if hp == 2:       # 桂は 1・2段目に打てない
            ok = ok & (y > 1)
        if hp == 0:
            ok = ok & ~pawn_files
        out[:, 20 + hp] |= ok

    out = out.view(n, 27, 81)

    # 打ち歩詰めの候補: 相手玉の 1 マス前（自分から見て下）に歩を打つ手
    okmask = (opp == K).view(n, 81)
    oksq = okmask.float().argmax(dim=1)
    ox, oy = oksq // 9, oksq % 9
    dsq = ox * 9 + (oy + 1).clamp(max=8)
    cand = okmask.any(dim=1) & (oy < 8)
    cand = cand & out[:, 20, :].gather(1, dsq.unsqueeze(1)).squeeze(1)
    return out, cand, dsq


def make_compiled_core(mode: str = "max-autotune-no-cudagraphs"):
    """
    torch.compile で合法手生成の本体をまとめて（カーネル融合して）速くする。
    GPU では Triton、CPU では C++ のコードが作られる。最初の 1 回はコンパイルに時間がかかる。
    """

    def fn(own, opp, own_hand, opp_hand, all_moves=True):
        return _core(own, opp, own_hand, opp_hand, all_moves, False)

    return torch.compile(fn, mode=mode, dynamic=False)


def _remove_uchifuzume(out, cand, dsq, own, opp, own_hand, opp_hand, core=None):
    """歩を打って王手したとき、相手に合法手がなければ（打ち歩詰め）その手を取り除く。"""
    idx = cand.nonzero(as_tuple=True)[0]
    m = idx.numel()
    if m == 0:
        return out
    # 形を 2 のべき乗にそろえる（compile 版の作り直しを減らすため）。足りない分は最初の局面をくり返す
    mm = 1 << (m - 1).bit_length()
    if mm != m:
        idx_p = torch.cat([idx, idx[:1].expand(mm - m)])
    else:
        idx_p = idx
    # 歩を打ったあとの局面を、相手の立場（180度回転）で作る
    own_s, opp_s = own[idx_p].clone().view(mm, 81), opp[idx_p].view(mm, 81)
    d = dsq[idx_p]
    own_s.scatter_(1, d.unsqueeze(1), torch.full_like(d, P, dtype=own_s.dtype).unsqueeze(1))
    hand_s = own_hand[idx_p].clone()
    hand_s[:, 0] -= 1
    new_own = opp_s.flip(1).view(mm, 9, 9)
    new_opp = own_s.flip(1).view(mm, 9, 9)
    if core is None:
        reply, _, _ = _core(new_own, new_opp, opp_hand[idx_p], hand_s, True, own.device.type == "cpu")
    else:
        reply, _, _ = core(new_own, new_opp, opp_hand[idx_p], hand_s, True)
    mated = ~reply.view(mm, -1).any(dim=1)[:m]
    # 詰んでいる局面だけ、歩を打つ手を消す（データに応じた分岐を避けて scatter で書く）
    flat = out.view(out.shape[0], 27 * 81)
    pos = 20 * 81 + dsq[idx]
    keep = flat[idx, pos] & ~mated
    flat[idx, pos] = keep
    return out


# ── 確認用: ラベル → cshogi の指し手（CPU） ─────────────────────
def labels_from_cshogi(board) -> set:
    """cshogi の Board の合法手（LegalAll）を方策ラベルの集合にする（テスト用）。"""
    from game.board import make_move_label

    return {make_move_label(m, board.turn) for m in board.legal_moves}
