"""
game/board.py
将棋の盤面操作・447bit入力特徴量生成・指し手ラベル変換を担当するモジュール。

【447bit盤面表現の設計】
  盤面 (405bit) : 81マス × 5bit
      bit4      : 1=相手の駒 / 0=自分の駒
      bit3-0    : 駒種 (0=空, 1=歩, ..., 14=龍)
  持ち駒 (42bit) : 自分21bit + 相手21bit
      歩   : 5bit (最大18枚)
      金   : 3bit (最大4枚)
      銀   : 3bit (最大4枚)
      桂   : 3bit (最大4枚)
      香   : 3bit (最大4枚)
      飛車 : 2bit (最大2枚)
      角   : 2bit (最大2枚)
  合計 405 + 42 = 447bit → Python list[int] (各要素0/1)
"""

from __future__ import annotations

import cshogi
from cshogi import (
    BLACK,
    WHITE,
    PIECE_TYPES,
    HAND_PIECES,
    move_is_drop,
    move_to,
    move_from,
    move_is_promotion,
    move_drop_hand_piece,
    BLACK_WIN,
    WHITE_WIN,
)
import numpy as np

# ───────────────────────────────────────────────
# 移動方向定数
# ───────────────────────────────────────────────
MOVE_DIRECTION = [
    UP,
    UP_LEFT,
    UP_RIGHT,
    LEFT,
    RIGHT,
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    UP2_LEFT,
    UP2_RIGHT,
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE,
] = range(20)

MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(HAND_PIECES)  # 20 + 7 = 27
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81                    # 2187

# ───────────────────────────────────────────────
# 447bit 盤面表現
# ───────────────────────────────────────────────

# 持ち駒の (最大枚数, 必要ビット幅) テーブル
# HAND_PIECES = (PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK)
_HAND_BITS: list[tuple[int, int]] = [
    (18, 5),  # 歩  : 最大18枚 → 5bit (0〜18 は 5bit で表現可能)
    (4,  3),  # 香  : 最大 4枚 → 3bit
    (4,  3),  # 桂  : 最大 4枚 → 3bit
    (4,  3),  # 銀  : 最大 4枚 → 3bit
    (4,  3),  # 金  : 最大 4枚 → 3bit
    (2,  2),  # 角  : 最大 2枚 → 2bit
    (2,  2),  # 飛車: 最大 2枚 → 2bit
]
# 合計 5+3+3+3+3+2+2 = 21bit × 2(自他) = 42bit
HAND_BITS_TOTAL  = sum(b for _, b in _HAND_BITS)  # 21
BOARD_BITS_TOTAL = 81 * 5                          # 405
TOTAL_BITS       = BOARD_BITS_TOTAL + HAND_BITS_TOTAL * 2  # 447


def _int_to_bits(value: int, width: int) -> list[int]:
    """value を width ビットの MSB-first リストに変換する。"""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def make_bit447(board: cshogi.Board) -> list[int]:
    """
    盤面を 447bit のリスト (各要素 0 or 1) に変換して返す。

    自分 = board.turn 側の駒が bit4=0、相手側が bit4=1。
    """
    bits: list[int] = []

    # ── 盤面 405bit ──────────────────────────────
    # cshogi では square=0 が 9一（左上）、square=80 が 1九（右下）。
    # 手番によらず常に「自分の陣地が下」になるよう、後手番は盤を回転する。
    turn = board.turn
    pieces = board.pieces  # length=81, piece=color_piece (0=空)

    for sq in range(81):
        # 後手番のとき盤を 180度回転（80-sq）
        raw_sq = (80 - sq) if turn == WHITE else sq
        piece = pieces[raw_sq]

        if piece == 0:
            # 空マス: 5bit 全て 0
            bits.extend([0, 0, 0, 0, 0])
        else:
            piece_type = cshogi.piece_to_piece_type(piece)
            # 手番と異なる色かどうかを判定
            if turn == BLACK:
                is_opponent = (piece != piece_type)  # 後手駒なら True
            else:
                is_opponent = (piece == piece_type)  # 先手駒なら True（後手視点）
            bits.append(1 if is_opponent else 0)
            bits.extend(_int_to_bits(piece_type, 4))

    # ── 持ち駒 42bit (自分 21bit → 相手 21bit) ──
    # board.pieces_in_hand は (先手の持ち駒tuple, 後手の持ち駒tuple)
    own_hand, opp_hand = (
        (board.pieces_in_hand[BLACK], board.pieces_in_hand[WHITE])
        if turn == BLACK
        else (board.pieces_in_hand[WHITE], board.pieces_in_hand[BLACK])
    )

    for hand in (own_hand, opp_hand):
        for piece_idx, (_, width) in enumerate(_HAND_BITS):
            count = hand[piece_idx]
            bits.extend(_int_to_bits(count, width))

    assert len(bits) == TOTAL_BITS, f"447bit error: got {len(bits)}"
    return bits


def bit447_to_numpy(board: cshogi.Board) -> np.ndarray:
    """make_bit447 の結果を float32 の 1-D numpy 配列で返す。"""
    return np.array(make_bit447(board), dtype=np.float32)


# ───────────────────────────────────────────────
# 指し手ラベル変換
# ───────────────────────────────────────────────

def make_move_label(move: int, color: int) -> int:
    """
    指し手を方策ネットワークの出力インデックスに変換する。
    返り値の範囲: 0 〜 MOVE_LABELS_NUM-1 (= 2186)
    """
    if not move_is_drop(move):
        to_sq: int = move_to(move)
        from_sq: int = move_from(move)
        if color == WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y

        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_LEFT
            elif dir_x > 0:
                move_direction = UP_RIGHT
            else:
                move_direction = UP_LEFT
        elif dir_y == 0:
            move_direction = RIGHT if dir_x > 0 else LEFT
        else:
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x > 0:
                move_direction = DOWN_RIGHT
            else:
                move_direction = DOWN_LEFT

        if move_is_promotion(move):
            move_direction += 10
    else:
        to_sq = move_to(move)
        if color == WHITE:
            to_sq = 80 - to_sq
        move_direction = len(MOVE_DIRECTION) + move_drop_hand_piece(move)

    return int(move_direction * 81 + to_sq)


# ───────────────────────────────────────────────
# 対局結果 → 報酬変換
# ───────────────────────────────────────────────

def make_result(game_result: int, color: int) -> float:
    """
    対局結果（BLACK_WIN / WHITE_WIN）を手番側の報酬 (0.0〜1.0) に変換する。
    引き分けは 0.5。
    """
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1.0
        if game_result == WHITE_WIN:
            return 0.0
    else:
        if game_result == BLACK_WIN:
            return 0.0
        if game_result == WHITE_WIN:
            return 1.0
    return 0.5
