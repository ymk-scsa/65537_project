// apple_core.hpp
// C++ 探索の共通部品
//   - cshogi(Apery) の盤面の初期化
//   - 447bit 特徴量（game/board.py の make_bit447 と同じ並び）
//   - 指し手ラベル（game/board.py の make_move_label と同じ計算）
//   - 局面の保持（sfen + 指し手列）
//   - 終局判定（詰み・千日手・最大手数・入玉宣言）
//   - 評価関数（NN）の呼び出し口 Evaluator
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "init.hpp"
#include "position.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "usi.hpp"
#include "mate.h"
#include "dfpn.h"

namespace apple {

constexpr int kFeatures = 447;
constexpr int kLabels = 27 * 81;  // 2187
constexpr int kMaxPly = 1024;

inline void init_tables() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        initTable();
        Position::initZobrist();
        HuffmanCodedPos::init();
    });
}

// ─────────────────────────────────────────────────────────
// 447bit 特徴量（手番側から見た盤。後手番は 180 度回転）
// ─────────────────────────────────────────────────────────
inline void make_features(const Position& pos, float* out) {
    const Color us = pos.turn();
    int k = 0;
    for (int sq = 0; sq < 81; ++sq) {
        const int raw = (us == White) ? 80 - sq : sq;
        const Piece pc = pos.piece(static_cast<Square>(raw));
        if (pc == Empty) {
            for (int i = 0; i < 5; ++i) out[k++] = 0.0f;
            continue;
        }
        const int pt = static_cast<int>(pc) & 15;
        const bool white_piece = static_cast<int>(pc) >= 16;
        const bool opponent = (us == Black) ? white_piece : !white_piece;
        out[k++] = opponent ? 1.0f : 0.0f;
        for (int b = 3; b >= 0; --b) out[k++] = static_cast<float>((pt >> b) & 1);
    }
    static const int widths[7] = {5, 3, 3, 3, 3, 2, 2};  // 歩 香 桂 銀 金 角 飛
    const Hand hands[2] = {pos.hand(us), pos.hand(oppositeColor(us))};
    for (int side = 0; side < 2; ++side) {
        for (int hp = 0; hp < 7; ++hp) {
            const int cnt = static_cast<int>(hands[side].numOf(static_cast<HandPiece>(hp)));
            for (int b = widths[hp] - 1; b >= 0; --b) out[k++] = static_cast<float>((cnt >> b) & 1);
        }
    }
}

// ─────────────────────────────────────────────────────────
// 指し手ラベル（dir * 81 + to）。game/board.py と同じ方向の決め方。
// ─────────────────────────────────────────────────────────
enum Dir {
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT
};

inline int make_move_label(const Move move, const Color color) {
    int to = static_cast<int>(move.to());
    if (!move.isDrop()) {
        int from = static_cast<int>(move.from());
        if (color == White) {
            to = 80 - to;
            from = 80 - from;
        }
        const int to_x = to / 9, to_y = to % 9;
        const int from_x = from / 9, from_y = from % 9;
        const int dx = to_x - from_x, dy = to_y - from_y;
        int d;
        if (dy < 0) {
            if (dx == 0) d = UP;
            else if (dy == -2 && dx == 1) d = UP2_RIGHT;
            else if (dy == -2 && dx == -1) d = UP2_LEFT;
            else if (dx > 0) d = UP_RIGHT;
            else d = UP_LEFT;
        } else if (dy == 0) {
            d = dx > 0 ? RIGHT : LEFT;
        } else {
            if (dx == 0) d = DOWN;
            else if (dx > 0) d = DOWN_RIGHT;
            else d = DOWN_LEFT;
        }
        if (move.isPromotion()) d += 10;
        return d * 81 + to;
    }
    if (color == White) to = 80 - to;
    const int d = 20 + static_cast<int>(move.handPieceDropped());
    return d * 81 + to;
}

// ─────────────────────────────────────────────────────────
// 評価関数の呼び出し口
//   feats : n × 447
//   turns : n（0=先手番, 1=後手番）
//   policy: n × 2187 のロジット（出力）
//   value : n（手番側から見た -1〜+1、出力）
// ─────────────────────────────────────────────────────────
struct Evaluator {
    virtual ~Evaluator() = default;
    virtual void evaluate(int n, const float* feats, const int* turns, float* policy, float* value) = 0;
};

// ─────────────────────────────────────────────────────────
// 探索の設定
// ─────────────────────────────────────────────────────────
struct SearchConfig {
    int batch_size = 32;           // NN にまとめて渡す局面数
    float c_puct = 1.5f;           // MCTS の探索の広さ
    float c_puct_base = 19652.0f;  // 訪問回数に応じて c_puct を少し増やす（AlphaZero 式）
    float fpu_reduction = 0.2f;    // 未訪問の手の評価を親の評価からどれだけ下げるか
    float softmax_temperature = 1.0f;
    float draw_value_black = 0.4f; // 引き分けの勝ち点（電竜戦: 先手 0.4）
    float draw_value_white = 0.6f; // 引き分けの勝ち点（電竜戦: 後手 0.6）
    int max_moves = 512;           // 最大手数（これを超えたら引き分け）
    bool leaf_mate1 = true;        // 葉で 1手詰めを調べる
    int root_mate_nodes = 30000;   // ルートで df-pn 詰み探索をするノード数（0 で無効）
    bool declare_nyugyoku = true;  // 27点法で宣言勝ちできるなら "win" を返す
    // NPLS の重み（Python 版と同じ）
    float npls_w1 = 1.0f;
    float npls_w2 = 0.5f;
    float npls_w3 = 0.1f;

    float draw_score(Color c) const {
        // 手番側から見た引き分けの値（-1〜+1）
        const float d = (c == Black) ? draw_value_black : draw_value_white;
        return 2.0f * d - 1.0f;
    }
};

struct SearchResult {
    std::string bestmove = "resign";
    std::string ponder;
    int playouts = 0;      // MCTS: シミュレーション回数 / NPLS: 展開したノード数
    int evaluated = 0;     // NN で評価した局面数
    int collisions = 0;
    float value = 0.0f;    // ルート手番から見た評価（-1〜+1）
    double time_ms = 0.0;
    std::vector<std::string> pv;
    std::string reason;    // "search" / "mate" / "nyugyoku" / "only_move" / "no_moves"
};

// ─────────────────────────────────────────────────────────
// 対局中の局面（履歴の StateInfo を保持して千日手判定に使う）
// ─────────────────────────────────────────────────────────
class GameBoard {
public:
    GameBoard() { init_tables(); set_startpos({}); }

    void set_startpos(const std::vector<std::string>& moves) { set_sfen(DefaultStartPositionSFEN, moves); }

    // sfen と指し手列（USI 表記）で局面を作る。失敗したら false。
    bool set_sfen(const std::string& sfen, const std::vector<std::string>& moves) {
        states_.clear();
        pos_.set(sfen);
        for (const auto& m : moves) {
            const Move mv = usiToMove(pos_, m);
            if (mv == Move::moveNone()) return false;
            states_.emplace_back();
            pos_.doMove(mv, states_.back());
        }
        return true;
    }

    Position& pos() { return pos_; }
    const Position& pos() const { return pos_; }

private:
    Position pos_;
    std::deque<StateInfo> states_;  // deque なのでアドレスが変わらない
};

// 葉ノードの終局判定。終局なら true を返し、v に手番側から見た値を入れる。
// is_root のときは千日手判定をしない（指す前の局面なので）。
inline bool check_terminal(Position& pos, const SearchConfig& cfg, bool is_root, float& v) {
    const Color us = pos.turn();
    if (!is_root) {
        switch (pos.isDraw(16)) {
            case RepetitionDraw: v = cfg.draw_score(us); return true;
            case RepetitionWin: case RepetitionSuperior: v = 1.0f; return true;
            case RepetitionLose: case RepetitionInferior: v = -1.0f; return true;
            default: break;
        }
        if (static_cast<int>(pos.gamePly()) > cfg.max_moves) {  // 最大手数に達した
            v = cfg.draw_score(us);
            return true;
        }
    }
    const MoveList<Legal> ml(pos);
    if (ml.size() == 0) { v = -1.0f; return true; }  // 詰み
    if (cfg.declare_nyugyoku && nyugyoku(pos, LAW_27) == NyugyokuWin) { v = 1.0f; return true; }
    if (!is_root && cfg.leaf_mate1 && !pos.inCheck() &&
        static_cast<int>(pos.gamePly()) < cfg.max_moves) {
        if (pos.mateMoveIn1Ply()) { v = 1.0f; return true; }
    }
    return false;
}

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 合法手と事前確率（ロジットの softmax）を作る
inline void legal_priors(const Position& pos, const float* logits, float temperature,
                         std::vector<Move>& moves, std::vector<float>& priors) {
    moves.clear();
    priors.clear();
    const Color us = pos.turn();
    float mx = -1e30f;
    for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
        const Move m = ml.move();
        const float l = logits[make_move_label(m, us)] / temperature;
        moves.push_back(m);
        priors.push_back(l);
        if (l > mx) mx = l;
    }
    float sum = 0.0f;
    for (auto& p : priors) { p = std::exp(p - mx); sum += p; }
    if (sum > 0) for (auto& p : priors) p /= sum;
}

// ルートの詰み探索（見つかれば詰ます手を返す）
inline Move root_mate_search(Position& pos, const SearchConfig& cfg) {
    if (cfg.root_mate_nodes <= 0) return Move::moveNone();
    if (const Move m1 = pos.mateMoveIn1Ply()) {
        if (static_cast<int>(pos.gamePly()) < cfg.max_moves) return m1;
    }
    static thread_local DfPn dfpn;
    dfpn.set_maxdepth(31);
    dfpn.set_max_search_node(static_cast<uint32_t>(cfg.root_mate_nodes));
    dfpn.set_draw_ply(cfg.max_moves - 1);  // 最大手数ちょうどの詰みは引き分け扱い
    if (dfpn.dfpn(pos)) {
        const Move m = dfpn.dfpn_move(pos);
        if (m != Move::moveNone()) return m;
    }
    return Move::moveNone();
}

}  // namespace apple
