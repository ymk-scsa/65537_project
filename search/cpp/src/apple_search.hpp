// apple_search.hpp
// C++ 版の探索（MCTS / NPLS）。NN の評価は Evaluator 経由でまとめて（バッチで）行う。
//
// Python 版（search/mcts.py, search/npls.py）からの主な変更点
//   - 盤面を毎回 sfen 文字列から作り直さず、指し手を進める/戻す（doMove/undoMove）で扱う
//   - 葉ノードを batch_size 個ためてから NN に1回で渡す（バーチャルロス方式）
//   - 終局判定: 詰み・千日手（手番ごとの引き分けの価値）・最大手数・入玉宣言(27点法)・1手詰め
//   - ルートで df-pn の詰み探索
//   - MCTS の価値の符号を「その手を指した側から見た値」にそろえた
//     （Python 版 mcts.py は相手から見た値で手を選んでいた）
#pragma once

#include <algorithm>
#include <memory>
#include <queue>

#include "apple_core.hpp"

namespace apple {

// ─────────────────────────────────────────────────────────
// 共通: NN に渡すバッチ
// ─────────────────────────────────────────────────────────
struct Batch {
    std::vector<float> feats;
    std::vector<int> turns;
    std::vector<float> policy;
    std::vector<float> value;
    int n = 0;

    void reset(int cap) {
        n = 0;
        feats.resize(static_cast<size_t>(cap) * kFeatures);
        turns.resize(cap);
        policy.resize(static_cast<size_t>(cap) * kLabels);
        value.resize(cap);
    }
    int add(const Position& pos) {
        make_features(pos, &feats[static_cast<size_t>(n) * kFeatures]);
        turns[n] = static_cast<int>(pos.turn());
        return n++;
    }
    void run(Evaluator& ev) {
        if (n > 0) ev.evaluate(n, feats.data(), turns.data(), policy.data(), value.data());
    }
    const float* logits(int i) const { return &policy[static_cast<size_t>(i) * kLabels]; }
};

// ─────────────────────────────────────────────────────────
// MCTS（PUCT）
// ─────────────────────────────────────────────────────────
class MCTSSearcher {
public:
    struct Node {
        enum State : uint8_t { Unexpanded, Pending, Expanded, Terminal };
        State state = Unexpanded;
        float value = 0.0f;  // NN の評価 or 終局値（このノードの手番側から見た値）
        int visits = 0;      // バーチャルロスを含む子の訪問回数の合計
        std::vector<Move> moves;
        std::vector<float> prior;
        std::vector<int> n;
        std::vector<float> w;  // その手を指した側（このノードの手番側）から見た価値の合計
        std::vector<std::unique_ptr<Node>> child;
    };

    explicit MCTSSearcher(Evaluator& ev) : ev_(ev), st_(kMaxPly) {}

    SearchConfig& config() { return cfg_; }
    void stop() { stop_.store(true); }

    SearchResult search(GameBoard& board, double time_limit_ms, int max_playouts) {
        const auto t0 = Clock::now();
        stop_.store(false);
        SearchResult res;
        Position& pos = board.pos();

        // ルートの特別な判定
        {
            const MoveList<Legal> ml(pos);
            if (ml.size() == 0) { res.reason = "no_moves"; res.bestmove = "resign"; return res; }
            if (cfg_.declare_nyugyoku && nyugyoku(pos, LAW_27) == NyugyokuWin) {
                res.reason = "nyugyoku"; res.bestmove = "win"; return res;
            }
            if (ml.size() == 1) {
                res.reason = "only_move"; res.bestmove = ml.move().toUSI();
                res.time_ms = elapsed_ms(t0); return res;
            }
            const Move mate = root_mate_search(pos, cfg_);
            if (mate) {
                res.reason = "mate"; res.bestmove = mate.toUSI(); res.value = 1.0f;
                res.pv.push_back(res.bestmove);
                res.time_ms = elapsed_ms(t0); return res;
            }
        }

        root_ = std::make_unique<Node>();
        batch_.reset(cfg_.batch_size);
        pending_.clear();
        int playouts = 0;

        while (!stop_.load()) {
            // ① 葉を集める
            batch_.n = 0;
            pending_.clear();
            int attempts = 0;
            while (batch_.n < cfg_.batch_size && attempts < cfg_.batch_size * 2) {
                ++attempts;
                const int r = simulate(pos);
                if (r == 1) ++playouts;            // 終局ノードで完了
                else if (r == 2) ++res.collisions;  // 衝突
                if (root_->state == Node::Pending) break;  // 最初はルートだけ評価
            }
            // ② まとめて評価して展開・逆伝播
            if (batch_.n > 0) {
                batch_.run(ev_);
                res.evaluated += batch_.n;
                for (auto& pe : pending_) {
                    expand(pe.node, pe.pos_index, pe.path, pos);
                    ++playouts;
                }
            }
            if (elapsed_ms(t0) >= time_limit_ms) break;
            if (max_playouts > 0 && playouts >= max_playouts) break;
        }

        res.playouts = playouts;
        res.time_ms = elapsed_ms(t0);
        finish(pos, res);
        return res;
    }

private:
    struct PendingEval {
        Node* node;
        int pos_index;
        std::vector<std::pair<Node*, int>> path;
    };

    // 1回のシミュレーション。0=評価待ちに追加, 1=終局で逆伝播済み, 2=衝突
    int simulate(Position& pos) {
        std::vector<std::pair<Node*, int>> path;
        StateInfo* st = st_.data();
        std::vector<Move> played;
        Node* node = root_.get();
        int result = 0;
        while (true) {
            if (node->state == Node::Terminal) {
                backprop(path, node->value);
                result = 1;
                break;
            }
            if (node->state == Node::Pending) {
                revert(path);
                result = 2;
                break;
            }
            if (node->state == Node::Unexpanded) {
                float tv;
                if (check_terminal(pos, cfg_, node == root_.get(), tv)) {
                    node->state = Node::Terminal;
                    node->value = tv;
                    backprop(path, tv);
                    result = 1;
                    break;
                }
                node->state = Node::Pending;
                const int idx = batch_.add(pos);
                pending_.push_back({node, idx, path});
                result = 0;
                break;
            }
            // 展開済み: PUCT で手を選ぶ
            const int i = select(node);
            node->n[i] += 1;      // バーチャルロス
            node->w[i] -= 1.0f;
            node->visits += 1;
            path.emplace_back(node, i);
            if (!node->child[i]) node->child[i] = std::make_unique<Node>();
            const Move m = node->moves[i];
            pos.doMove(m, st[played.size()]);
            played.push_back(m);
            node = node->child[i].get();
            if (played.size() >= static_cast<size_t>(kMaxPly - 1)) { revert(path); result = 2; break; }
        }
        for (auto it = played.rbegin(); it != played.rend(); ++it) pos.undoMove(*it);
        return result;
    }

    int select(const Node* node) const {
        const float sqrt_n = std::sqrt(static_cast<float>(std::max(node->visits, 1)));
        const float c = cfg_.c_puct + std::log((node->visits + cfg_.c_puct_base + 1.0f) / cfg_.c_puct_base);
        const float fpu = node->value - cfg_.fpu_reduction;
        int best = 0;
        float best_score = -1e30f;
        for (size_t i = 0; i < node->moves.size(); ++i) {
            const int n = node->n[i];
            const float q = n > 0 ? node->w[i] / n : fpu;
            const float u = c * node->prior[i] * sqrt_n / (1.0f + n);
            const float s = q + u;
            if (s > best_score) { best_score = s; best = static_cast<int>(i); }
        }
        return best;
    }

    // v: 葉ノードの手番側から見た値
    void backprop(const std::vector<std::pair<Node*, int>>& path, float v) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            v = -v;  // 親の手番（その手を指した側）から見た値
            Node* nd = it->first;
            nd->w[it->second] += 1.0f + v;  // バーチャルロスを戻して本当の値を足す
        }
    }

    void revert(const std::vector<std::pair<Node*, int>>& path) {
        for (const auto& pr : path) {
            pr.first->n[pr.second] -= 1;
            pr.first->w[pr.second] += 1.0f;
            pr.first->visits -= 1;
        }
    }

    void expand(Node* node, int idx, const std::vector<std::pair<Node*, int>>& path, Position& root_pos) {
        // 葉の局面に移動して合法手を作る
        StateInfo* st = st_.data();
        std::vector<Move> played;
        for (const auto& pr : path) {
            const Move m = pr.first->moves[pr.second];
            root_pos.doMove(m, st[played.size()]);
            played.push_back(m);
        }
        legal_priors(root_pos, batch_.logits(idx), cfg_.softmax_temperature, node->moves, node->prior);
        for (auto it = played.rbegin(); it != played.rend(); ++it) root_pos.undoMove(*it);

        const size_t k = node->moves.size();
        node->n.assign(k, 0);
        node->w.assign(k, 0.0f);
        node->child.clear();
        node->child.resize(k);
        node->value = batch_.value[idx];
        node->state = Node::Expanded;
        backprop(path, node->value);
    }

    void finish(Position& pos, SearchResult& res) {
        Node* node = root_.get();
        if (node->state != Node::Expanded || node->moves.empty()) {
            const MoveList<Legal> ml(pos);
            res.bestmove = ml.size() ? ml.move().toUSI() : "resign";
            res.reason = "search";
            return;
        }
        int best = best_child(node);
        res.bestmove = node->moves[best].toUSI();
        res.value = node->n[best] > 0 ? node->w[best] / node->n[best] : node->value;
        res.reason = "search";
        // 読み筋
        Node* cur = node;
        int depth = 0;
        while (cur && cur->state == Node::Expanded && !cur->moves.empty() && depth < 32) {
            const int b = best_child(cur);
            if (cur->n[b] == 0) break;
            res.pv.push_back(cur->moves[b].toUSI());
            cur = cur->child[b].get();
            ++depth;
        }
        if (res.pv.size() >= 2) res.ponder = res.pv[1];
    }

    static int best_child(const Node* node) {
        int best = 0;
        for (size_t i = 1; i < node->moves.size(); ++i) {
            const int a = node->n[i], b = node->n[best];
            if (a > b) best = static_cast<int>(i);
            else if (a == b) {
                const float qa = a ? node->w[i] / a : -2.0f;
                const float qb = b ? node->w[best] / b : -2.0f;
                if (qa > qb || (qa == qb && node->prior[i] > node->prior[best])) best = static_cast<int>(i);
            }
        }
        return best;
    }

    Evaluator& ev_;
    SearchConfig cfg_;
    std::atomic<bool> stop_{false};
    std::unique_ptr<Node> root_;
    Batch batch_;
    std::vector<PendingEval> pending_;
    std::vector<StateInfo> st_;  // doMove 用の作業領域
};

// ─────────────────────────────────────────────────────────
// NPLS（Python 版 search/npls.py の移植）
//   優先度 = -(w1 * 方策 + w2 * 親の価値 + w3 / (親の訪問数 + 1))
//   いちばん有望なノードから順に展開し、最後はルートの子の訪問数で手を選ぶ。
//   C++ 版では優先度の高い順に batch_size 個まとめて評価する。
// ─────────────────────────────────────────────────────────
class NPLSSearcher {
public:
    struct Node {
        Move move = Move::moveNone();
        Node* parent = nullptr;
        float p = 0.0f;
        float v = 0.0f;
        float priority = 0.0f;
        int n = 0;
        std::vector<std::unique_ptr<Node>> children;
    };

    explicit NPLSSearcher(Evaluator& ev) : ev_(ev), st_(kMaxPly) {}

    SearchConfig& config() { return cfg_; }
    void stop() { stop_.store(true); }

    SearchResult search(GameBoard& board, double time_limit_ms, int max_nodes) {
        const auto t0 = Clock::now();
        stop_.store(false);
        SearchResult res;
        Position& pos = board.pos();
        {
            const MoveList<Legal> ml(pos);
            if (ml.size() == 0) { res.reason = "no_moves"; res.bestmove = "resign"; return res; }
            if (cfg_.declare_nyugyoku && nyugyoku(pos, LAW_27) == NyugyokuWin) {
                res.reason = "nyugyoku"; res.bestmove = "win"; return res;
            }
            if (ml.size() == 1) {
                res.reason = "only_move"; res.bestmove = ml.move().toUSI();
                res.time_ms = elapsed_ms(t0); return res;
            }
            const Move mate = root_mate_search(pos, cfg_);
            if (mate) {
                res.reason = "mate"; res.bestmove = mate.toUSI(); res.value = 1.0f;
                res.pv.push_back(res.bestmove);
                res.time_ms = elapsed_ms(t0); return res;
            }
        }

        root_ = std::make_unique<Node>();
        Batch batch;
        batch.reset(std::max(cfg_.batch_size, 1));
        // ルートを評価
        batch.add(pos);
        batch.run(ev_);
        res.evaluated += 1;
        root_->v = batch.value[0];
        std::priority_queue<Entry> queue;
        make_children(root_.get(), pos, batch.logits(0), /*root=*/true, queue);
        root_->n = 1;

        int searched = 0;
        std::vector<Node*> popped;
        std::vector<int> slot;
        while (!queue.empty() && !stop_.load()) {
            if (elapsed_ms(t0) >= time_limit_ms) break;
            if (max_nodes > 0 && searched >= max_nodes) break;
            // 優先度の高い順に取り出す
            popped.clear();
            slot.clear();
            batch.n = 0;
            while (!queue.empty() && static_cast<int>(popped.size()) < cfg_.batch_size) {
                Node* nd = queue.top().node;
                queue.pop();
                StateInfo* st = st_.data();
                std::vector<Move> line;
                for (Node* c = nd; c->parent; c = c->parent) line.push_back(c->move);
                std::reverse(line.begin(), line.end());
                for (size_t i = 0; i < line.size(); ++i) pos.doMove(line[i], st[i]);
                float tv;
                if (check_terminal(pos, cfg_, false, tv)) {
                    nd->v = tv;
                    slot.push_back(-1);
                } else {
                    slot.push_back(batch.add(pos));
                }
                for (auto it = line.rbegin(); it != line.rend(); ++it) pos.undoMove(*it);
                popped.push_back(nd);
            }
            batch.run(ev_);
            res.evaluated += batch.n;
            for (size_t k = 0; k < popped.size(); ++k) {
                Node* nd = popped[k];
                if (slot[k] >= 0) {
                    nd->v = batch.value[slot[k]];
                    StateInfo* st = st_.data();
                    std::vector<Move> line;
                    for (Node* c = nd; c->parent; c = c->parent) line.push_back(c->move);
                    std::reverse(line.begin(), line.end());
                    for (size_t i = 0; i < line.size(); ++i) pos.doMove(line[i], st[i]);
                    make_children(nd, pos, batch.logits(slot[k]), false, queue);
                    for (auto it = line.rbegin(); it != line.rend(); ++it) pos.undoMove(*it);
                }
                for (Node* c = nd; c; c = c->parent) c->n += 1;
                ++searched;
            }
        }

        res.playouts = searched;
        res.time_ms = elapsed_ms(t0);
        // ルートの子のうち訪問数が最大のもの（同じなら方策が大きいもの）
        Node* best = nullptr;
        for (auto& c : root_->children) {
            if (!best || c->n > best->n || (c->n == best->n && c->p > best->p)) best = c.get();
        }
        if (best) {
            res.bestmove = best->move.toUSI();
            res.value = -best->v;
            res.pv.push_back(res.bestmove);
            Node* ponder = nullptr;
            for (auto& c : best->children) {
                if (!ponder || c->n > ponder->n) ponder = c.get();
            }
            if (ponder) { res.ponder = ponder->move.toUSI(); res.pv.push_back(res.ponder); }
        }
        res.reason = "search";
        return res;
    }

private:
    struct Entry {
        float priority;
        Node* node;
        bool operator<(const Entry& o) const { return priority > o.priority; }  // 小さい方を先に
    };

    void make_children(Node* nd, const Position& pos, const float* logits, bool root,
                       std::priority_queue<Entry>& queue) {
        std::vector<Move> moves;
        std::vector<float> priors;
        legal_priors(pos, logits, cfg_.softmax_temperature, moves, priors);
        nd->children.reserve(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) {
            auto c = std::make_unique<Node>();
            c->move = moves[i];
            c->parent = nd;
            c->p = priors[i];
            if (root) c->priority = -(cfg_.npls_w1 * c->p + cfg_.npls_w2 * 0.0f + cfg_.npls_w3 * 1.0f);
            else c->priority = -(cfg_.npls_w1 * c->p + cfg_.npls_w2 * nd->v + cfg_.npls_w3 * (1.0f / (nd->n + 1)));
            queue.push({c->priority, c.get()});
            nd->children.push_back(std::move(c));
        }
    }

    Evaluator& ev_;
    SearchConfig cfg_;
    std::atomic<bool> stop_{false};
    std::unique_ptr<Node> root_;
    std::vector<StateInfo> st_;
};

}  // namespace apple
