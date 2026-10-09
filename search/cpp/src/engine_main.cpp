// engine_main.cpp
// 全部 C++ の USI エンジン（LibTorch 版）
//   - 探索: apple_search.hpp（MCTS / NPLS）
//   - 推論: LibTorch で TorchScript モデル（export_torchscript.py で作る）を読む
//   - 時間: time_manager.hpp（電竜戦の 1秒未満切り捨て＋加算に合わせる）
//
// ビルド: bash search/cpp/build_engine.sh
// 実行  : search/cpp/apple_engine  （USI で通信）
//   setoption name ModelPath value weights/appleS.ts
#include <torch/script.h>
#include <ATen/Parallel.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include "apple_search.hpp"
#include "time_manager.hpp"

using namespace apple;

class TorchEvaluator : public Evaluator {
public:
    bool load(const std::string& path) {
        try {
            module_ = torch::jit::load(path, torch::kCPU);
            module_.eval();
            loaded_ = true;
        } catch (const std::exception& e) {
            std::cout << "info string model load error: " << e.what() << std::endl;
            loaded_ = false;
        }
        return loaded_;
    }
    bool loaded() const { return loaded_; }

    void evaluate(int n, const float* feats, const int* turns, float* policy, float* value) override {
        if (!loaded_) {  // モデルがないときは一様な方策・評価0
            std::fill(policy, policy + static_cast<size_t>(n) * kLabels, 0.0f);
            std::fill(value, value + n, 0.0f);
            return;
        }
        torch::InferenceMode guard;
        // バッチの大きさを 2 のべき乗にそろえる（形が毎回変わると TorchScript の最適化がやり直しになるため）
        int m = 1;
        while (m < n) m <<= 1;
        if (static_cast<int>(xbuf_.size()) < m * kFeatures) { xbuf_.resize(m * kFeatures); tbuf_.resize(m); }
        std::memcpy(xbuf_.data(), feats, sizeof(float) * n * kFeatures);
        std::fill(xbuf_.begin() + static_cast<size_t>(n) * kFeatures, xbuf_.begin() + static_cast<size_t>(m) * kFeatures, 0.0f);
        for (int i = 0; i < m; ++i) tbuf_[i] = i < n ? turns[i] : 0;
        auto x = torch::from_blob(xbuf_.data(), {m, kFeatures}, torch::kFloat32);
        auto t = torch::from_blob(tbuf_.data(), {m}, torch::kInt64);
        auto out = module_.forward({x, t}).toTuple();
        auto p = out->elements()[0].toTensor().contiguous().to(torch::kFloat32);
        auto v = out->elements()[1].toTensor().reshape({m}).contiguous().to(torch::kFloat32);
        std::memcpy(policy, p.data_ptr<float>(), sizeof(float) * n * kLabels);
        std::memcpy(value, v.data_ptr<float>(), sizeof(float) * n);
    }

private:
    torch::jit::script::Module module_;
    bool loaded_ = false;
    std::vector<float> xbuf_;
    std::vector<int64_t> tbuf_;
};

struct Options {
    std::string model_path = "weights/model.ts";
    std::string algo = "mcts";
    int batch_size = 32;
    int threads = 0;
    int draw_black = 40;  // 引き分けの勝ち点 ×100
    int draw_white = 60;
    int max_moves = 512;
    int root_mate_nodes = 30000;
    int margin_ms = 400;
    bool floor_rule = true;
    int c_puct_x100 = 150;
};

class Engine {
public:
    Engine() : mcts_(ev_), npls_(ev_) {}

    void loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream is(line);
            std::string cmd;
            is >> cmd;
            if (cmd == "usi") {
                std::cout << "id name Apple-RayGraph (C++/LibTorch)\n"
                          << "id author 65537_project\n"
                          << "option name ModelPath type string default " << opt_.model_path << "\n"
                          << "option name Algo type combo default mcts var mcts var npls\n"
                          << "option name BatchSize type spin default 32 min 1 max 1024\n"
                          << "option name Threads type spin default 0 min 0 max 256\n"
                          << "option name DrawValueBlack type spin default 40 min 0 max 100\n"
                          << "option name DrawValueWhite type spin default 60 min 0 max 100\n"
                          << "option name MaxMoves type spin default 512 min 1 max 10000\n"
                          << "option name RootMateNodes type spin default 30000 min 0 max 10000000\n"
                          << "option name MarginMs type spin default 400 min 0 max 5000\n"
                          << "option name FloorRule type check default true\n"
                          << "option name CPuct type spin default 150 min 1 max 1000\n"
                          << "usiok" << std::endl;
            } else if (cmd == "setoption") {
                setoption(is);
            } else if (cmd == "isready") {
                wait_search();
                prepare();
                std::cout << "readyok" << std::endl;
            } else if (cmd == "usinewgame") {
                wait_search();
            } else if (cmd == "position") {
                wait_search();
                position(is);
            } else if (cmd == "go") {
                wait_search();
                go(is);
            } else if (cmd == "stop") {
                mcts_.stop();
                npls_.stop();
                wait_search();
            } else if (cmd == "ponderhit") {
                // ponder は使わない
            } else if (cmd == "gameover") {
                mcts_.stop(); npls_.stop(); wait_search();
            } else if (cmd == "quit") {
                mcts_.stop(); npls_.stop(); wait_search();
                break;
            }
        }
    }

private:
    void setoption(std::istringstream& is) {
        std::string tok, name, value;
        is >> tok;  // name
        while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
        std::getline(is, value);
        if (!value.empty() && value[0] == ' ') value.erase(0, 1);
        try {
            if (name == "ModelPath") opt_.model_path = value;
            else if (name == "Algo") opt_.algo = value;
            else if (name == "BatchSize") opt_.batch_size = std::stoi(value);
            else if (name == "Threads") opt_.threads = std::stoi(value);
            else if (name == "DrawValueBlack") opt_.draw_black = std::stoi(value);
            else if (name == "DrawValueWhite") opt_.draw_white = std::stoi(value);
            else if (name == "MaxMoves") opt_.max_moves = std::stoi(value);
            else if (name == "RootMateNodes") opt_.root_mate_nodes = std::stoi(value);
            else if (name == "MarginMs") opt_.margin_ms = std::stoi(value);
            else if (name == "FloorRule") opt_.floor_rule = (value == "true");
            else if (name == "CPuct") opt_.c_puct_x100 = std::stoi(value);
        } catch (...) {
        }
    }

    void prepare() {
        if (opt_.threads > 0) {
            at::set_num_threads(opt_.threads);
        }
        if (!ev_.loaded() || loaded_path_ != opt_.model_path) {
            if (ev_.load(opt_.model_path)) {
                loaded_path_ = opt_.model_path;
                std::cout << "info string loaded " << opt_.model_path << std::endl;
            }
        }
        for (SearchConfig* c : {&mcts_.config(), &npls_.config()}) {
            c->batch_size = opt_.batch_size;
            c->draw_value_black = opt_.draw_black / 100.0f;
            c->draw_value_white = opt_.draw_white / 100.0f;
            c->max_moves = opt_.max_moves;
            c->root_mate_nodes = opt_.root_mate_nodes;
            c->c_puct = opt_.c_puct_x100 / 100.0f;
        }
        // 1回推論して温めておく
        std::vector<float> f(kFeatures, 0.0f), p(kLabels), v(1);
        int t = 0;
        ev_.evaluate(1, f.data(), &t, p.data(), v.data());
    }

    void position(std::istringstream& is) {
        std::string tok, sfen;
        std::vector<std::string> moves;
        is >> tok;
        if (tok == "startpos") {
            sfen = DefaultStartPositionSFEN;
            is >> tok;  // moves
        } else if (tok == "sfen") {
            while (is >> tok && tok != "moves") sfen += tok + " ";
        }
        while (is >> tok) moves.push_back(tok);
        if (!board_.set_sfen(sfen, moves)) std::cout << "info string position error" << std::endl;
    }

    void go(std::istringstream& is) {
        int btime = 0, wtime = 0, binc = 0, winc = 0, byoyomi = 0, movetime = 0, nodes = 0;
        bool infinite = false;
        std::string tok;
        while (is >> tok) {
            if (tok == "btime") is >> btime;
            else if (tok == "wtime") is >> wtime;
            else if (tok == "binc") is >> binc;
            else if (tok == "winc") is >> winc;
            else if (tok == "byoyomi") is >> byoyomi;
            else if (tok == "movetime") is >> movetime;
            else if (tok == "nodes") is >> nodes;
            else if (tok == "infinite") infinite = true;
        }
        const bool black = board_.pos().turn() == Black;
        TimeParams tp;
        tp.margin_ms = opt_.margin_ms;
        tp.floor_rule = opt_.floor_rule;
        tp.max_moves = opt_.max_moves;
        double limit = infinite ? 1e12
                                : think_time_ms(black ? btime : wtime, black ? binc : winc, byoyomi, movetime,
                                                board_.pos().gamePly(), tp);
        if (!infinite && btime == 0 && wtime == 0 && byoyomi == 0 && movetime == 0 && nodes > 0) limit = 1e12;
        searching_ = true;
        worker_ = std::thread([this, limit, nodes] {
            SearchResult r = (opt_.algo == "npls") ? npls_.search(board_, limit, nodes)
                                                    : mcts_.search(board_, limit, nodes);
            report(r);
        });
    }

    void report(const SearchResult& r) {
        std::ostringstream os;
        const double t = std::max(r.time_ms, 1e-3);
        if (r.reason == "mate") {
            os << "info score mate + pv " << r.bestmove << "\n";
        } else if (r.reason == "search") {
            const double p = std::clamp((r.value + 1.0) * 0.5, 1e-6, 1 - 1e-6);
            const int cp = static_cast<int>(std::lround(-600.0 * std::log(1.0 / p - 1.0)));
            os << "info nodes " << r.evaluated << " nps " << static_cast<long>(r.evaluated * 1000.0 / t)
               << " time " << static_cast<long>(t) << " score cp " << cp << " pv";
            for (auto& m : r.pv) os << " " << m;
            os << "\n";
        }
        os << "info string " << opt_.algo << " " << r.reason << " playouts=" << r.playouts
           << " evaluated=" << r.evaluated << " time=" << static_cast<long>(t) << "ms\n";
        os << "bestmove " << r.bestmove;
        std::cout << os.str() << std::endl;
    }

    void wait_search() {
        if (worker_.joinable()) worker_.join();
        searching_ = false;
    }

    TorchEvaluator ev_;
    MCTSSearcher mcts_;
    NPLSSearcher npls_;
    GameBoard board_;
    Options opt_;
    std::string loaded_path_;
    std::thread worker_;
    bool searching_ = false;
};

int main() {
    std::ios::sync_with_stdio(false);
    init_tables();
    Engine e;
    e.loop();
    return 0;
}
