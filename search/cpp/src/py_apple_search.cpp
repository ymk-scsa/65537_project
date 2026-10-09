// py_apple_search.cpp
// Python から C++ 探索を使うためのモジュール（pybind11）。
// NN の評価は Python の関数（PyTorch）を呼び出してバッチで行う。
//
//   import apple_search_cpp as A
//   s = A.Searcher(evaluate_fn, "mcts")   # evaluate_fn(feats[N,447] float32, turns[N] int64)
//                                         #   -> (policy_logits[N,2187] float32, value[N] float32)
//   s.set_position("startpos", ["7g7f"])
//   res = s.go(1000.0, 0)                 # 時間(ms), 上限(0=なし)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>

#include "apple_search.hpp"

namespace py = pybind11;
using namespace apple;

class PyEvaluator : public Evaluator {
public:
    explicit PyEvaluator(py::function fn) : fn_(std::move(fn)) {}
    void evaluate(int n, const float* feats, const int* turns, float* policy, float* value) override {
        py::gil_scoped_acquire gil;
        py::array_t<float> f({n, kFeatures});
        std::memcpy(f.mutable_data(), feats, sizeof(float) * n * kFeatures);
        py::array_t<int64_t> t(n);
        for (int i = 0; i < n; ++i) t.mutable_data()[i] = turns[i];
        py::tuple out = fn_(f, t);
        auto p = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(out[0]);
        auto v = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(out[1]);
        if (!p || !v || p.size() < static_cast<py::ssize_t>(n) * kLabels || v.size() < n)
            throw std::runtime_error("evaluate_fn は (policy[N,2187], value[N]) を返してください");
        std::memcpy(policy, p.data(), sizeof(float) * n * kLabels);
        std::memcpy(value, v.data(), sizeof(float) * n);
    }

private:
    py::function fn_;
};

class PySearcher {
public:
    PySearcher(py::function fn, const std::string& algo)
        : ev_(std::move(fn)), algo_(algo), mcts_(ev_), npls_(ev_) {
        if (algo_ != "mcts" && algo_ != "npls") throw std::invalid_argument("algo は 'mcts' か 'npls'");
    }

    bool set_position(const std::string& sfen, const std::vector<std::string>& moves) {
        if (sfen == "startpos" || sfen.empty()) return board_.set_sfen(DefaultStartPositionSFEN, moves);
        std::string s = sfen;
        if (s.rfind("sfen ", 0) == 0) s = s.substr(5);
        return board_.set_sfen(s, moves);
    }

    py::dict go(double time_ms, int limit) {
        SearchResult r;
        {
            py::gil_scoped_release release;  // 探索中も stop() を受け付けられるように
            r = (algo_ == "mcts") ? mcts_.search(board_, time_ms, limit) : npls_.search(board_, time_ms, limit);
        }
        py::dict d;
        d["bestmove"] = r.bestmove;
        d["ponder"] = r.ponder;
        d["playouts"] = r.playouts;
        d["evaluated"] = r.evaluated;
        d["collisions"] = r.collisions;
        d["value"] = r.value;
        d["time_ms"] = r.time_ms;
        d["pv"] = r.pv;
        d["reason"] = r.reason;
        return d;
    }

    void stop() { mcts_.stop(); npls_.stop(); }

    SearchConfig& cfg() { return algo_ == "mcts" ? mcts_.config() : npls_.config(); }
    std::string sfen() const { return board_.pos().toSFEN(); }
    int turn() const { return static_cast<int>(board_.pos().turn()); }
    int ply() const { return static_cast<int>(board_.pos().gamePly()); }

private:
    PyEvaluator ev_;
    std::string algo_;
    MCTSSearcher mcts_;
    NPLSSearcher npls_;
    GameBoard board_;
};

// ─────────────────────────────────────────────────────────
// 学習用の高速な特徴量作成（HCPE）
// ─────────────────────────────────────────────────────────
static py::tuple hcpe_batch(py::array_t<uint8_t, py::array::c_style> hcp, py::array_t<uint16_t> move16,
                            bool with_turn) {
    init_tables();
    const int n = static_cast<int>(hcp.shape(0));
    const int width = kFeatures + (with_turn ? 1 : 0);
    py::array_t<float> feats({n, width});
    py::array_t<int64_t> labels(n);
    py::array_t<int64_t> turns(n);
    auto m = move16.unchecked<1>();
    {
        py::gil_scoped_release release;
        Position pos;
        for (int i = 0; i < n; ++i) {
            pos.set_hcp(reinterpret_cast<char*>(const_cast<uint8_t*>(hcp.data(i))));
            float* f = feats.mutable_data(i);
            make_features(pos, f);
            if (with_turn) f[kFeatures] = static_cast<float>(pos.turn());
            const Move mv = move16toMove(Move(m(i)), pos);
            labels.mutable_data()[i] = make_move_label(mv, pos.turn());
            turns.mutable_data()[i] = static_cast<int64_t>(pos.turn());
        }
    }
    return py::make_tuple(feats, labels, turns);
}

static py::array_t<float> features_from_sfen(const std::string& sfen) {
    init_tables();
    Position pos(sfen);
    py::array_t<float> f(kFeatures);
    make_features(pos, f.mutable_data());
    return f;
}

static int move_label_from_usi(const std::string& sfen, const std::string& usi) {
    init_tables();
    Position pos(sfen);
    const Move m = usiToMove(pos, usi);
    if (!m) throw std::invalid_argument("不正な指し手: " + usi);
    return make_move_label(m, pos.turn());
}

PYBIND11_MODULE(apple_search_cpp, mod) {
    mod.doc() = "C++ 版の探索（MCTS / NPLS）と特徴量作成";
    init_tables();

    py::class_<SearchConfig>(mod, "SearchConfig")
        .def_readwrite("batch_size", &SearchConfig::batch_size)
        .def_readwrite("c_puct", &SearchConfig::c_puct)
        .def_readwrite("c_puct_base", &SearchConfig::c_puct_base)
        .def_readwrite("fpu_reduction", &SearchConfig::fpu_reduction)
        .def_readwrite("softmax_temperature", &SearchConfig::softmax_temperature)
        .def_readwrite("draw_value_black", &SearchConfig::draw_value_black)
        .def_readwrite("draw_value_white", &SearchConfig::draw_value_white)
        .def_readwrite("max_moves", &SearchConfig::max_moves)
        .def_readwrite("leaf_mate1", &SearchConfig::leaf_mate1)
        .def_readwrite("root_mate_nodes", &SearchConfig::root_mate_nodes)
        .def_readwrite("declare_nyugyoku", &SearchConfig::declare_nyugyoku)
        .def_readwrite("npls_w1", &SearchConfig::npls_w1)
        .def_readwrite("npls_w2", &SearchConfig::npls_w2)
        .def_readwrite("npls_w3", &SearchConfig::npls_w3);

    py::class_<PySearcher>(mod, "Searcher")
        .def(py::init<py::function, const std::string&>(), py::arg("evaluate_fn"), py::arg("algo") = "mcts")
        .def("set_position", &PySearcher::set_position, py::arg("sfen"), py::arg("moves"))
        .def("go", &PySearcher::go, py::arg("time_ms"), py::arg("limit") = 0)
        .def("stop", &PySearcher::stop)
        .def_property_readonly("config", &PySearcher::cfg, py::return_value_policy::reference_internal)
        .def("sfen", &PySearcher::sfen)
        .def("turn", &PySearcher::turn)
        .def("ply", &PySearcher::ply);

    mod.def("hcpe_batch", &hcpe_batch, py::arg("hcp"), py::arg("move16"), py::arg("with_turn") = false,
            "HCPE の hcp(N,32) と bestMove16(N) から (特徴量, 指し手ラベル, 手番) を作る");
    mod.def("features_from_sfen", &features_from_sfen);
    mod.def("move_label_from_usi", &move_label_from_usi);
    mod.attr("NUM_FEATURES") = kFeatures;
    mod.attr("NUM_LABELS") = kLabels;
}
