// time_manager.hpp
// 電竜戦ルール向けの思考時間の決め方
//
//   電竜戦では 1手の消費時間は「1秒未満切り捨て」で、そのあと加算（フィッシャー）が入る。
//   例: 加算 2秒のとき、2.9秒考えても消費は 2秒 → すぐ 2秒戻るので持ち時間は減らない。
//   そこで、思考時間を「◯秒 + (1000 - margin) ミリ秒」にそろえて、切り捨てを最大限に使う。
//   margin は通信の遅れなどの余裕（既定 400ms）。
#pragma once

#include <algorithm>

namespace apple {

struct TimeParams {
    int margin_ms = 400;      // 通信遅延などの余裕
    int min_think_ms = 100;   // 最低思考時間
    bool floor_rule = true;   // 1秒未満切り捨てを使う
    int max_moves = 512;
};

// remaining: 自分の残り時間, inc: 1手ごとの加算, byoyomi: 秒読み, movetime: 固定時間, ply: 現在の手数
inline int think_time_ms(int remaining, int inc, int byoyomi, int movetime, int ply, const TimeParams& tp) {
    const int margin = std::max(tp.margin_ms, 0);
    if (movetime > 0) return std::max(movetime - margin, std::min(movetime, tp.min_think_ms));

    // 残りの自分の手数の見積もり（最大手数が近いほど少なく見る）
    const int own_left = std::max(1, (tp.max_moves - ply) / 2);
    const int est = std::clamp(own_left, 1, remaining < 30000 ? 60 : 40);
    const int frac = 1000 - std::min(margin, 900);  // 1秒の端数として使える部分

    long target;
    if (inc > 0) {
        target = static_cast<long>(inc) + (tp.floor_rule ? frac : 0) + remaining / est;
    } else if (byoyomi > 0) {
        target = static_cast<long>(byoyomi) - margin + remaining / est;
    } else {
        target = (tp.floor_rule ? frac : 0) + remaining / (est + 10);
    }

    if (tp.floor_rule && inc + remaining > 0) {
        // 端数を frac にそろえる（k 秒 + frac ミリ秒）
        const long k = std::max(0L, (target - frac) / 1000);
        target = k * 1000 + frac;
        // 消費（切り捨て後）が残り時間を超えないように
        const long cap = (static_cast<long>(remaining) / 1000) * 1000 + frac;
        target = std::min(target, cap);
    } else if (byoyomi <= 0) {
        target = std::min<long>(target, std::max(0, remaining - margin));
    }
    return static_cast<int>(std::max<long>(target, tp.min_think_ms));
}

}  // namespace apple
