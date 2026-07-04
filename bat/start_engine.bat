@echo off
REM ============================================================
REM GNN将棋AI 大会用起動バッチファイル
REM ============================================================
REM 各パラメータの意味:
REM   --time_rate     持ち時間使用率 (0.01〜1.0)
REM                   0.5 = 残り持ち時間の50%を1手に使う（保守的）
REM                   0.8 = 残り持ち時間の80%を1手に使う（積極的）
REM
REM   --draw_value    千日手の評価値 (cp単位)
REM                    0  = 千日手は互角（標準）
REM                   -50 = 千日手を避ける（攻め将棋向け）
REM                   +50 = 千日手を歓迎する（入玉狙い）
REM
REM   --impasse_score 持将棋の入玉宣言に必要な最低点数
REM                   通常の大会 = 28点（デフォルト）
REM                   一部大会   = 31点など規定に合わせる
REM
REM   --impasse_value 持将棋局面の評価値 (cp単位)
REM                    0  = 持将棋は互角（標準）
REM                   +50 = 持将棋を歓迎する
REM
REM   --max_moves     最大手数（大会規定に合わせる）
REM                   標準大会 = 320手
REM                   電竜戦   = 512手（例）
REM
REM   --use_search    探索部の有無
REM                   1 = 探索あり（通常）
REM                   0 = 評価関数のみ（超高速・低精度）
REM
REM   --search_type   探索アルゴリズム
REM                   mcts   = モンテカルロ木探索（標準）
REM                   npls   = UW-NPLS（独自探索、推奨）
REM                   mc_rzf = MC-RZF探索実験
REM                   ads_ab = ADS-AB探索実験
REM ============================================================

REM ── Python インタープリタのパスを指定 ──
set PYTHON=python

REM ── プロジェクトルートへ移動 ──
cd /d %~dp0..

REM ============================================================
REM 【標準設定】通常大会向け
REM ============================================================
%PYTHON% agent/main.py ^
  --search_type npls ^
  --mode 30blocks ^
  --weights weights/checkpoint.pth ^
  --name "GNN-Shogi" ^
  --time_rate 0.5 ^
  --draw_value 0 ^
  --impasse_score 28 ^
  --impasse_value 0 ^
  --max_moves 320 ^
  --use_search 1

REM ============================================================
REM 【設定例2】電竜戦向け（最大512手・積極的な時間使用）
REM ============================================================
REM %PYTHON% agent/main.py ^
REM   --search_type npls ^
REM   --mode 30blocks ^
REM   --weights weights/checkpoint.pth ^
REM   --name "GNN-Shogi" ^
REM   --time_rate 0.7 ^
REM   --draw_value -30 ^
REM   --impasse_score 28 ^
REM   --impasse_value 0 ^
REM   --max_moves 512 ^
REM   --use_search 1

REM ============================================================
REM 【設定例3】探索なし高速モード（予選・テスト用）
REM ============================================================
REM %PYTHON% agent/main.py ^
REM   --search_type mcts ^
REM   --mode 30blocks ^
REM   --weights weights/checkpoint.pth ^
REM   --name "GNN-Shogi-Fast" ^
REM   --time_rate 0.3 ^
REM   --draw_value 0 ^
REM   --impasse_score 28 ^
REM   --impasse_value 0 ^
REM   --max_moves 320 ^
REM   --use_search 0
