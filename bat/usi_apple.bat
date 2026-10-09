@echo off
REM ============================================================
REM Apple 版 USI エンジン（C++ 探索 + PyTorch 推論）
REM   将棋GUI にはこのファイルをエンジンとして登録する。
REM   事前に bat\build_cpp.bat で C++ モジュールをビルドしておくこと。
REM ============================================================
REM 主な引数:
REM   --search_type  mcts_cpp（おすすめ） / npls_cpp
REM   --mode         appleS（標準） / appleXS（軽い） / appleM（重い）
REM   --time_mode    denryu = 1秒未満切り捨て＋加算を使う（電竜戦向け）
REM   --draw_black / --draw_white  引き分けの勝ち点（電竜戦 0.4 / 0.6）
REM   --batch_size   NN にまとめて渡す局面数
REM   --threads      PyTorch の CPU スレッド数（0=自動）
REM ============================================================
set PYTHONIOENCODING=utf-8
call "%~dp0wsl_python.bat" usi.py ^
  --search_type mcts_cpp ^
  --mode appleS ^
  --weights weights/apple_s.pth ^
  --name "Apple-RayGraph" ^
  --time_mode denryu ^
  --margin_ms 400 ^
  --max_moves 512 ^
  --draw_black 0.4 ^
  --draw_white 0.6 ^
  --batch_size 32 ^
  --threads 0
