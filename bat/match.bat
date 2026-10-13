@echo off
REM ============================================================
REM GNN-Shogi match launcher
REM Python: WSL Ubuntu-22.04 + project .venv
REM ============================================================
REM Args:
REM   --games --think_time_ms --max_moves --swap_colors --save_csa
REM   --black_name --black_search --black_mode --black_weights
REM   --white_name --white_search --white_mode --white_weights
REM ============================================================

REM [default] same weights, 1 game, 1s/move, save CSA
call "%~dp0wsl_python.bat" play/match.py ^
  --games 1 ^
  --think_time_ms 1000 ^
  --max_moves 320 ^
  --save_csa ^
  --black_name "GNN-Black" ^
  --black_search npls ^
  --black_mode modelC ^
  --black_weights weights/checkpoint.pth ^
  --white_name "GNN-White" ^
  --white_search npls ^
  --white_mode modelC ^
  --white_weights weights/checkpoint.pth

REM [example] multi-game with color swap
REM call "%~dp0wsl_python.bat" play/match.py ^
REM   --games 10 ^
REM   --think_time_ms 500 ^
REM   --swap_colors ^
REM   --save_csa ^
REM   --black_weights weights/checkpoint.pth ^
REM   --white_weights weights/checkpoint.pth

REM [example] NPLS vs MCTS
REM call "%~dp0wsl_python.bat" play/match.py ^
REM   --games 4 ^
REM   --think_time_ms 1000 ^
REM   --swap_colors ^
REM   --save_csa ^
REM   --black_name "NPLS" ^
REM   --black_search npls ^
REM   --black_weights weights/checkpoint.pth ^
REM   --white_name "MCTS" ^
REM   --white_search mcts ^
REM   --white_weights weights/checkpoint.pth

pause
