@echo off
REM ============================================================
REM GNN-Shogi USI engine launcher
REM Python: WSL Ubuntu-22.04 + project .venv
REM ============================================================
REM Args:
REM   --time_rate       time usage rate 0.01-1.0
REM   --draw_value      repetition eval
REM   --impasse_score   impasse declaration score
REM   --impasse_value   impasse eval
REM   --max_moves       max moves
REM   --use_search      1=search, 0=policy only
REM   --search_type     mcts / npls / mc_rzf
REM   --mode            modelC etc.
REM ============================================================

REM [default] standard tournament
call "%~dp0wsl_python.bat" usi.py ^
  --search_type npls ^
  --mode modelC ^
  --weights weights/checkpoint.pth ^
  --name "GNN-Shogi" ^
  --time_rate 0.5 ^
  --draw_value 0 ^
  --impasse_score 28 ^
  --impasse_value 0 ^
  --max_moves 320 ^
  --use_search 1

REM [example] denryusen-like
REM call "%~dp0wsl_python.bat" usi.py ^
REM   --search_type npls ^
REM   --mode modelC ^
REM   --weights weights/checkpoint.pth ^
REM   --name "GNN-Shogi" ^
REM   --time_rate 0.7 ^
REM   --draw_value -30 ^
REM   --impasse_score 28 ^
REM   --impasse_value 0 ^
REM   --max_moves 512 ^
REM   --use_search 1

REM [example] fast no-search
REM call "%~dp0wsl_python.bat" usi.py ^
REM   --search_type mcts ^
REM   --mode fastA ^
REM   --weights weights/checkpoint.pth ^
REM   --name "GNN-Shogi-Fast" ^
REM   --time_rate 0.3 ^
REM   --draw_value 0 ^
REM   --impasse_score 28 ^
REM   --impasse_value 0 ^
REM   --max_moves 320 ^
REM   --use_search 0
