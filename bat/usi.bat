@echo off
REM ============================================================
REM GNN-Shogi USI engine for ShogiGUI / Shogidokoro
REM Register THIS file as the engine executable in the GUI.
REM Python: WSL Ubuntu-22.04 + project .venv
REM ============================================================
set PYTHONIOENCODING=utf-8
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
