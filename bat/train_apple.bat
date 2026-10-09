@echo off
REM ============================================================
REM Apple 版（RayGraph ハイブリッド + 手番別ヘッド）の教師あり学習
REM   C++ モジュール（bat\build_cpp.bat）をビルドしておくと、データ読み込みが速くなる。
REM   GPU なし（CPU）を想定して、小さいモデル appleS を使う。
REM ============================================================
call "%~dp0wsl_python.bat" train/supervised_train.py ^
  --train_data train_data/2024_train.hcpe ^
  --test_data train_data/2024_test.hcpe ^
  --data_format hcpe ^
  --mode appleS ^
  --epochs 1 ^
  --batch_size 256 ^
  --lr 0.001 ^
  --resume 1 ^
  --weights weights/apple_s.pth
pause
