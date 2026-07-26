@echo off
REM ============================================================
REM Continue supervised training from existing checkpoint
REM Python: WSL Ubuntu-22.04 + project .venv
REM ============================================================
REM Loads weights/checkpoint.pth and keeps training.
REM Data: 2023+2024 floodgate HCPE (Suisho is optional below)
REM ============================================================

REM call "%~dp0wsl_python.bat" train/supervised_train.py ^
REM   --train_data train_data/2023_train.hcpe train_data/2024_train.hcpe ^
REM   --test_data train_data/2024_test.hcpe ^
REM   --data_format hcpe ^
REM   --mode modelC ^
REM   --epochs 10 ^
REM   --batch_size 256 ^
REM   --lr 0.0003 ^
REM   --resume 1 ^
REM   --save_every 1 ^
REM   --weights weights/checkpoint.pth

REM Optional heavier run (Suisho, long):
call "%~dp0wsl_python.bat" train/supervised_train.py ^
  --train_data train_data/Suisho_train.hcpe ^
  --test_data train_data/Suisho_test.hcpe ^
  --data_format hcpe ^
  --mode modelC ^
  --epochs 3 ^
  --batch_size 128 ^
  --lr 0.0002 ^
  --resume 1 ^
  --save_every 1 ^
  --weights weights/checkpoint.pth

pause
