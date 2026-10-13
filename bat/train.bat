@echo off
REM ============================================================
REM GNN将棋AI 教師あり学習バッチ
REM Python: WSL Ubuntu-22.04 + プロジェクト .venv
REM ============================================================
REM 使い方: このファイルをダブルクリック、またはコマンドから実行
REM
REM 主な引数:
REM   --train_data    学習 HCPE/PSV ファイル
REM   --test_data     テスト HCPE/PSV ファイル
REM   --data_format   hcpe / psv
REM   --mode          モデル構成 (modelC など)
REM   --epochs        エポック数
REM   --batch_size    バッチサイズ
REM   --lr            学習率
REM   --limit         件数制限（動作確認用。本番は付けない）
REM   --weights       保存先 (default: weights/checkpoint.pth)
REM ============================================================

REM ============================================================
REM 【標準設定】2024 floodgate HCPE・1 epoch
REM ============================================================
call "%~dp0wsl_python.bat" train/supervised_train.py ^
  --train_data train_data/2024_train.hcpe ^
  --test_data train_data/2024_test.hcpe ^
  --data_format hcpe ^
  --mode modelC ^
  --epochs 1 ^
  --batch_size 256 ^
  --lr 0.001 ^
  --resume 1 ^
  --weights weights/checkpoint.pth

REM ============================================================
REM 【設定例2】動作確認（件数制限）
REM ============================================================
REM call "%~dp0wsl_python.bat" train/supervised_train.py ^
REM   --train_data train_data/2024_train.hcpe ^
REM   --test_data train_data/2024_test.hcpe ^
REM   --mode modelC ^
REM   --epochs 1 ^
REM   --batch_size 64 ^
REM   --limit 4096 ^
REM   --weights weights/checkpoint.pth

REM ============================================================
REM 【設定例3】水匠 HCPE
REM ============================================================
REM call "%~dp0wsl_python.bat" train/supervised_train.py ^
REM   --train_data train_data/Suisho_train.hcpe ^
REM   --test_data train_data/Suisho_test.hcpe ^
REM   --data_format hcpe ^
REM   --mode modelC ^
REM   --epochs 1 ^
REM   --batch_size 256 ^
REM   --weights weights/checkpoint.pth

REM ============================================================
REM 【設定例4】PSV（水匠）
REM ============================================================
REM call "%~dp0wsl_python.bat" train/supervised_train.py ^
REM   --train_data train_data/Suisho10Mn_psv.bin ^
REM   --data_format psv ^
REM   --mode modelC ^
REM   --epochs 1 ^
REM   --batch_size 256 ^
REM   --psv_score_scale 600 ^
REM   --weights weights/checkpoint.pth

pause
