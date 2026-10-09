@echo off
REM ============================================================
REM C++ 部分のビルド（WSL Ubuntu-22.04 + プロジェクト .venv）
REM   1) search/cpp/build.sh        : Python から使う C++ 探索モジュール
REM   2) search/cpp/build_engine.sh : 全部 C++ の USI エンジン（LibTorch）
REM 事前に WSL 側で: sudo apt install -y build-essential python3-dev
REM                  .venv で pip install pybind11
REM ============================================================
setlocal EnableExtensions
if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu-22.04"
cd /d "%~dp0.."
for /f "usebackq delims=" %%i in (`wsl -d %WSL_DISTRO% wslpath -a "%CD%" 0^<nul`) do set "WSL_PROJECT=%%i"
wsl -d %WSL_DISTRO% --cd "%WSL_PROJECT%" -- bash -c "source .venv/bin/activate && bash search/cpp/build.sh && bash search/cpp/build_engine.sh"
pause
