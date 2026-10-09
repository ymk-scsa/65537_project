@echo off
REM ============================================================
REM 全部 C++ の USI エンジン（LibTorch 版）
REM   事前に bat\build_cpp.bat でビルドし、
REM   python search/cpp/export_torchscript.py でモデル(.ts)を書き出しておくこと。
REM   GUI のエンジン設定で ModelPath にモデルの場所（WSL 側のパス）を指定する。
REM ============================================================
setlocal EnableExtensions
if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu-22.04"
cd /d "%~dp0.."
for /f "usebackq delims=" %%i in (`wsl -d %WSL_DISTRO% wslpath -a "%CD%" 0^<nul`) do set "WSL_PROJECT=%%i"
wsl -d %WSL_DISTRO% --cd "%WSL_PROJECT%" -- ./search/cpp/apple_engine
exit /b %ERRORLEVEL%
