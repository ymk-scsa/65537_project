@echo off
REM ============================================================
REM WSL (Ubuntu-22.04) + project .venv python launcher
REM Usage: call "%~dp0wsl_python.bat" script.py [args...]
REM
REM Optional env:
REM   WSL_DISTRO  default Ubuntu-22.04
REM   WSL_VENV    absolute WSL path to venv (default: project/.venv)
REM
REM Path lookup wsl calls must not consume stdin (needed for USI pipes).
REM ============================================================
setlocal EnableExtensions

if not defined WSL_DISTRO set "WSL_DISTRO=Ubuntu-22.04"

cd /d "%~dp0.."

for /f "usebackq delims=" %%i in (`wsl -d %WSL_DISTRO% wslpath -a "%CD%" 0^<nul`) do set "WSL_PROJECT=%%i"
for /f "usebackq delims=" %%i in (`wsl -d %WSL_DISTRO% wslpath -a "%~dp0run_wsl.sh" 0^<nul`) do set "WSL_RUNNER=%%i"

if not defined WSL_PROJECT (
  echo ERROR: failed to resolve WSL project path via wslpath.
  exit /b 1
)
if not defined WSL_RUNNER (
  echo ERROR: failed to resolve WSL runner path via wslpath.
  exit /b 1
)

wsl -d %WSL_DISTRO% -- bash "%WSL_RUNNER%" %*
exit /b %ERRORLEVEL%
