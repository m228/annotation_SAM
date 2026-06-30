@echo off
REM QuickLabel launcher (Windows). Finds venv automatically.
setlocal
set "ROOT=%~dp0"

REM Try local .venv first, then parent .venv
set "VENV=%ROOT%.venv\Scripts\python.exe"
if not exist "%VENV%" set "VENV=%ROOT%..\.venv\Scripts\python.exe"
if not exist "%VENV%" (
  echo Python venv not found. Run setup.ps1 first.
  pause
  exit /b 1
)

cd /d "%ROOT%"
echo Starting QuickLabel on http://127.0.0.1:8765
"%VENV%" -m backend.server
