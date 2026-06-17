@echo off
REM QuickLabel launcher (Windows). Self-contained; uses the project venv.
setlocal
set "ROOT=%~dp0"
set "VENV=C:\Users\New\Documents\annotation_SAM\.venv\Scripts\python.exe"
if not exist "%VENV%" (
  echo Python venv not found at "%VENV%". See README.
  exit /b 1
)
cd /d "%ROOT%"
echo Starting QuickLabel...
"%VENV%" -m backend.server
