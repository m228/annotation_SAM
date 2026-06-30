@echo off
REM QuickLabel updater wrapper (ASCII only - no cyrillic in .bat files)
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %*
