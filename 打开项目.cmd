@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Project Python environment is missing. See docs\COMMANDS.md.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m product_search.desktop
endlocal
