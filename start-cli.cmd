@echo off
cd /d "%~dp0"

if not exist ".venv312-run\Scripts\anomaly-menu.exe" (
  echo Project CLI is not installed yet.
  echo Run: .venv312-run\Scripts\python.exe -m pip install -e .
  pause
  exit /b 1
)

".venv312-run\Scripts\anomaly-menu.exe"
echo.
echo CLI exited.
pause
