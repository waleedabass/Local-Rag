@echo off
setlocal
cd /d "%~dp0"

REM Optional: activate local venv if present
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

echo [1/2] Ingesting PDFs from .\source_docs ...
python ingest.py
if errorlevel 1 (
  echo Ingest failed. Fix errors above, then retry.
  pause
  exit /b 1
)

echo.
echo [2/2] Starting server at http://127.0.0.1:8000
echo Open the URL in your browser. Ensure Ollama is running with: ollama pull llama3.2:3b
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
