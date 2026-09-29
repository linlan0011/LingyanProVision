@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0run_app.py"
if errorlevel 1 pause
