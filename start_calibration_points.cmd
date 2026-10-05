@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m calibration.collect_mount_points %*
if errorlevel 1 pause
