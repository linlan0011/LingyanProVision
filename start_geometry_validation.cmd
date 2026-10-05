@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -X utf8 -m calibration.collect_mount_points --camera usb0bda3035_1080p --purpose validation %*
if errorlevel 1 pause
