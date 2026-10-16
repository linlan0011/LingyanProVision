@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m calibration.collect_mount_points --camera usb0bda3035_1080p %*
if errorlevel 1 pause
