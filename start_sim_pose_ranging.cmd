@echo off
cd /d "%~dp0"
call "%~dp0start_app.cmd" --camera usb0bda3035_1080p --ranging --target-z-mm 0 %*
