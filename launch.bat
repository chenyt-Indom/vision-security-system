@echo off
title Vision Security System
set PYTHON=C:\Users\19853\AppData\Local\Programs\Python\Python312\python.exe
cd /d "D:\视觉安防系统"

echo ============================================
echo   Vision Security System
echo   AI Smoking Detection
echo ============================================
echo.
echo [1] Start GUI Monitor Mode
echo [2] Start Headless Background Mode
echo [3] Open Admin Review Panel
echo [4] Exit
echo.
set /p choice="Select (1-4): "

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto headless
if "%choice%"=="3" goto admin
if "%choice%"=="4" goto end
goto end

:gui
echo Starting GUI Monitor Mode...
%PYTHON% main.py
goto end

:headless
echo Starting Headless Background Mode...
echo Detection running in background, screenshots saved to alerts/
echo Press Ctrl+C to stop
%PYTHON% headless_mode.py
goto end

:admin
echo Starting Admin Review Panel...
%PYTHON% standalone_admin.py
goto end

:end
echo Goodbye!
pause