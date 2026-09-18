@echo off
setlocal enabledelayedexpansion
title Hand Gesture Mouse Control
cd /d "%~dp0"

echo ========================================================
echo   Touchless Hand Gesture Mouse Control System
echo ========================================================

REM 1. If standalone pre-built executable exists, launch it directly!
REM This works on ANY Windows PC without requiring Python installed.
if exist "%~dp0dist\HandGestureMouse\HandGestureMouse.exe" (
    echo [OK] Found standalone application bundle.
    echo Launching HandGestureMouse.exe...
    echo.
    start "" "%~dp0dist\HandGestureMouse\HandGestureMouse.exe" %*
    exit /b 0
)

REM 2. Search for Python on this machine dynamically
set "PY_CMD="

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :PYTHON_FOUND
)

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3"
    goto :PYTHON_FOUND
)

for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        set "PY_CMD="%%D\python.exe""
        goto :PYTHON_FOUND
    )
)

for /d %%D in ("%ProgramFiles%\Python3*") do (
    if exist "%%D\python.exe" (
        set "PY_CMD="%%D\python.exe""
        goto :PYTHON_FOUND
    )
)

if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD="%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe""
        goto :PYTHON_FOUND
    )
)

:PYTHON_NOT_FOUND
echo.
echo [!] Python was not found on this computer.
echo.
echo To run this application:
echo  1. Run this command in PowerShell or Terminal to install Python automatically:
echo     winget install Python.Python.3.11 -e
echo  2. Or download Python from: https://www.python.org/downloads/
echo     (Make sure to check "Add Python to PATH" during installation)
echo.
pause
exit /b 1

:PYTHON_FOUND
echo [OK] Using Python: %PY_CMD%

%PY_CMD% -c "import cv2, mediapipe, numpy, pyautogui, pynput" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [*] Installing required packages from requirements.txt...
    echo     This is only needed once on first run on a new computer.
    %PY_CMD% -m pip install -r "%~dp0requirements.txt"
    if !errorlevel! neq 0 (
        echo [!] Failed to install dependencies automatically.
        pause
        exit /b 1
    )
)

echo.
echo Launching application...
echo.
%PY_CMD% "%~dp0main.py" %*

if %errorlevel% neq 0 (
    echo.
    echo Application exited with code %errorlevel%.
    pause
)
