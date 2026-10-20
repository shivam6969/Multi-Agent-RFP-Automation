@echo off
rem Start SYNAPSE on this laptop (double-click, or run "start.bat" in a terminal).
rem First run creates the Python environment in venv\ and installs packages.
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [start] Creating the Python environment in venv\ ^(first run only^) ...
    py -3.12 -m venv venv 2>nul || python -m venv venv
    if not exist venv\Scripts\python.exe (
        echo.
        echo [start] PROBLEM: Python 3.12 was not found. Install it from https://www.python.org/downloads/
        echo         ^(tick "Add python.exe to PATH"^), then double-click start.bat again.
        pause
        exit /b 1
    )
)

venv\Scripts\python.exe start.py %*
if errorlevel 1 pause
