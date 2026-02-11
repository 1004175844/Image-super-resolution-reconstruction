@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 (
        set "PYTHON_CMD=py -3"
    )
)

if "%PYTHON_CMD%"=="" (
    echo Python not found.
    echo Please run install_dependencies.bat first (it can install Python automatically).
    pause
    exit /b 1
)

echo Starting app.py with: %PYTHON_CMD%
%PYTHON_CMD% app.py
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% neq 0 (
    echo.
    echo app.py exited with code %EXIT_CODE%.
    echo If dependencies are missing, run install_dependencies.bat first.
)

pause
exit /b %EXIT_CODE%
