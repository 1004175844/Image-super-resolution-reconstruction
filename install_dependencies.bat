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
    ) else (
        echo Python not found in PATH.
        where winget >nul 2>nul
        if %ERRORLEVEL%==0 (
            echo Trying to install Python 3.12 with winget...
            winget install -e --id Python.Python.3.12 --scope user --silent
            if %ERRORLEVEL% neq 0 (
                echo Failed to install Python via winget.
                echo Please install Python manually, then run this script again.
                pause
                exit /b 1
            )
            where py >nul 2>nul
            if %ERRORLEVEL%==0 (
                set "PYTHON_CMD=py -3"
            ) else (
                where python >nul 2>nul
                if %ERRORLEVEL%==0 set "PYTHON_CMD=python"
            )
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo Python is still unavailable.
    echo Please install Python 3.10+ manually and re-run this script.
    pause
    exit /b 1
)

echo Using Python command: %PYTHON_CMD%
echo Installing dependencies from requirements.txt...

%PYTHON_CMD% -m ensurepip --upgrade >nul 2>nul
%PYTHON_CMD% -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% neq 0 (
    echo Failed to upgrade pip/setuptools/wheel.
    pause
    exit /b 1
)

%PYTHON_CMD% -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo All dependencies installed successfully.
pause
exit /b 0
