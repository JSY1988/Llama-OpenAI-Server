@echo off
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_HOME=C:\Program Files\Python311
set "PYTHON_EXE=%PYTHON_HOME%\python.exe"

(
echo home = %PYTHON_HOME%
echo include-system-site-packages = false
echo version = 3.11.9
echo executable = %PYTHON_EXE%
echo command = %PYTHON_EXE% -m venv "%VENV_DIR%"
) > "%VENV_DIR%\pyvenv.cfg" 2>nul


set "VENV_DIR=%~dp0.venv"
set "ACTIVATE_BAT=%VENV_DIR%\Scripts\activate.bat"
if not exist "%ACTIVATE_BAT%" (
    echo Error: activate.bat not found
    pause
    exit /b 1
)

set "TEMP_FILE=%ACTIVATE_BAT%.tmp"
set "NEW_SET=set "VIRTUAL_ENV=%VENV_DIR%""

(
    for /f "usebackq delims=" %%a in ("%ACTIVATE_BAT%") do (
        set "line=%%a"
        rem Check if line contains VIRTUAL_ENV=
        echo !line! | findstr /i /c:"VIRTUAL_ENV=" >nul
        if !errorlevel! equ 0 (
            echo !NEW_SET!
        ) else (
            echo !line!
        )
    )
) > "%TEMP_FILE%"

if exist "%TEMP_FILE%" (
    move /y "%TEMP_FILE%" "%ACTIVATE_BAT%" >nul
    echo Successfully
) else (
    echo Failed
)
pause