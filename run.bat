@echo off
title Credit Card Default Predictor - AI Application Lab
cd /d "%~dp0"

echo ======================================================================
echo           CREDIT CARD DEFAULT PREDICTOR - LAUNCHER
echo                 AI Application Lab - Chandigarh University
echo ======================================================================
echo.

REM 1. Detect Python
set "PYTHON_EXE="

REM Try py -3.10
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py -3.10"
    goto :PYTHON_FOUND
)

REM Try py -3
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py -3"
    goto :PYTHON_FOUND
)

REM Try python
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto :PYTHON_FOUND
)

REM Common standard install locations
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :PYTHON_FOUND
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :PYTHON_FOUND
)

if exist "%ProgramFiles%\Python310\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python310\python.exe"
    goto :PYTHON_FOUND
)

if exist "%ProgramFiles%\Python311\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python311\python.exe"
    goto :PYTHON_FOUND
)

:PYTHON_NOT_FOUND
echo [ERROR] Python was not found on your computer.
echo Please install Python 3.10 or higher from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:PYTHON_FOUND
echo Found Python: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

REM 2. Check if dependencies are installed
%PYTHON_EXE% -c "import streamlit, xgboost, fpdf, pptx" >nul 2>&1
if not errorlevel 1 (
    echo All dependencies are verified!
    goto :CHECK_MODELS
)

echo [Step 1 of 3] Installing required packages from requirements.txt...
%PYTHON_EXE% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies reported warnings. Continuing...
)

:CHECK_MODELS
REM 3. Verify if models are trained
if exist "models\xgboost.joblib" goto :LAUNCH_APP

echo.
echo [Step 2 of 3] Training ML classification models...
%PYTHON_EXE% model.py
if errorlevel 1 (
    echo [ERROR] Model training failed.
    pause
    exit /b 1
)

:LAUNCH_APP
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit" >nul 2>&1
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

echo.
echo ======================================================================
echo  [Step 3 of 3] Launching web application in your browser...
echo  App URL: http://localhost:8501
echo  NOTE: Keep this terminal window open while using the application.
echo ======================================================================
echo.

%PYTHON_EXE% -m streamlit run app.py --browser.gatherUsageStats false

echo.
echo Application stopped.
pause
