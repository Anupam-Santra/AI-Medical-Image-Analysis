@echo off
title AI Pneumonia Detector Launcher
echo ===================================================
echo Starting AI-Powered Medical Image Analysis System...
echo ===================================================

:: Activate the virtual environment
call venv\Scripts\activate

:: Check if activation was successful
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate the virtual environment. 
    echo Please make sure the 'venv' folder exists in this directory.
    pause
    exit /b
)

:: Run the Streamlit application
echo Launching the web interface...
streamlit run app.py

:: Pause if the server crashes or is stopped manually
pause