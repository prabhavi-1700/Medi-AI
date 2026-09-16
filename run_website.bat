@echo off
title DermAI - Skin Disease Classification Server
echo =======================================================
echo Starting DermAI Skin Disease Screening Web Application
echo =======================================================
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting Flask web server at http://127.0.0.1:5000 ...
python app.py
pause
