@echo off
rem Double-click on the morning of the demo. It must end with READY FOR THE DEMO.
cd /d "%~dp0"
call "%USERPROFILE%\.venvs\spoofdet\Scripts\activate.bat"
set SPOOFDET_DATA=C:\spoofdet-data
set PYTHONIOENCODING=utf-8
python scripts\demo_check.py
pause
