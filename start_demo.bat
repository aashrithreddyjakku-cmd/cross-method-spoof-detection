@echo off
rem Double-click to start the detector demo, then open http://127.0.0.1:7860 in a browser. Close this window to stop it.
cd /d "%~dp0"
call "%USERPROFILE%\.venvs\spoofdet\Scripts\activate.bat"
set SPOOFDET_DATA=C:\spoofdet-data
set PYTHONIOENCODING=utf-8
set GRADIO_ANALYTICS_ENABLED=False
if exist results\scores\p0_lcnn_mel64_w400_s1\DONE (
  python -m spoofdet.app.demo --ckpt checkpoints\p0_lcnn_mel64_w400_s1\best.pt
) else (
  python -m spoofdet.app.demo --ckpt C:\spoofdet-data\backup_2026-10-20\checkpoints\p0_lcnn_mel64_w400_s1\best.pt --threshold 0.12
)
pause
