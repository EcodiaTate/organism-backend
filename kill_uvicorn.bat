@echo off
REM Kill all uvicorn processes (Windows equivalent of kill_uvicorn.sh)

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Get all Python processes and check for uvicorn
for /f "tokens=2" %%A in ('wmic process list brief find /i "python" ^| find /v "findstr"') do (
    set found=1
)

if defined found (
    echo Killing uvicorn processes...
    wmic process where "commandline like '%%uvicorn%%'" delete /nointeractive >nul 2>&1
    timeout /t 1 /nobreak >nul

    REM Check if any are still running (force kill)
    for /f "tokens=2" %%A in ('wmic process list brief find /i "uvicorn"') do (
        echo Force-killing remaining uvicorn processes...
        taskkill /FI "COMMANDLINE eq *uvicorn*" /F >nul 2>&1
        goto :done
    )
)

echo No uvicorn processes found.

:done
echo Done.
pause
