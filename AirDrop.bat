@echo off
rem ============================================================================
rem  Ultrasonic AirDrop - one launcher for sending, receiving and diagnostics.
rem  Double-click to open the menu, or drag files onto this .bat to send them.
rem ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title Ultrasonic AirDrop

set "PY=%~dp0.venv\Scripts\python.exe"
set "RECV_DIR=%USERPROFILE%\Downloads\Ultrasonic AirDrop"
set "PROFILE=nu"

if not exist "%PY%" call :setup
if not exist "%PY%" goto :fatal

rem Files dropped onto the .bat: go straight to sending them.
if "%~1"=="" goto :menu
set "FILES=%*"
goto :send_with_files

:menu
cls
echo ============================================================
echo   Ultrasonic AirDrop          profile: %PROFILE%
echo   (nu = near-ultrasonic 17.5-20 kHz, au = audible 3-5 kHz)
echo ============================================================
echo    1  Open the app (GUI)
echo    2  RECEIVE files
echo    3  SEND files
echo    4  Estimate transfer time for files (plays nothing)
echo    5  Hardware self-check (plays test sweeps)
echo    6  Link test - this PC LISTENS
echo    7  Link test - this PC SENDS probes
echo    8  List audio devices
echo    9  Switch profile (nu / au)
echo    S  Simulate a transfer (no sound, tests the software only)
echo    U  Update / reinstall
echo    Q  Quit
echo.
choice /c 123456789SUQ /n /m "Choose: "
if errorlevel 12 goto :eof
if errorlevel 11 goto :reinstall
if errorlevel 10 goto :simulate
if errorlevel 9 goto :toggle_profile
if errorlevel 8 goto :devices
if errorlevel 7 goto :linktest_send
if errorlevel 6 goto :linktest_receive
if errorlevel 5 goto :check
if errorlevel 4 goto :estimate
if errorlevel 3 goto :send
if errorlevel 2 goto :receive
if errorlevel 1 goto :gui
goto :menu

rem ---------------------------------------------------------------------------
:gui
start "" "%~dp0.venv\Scripts\pythonw.exe" -m ultrasonic_airdrop
goto :menu

:receive
echo.
echo Files will be saved to: %RECV_DIR%
set "NEWDIR="
set /p "NEWDIR=Press Enter to keep it, or type another folder: "
if defined NEWDIR set "RECV_DIR=%NEWDIR:"=%"
echo.
set "ACCEPT="
choice /c YN /n /m "Accept every offer automatically? [Y/N] "
if errorlevel 2 goto :receive_run
set "ACCEPT=--accept-all"
:receive_run
echo.
echo Listening... The other PC can now send.
echo Press Ctrl+C to stop listening (answer N to "Terminate batch job?" to return to this menu).
"%PY%" -m ultrasonic_airdrop.cli receive --out "%RECV_DIR%" --keep-listening %ACCEPT%
echo.
choice /c YN /n /m "Open the received folder? [Y/N] "
if errorlevel 2 goto :pause_menu
if exist "%RECV_DIR%" explorer "%RECV_DIR%"
goto :pause_menu

:send
call :pick_files
if not defined FILES (
    echo No files selected.
    goto :pause_menu
)
:send_with_files
echo.
echo Files: %FILES%
echo.
echo   1  Interactive (receiver confirms every window - recommended)
echo   2  Broadcast (one-way; use when the other PC cannot answer)
set "MODEOPT="
set "ENCOPT="
choice /c 12 /n /m "Transfer mode: "
if errorlevel 2 set "MODEOPT=--broadcast"
choice /c YN /n /m "Encrypt with a shared passphrase? [Y/N] "
if errorlevel 2 goto :send_run
set "ENCOPT=--encrypt"
:send_run
echo.
echo The estimate is shown next; answer y to start playing sound.
"%PY%" -m ultrasonic_airdrop.cli send %FILES% --profile %PROFILE% %MODEOPT% %ENCOPT% --name "%COMPUTERNAME%"
goto :pause_menu

:estimate
call :pick_files
if not defined FILES (
    echo No files selected.
    goto :pause_menu
)
"%PY%" -m ultrasonic_airdrop.cli --log-level WARNING estimate %FILES% --profile %PROFILE%
goto :pause_menu

:check
echo.
echo This plays 1 s of silence and two 3-second sweeps (high tones) through the speaker.
choice /c YN /n /m "Continue? [Y/N] "
if errorlevel 2 goto :menu
"%PY%" -m ultrasonic_airdrop.cli check
goto :pause_menu

:linktest_receive
echo Listening for probe frames. Start "Link test - SENDS" on the other PC. Ctrl+C to stop.
"%PY%" -m ultrasonic_airdrop.cli linktest receive --profile %PROFILE%
goto :pause_menu

:linktest_send
echo This plays probe frames for 20 seconds.
choice /c YN /n /m "Continue? [Y/N] "
if errorlevel 2 goto :menu
"%PY%" -m ultrasonic_airdrop.cli linktest send --profile %PROFILE% --duration 20
goto :pause_menu

:devices
"%PY%" -m ultrasonic_airdrop.cli devices
goto :pause_menu

:toggle_profile
if "%PROFILE%"=="nu" (set "PROFILE=au") else (set "PROFILE=nu")
goto :menu

:simulate
call :pick_files
if not defined FILES (
    echo No files selected.
    goto :pause_menu
)
echo Simulating with SNR 15 dB, 150 ppm clock offset, RT60 0.3 s room echo...
"%PY%" -m ultrasonic_airdrop.cli --log-level WARNING simulate %FILES% --snr 15 --ppm 150 --reverb 0.3 --profile %PROFILE%
goto :pause_menu

:reinstall
call :setup force
goto :pause_menu

:pause_menu
echo.
pause
goto :menu

rem ---------------------------------------------------------------------------
rem  :pick_files  -> sets FILES to a space-separated list of quoted paths
rem ---------------------------------------------------------------------------
:pick_files
set "FILES="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Multiselect = $true; $d.Title = 'Choose files'; if ($d.ShowDialog() -eq 'OK') { ($d.FileNames | ForEach-Object { [char]34 + $_ + [char]34 }) -join ' ' }"`) do set "FILES=%%F"
if defined FILES goto :eof
echo (File dialog unavailable or cancelled.)
set /p "FILES=Type or paste a file path (drag a file into this window), or press Enter to cancel: "
goto :eof

rem ---------------------------------------------------------------------------
rem  :setup  -> creates .venv and installs the app
rem ---------------------------------------------------------------------------
:setup
set "BASEPY="
rem Prefer a supported version via the Windows launcher, then plain "python".
for %%V in (3.13 3.12 3.11) do if not defined BASEPY py -%%V -c "import sys" >nul 2>nul && set "BASEPY=py -%%V"
if not defined BASEPY python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul && set "BASEPY=python"
if not defined BASEPY (
    echo Python 3.11, 3.12 or 3.13 was not found. Install one from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH", then run this file again.
    goto :eof
)
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Creating the virtual environment in .venv ...
    %BASEPY% -m venv "%~dp0.venv"
)
if not exist "%PY%" (
    echo Could not create the virtual environment.
    goto :eof
)
echo Installing Ultrasonic AirDrop and its dependencies (one-time, needs internet) ...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -e "%~dp0."
if errorlevel 1 echo Installation failed - see the messages above.
goto :eof

:fatal
echo.
echo Setup did not complete. Fix the problem above and run AirDrop.bat again.
pause
exit /b 1
