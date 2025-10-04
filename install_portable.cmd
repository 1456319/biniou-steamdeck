@echo off
setlocal

REM Temporarily map the script's network path to a drive letter if needed.
pushd "%~dp0"

REM **********************************************************************
REM ******* FINAL OFFLINE BINIOU INSTALLER - NO ADMIN RIGHTS REQUIRED ******
REM **********************************************************************
REM * This script installs Biniou and all dependencies from local files.
REM * It requires NO internet connection to run.
REM *
REM * BEFORE RUNNING:
REM * 1. Create a "tools" folder containing the 6 required tool/repo archives.
REM * 2. Create a "wheels" folder containing all required Python .whl packages.
REM **********************************************************************

REM Set the root directory to the current directory (now a reliable local path).
set "ROOT_DIR=%cd%"

echo #####################################################################
echo #                                                                   #
echo #              Biniou Final Offline Installer                       #
echo #                                                                   #
echo #####################################################################
echo.
echo This script will install Biniou using the archives you have downloaded.
echo.
echo Press any key to start the installation.
pause > nul
echo.

REM --- Define file names for the pre-downloaded tools ---
set "FILE_PYTHON=python-3.11.9-embed-amd64.zip"
set "FILE_GIT=PortableGit-2.45.2-64-bit.7z"
set "FILE_MINGW=winlibs-x86_64-posix-seh-gcc-13.2.0-mingw-w64ucrt-11.0.1-r1.7z"
set "FILE_FFMPEG=ffmpeg-master-latest-win64-gpl.zip"
set "FILE_7ZIP=7z-extra.7z"
set "FILE_REPO=biniou-main.zip"

REM --- Define paths for portable tools ---
set "TOOLS_DIR=%ROOT_DIR%\tools"
set "WHEELS_DIR=%ROOT_DIR%\wheels"
set "PYTHON_DIR=%TOOLS_DIR%\python"
set "GIT_DIR=%TOOLS_DIR%\git"
set "MINGW_DIR=%TOOLS_DIR%\mingw64"
set "FFMPEG_DIR=%TOOLS_DIR%\ffmpeg"
set "SEVENZIP_DIR=%TOOLS_DIR%\7z"

REM --- Check for required folders ---
if not exist "%TOOLS_DIR%" (
    echo ERROR: The "tools" directory does not exist.
    pause
    exit /b 1
)
if not exist "%WHEELS_DIR%" (
    echo ERROR: The "wheels" directory does not exist.
    pause
    exit /b 1
)

REM **************************************************
REM *** 1. EXTRACT 7-ZIP (to extract everything else)
REM **************************************************
echo [1/6] Setting up 7-Zip...
if exist "%SEVENZIP_DIR%\7z.exe" (
    echo 7-Zip is already set up.
) else (
    if not exist "%TOOLS_DIR%\%FILE_7ZIP%" (
        echo ERROR: %FILE_7ZIP% not found in the 'tools' folder.
        pause
        exit /b 1
    )
    echo Extracting 7-Zip...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%TOOLS_DIR%\%FILE_7ZIP%' -DestinationPath '%SEVENZIP_DIR%' -Force"
    if not exist "%SEVENZIP_DIR%\7z.exe" (
        echo ERROR: Failed to extract 7-Zip.
        pause
        exit /b 1
    )
)
set "SEVENZIP_EXE=%SEVENZIP_DIR%\7z.exe"
echo.


REM **************************************************
REM *** 2. EXTRACT PORTABLE PYTHON
REM **************************************************
echo [2/6] Setting up Python...
if exist "%PYTHON_DIR%\python.exe" (
    echo Python is already set up.
) else (
    if not exist "%TOOLS_DIR%\%FILE_PYTHON%" (
        echo ERROR: %FILE_PYTHON% not found in the 'tools' folder.
        pause
        exit /b 1
    )
    echo Extracting Python...
    "%SEVENZIP_EXE%" x "%TOOLS_DIR%\%FILE_PYTHON%" -o"%PYTHON_DIR%" -y > nul
    if not exist "%PYTHON_DIR%\python.exe" (
        echo ERROR: Failed to extract Python.
        pause
        exit /b 1
    )
)
echo.

REM **************************************************
REM *** 3. EXTRACT PORTABLE GIT (Needed for OpenSSL)
REM **************************************************
echo [3/6] Setting up Git...
if exist "%GIT_DIR%\cmd\git.exe" (
    echo Git is already set up.
) else (
    if not exist "%TOOLS_DIR%\%FILE_GIT%" (
        echo ERROR: %FILE_GIT% not found in the 'tools' folder.
        pause
        exit /b 1
    )
    echo Extracting Git...
    "%SEVENZIP_EXE%" x "%TOOLS_DIR%\%FILE_GIT%" -o"%GIT_DIR%" -y > nul
    if not exist "%GIT_DIR%\cmd\git.exe" (
        echo ERROR: Failed to extract Git.
        pause
        exit /b 1
    )
)
echo.

REM **************************************************
REM *** 4. EXTRACT MINGW-W64 COMPILER
REM **************************************************
echo [4/6] Setting up MinGW-w64 Compiler...
if exist "%MINGW_DIR%\bin\gcc.exe" (
    echo MinGW-w64 is already set up.
) else (
    if not exist "%TOOLS_DIR%\%FILE_MINGW%" (
        echo ERROR: %FILE_MINGW% not found in the 'tools' folder.
        pause
        exit /b 1
    )
    echo Extracting MinGW-w64...
    "%SEVENZIP_EXE%" x "%TOOLS_DIR%\%FILE_MINGW%" -o"%MINGW_DIR%" -y > nul
    if not exist "%MINGW_DIR%\bin\gcc.exe" (
        echo ERROR: Failed to extract MinGW-w64.
        pause
        exit /b 1
    )
)
echo.

REM **************************************************
REM *** 5. EXTRACT FFMPEG
REM **************************************************
echo [5/6] Setting up FFmpeg...
if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" (
    echo FFmpeg is already set up.
) else (
    if not exist "%TOOLS_DIR%\%FILE_FFMPEG%" (
        echo ERROR: %FILE_FFMPEG% not found in the 'tools' folder.
        pause
        exit /b 1
    )
    echo Extracting FFmpeg...
    mkdir "%FFMPEG_DIR%_temp"
    "%SEVENZIP_EXE%" x "%TOOLS_DIR%\%FILE_FFMPEG%" -o"%FFMPEG_DIR%_temp" -y > nul
    
    pushd "%FFMPEG_DIR%_temp"
    for /d %%i in (*) do (
        move "%%i\*" ".."
        rd "%%i"
    )
    popd
    move "%FFMPEG_DIR%_temp\*" "%FFMPEG_DIR%\" > nul
    rmdir /s /q "%FFMPEG_DIR%_temp"

    if not exist "%FFMPEG_DIR%\bin\ffmpeg.exe" (
        echo ERROR: Failed to extract FFmpeg.
        pause
        exit /b 1
    )
)
echo.
echo All tools have been extracted successfully.
echo.

REM --- Define Biniou directory ---
set "BINIOU_DIR=%ROOT_DIR%\biniou"

REM **************************************************
REM *** 6. EXTRACT BINIOU REPOSITORY
REM **************************************************
echo [6/6] Setting up Biniou source code...
if exist "%BINIOU_DIR%" (
    echo Biniou directory already exists.
) else (
    if not exist "%TOOLS_DIR%\%FILE_REPO%" (
        echo ERROR: %FILE_REPO% not found in the 'tools' folder.
        pause
        exit /b 1
    )
    echo Extracting Biniou repository zip...
    REM Extract to a temporary dir first
    mkdir "%ROOT_DIR%\biniou-main"
    "%SEVENZIP_EXE%" x "%TOOLS_DIR%\%FILE_REPO%" -o"%ROOT_DIR%" -y > nul
    
    REM Rename the extracted folder
    move "%ROOT_DIR%\biniou-main" "%BINIOU_DIR%"

    if not exist "%BINIOU_DIR%\webui.cmd" (
        echo ERROR: Failed to extract the Biniou repository correctly.
        pause
        exit /b 1
    )
)
echo.

REM **************************************************
REM *** SETTING UP THE ENVIRONMENT FOR THIS SESSION
REM **************************************************
echo Setting up the environment for the installation process...
set "PATH=%PYTHON_DIR%;%GIT_DIR%\cmd;%MINGW_DIR%\bin;%FFMPEG_DIR%\bin;%PATH%"

echo.
echo Verifying that the correct tools are in the PATH:
where python
where git
where gcc
where ffmpeg
echo.

echo Environment is configured. The installation will now proceed.
echo.
cd "%BINIOU_DIR%"

REM **************************************************
REM *** 7. CREATING DIRECTORIES
REM **************************************************
echo [7/8] Creating necessary directories...
if not exist "%BINIOU_DIR%\outputs" mkdir "%BINIOU_DIR%\outputs"
if not exist "%BINIOU_DIR%\ssl" mkdir "%BINIOU_DIR%\ssl"
if not exist "%BINIOU_DIR%\models\Audiocrack" mkdir "%BINIOU_DIR%\models\Audiocrack"
echo Directories created.
echo.

REM **************************************************
REM *** 8. INSTALLING PYTHON VIRTUAL ENVIRONMENT & DEPENDENCIES
REM **************************************************
echo [8/8] Setting up Python virtual environment and installing dependencies...

REM Create SSL certificate using portable git's openssl
echo Creating SSL certificate...
for /f "delims=" %%F in ('dir /b /s "%GIT_DIR%\usr\bin\openssl.exe"') do (
  set "OPENSSL_PATH=%%F"
)

if defined OPENSSL_PATH (
    "%OPENSSL_PATH%" req -x509 -newkey rsa:4096 -keyout "%BINIOU_DIR%\ssl\key.pem" -out "%BINIOU_DIR%\ssl\cert.pem" -sha256 -days 3650 -nodes -subj "/C=FR/ST=Paris/L=Paris/O=Biniou/OU=/CN="
) else (
    echo ERROR: Could not find openssl.exe in the portable Git directory.
    pause
    exit /b 1
)


REM The embeddable python package does not include the venv module by default.
echo Enabling site packages for venv creation...
powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-Content -Path '%PYTHON_DIR%\python311._pth') -replace '#import site', 'import site' | Set-Content -Path '%PYTHON_DIR%\python311._pth'"

REM Create Python virtual environment
echo Creating Python virtual environment...
python -m venv "%BINIOU_DIR%\env"
if errorlevel 1 (
    echo ERROR: Failed to create Python virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment
call "%BINIOU_DIR%\env\Scripts\activate.bat"

echo Installing Python packages from local 'wheels' folder...
python -m pip install --upgrade pip
pip install wheel

REM --- Install all other packages from the local wheels folder ---
echo Installing PyTorch and other dependencies...
pip install --no-index --find-links="%WHEELS_DIR%" -r requirements.txt
pip install --no-index --find-links="%WHEELS_DIR%" torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu
pip install --no-index --find-links="%WHEELS_DIR%" llama-cpp-python

if errorlevel 1 (
    echo ERROR: Failed to install Python packages from the 'wheels' folder.
    echo Please ensure all required .whl files are present.
    pause
    exit /b 1
)

echo.
echo #####################################################################
echo #                                                                   #
echo #              Installation Complete!                               #
echo #                                                                   #
echo #####################################################################
echo.
echo You can now launch Biniou by double-clicking the
echo "webui.cmd" file inside the "%BINIOU_DIR%" directory.
echo.
echo Press any key to exit the installer.
pause > nul

REM Return to the original directory and unmap the network drive if one was created.
popd
endlocal
