@echo off
setlocal

REM **********************************************************************
REM *********** PORTABLE BINIOU INSTALLER - NO ADMIN RIGHTS REQUIRED ***********
REM **********************************************************************
REM * This script downloads and installs Biniou and its dependencies
REM * into a self-contained portable directory. No administrator
REM * rights are required.
REM **********************************************************************

REM Set the root directory to wherever this script is located
set "ROOT_DIR=%~dp0"

echo #####################################################################
echo #                                                                   #
echo #              Biniou Portable Installer                            #
echo #                                                                   #
echo #####################################################################
echo.
echo This script will download and set up a portable version of Biniou.
echo All files will be contained in this directory: %ROOT_DIR%
echo.
echo Press any key to start the installation.
pause > nul
echo.

REM --- Define URLs for portable tools ---
set "URL_PYTHON=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "URL_GIT=https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/PortableGit-2.45.2-64-bit.7z.exe"
set "URL_MINGW=https://github.com/winlibs/w64/releases/download/13.2.0-16.0.0-11.0.1-ucrt-r1/winlibs-x86_64-posix-seh-gcc-13.2.0-mingw-w64ucrt-11.0.1-r1.7z"
set "URL_FFMPEG=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "URL_7ZIP=https://www.7-zip.org/a/7z2405-x64.exe"

REM --- Define paths for portable tools ---
set "TOOLS_DIR=%ROOT_DIR%tools"
set "PYTHON_DIR=%TOOLS_DIR%\python"
set "GIT_DIR=%TOOLS_DIR%\git"
set "MINGW_DIR=%TOOLS_DIR%\mingw64"
set "FFMPEG_DIR=%TOOLS_DIR%\ffmpeg"
set "SEVENZIP_DIR=%TOOLS_DIR%\7z"

REM --- Create tools directory ---
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"

REM **************************************************
REM *** 1. DOWNLOAD AND EXTRACT 7-ZIP (for .7z files)
REM **************************************************
echo [1/5] Setting up 7-Zip...
if exist "%SEVENZIP_DIR%\7z.exe" (
    echo 7-Zip is already installed.
) else (
    echo Downloading 7-Zip extractor...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "(New-Object System.Net.WebClient).DownloadFile('%URL_7ZIP%', '%TOOLS_DIR%\7z_installer.exe')"
    if errorlevel 1 (
        echo ERROR: Failed to download 7-Zip. Please check your network connection.
        pause
        exit /b 1
    )

    echo Extracting 7-Zip...
    start /wait "" "%TOOLS_DIR%\7z_installer.exe" -o"%SEVENZIP_DIR%" -y
    del "%TOOLS_DIR%\7z_installer.exe"
    if not exist "%SEVENZIP_DIR%\7z.exe" (
        echo ERROR: Failed to extract 7-Zip.
        pause
        exit /b 1
    )
)
set "SEVENZIP_EXE=%SEVENZIP_DIR%\7z.exe"
echo.

REM **************************************************
REM *** 2. DOWNLOAD AND EXTRACT PORTABLE PYTHON
REM **************************************************
echo [2/5] Setting up Python...
if exist "%PYTHON_DIR%\python.exe" (
    echo Python is already installed.
) else (
    echo Downloading Python (Embeddable)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "(New-Object System.Net.WebClient).DownloadFile('%URL_PYTHON%', '%TOOLS_DIR%\python.zip')"
    if errorlevel 1 (
        echo ERROR: Failed to download Python.
        pause
        exit /b 1
    )

    echo Extracting Python...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%TOOLS_DIR%\python.zip' -DestinationPath '%PYTHON_DIR%' -Force"
    del "%TOOLS_DIR%\python.zip"
    if not exist "%PYTHON_DIR%\python.exe" (
        echo ERROR: Failed to extract Python.
        pause
        exit /b 1
    )
)
echo.

REM **************************************************
REM *** 3. DOWNLOAD AND EXTRACT PORTABLE GIT
REM **************************************************
echo [3/5] Setting up Git...
if exist "%GIT_DIR%\cmd\git.exe" (
    echo Git is already installed.
) else (
    echo Downloading Portable Git...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "(New-Object System.Net.WebClient).DownloadFile('%URL_GIT%', '%TOOLS_DIR%\git_portable.7z.exe')"
    if errorlevel 1 (
        echo ERROR: Failed to download Git.
        pause
        exit /b 1
    )

    echo Extracting Git...
    start /wait "" "%TOOLS_DIR%\git_portable.7z.exe" -o"%GIT_DIR%" -y
    del "%TOOLS_DIR%\git_portable.7z.exe"
    if not exist "%GIT_DIR%\cmd\git.exe" (
        echo ERROR: Failed to extract Git.
        pause
        exit /b 1
    )
)
echo.

REM **************************************************
REM *** 4. DOWNLOAD AND EXTRACT MINGW-W64 COMPILER
REM **************************************************
echo [4/5] Setting up MinGW-w64 Compiler...
if exist "%MINGW_DIR%\bin\gcc.exe" (
    echo MinGW-w64 is already installed.
) else (
    echo Downloading MinGW-w64...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "(New-Object System.Net.WebClient).DownloadFile('%URL_MINGW%', '%TOOLS_DIR%\mingw64.7z')"
    if errorlevel 1 (
        echo ERROR: Failed to download MinGW-w64.
        pause
        exit /b 1
    )

    echo Extracting MinGW-w64...
    "%SEVENZIP_EXE%" x "%TOOLS_DIR%\mingw64.7z" -o"%MINGW_DIR%" -y
    del "%TOOLS_DIR%\mingw64.7z"
    if not exist "%MINGW_DIR%\bin\gcc.exe" (
        echo ERROR: Failed to extract MinGW-w64.
        pause
        exit /b 1
    )
)
echo.

REM **************************************************
REM *** 5. DOWNLOAD AND EXTRACT FFMPEG
REM **************************************************
echo [5/5] Setting up FFmpeg...
if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" (
    echo FFmpeg is already installed.
) else (
    echo Downloading FFmpeg...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "(New-Object System.Net.WebClient).DownloadFile('%URL_FFMPEG%', '%TOOLS_DIR%\ffmpeg.zip')"
    if errorlevel 1 (
        echo ERROR: Failed to download FFmpeg.
        pause
        exit /b 1
    )

    echo Extracting FFmpeg...
    REM Extract to a temp dir and move contents up one level for a cleaner path
    mkdir "%FFMPEG_DIR%_temp"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%TOOLS_DIR%\ffmpeg.zip' -DestinationPath '%FFMPEG_DIR%_temp' -Force"
    del "%TOOLS_DIR%\ffmpeg.zip"

    REM The files are in a nested directory, e.g., ffmpeg-master-latest-win64-gpl
    pushd "%FFMPEG_DIR%_temp"
    for /d %%i in (*) do (
        move "%%i\*" ".."
        rd "%%i"
    )
    popd
    move "%FFMPEG_DIR%_temp\*" "%FFMPEG_DIR%\"
    rmdir "%FFMPEG_DIR%_temp"

    if not exist "%FFMPEG_DIR%\bin\ffmpeg.exe" (
        echo ERROR: Failed to extract FFmpeg.
        pause
        exit /b 1
    )
)
echo.
echo All tools have been downloaded and extracted successfully.
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

REM --- Define Biniou directory ---
set "BINIOU_DIR=%ROOT_DIR%biniou"

REM **************************************************
REM *** 6. CLONING BINIOU REPOSITORY
REM **************************************************
echo [6/8] Cloning Biniou repository...
if exist "%BINIOU_DIR%\.git" (
    echo Biniou repository already exists.
) else (
    git clone --branch main https://github.com/Woolverine94/biniou.git "%BINIOU_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to clone the repository.
        pause
        exit /b 1
    )
)
cd "%BINIOU_DIR%"
git config --global --add safe.directory "%BINIOU_DIR%"
echo.

REM **************************************************
REM *** 7. CREATING DIRECTORIES
REM **************************************************
echo [7/8] Creating necessary directories...
if not exist "%BINIOU_DIR%\outputs" mkdir "%BINIOU_DIR%\outputs"
if not exist "%BINIOU_DIR%\ssl" mkdir "%BINIOU_DIR%\ssl"
if not exist "%BINIOU_DIR%\models\Audiocraft" mkdir "%BINIOU_DIR%\models\Audiocraft"
echo Directories created.
echo.

REM **************************************************
REM *** 8. INSTALLING PYTHON VIRTUAL ENVIRONMENT & DEPENDENCIES
REM **************************************************
echo [8/8] Setting up Python virtual environment and installing dependencies...

REM Create SSL certificate using portable git's openssl
echo Creating SSL certificate...
rem Find openssl.exe inside the git directory
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
REM We need to uncomment the import of site in python311._pth
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

echo Installing Python packages...
python -m pip install --upgrade pip
pip install wheel

echo Installing PyTorch...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

echo Compiling and installing llama-cpp-python...
REM --- This is the critical step: tell pip to use the portable MinGW compiler ---
set "CMAKE_ARGS=-G "MinGW Makefiles""
set "FORCE_CMAKE=1"
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-cache-dir

echo Installing remaining requirements...
pip install -r requirements.txt

REM Cleanup environment variables
set "CMAKE_ARGS="
set "FORCE_CMAKE="

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

endlocal