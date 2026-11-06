@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo MCP Image to 3D installer

rem --- prerequisites ---------------------------------------------------------
where docker >nul 2>nul
if errorlevel 1 (
    echo Docker is required. Install Docker Desktop first.
    goto fail
)
docker compose version >nul 2>nul
if errorlevel 1 (
    echo Docker Compose v2 is required ^(the "docker compose" command^).
    goto fail
)

set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python 3.10+ is required.
        goto fail
    )
    set "PY=py -3"
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo Python 3.10+ is required.
    goto fail
)

rem --- host environment ------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto fail
)
set "VPY=.venv\Scripts\python.exe"
rem On Windows pip retries endlessly when it cannot write its cache, and these few
rem small packages do not need one.
set "PIP_NO_CACHE_DIR=1"

"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%VPY%" -m pip install -r requirements-dev.txt
if errorlevel 1 goto fail

if defined HF_TOKEN goto token_ok
if exist "%USERPROFILE%\.cache\huggingface\token" goto token_ok
if defined HF_HOME if exist "%HF_HOME%\token" goto token_ok

echo.
echo No Hugging Face token found.
echo Accept the DINOv3 terms first:
echo   https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m
echo Then log in:
if exist ".venv\Scripts\hf.exe" (
    ".venv\Scripts\hf.exe" auth login
) else if exist ".venv\Scripts\huggingface-cli.exe" (
    ".venv\Scripts\huggingface-cli.exe" login
) else (
    echo Neither "hf" nor "huggingface-cli" is available.
    goto fail
)
if errorlevel 1 goto fail

:token_ok
rem --- weights ---------------------------------------------------------------
"%VPY%" tools\prefetch.py
if errorlevel 1 goto fail
"%VPY%" tools\prepare_local_weights.py
if errorlevel 1 goto fail

rem --- service ---------------------------------------------------------------
docker compose up -d --build
if errorlevel 1 goto fail

echo Waiting for the model to become ready...
powershell -NoProfile -ExecutionPolicy Bypass -Command "for($i=0;$i -lt 60;$i++){try{$s=(Invoke-RestMethod -Uri 'http://127.0.0.1:8043/health' -TimeoutSec 5).ready}catch{$s=$false}; if($s){exit 0}; Start-Sleep -Seconds 3}; exit 1"
if errorlevel 1 (
    echo The model did not become ready. Recent logs:
    docker compose logs --tail=50 mcp-image-to-3d
    goto fail
)

echo.
echo MCP Image to 3D is ready: http://localhost:8043
echo MCP endpoint:            http://localhost:8043/mcp
if not defined CI pause
exit /b 0

:fail
echo.
echo Install failed.
if not defined CI pause
exit /b 1
