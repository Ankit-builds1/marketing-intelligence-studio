@echo off
REM ============================================================
REM Causal Inference / MMM project - one-click environment setup
REM ============================================================
REM This script creates a Python virtual environment and installs
REM all required dependencies. Run from inside the project folder.

setlocal

echo.
echo [1/4] Locating a compatible Python (3.11 or 3.12 preferred)...
where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'py' launcher not found. Install Python from python.org.
    exit /b 1
)

REM Prefer 3.11 then 3.12 then 3.10 then whatever default 'py' resolves to.
set PY_CMD=
for %%V in (3.11 3.12 3.10) do (
    py -%%V --version >nul 2>&1
    if not errorlevel 1 (
        if "%PY_CMD%"=="" set PY_CMD=py -%%V
    )
)
if "%PY_CMD%"=="" set PY_CMD=py
echo Using Python: %PY_CMD%

echo.
echo [2/4] Creating virtual environment in .\venv ...
%PY_CMD% -m venv venv
if errorlevel 1 (
    echo ERROR: venv creation failed.
    exit /b 1
)

echo.
echo [3/4] Upgrading pip ...
call venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

echo.
echo [4/4] Installing requirements ...
call venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Some packages failed. If you are on Python 3.14, install Python 3.11
    echo or 3.12 from python.org and re-run this script.
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS - Environment ready.
echo.
echo To activate:        venv\Scripts\activate
echo To open notebook:   jupyter notebook notebooks\causal_inference_mmm.ipynb
echo To launch app:      streamlit run app\budget_optimizer_app.py
echo ============================================================
endlocal
