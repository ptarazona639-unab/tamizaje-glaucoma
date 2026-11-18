@echo off
REM Abre la aplicación de tamizaje. La primera vez instala las dependencias.
cd /d "%~dp0"
if not exist .venv (
    py -3 -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
python app.py
