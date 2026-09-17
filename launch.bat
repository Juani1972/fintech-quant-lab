@echo off
REM ============================================================
REM  Fintech Quant Lab - Lanzador de la GUI (Windows)
REM ============================================================
REM  Usa SIEMPRE la ruta completa a venv\Scripts\python.exe en vez de
REM  "call venv\Scripts\activate.bat" + "python" a secas. Es más
REM  robusto: activate.bat depende de que modificar el PATH de la
REM  sesión de cmd.exe funcione correctamente, y en algunos entornos
REM  (antivirus, otra instalación de Python por delante en el PATH,
REM  políticas corporativas) eso falla en silencio -- "python" acaba
REM  resolviendo a un intérprete distinto al del venv, y entonces
REM  cualquier comprobación de dependencias da un falso negativo cada
REM  vez, aunque sí estén instaladas dentro del venv. Con la ruta
REM  completa no hay ambigüedad posible.

set VENV_PY=venv\Scripts\python.exe

REM 1. Crear entorno virtual si no existe
if not exist "venv" (
    echo 📦 No se encontró el entorno virtual -- creándolo por ti...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Python 3 no está instalado. Instálalo antes de continuar
        echo    ^(https://www.python.org/downloads/^) y vuelve a ejecutar este archivo.
        pause
        exit /b 1
    )
    python -m venv venv
    if errorlevel 1 (
        echo ❌ No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

if not exist "%VENV_PY%" (
    echo ❌ El entorno virtual existe pero está incompleto ^(no se encuentra
    echo    %VENV_PY%^). Borra la carpeta "venv" y ejecuta este archivo de nuevo.
    pause
    exit /b 1
)

REM 2. Comprobar que streamlit se puede IMPORTAR de verdad con el
REM    intérprete del venv -- no basta con que "pip" lo tenga listado
REM    como instalado: puede fallar al cargarse por otra causa (DLL
REM    rota, versión incompatible entre paquetes). En ese caso
REM    reinstalar no arregla nada, así que distinguimos los dos casos
REM    en vez de reinstalar a ciegas siempre.
"%VENV_PY%" -c "import streamlit" 2>streamlit_check_error.tmp
if errorlevel 1 (
    findstr /C:"ModuleNotFoundError" streamlit_check_error.tmp >nul 2>&1
    if errorlevel 1 (
        echo ❌ streamlit está instalado pero falla al cargarse. Este es
        echo    el error real de Python ^(reinstalar no lo arreglaría,
        echo    así que no lo hacemos^):
        echo.
        type streamlit_check_error.tmp
        del streamlit_check_error.tmp >nul 2>&1
        echo.
        echo 👉 Copia el error de arriba y compártelo para diagnosticar la
        echo    causa exacta ^(DLL del sistema que falta, conflicto de
        echo    versiones entre paquetes, antivirus bloqueando algo...^).
        pause
        exit /b 1
    )

    del streamlit_check_error.tmp >nul 2>&1
    echo 📥 streamlit no está instalado -- instalando dependencias...
    "%VENV_PY%" -m pip install --upgrade pip >nul
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ❌ La instalación de dependencias falló ^(revisa el error de arriba^).
        pause
        exit /b 1
    )
    echo ✅ Dependencias instaladas correctamente.
) else (
    del streamlit_check_error.tmp >nul 2>&1
)

REM 3. Iniciar Streamlit en segundo plano
REM    "%VENV_PY%" -m streamlit (no streamlit.exe a secas): así Python
REM    añade el directorio actual (la raíz del proyecto) a sys.path,
REM    que es donde vive el paquete "app" -- si no, falla con
REM    "ModuleNotFoundError: No module named 'app'" al importar
REM    app.config.
echo.
echo 🌐 Iniciando Fintech Quant Lab en http://localhost:8501 ...
echo.
start "FintechQuantLab" /B "%VENV_PY%" -m streamlit run app/main.py --server.port=8501 --server.headless=true

echo Esperando a que el servidor arranque...
timeout /t 8 /nobreak >nul

set URL=http://localhost:8501
echo Abriendo navegador...
start "" "%URL%"

echo.
echo Aplicación abierta. Si el navegador no se abrió, ve manualmente a:
echo    %URL%
echo.
echo Para cerrar, usa el botón dentro de la GUI.
echo.
pause
