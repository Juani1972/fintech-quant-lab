@echo off
REM ============================================================
REM  Fintech Quant Lab - Lanzador de la GUI (Windows)
REM  Abre Streamlit y el navegador en modo maximizado
REM ============================================================

REM 1. Verificar entorno virtual
if not exist "venv" (
    echo ❌ No se encontró el entorno virtual.
    echo 👉 Ejecuta primero: install.bat
    pause
    exit /b 1
)

REM 2. Activar entorno virtual
call venv\Scripts\activate.bat

REM 3. Iniciar Streamlit en segundo plano
REM    IMPORTANTE: "python -m streamlit" (no "streamlit" a secas). El
REM    ejecutable streamlit.exe vive en venv\Scripts\, así que si se
REM    invoca directamente, Python añade ESA carpeta a sys.path en vez
REM    de la raíz del proyecto -- y falla con
REM    "ModuleNotFoundError: No module named 'app'" al importar
REM    app.config. "python -m streamlit" sí añade el directorio actual
REM    (la raíz del proyecto) a sys.path, que es donde vive el paquete app.
echo 🌐 Iniciando servidor Streamlit en http://localhost:8501 ...
start /B python -m streamlit run app/main.py --server.port=8501 --server.headless=true

REM 4. Esperar a que el servidor esté listo
timeout /t 4 /nobreak >nul

REM 5. Abrir navegador maximizado
set URL=http://localhost:8501

REM Intentar con Chrome
start chrome --start-maximized %URL% 2>nul
if errorlevel 1 (
    REM Intentar con Edge
    start msedge --start-maximized %URL% 2>nul
    if errorlevel 1 (
        REM Fallback: navegador por defecto
        start %URL%
    )
)

echo ✅ Aplicación abierta. Para cerrar, usa el botón dentro de la GUI.
pause
