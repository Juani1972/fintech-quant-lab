@echo off
REM ============================================================
REM  Fintech Quant Lab - Instalador (Windows)
REM  Crea el entorno virtual e instala todas las dependencias.
REM  Ejecutar UNA sola vez. Despues usa launch.bat.
REM ============================================================

echo.
echo ============================================================
echo   FINTECH QUANT LAB - INSTALACION
echo ============================================================
echo.

REM 1. Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3 no esta instalado.
    echo.
    echo Descargalo de: https://www.python.org/downloads/
    echo IMPORTANTE: marca "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)
echo [OK] Python detectado.

REM 2. Crear entorno virtual si no existe
if not exist "venv" (
    echo [1/3] Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado.
) else (
    echo [1/3] El entorno virtual ya existe, se reutiliza.
)

REM 3. Actualizar pip
echo [2/3] Actualizando pip...
venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1

REM 4. Instalar dependencias
echo [3/3] Instalando dependencias (puede tardar 2-3 minutos)...
echo.
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo la instalacion de dependencias.
    echo.
    echo Causas tipicas:
    echo   - Sin conexion a internet
    echo   - Antivirus bloqueando pip
    echo   - Falta Microsoft Visual C++ Build Tools
    echo.
    echo Solucion: borra la carpeta "venv" y vuelve a ejecutar este archivo.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   INSTALACION COMPLETADA
echo ============================================================
echo.
echo Para lanzar la aplicacion, ejecuta: launch.bat
echo.
pause