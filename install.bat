@echo off
REM ============================================================
REM  Fintech Quant Lab - Script de instalación (Windows)
REM  Crea el entorno virtual e instala las dependencias
REM ============================================================

echo 🚀 Iniciando instalación de Fintech Quant Lab...

REM 1. Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3 no está instalado. Instálalo antes de continuar.
    pause
    exit /b 1
)

echo ✅ Python detectado.

REM 2. Crear entorno virtual
if not exist "venv" (
    echo 📦 Creando entorno virtual...
    python -m venv venv
) else (
    echo 📦 El entorno virtual ya existe.
)

REM 3. Activar entorno virtual
call venv\Scripts\activate.bat

REM 4. Actualizar pip
echo ⬆️  Actualizando pip...
python -m pip install --upgrade pip

REM 5. Instalar dependencias
echo 📥 Instalando dependencias desde requirements.txt...
pip install -r requirements.txt

echo.
echo ✅ Instalación completada.
echo 👉 Para lanzar la aplicación, ejecuta: launch.bat
pause
