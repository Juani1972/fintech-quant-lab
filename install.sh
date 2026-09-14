#!/usr/bin/env bash
# ============================================================
#  Fintech Quant Lab - Script de instalación
#  Crea el entorno virtual e instala las dependencias
# ============================================================

set -e

echo "🚀 Iniciando instalación de Fintech Quant Lab..."

# 1. Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Instálalo antes de continuar."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python detectado: $PYTHON_VERSION"

# 2. Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
else
    echo "📦 El entorno virtual ya existe."
fi

# 3. Activar entorno virtual
source venv/bin/activate

# 4. Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# 5. Instalar dependencias
echo "📥 Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Instalación completada."
echo "👉 Para lanzar la aplicación, ejecuta: ./launch.sh"
