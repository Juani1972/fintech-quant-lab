#!/usr/bin/env bash
# ============================================================
#  Fintech Quant Lab - Lanzador de la GUI
# ============================================================
#  Usa SIEMPRE la ruta completa a venv/bin/python en vez de
#  "source venv/bin/activate" + "python" a secas. Es más robusto:
#  activate depende de que modificar $PATH de la sesión actual
#  funcione correctamente, y en algunos entornos (otro Python por
#  delante en el PATH, shells no estándar) eso falla en silencio --
#  "python" acaba resolviendo a un intérprete distinto al del venv.
#  Con la ruta completa no hay ambigüedad posible.

VENV_PY="venv/bin/python"

# 1. Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 No se encontró el entorno virtual -- creándolo por ti..."
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 no está instalado. Instálalo antes de continuar."
        exit 1
    fi
    python3 -m venv venv
fi

if [ ! -x "$VENV_PY" ]; then
    echo "❌ El entorno virtual existe pero está incompleto (no se encuentra"
    echo "   $VENV_PY). Borra la carpeta 'venv' y ejecuta este script de nuevo."
    exit 1
fi

# 2. Comprobar que streamlit se puede IMPORTAR de verdad con el
#    intérprete del venv -- no basta con que "pip" lo tenga listado
#    como instalado: puede fallar al cargarse por otra causa (versión
#    incompatible entre paquetes, librería del sistema que falta). En
#    ese caso reinstalar no arregla nada, así que distinguimos los dos
#    casos en vez de reinstalar a ciegas siempre.
streamlit_error=$("$VENV_PY" -c "import streamlit" 2>&1)
if [ $? -ne 0 ]; then
    if ! echo "$streamlit_error" | grep -q "ModuleNotFoundError"; then
        echo "❌ streamlit está instalado pero falla al cargarse. Este es"
        echo "   el error real de Python (reinstalar no lo arreglaría,"
        echo "   así que no lo hacemos):"
        echo ""
        echo "$streamlit_error"
        echo ""
        echo "👉 Copia el error de arriba y compártelo para diagnosticar la"
        echo "   causa exacta (versión incompatible entre paquetes, librería"
        echo "   del sistema que falta, etc.)."
        exit 1
    fi

    echo "📥 streamlit no está instalado -- instalando dependencias..."
    "$VENV_PY" -m pip install --upgrade pip &> /dev/null
    if ! "$VENV_PY" -m pip install -r requirements.txt; then
        echo ""
        echo "❌ La instalación de dependencias falló (revisa el error de arriba)."
        exit 1
    fi
    echo "✅ Dependencias instaladas correctamente."
fi

# 3. Iniciar Streamlit en segundo plano
#    "$VENV_PY" -m streamlit (no streamlit a secas): así Python añade
#    el directorio actual (la raíz del proyecto) a sys.path, que es
#    donde vive el paquete "app" -- si no, falla con
#    "ModuleNotFoundError: No module named 'app'" al importar app.config.
echo ""
echo "🌐 Iniciando Fintech Quant Lab en http://localhost:8501 ..."
echo ""
"$VENV_PY" -m streamlit run app/main.py --server.port=8501 --server.headless=true &
STREAMLIT_PID=$!

echo "Esperando a que el servidor arranque..."
sleep 5

URL="http://localhost:8501"
echo "Abriendo navegador..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    open -a "Google Chrome" --args --start-maximized "$URL" 2>/dev/null || \
    open -a "Safari" "$URL" 2>/dev/null || \
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    google-chrome --start-maximized "$URL" 2>/dev/null || \
    chromium-browser --start-maximized "$URL" 2>/dev/null || \
    firefox --new-window "$URL" 2>/dev/null || \
    xdg-open "$URL"
fi

echo ""
echo "Aplicación abierta. Si el navegador no se abrió, ve manualmente a:"
echo "   $URL"
echo ""
echo "Para cerrar, usa el botón dentro de la GUI."
echo ""

# 4. Esperar a que el proceso de Streamlit termine
wait $STREAMLIT_PID
echo "👋 Servidor detenido."
