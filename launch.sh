#!/usr/bin/env bash
# ============================================================
#  Fintech Quant Lab - Lanzador de la GUI
#  Abre Streamlit y el navegador en modo maximizado
# ============================================================

set -e

# 1. Verificar entorno virtual
if [ ! -d "venv" ]; then
    echo "❌ No se encontró el entorno virtual."
    echo "👉 Ejecuta primero: ./install.sh"
    exit 1
fi

# 2. Activar entorno virtual
source venv/bin/activate

# 3. Iniciar Streamlit en segundo plano
#    IMPORTANTE: "python -m streamlit" (no "streamlit" a secas). El
#    ejecutable streamlit vive en venv/bin/, así que si se invoca
#    directamente, Python añade ESA carpeta a sys.path en vez de la
#    raíz del proyecto -- y falla con
#    "ModuleNotFoundError: No module named 'app'" al importar
#    app.config. "python -m streamlit" sí añade el directorio actual
#    (la raíz del proyecto) a sys.path, que es donde vive el paquete app.
echo "🌐 Iniciando servidor Streamlit en http://localhost:8501 ..."
python -m streamlit run app/main.py --server.port=8501 --server.headless=true &
STREAMLIT_PID=$!

# 4. Esperar a que el servidor esté listo
sleep 3

# 5. Abrir navegador maximizado
URL="http://localhost:8501"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open -a "Google Chrome" --args --start-maximized "$URL" 2>/dev/null || \
    open -a "Safari" "$URL" 2>/dev/null || \
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    google-chrome --start-maximized "$URL" 2>/dev/null || \
    chromium-browser --start-maximized "$URL" 2>/dev/null || \
    firefox --new-window "$URL" 2>/dev/null || \
    xdg-open "$URL"
fi

echo "✅ Aplicación abierta. Para cerrar, usa el botón dentro de la GUI."

# 6. Esperar a que el proceso de Streamlit termine
wait $STREAMLIT_PID
echo "👋 Servidor detenido."
