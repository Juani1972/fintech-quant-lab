FROM python:3.11-slim

WORKDIR /app

COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .

# Usuario sin privilegios: por defecto el contenedor corría como root,
# lo cual no es necesario para servir una app Streamlit y es una mala
# práctica de seguridad (si el proceso se ve comprometido, el atacante
# hereda privilegios de root dentro del contenedor).
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
