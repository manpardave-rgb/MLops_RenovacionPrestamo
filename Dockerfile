# Dockerfile — Imagen del servicio API (FastAPI)
# Usado por docker-compose.preprod.yml (servicio: api)
FROM python:3.10-slim

LABEL maintainer="MLOps Renovación de Préstamo — Banco Wiesse"
LABEL description="API de clasificación de propensión de renovación de préstamo"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=preprod

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python — copiar antes del código para aprovechar el cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la API y del pipeline
COPY api/ ./api/
COPY src/ ./src/

# Directorio de artefactos (el volumen Docker lo llenará en runtime)
RUN mkdir -p artifacts

EXPOSE 8000

# Health check interno
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
