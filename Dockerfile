# MODUL 4 - LAB SESI 2 - Langkah 4: Packaging
# Sebelum build, salin model terbaik dari mlruns/ ke folder ./model (lihat panduan)
FROM python:3.11.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies before copying application code and model weights.
COPY requirements-api.txt ./requirements-api.txt
COPY model/requirements.txt ./model/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY scripts/serve.py ./serve.py
COPY model ./model
RUN test -f model/MLmodel && test -f model/model.pkl

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
