"""
MODUL 4 - LAB SESI 2 - Langkah 4: Packaging - FastAPI

Uji lokal dulu sebelum build Docker:
    uvicorn serve:app --host 0.0.0.0 --port 8080

Cek:
    GET  http://localhost:8080/health   -> {"status": "ok"}
    POST http://localhost:8080/predict  -> {"prediction": ...}
"""
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

# Sebelum di-package ke Docker, path ini diganti ke "./model"
# (lihat Dockerfile: COPY PATH_MODEL ./model)
MODEL_PATH = "./model"

app = FastAPI(title="MLOps Training - Model Serving")
model = mlflow.sklearn.load_model(MODEL_PATH)


class PredictRequest(BaseModel):
    age: int
    income: float
    lama_bekerja_tahun: float = 0
    skor_kredit_internal: int = 650
    jumlah_pinjaman_aktif: int = 0
    jumlah_tanggungan: int = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    import pandas as pd
    X = pd.DataFrame([req.dict()])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {"prediction": int(pred), "probability": round(float(proba), 4)}
