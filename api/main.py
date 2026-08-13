from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "ikigai_final_package.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    package = joblib.load(MODEL_PATH)

    model = package["model"]
    vectorizer = package["vectorizer"]
    thresholds = package["thresholds"]
    labels = package["labels"]

except FileNotFoundError:
    raise RuntimeError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

except KeyError as e:
    raise RuntimeError(
        f"Key '{e.args[0]}' tidak ditemukan di model package."
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="MindRest AI API",
    description=(
        "AI service for Ikigai classification using "
        "TF-IDF and One-vs-Rest Logistic Regression."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Teks yang akan dianalisis oleh model Ikigai.",
        examples=[
            "Saya suka membuat desain UI dan ingin menjadikannya pekerjaan."
        ],
    )


class PredictionResult(BaseModel):
    score: float
    threshold: float
    matched: bool


class PredictionResponse(BaseModel):
    text: str
    predictions: dict[str, PredictionResult]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/",
    summary="Health Check",
    description="Memeriksa apakah MindRest AI API sedang berjalan.",
)
def root():
    return {
        "status": "ok",
        "service": "MindRest AI API",
        "version": "1.0.0",
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Ikigai",
    description=(
        "Menganalisis teks pengguna dan menentukan kecocokan "
        "terhadap empat dimensi Ikigai: Love, Good At, "
        "World Needs, dan Paid For."
    ),
)
def predict(request: PredictionRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text tidak boleh kosong.",
        )

    try:
        # TF-IDF transformation
        X = vectorizer.transform([text])

        # Model prediction
        probabilities = model.predict_proba(X)[0]

        predictions = {}

        for label, probability in zip(labels, probabilities):

            threshold = float(thresholds[label])
            score = float(probability)

            predictions[label] = {
                "score": round(score, 4),
                "threshold": threshold,
                "matched": bool(score >= threshold),
            }

        return {
            "text": text,
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction gagal: {str(e)}",
        )