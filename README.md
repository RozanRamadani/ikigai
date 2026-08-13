# MindRest AI Service

AI Microservice untuk **MindRest** yang digunakan untuk melakukan klasifikasi teks berdasarkan konsep **Ikigai**.

Service ini menerima teks dari Backend MindRest, kemudian melakukan klasifikasi menggunakan model Machine Learning berbasis **TF-IDF + Logistic Regression** untuk menghasilkan prediksi pada empat aspek Ikigai:

- Love
- Good At
- World Needs
- Paid For

## Tech Stack

- Python 3
- FastAPI
- Uvicorn
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- Pydantic

## Architecture

```text
Frontend
    |
    v
Backend MindRest
    |
    | POST /predict
    v
+----------------------+
|    MindRest AI API   |
|                      |
|  TF-IDF Vectorizer   |
|          |           |
|          v           |
| Logistic Regression  |
|          |           |
|          v           |
| Threshold Analysis   |
+----------------------+
    |
    v
Prediction Result
    |
    v
Backend MindRest
```

## Project Structure

```text
ikigai/
│
├── api/
│   ├── main.py
│   └── test_api.py
│
├── ikigai_final_package.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

> `.venv/` tidak disertakan dalam repository karena merupakan environment lokal.

## Machine Learning Model

Model menggunakan:

### TF-IDF

TF-IDF digunakan untuk mengubah teks menjadi representasi numerik berdasarkan tingkat kepentingan kata dalam dataset.

### Logistic Regression

Model Logistic Regression digunakan untuk melakukan klasifikasi multi-label terhadap empat aspek Ikigai.

### Classification Labels

```text
love
good_at
world_needs
paid_for
```

### Optimized Threshold

Threshold telah ditentukan melalui **5-Fold Out-of-Fold Threshold Tuning**.

| Label | Threshold |
|---|---:|
| Love | 0.47 |
| Good At | 0.48 |
| World Needs | 0.51 |
| Paid For | 0.46 |

Hasil evaluasi:

```text
Default Threshold Macro F1 : 0.8888
Optimal Threshold Macro F1 : 0.8958
Improvement                 : +0.0070
```

## API

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "ok",
  "service": "MindRest AI API"
}
```

### Prediction

```http
POST /predict
```

Request:

```json
{
  "text": "Saya suka membuat desain UI, cukup mahir menggunakannya, dan ingin menjadikannya pekerjaan profesional."
}
```

Response:

```json
{
  "text": "Saya suka membuat desain UI, cukup mahir menggunakannya, dan ingin menjadikannya pekerjaan profesional.",
  "predictions": {
    "love": {
      "score": 0.7891,
      "threshold": 0.47,
      "matched": true
    },
    "good_at": {
      "score": 0.7243,
      "threshold": 0.48,
      "matched": true
    },
    "world_needs": {
      "score": 0.292,
      "threshold": 0.51,
      "matched": false
    },
    "paid_for": {
      "score": 0.6194,
      "threshold": 0.46,
      "matched": true
    }
  }
}
```

## Running Locally

### 1. Clone Repository

```bash
git clone https://github.com/RozanRamadani/ikigai.git
cd ikigai
```

### 2. Create Virtual Environment

Windows:

```powershell
python -m venv api/.venv
```

Activate:

```powershell
.\api\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run API

Masuk ke folder API:

```powershell
cd api
```

Kemudian:

```powershell
python -m uvicorn main:app --reload
```

API akan berjalan pada:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

## Testing

Automated API testing tersedia pada:

```text
api/test_api.py
```

Jalankan:

```powershell
python test_api.py
```

Current test coverage:

```text
[PASS] Normal prediction
[PASS] Multiple aspects
[PASS] Short text
[PASS] Empty text
[PASS] Missing text
[PASS] Long text

RESULT: 6/6 TEST PASSED
```

## Integration with MindRest

AI Service ini dirancang untuk diintegrasikan dengan Backend MindRest menggunakan HTTP REST API.

Flow utama:

```text
User
  |
  v
Frontend
  |
  | Journal / Text Input
  v
MindRest Backend
  |
  | POST /predict
  v
MindRest AI Service
  |
  | Classification
  v
Prediction Result
  |
  v
MindRest Backend
  |
  v
Frontend
```

Backend bertanggung jawab sebagai penghubung antara frontend dan AI service.

AI service bertanggung jawab untuk:

1. Menerima teks.
2. Melakukan preprocessing melalui TF-IDF vectorizer.
3. Melakukan prediksi menggunakan Machine Learning model.
4. Menerapkan threshold per label.
5. Mengembalikan hasil klasifikasi kepada Backend.

## Model Evaluation

Model final telah melalui:

- Dataset preparation
- TF-IDF feature extraction
- Logistic Regression training
- 5-Fold Cross Validation
- Threshold Analysis
- Out-of-Fold Threshold Tuning
- Final Model Packaging
- API Testing

Final OOF evaluation:

```text
Precision : 0.9071
Recall    : 0.8867
Macro F1  : 0.8958
```

## Development Status

```text
Machine Learning Model       ✅
Threshold Optimization      ✅
Final Model Package          ✅
FastAPI Service              ✅
Prediction Endpoint          ✅
Swagger Documentation        ✅
Automated API Testing        ✅ 6/6
Backend Integration          ⏳
Production Deployment       ⏳
```

## Project

**MindRest AI Service**

Part of the MindRest application ecosystem.

Developed as an AI/Machine Learning service for the Ikigai classification feature.
