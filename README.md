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
