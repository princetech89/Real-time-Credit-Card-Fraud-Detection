from fastapi import FastAPI, Body
import joblib
import pandas as pd
import os

app = FastAPI(title="Credit Default Prediction API")

# Load trained pipeline
pipeline = joblib.load("credit_default_pipeline.pkl")


@app.get("/")
def home():
    return {"message": "Credit Default Prediction API is running"}


@app.post("/predict")
def predict(features: dict = Body(...)):
    df = pd.DataFrame([features])
    prob = pipeline.predict_proba(df)[0][1]

    return {
        "default_probability": float(prob),
        "prediction": int(prob >= 0.5)
    }


# 🔹 REQUIRED FOR RENDER / FREE HOSTING
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
