from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("random_forest_model.bin")
scaler = joblib.load("scaler_pca.bin")
pca = joblib.load("pca.bin")

class AirQualityInput(BaseModel):
    PM25: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    O3: float
    Temperature: float
    Humidity: int
    WindSpeed: float

@app.post("/predict")
def predict(input_data: AirQualityInput):
    try:
        print("Datos recibidos:", input_data)

        row = np.array([
            input_data.PM25,
            input_data.PM10,
            input_data.NO2,
            input_data.SO2,
            input_data.CO,
            input_data.O3,
            input_data.Temperature,
            input_data.Humidity,
            input_data.WindSpeed
        ]).reshape(1, -1)

        X_scaled = scaler.transform(row)
        print("Datos escalados:", X_scaled)

        X_pca = pca.transform(X_scaled)
        print("Datos después de PCA:", X_pca)

        prediction = model.predict(X_pca)[0]
        print("Predicción:", prediction)

        return {"prediction": str(prediction)}
    except Exception as e:
        print("Error en la predicción:", e)
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")
