from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo y transformadores
model = joblib.load("random_forest_model.bin")     # Modelo entrenado
scaler = joblib.load("scaler_pca.bin")             # Escalador
pca = joblib.load("pca.bin")                       # Reducción PCA

class AirQualityInput(BaseModel):
    PM25: float
    PM10: float
    NO2: float
    O3: float
    SO2: float
    CO: float

@app.post("/predict")
def predict(input_data: AirQualityInput):
    try:
        print("Datos recibidos:", input_data)
        # Crear el vector de entrada
        row = np.array([
            input_data.PM25,
            input_data.PM10,
            input_data.NO2,
            input_data.O3,
            input_data.SO2,
            input_data.CO
        ]).reshape(1, -1)

        # Escalar
        X_scaled = scaler.transform(row)
        print("Datos escalados:", X_scaled)

        # Reducir dimensiones con PCA
        X_pca = pca.transform(X_scaled)
        print("Datos después de PCA:", X_pca)

        # Predecir
        prediction = model.predict(X_pca)[0]
        print("Predicción:", prediction)

        return {"prediction": str(prediction)}
    except Exception as e:
        print("Error en la predicción:", e)
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")
