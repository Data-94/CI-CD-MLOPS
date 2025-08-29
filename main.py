from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"mesaj": "IsolationForest ML model API'sine hoşgeldiniz!"}

# 🎯 Input Schema
class MLModelSchema(BaseModel):
    u_q: float
    coolant: float
    stator_winding: float
    u_d: float
    stator_tooth: float
    motor_speed: float
    i_d: float
    i_q: float
    pm: float
    stator_yoke: float

# 🎯 JSON -> DataFrame
def _to_df(predict_values: MLModelSchema) -> pd.DataFrame:
    data = predict_values.model_dump() if hasattr(predict_values, "model_dump") else predict_values.dict()
    return pd.DataFrame([data])

# 🎯 
def _map_to_label(val) -> int:
    
    return 0 if val == 1 else 1  

# 🎯 IsolationForest modeli yükle
def _load_if_model():
    with open("IsolationForest_model.pkl", "rb") as f:
        return pickle.load(f)

# 🎯 Scaler yükle
def _load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

# 🎯 IsolationForest Prediction
@app.post("/predict/isolationforest/")
def isolationforest_predict(predict_values: MLModelSchema):
    model = _load_if_model()
    scaler = _load_scaler()
    df = _to_df(predict_values)
    df_scaled = scaler.transform(df)  
    pred = model.predict(df_scaled)
    return {"Predict": _map_to_label(pred[0])}
