from fastapi.testclient import TestClient
from main import app  

client = TestClient(app)

def test_isolationforest_model():
    payload = {
        "u_q": -0.32,
        "coolant": 70.0,
        "stator_winding": 19.0,
        "u_d": -0.30,
        "stator_tooth": 18.2,
        "motor_speed": 1600.0,
        "i_d": 0.003,
        "i_q": 0.001,
        "pm": 24.6,
        "stator_yoke": 18.33
    }
    
    # /predict/isolationforest/ endpoint'ine post isteği
    r = client.post("/predict/isolationforest/", json=payload)
    
    assert r.status_code == 200
    # "Predict" key'i olduğuna emin ol
    assert "Predict" in r.json()
    # Değerin 0(normal) veya 1(anormal) olduğuna emin ol
    assert r.json()["Predict"] in [0, 1]
