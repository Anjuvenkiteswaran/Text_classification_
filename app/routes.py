# app/routes.py
from fastapi import APIRouter, HTTPException
from model.predict import ModelPredictor
from model.monitor import monitor_prediction_time

router = APIRouter()

# Initialize the predictor and monitor
predictor = ModelPredictor("model/svm_model.pkl")
monitor = monitor_prediction_time()

@router.post("/predict/")
@monitor  # Assuming this is a decorator
def predict(text: str):
    try:
        result = predictor.predict(text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
