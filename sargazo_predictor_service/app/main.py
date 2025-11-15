import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predictor import SargazoPredictor


class SequenceRequest(BaseModel):
    sequence: List[List[float]]


app = FastAPI(title="Sargazo Predictor Service", version="0.1")

predictor: SargazoPredictor = None


@app.on_event("startup")
def load_predictor():
    global predictor
    base = os.environ.get("SARGAZO_MODEL_DIR", os.path.join("models", "coordinates"))
    predictor = SargazoPredictor(
        model_path=os.path.join(base, "sargazo_lstm_model.h5"),
        scaler_path=os.path.join(base, "sargazo_scaler.pkl"),
        config_path=os.path.join(base, "sargazo_config.json"),
    )


@app.get("/health")
def health():
    return {"status": "ok", "n_steps": predictor.N_STEPS if predictor else None}


@app.post("/predict")
def predict(req: SequenceRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    try:
        scaled = predictor.preprocess_sequence(req.sequence)
        out = predictor.predict_next_position(scaled)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
