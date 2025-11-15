import os
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predictor import SargazoPredictor
from .biomasa_predictor import BiomasaPredictor


class SequenceRequest(BaseModel):
    sequence: List[List[float]]


class BiomasaRequest(BaseModel):
    lat: float
    lon: float
    avg_sea_surface_temperature: float
    avg_ocean_current_velocity: float
    avg_ocean_current_direction: float


app = FastAPI(title="Sargazo Predictor Service", version="0.1")

predictor: SargazoPredictor = None
biomasa_predictor: BiomasaPredictor = None


@app.on_event("startup")
def load_predictor():
    global predictor, biomasa_predictor
    
    # Load coordinate predictor
    base = os.environ.get("SARGAZO_MODEL_DIR", os.path.join("models", "coordinates"))
    predictor = SargazoPredictor(
        model_path=os.path.join(base, "sargazo_lstm_model.h5"),
        scaler_path=os.path.join(base, "sargazo_scaler.pkl"),
        config_path=os.path.join(base, "sargazo_config.json"),
    )
    
    # Load biomasa predictor
    biomasa_base = os.environ.get("SARGAZO_BIOMASA_MODEL_DIR", os.path.join("models", "biomasa"))
    biomasa_predictor = BiomasaPredictor(
        model_path=os.path.join(biomasa_base, "sargassum_xgb_model.pkl"),
        config_path=os.path.join(biomasa_base, "sargassum_model_config.json"),
    )


@app.get("/health")
def health():
    coordinate_status = "ready" if predictor is not None else "not loaded"
    biomasa_status = "ready" if biomasa_predictor is not None else "not loaded"
    return {
        "status": "ok",
        "coordinate_predictor": coordinate_status,
        "biomasa_predictor": biomasa_status,
        "n_steps": predictor.N_STEPS if predictor else None
    }


@app.post("/predict-coordinate")
def predict_coordinate(req: SequenceRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    try:
        scaled = predictor.preprocess_sequence(req.sequence)
        out = predictor.predict_next_position(scaled)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-biomass")
def predict_biomass(request: BiomasaRequest):
    if biomasa_predictor is None:
        raise HTTPException(status_code=503, detail="Biomasa predictor not loaded")
    try:
        features = {
            "lat": request.lat,
            "lon": request.lon,
            "avg_sea_surface_temperature": request.avg_sea_surface_temperature,
            "avg_ocean_current_velocity": request.avg_ocean_current_velocity,
            "avg_ocean_current_direction": request.avg_ocean_current_direction,
        }
        biomass = biomasa_predictor.predict(features)
        return {
            "sargassum_biomass": float(biomass)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
