import os
import json
import pickle
from typing import List, Dict

import numpy as np


class BiomasaPredictor:
    """Carga y envuelve el modelo XGBoost para predicción de biomasa de sargazo.

    Uso:
      predictor = BiomasaPredictor(model_path, config_path)
      result = predictor.predict(features_dict)
    """

    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self._load_components()

    def _load_components(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.features = list(self.config["features"])
        self.target = self.config.get("target", "sargassum_biomass")
        self.model_type = self.config.get("model_type", "XGBRegressor")

        # load model (joblib pickle)
        import joblib
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def predict(self, features: Dict[str, float]) -> float:
        """Predict sargassum biomass given input features.

        features: dict with keys matching self.features
          Example: {
            "lat": 21.5,
            "lon": -87.2,
            "avg_sea_surface_temperature": 28.5,
            "avg_ocean_current_velocity": 0.35,
            "avg_ocean_current_direction": 180.0
          }

        Returns: predicted biomass as float
        """
        # validate input
        missing = [f for f in self.features if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # build input array in correct order
        X = np.array([[features[f] for f in self.features]], dtype=float)

        # predict
        pred = self.model.predict(X)
        biomass = float(pred[0])

        return biomass
