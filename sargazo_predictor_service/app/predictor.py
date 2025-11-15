import os
import json
import pickle
from typing import List, Dict

import numpy as np

from tensorflow.keras.models import load_model


class SargazoPredictor:
    """Carga y envuelve el modelo LSTM, el scaler y la configuración.

    Uso:
      predictor = SargazoPredictor(model_path, scaler_path, config_path)
      scaled_input = predictor.preprocess_sequence(sequence)
      result = predictor.predict_next_position(scaled_input)
    """

    def __init__(self, model_path: str, scaler_path: str, config_path: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self._load_components()

    def _load_components(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.N_STEPS = int(self.config["N_STEPS"])
        self.FEATURES = list(self.config["FEATURES"])
        self.TARGETS = list(self.config["TARGETS"])
        self.ALL_COLS = list(self.config["ALL_COLS"])

        # sanity check: FEATURES must be a subset of ALL_COLS
        for f in self.FEATURES:
            if f not in self.ALL_COLS:
                raise ValueError(f"Feature '{f}' listed in FEATURES not found in ALL_COLS in config")

        # load scaler
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # load keras model
        try:
            # Try normal load (may attempt to deserialize compile/metrics)
            self.model = load_model(self.model_path)
        except Exception as e:
            # Fallback: load without compile to avoid deserialization issues for legacy H5 files
            # This is safe for inference-only usage.
            try:
                self.model = load_model(self.model_path, compile=False)
                print("Warning: model loaded with compile=False due to deserialization issue; good for inference.")
            except Exception:
                # re-raise original exception for visibility
                raise e

        # precompute indices
        self.feature_indices = [self.ALL_COLS.index(c) for c in self.FEATURES]
        self.lat_next_idx = self.ALL_COLS.index(self.TARGETS[0])
        self.lon_next_idx = self.ALL_COLS.index(self.TARGETS[1])

    def preprocess_sequence(self, new_data_sequence: List[List[float]]):
        """Scale and shape the input sequence for the model.

        The function is configuration-driven: the number and names of features
        expected are read from the `sargazo_config.json` (self.FEATURES).

        new_data_sequence: list of N_STEPS rows, each row with len(self.FEATURES) floats.

        Returns: ndarray shaped (1, N_STEPS, N_FEATURES)
        """
        arr = np.array(new_data_sequence, dtype=float)
        if arr.ndim != 2:
            raise ValueError("La secuencia debe ser una lista 2D: [ [f1,...], [f1,...], ... ]")

        if arr.shape[0] != self.N_STEPS:
            raise ValueError(f"La secuencia debe tener {self.N_STEPS} filas (pasos). Se recibieron {arr.shape[0]}")

        if arr.shape[1] != len(self.FEATURES):
            raise ValueError(
                f"Each row must have {len(self.FEATURES)} features as defined in config.FEATURES; received {arr.shape[1]}.")

        # dummy matrix con todas las columnas que espera el scaler
        dummy = np.zeros((self.N_STEPS, len(self.ALL_COLS)), dtype=float)
        # colocar las features en sus columnas correspondientes
        for i, feat in enumerate(self.FEATURES):
            col_idx = self.ALL_COLS.index(feat)
            dummy[:, col_idx] = arr[:, i]

        # transformar (scale)
        scaled_dummy = self.scaler.transform(dummy)

        # extraer solo las columnas de features en el orden correcto para el modelo
        scaled_features = scaled_dummy[:, self.feature_indices]

        return scaled_features.reshape(1, self.N_STEPS, len(self.FEATURES))

    def predict_next_position(self, scaled_input: np.ndarray) -> Dict[str, float]:
        """Recibe input escalado con forma (1,N_STEPS,N_FEATURES).

        Devuelve diccionario con latitud y longitud reales (no escaladas).
        """
        pred_scaled = self.model.predict(scaled_input)
        if pred_scaled.ndim == 2 and pred_scaled.shape[0] == 1:
            pred = pred_scaled[0]
        else:
            # soportar batchs inesperados
            pred = np.asarray(pred_scaled).reshape(-1)[: len(self.TARGETS)]

        # crear dummy para inverse_transform
        inv_dummy = np.zeros((1, len(self.ALL_COLS)), dtype=float)
        inv_dummy[0, self.lat_next_idx] = pred[0]
        inv_dummy[0, self.lon_next_idx] = pred[1]

        inv = self.scaler.inverse_transform(inv_dummy)

        lat_real = float(inv[0, self.lat_next_idx])
        lon_real = float(inv[0, self.lon_next_idx])

        return {"latitud_siguiente": lat_real, "longitud_siguiente": lon_real}
