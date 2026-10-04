from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.h5"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl"


def _parse_numbers(raw: str | Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert user input into a flat length-63 landmark vector."""
    if isinstance(raw, str):
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw)
        if not numbers:
            raise ValueError("No numeric values were found in the input.")
        values = [float(v) for v in numbers]
    else:
        values = np.asarray(raw, dtype=np.float32).reshape(-1).tolist()

    arr = np.asarray(values, dtype=np.float32)

    # Accept flattened vectors and common 21x3 landmark formats.
    if arr.size == 63:
        return arr.reshape(1, 63)
    if arr.size == 189:
        return arr.reshape(1, 189)

    # Some pasted examples may contain a larger number of values; keep the first 63
    # so the model can still run instead of crashing at the UI boundary.
    if arr.size > 63 and arr.size % 3 == 0:
        trimmed = arr[:63]
        return trimmed.reshape(1, 63)

    raise ValueError(
        f"Expected 63 landmark values (or 21x3 coordinates), but received {arr.size}."
    )


def _load_artifacts():
    """Load the saved model, scaler, and label encoder."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, encoder


def predict_gesture(landmarks: str | Sequence[float] | np.ndarray) -> dict:
    """Predict a single gesture from a flattened or nested landmark array."""
    values = _parse_numbers(landmarks)
    model, scaler, encoder = _load_artifacts()

    X = values.reshape(1, -1)
    if X.shape[1] == 189:
        X = X.reshape(1, -1)

    X_scaled = scaler.transform(X)
    probabilities = model.predict(X_scaled, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])

    dist = {
        str(label): float(prob)
        for label, prob in zip(encoder.classes_, probabilities)
    }

    ranked = dict(sorted(dist.items(), key=lambda item: item[1], reverse=True))
    return {
        "predicted_gesture": predicted_label,
        "confidence": confidence,
        "probabilities": ranked,
    }


def predict_from_csv_text(csv_text: str) -> dict:
    """Convenience wrapper that accepts comma, space, or newline-separated values."""
    return predict_gesture(csv_text)
