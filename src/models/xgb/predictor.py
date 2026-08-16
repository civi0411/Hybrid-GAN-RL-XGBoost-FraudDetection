"""
src/models/xgb/predictor.py
==============================
Load XGBoost model đã train và chạy inference trên dữ liệu mới.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBPredictor:
    def __init__(self, model_path: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        logger.info(f"Đã load XGBoost model từ: {model_path}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Trả về xác suất fraud (lớp 1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def batch_predict(self, X: pd.DataFrame, batch_size: int = 100_000) -> np.ndarray:
        """Predict theo batch, hữu ích khi dữ liệu inference rất lớn."""
        outputs = []
        for start in range(0, len(X), batch_size):
            chunk = X.iloc[start : start + batch_size]
            outputs.append(self.predict_proba(chunk))
        return np.concatenate(outputs) if outputs else np.array([])
