"""
src/pipeline/predict_pipeline.py
===================================
Pipeline dự đoán cho dữ liệu mới (dùng bởi api/ hoặc batch scoring script).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.preprocessor import TabularPreprocessor
from src.models.xgb.predictor import XGBPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictPipeline:
    def __init__(self, config: Dict[str, Any]):
        models_dir = config["paths"]["models_dir"]
        self.preprocessor = TabularPreprocessor.load(str(Path(models_dir) / "preprocessor.joblib"))
        self.predictor = XGBPredictor(str(Path(models_dir) / "xgboost_final.json"))

    def predict(self, raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        X, _ = self.preprocessor.transform_for_classifier(raw_df)
        proba = self.predictor.predict_proba(X)
        preds = (proba >= threshold).astype(int)

        result = raw_df.copy()
        result["fraud_probability"] = proba
        result["fraud_prediction"] = preds
        return result
