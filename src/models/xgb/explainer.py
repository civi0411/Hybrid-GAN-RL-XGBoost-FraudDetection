"""
src/models/xgb/explainer.py
==============================
Giải thích dự đoán của XGBoost bằng SHAP (TreeExplainer — nhanh và chính
xác cho các mô hình tree-based, không cần xấp xỉ như KernelExplainer).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBExplainer:
    def __init__(self, model: xgb.XGBClassifier, config: Dict[str, Any]):
        self.model = model
        self.config = config["xgboost"]["explainability"]
        self.explainer = shap.TreeExplainer(model)

    def compute_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Trả về (shap_values, X_sample) trên 1 subsample để tính toán nhanh hơn."""
        sample_size = min(self.config["shap_sample_size"], len(X))
        X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X
        logger.info(f"Đang tính SHAP values trên {len(X_sample)} mẫu...")
        shap_values = self.explainer.shap_values(X_sample)
        return shap_values, X_sample

    def top_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Trả về DataFrame [feature, mean_abs_shap] sắp xếp giảm dần, top_k theo config."""
        shap_values, X_sample = self.compute_shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)
        top_k = self.config["top_k_features"]
        return importance_df.head(top_k).reset_index(drop=True)

    def native_feature_importance(self) -> pd.DataFrame:
        """Feature importance gốc của XGBoost (gain-based), dùng để đối chiếu với SHAP."""
        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type="gain")
        df = pd.DataFrame(
            {"feature": list(score_dict.keys()), "gain": list(score_dict.values())}
        ).sort_values("gain", ascending=False)
        return df.reset_index(drop=True)
