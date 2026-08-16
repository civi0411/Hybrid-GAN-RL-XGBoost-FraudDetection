"""
src/models/xgb/trainer.py
============================
Huấn luyện XGBoost — bộ phân loại chính của toàn hệ thống. Xem lý do lựa
chọn XGBoost (thay vì thuần deep learning) trong config/xgboost.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["xgboost"]
        self.model: Optional[xgb.XGBClassifier] = None

    def _compute_scale_pos_weight(self, y: pd.Series) -> float:
        imbalance_cfg = self.config["imbalance"]
        setting = imbalance_cfg["scale_pos_weight"]
        if setting == "auto":
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            weight = n_neg / max(n_pos, 1)
            logger.info(f"scale_pos_weight tự tính = {weight:.3f} (neg={n_neg}, pos={n_pos})")
            return weight
        return float(setting)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        params = dict(self.config["params"])
        params["scale_pos_weight"] = self._compute_scale_pos_weight(y_train)
        params["enable_categorical"] = True

        n_estimators = params.pop("n_estimators")
        train_cfg = self.config["training"]

        model = xgb.XGBClassifier(
            **params,
            n_estimators=n_estimators,
            early_stopping_rounds=train_cfg["early_stopping_rounds"],
        )

        logger.info(f"Bắt đầu train XGBoost trên {len(X_train)} mẫu (val: {len(X_val)} mẫu)...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=train_cfg["verbose_eval"],
        )
        self.model = model
        logger.info(
            f"Train xong. best_iteration={getattr(model, 'best_iteration', None)}, "
            f"best_score={getattr(model, 'best_score', None)}"
        )
        return model

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Chưa train model, không có gì để lưu.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info(f"Đã lưu XGBoost model tại: {path}")
