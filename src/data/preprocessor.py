"""
src/data/preprocessor.py
=========================
Tiền xử lý dữ liệu GENERIC (tự suy ra numeric/categorical theo dtype +
ngưỡng cardinality trong config, KHÔNG hard-code tên cột của IEEE-CIS).

Cung cấp `TabularPreprocessor`, được dùng bởi:
  - src/models/xgb/*      -> transform_for_classifier()
  - src/models/gan/*      -> transform_for_gan() / inverse_transform_gan()

Thiết kế 2 kiểu representation khác nhau vì:
  - XGBoost xử lý tốt numeric thô + categorical dạng ordinal/pandas category
    (tree_method="hist" hỗ trợ categorical trực tiếp).
  - GAN (WGAN-GP) cần numeric đã chuẩn hoá về khoảng liên tục và categorical
    dạng one-hot (để sinh qua Gumbel-Softmax) — xem CTGAN/CTAB-GAN+.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TabularDataSpec:
    """Mô tả schema dữ liệu sau khi fit — dùng để build kiến trúc GAN động."""

    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    categorical_cardinalities: Dict[str, int] = field(default_factory=dict)
    target_column: str = "isFraud"

    @property
    def numeric_dim(self) -> int:
        return len(self.numeric_columns)

    @property
    def categorical_dims(self) -> List[int]:
        return [self.categorical_cardinalities[c] for c in self.categorical_columns]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "categorical_cardinalities": self.categorical_cardinalities,
            "target_column": self.target_column,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TabularDataSpec":
        return cls(**d)


class TabularPreprocessor:
    """
    Preprocessor generic dùng chung cho XGBoost và GAN.

    fit() tự động:
      - Loại các cột trong drop_columns / id_column / time_column / target_column
      - Coi 1 cột là categorical nếu: dtype object/category, HOẶC dtype numeric
        nhưng số unique <= low_cardinality_threshold
      - Impute missing (numeric: median/mean/constant; categorical: most_frequent/constant)
      - Fit StandardScaler cho numeric, LabelEncoder cho từng cột categorical
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        data_cfg = config["data"]
        self.target_column: str = config["target_column"]
        self.id_column: Optional[str] = config.get("id_column")
        self.time_column: Optional[str] = config.get("time_column")
        self.drop_columns: List[str] = list(data_cfg.get("drop_columns", []))
        self.low_cardinality_threshold: int = data_cfg.get("low_cardinality_threshold", 20)
        self.max_categorical_cardinality: int = data_cfg.get("max_categorical_cardinality", 100)
        self.numeric_strategy: str = data_cfg["missing_values"]["numeric_strategy"]
        self.categorical_strategy: str = data_cfg["missing_values"]["categorical_strategy"]
        self.numeric_fill_value = data_cfg["missing_values"].get("numeric_fill_value", -999)
        self.categorical_fill_value = data_cfg["missing_values"].get("categorical_fill_value", "missing")

        self.spec: Optional[TabularDataSpec] = None
        self._numeric_imputer: Optional[SimpleImputer] = None
        self._numeric_scaler: Optional[StandardScaler] = None
        self._categorical_imputers: Dict[str, SimpleImputer] = {}
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def _infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        exclude = set(self.drop_columns) | {self.target_column}
        if self.id_column:
            exclude.add(self.id_column)
        if self.time_column:
            exclude.add(self.time_column)

        numeric_cols, categorical_cols = [], []
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                n_unique = df[col].nunique(dropna=True)
                if n_unique <= self.low_cardinality_threshold:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                n_unique = df[col].nunique(dropna=True)
                if n_unique > self.max_categorical_cardinality:
                    logger.warning(
                        f"Cột '{col}' có cardinality {n_unique} > "
                        f"max_categorical_cardinality, vẫn giữ nhưng nên cân nhắc "
                        "hash-encoding riêng khi hoàn thiện feature engineering."
                    )
                categorical_cols.append(col)
        return numeric_cols, categorical_cols

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        numeric_cols, categorical_cols = self._infer_column_types(df)
        logger.info(
            f"Phát hiện {len(numeric_cols)} cột numeric, {len(categorical_cols)} cột categorical."
        )

        if numeric_cols:
            self._numeric_imputer = SimpleImputer(
                strategy=self.numeric_strategy
                if self.numeric_strategy != "constant"
                else "constant",
                fill_value=self.numeric_fill_value if self.numeric_strategy == "constant" else None,
            )
            numeric_imputed = self._numeric_imputer.fit_transform(df[numeric_cols])
            self._numeric_scaler = StandardScaler()
            self._numeric_scaler.fit(numeric_imputed)

        categorical_cardinalities: Dict[str, int] = {}
        for col in categorical_cols:
            imputer = SimpleImputer(
                strategy=self.categorical_strategy,
                fill_value=self.categorical_fill_value if self.categorical_strategy == "constant" else None,
            )
            values = imputer.fit_transform(df[[col]].astype(object)).ravel()
            self._categorical_imputers[col] = imputer

            encoder = LabelEncoder()
            encoder.fit(np.append(values.astype(str), "__unknown__"))
            self._label_encoders[col] = encoder
            categorical_cardinalities[col] = len(encoder.classes_)

        self.spec = TabularDataSpec(
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            categorical_cardinalities=categorical_cardinalities,
            target_column=self.target_column,
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Transform for XGBoost (numeric raw/scaled + categorical ordinal)
    # ------------------------------------------------------------------ #
    def transform_for_classifier(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        self._check_fitted()
        assert self.spec is not None

        out = pd.DataFrame(index=df.index)

        if self.spec.numeric_columns:
            numeric_imputed = self._numeric_imputer.transform(df[self.spec.numeric_columns])
            out[self.spec.numeric_columns] = numeric_imputed  # raw scale is fine for tree models

        for col in self.spec.categorical_columns:
            imputer = self._categorical_imputers[col]
            encoder = self._label_encoders[col]
            values = imputer.transform(df[[col]].astype(object)).ravel().astype(str)
            values = np.where(np.isin(values, encoder.classes_), values, "__unknown__")
            out[col] = encoder.transform(values)
            out[col] = out[col].astype("category")

        y = df[self.target_column] if self.target_column in df.columns else None
        return out, y

    # ------------------------------------------------------------------ #
    # Transform for GAN (numeric standardized + categorical one-hot)
    # ------------------------------------------------------------------ #
    def transform_for_gan(self, df: pd.DataFrame) -> Dict[str, Any]:
        self._check_fitted()
        assert self.spec is not None

        result: Dict[str, Any] = {}

        if self.spec.numeric_columns:
            numeric_imputed = self._numeric_imputer.transform(df[self.spec.numeric_columns])
            result["numeric"] = self._numeric_scaler.transform(numeric_imputed).astype("float32")
        else:
            result["numeric"] = np.zeros((len(df), 0), dtype="float32")

        onehots = []
        for col in self.spec.categorical_columns:
            imputer = self._categorical_imputers[col]
            encoder = self._label_encoders[col]
            values = imputer.transform(df[[col]].astype(object)).ravel().astype(str)
            values = np.where(np.isin(values, encoder.classes_), values, "__unknown__")
            idx = encoder.transform(values)
            n_classes = len(encoder.classes_)
            onehot = np.zeros((len(df), n_classes), dtype="float32")
            onehot[np.arange(len(df)), idx] = 1.0
            onehots.append(onehot)
        result["categorical"] = onehots

        if self.target_column in df.columns:
            result["target"] = df[self.target_column].to_numpy()
        return result

    def inverse_transform_gan(
        self, numeric_arr: np.ndarray, categorical_onehots: List[np.ndarray]
    ) -> pd.DataFrame:
        """Chuyển output GAN (numeric chuẩn hoá + one-hot) về DataFrame dễ đọc."""
        self._check_fitted()
        assert self.spec is not None

        out = pd.DataFrame(index=range(numeric_arr.shape[0]))
        if self.spec.numeric_columns:
            numeric_original = self._numeric_scaler.inverse_transform(numeric_arr)
            out[self.spec.numeric_columns] = numeric_original

        for col, onehot in zip(self.spec.categorical_columns, categorical_onehots):
            idx = np.argmax(onehot, axis=1)
            encoder = self._label_encoders[col]
            out[col] = encoder.inverse_transform(idx)
        return out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_numeric_array(self, X: pd.DataFrame) -> np.ndarray:
        """
        Chuyển 1 DataFrame ở dạng classifier-ready (output của
        transform_for_classifier, categorical là pandas 'category' dtype)
        thành ma trận float thuần tuý — dùng làm observation cho RL agent.
        """
        arr = np.zeros((len(X), len(X.columns)), dtype="float32")
        for i, col in enumerate(X.columns):
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                arr[:, i] = X[col].cat.codes.to_numpy(dtype="float32")
            else:
                arr[:, i] = X[col].to_numpy(dtype="float32")
        return arr

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Đã lưu preprocessor tại: {path}")

    @staticmethod
    def load(path: str) -> "TabularPreprocessor":
        return joblib.load(path)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TabularPreprocessor chưa được fit(). Gọi .fit(df) trước.")
