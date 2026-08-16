"""
src/utils/validators.py
========================
Các hàm kiểm tra dữ liệu (generic, KHÔNG gắn cụ thể với schema IEEE-CIS).
Dùng để chặn sớm các lỗi phổ biến trước khi đưa dữ liệu vào GAN/XGBoost/RL.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raise khi dữ liệu đầu vào không đạt các điều kiện tối thiểu."""


def validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Thiếu các cột bắt buộc: {missing}")


def validate_no_all_nan_columns(df: pd.DataFrame) -> List[str]:
    """Trả về danh sách cột toàn NaN (cảnh báo, không raise) để caller tự quyết định drop."""
    return [c for c in df.columns if df[c].isna().all()]


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    validate_required_columns(df, [target_column])
    unique_vals = set(pd.unique(df[target_column].dropna()))
    if not unique_vals.issubset({0, 1}):
        raise DataValidationError(
            f"Cột target '{target_column}' phải là nhị phân (0/1), nhận được: {unique_vals}"
        )


def validate_finite_numeric(df: pd.DataFrame, numeric_columns: Sequence[str]) -> List[str]:
    """Trả về danh sách cột numeric có giá trị inf (để caller xử lý/log cảnh báo)."""
    bad_cols = []
    for c in numeric_columns:
        if c in df.columns and np.isinf(df[c].to_numpy(dtype="float64", na_value=0.0)).any():
            bad_cols.append(c)
    return bad_cols


def validate_class_balance(df: pd.DataFrame, target_column: str, min_positive_count: int = 10) -> None:
    positive_count = int((df[target_column] == 1).sum())
    if positive_count < min_positive_count:
        raise DataValidationError(
            f"Số lượng mẫu positive ({positive_count}) quá ít (< {min_positive_count}) "
            "để train GAN/RL một cách có ý nghĩa."
        )
