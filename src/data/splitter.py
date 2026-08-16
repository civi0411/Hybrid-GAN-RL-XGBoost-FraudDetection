"""
src/data/splitter.py
=====================
Chia train/val/test.

Cơ sở lựa chọn: Với dữ liệu giao dịch có yếu tố thời gian (như IEEE-CIS,
TransactionDT), nhiều solution top của cuộc thi (vd. VedangW/ieee-cis-fraud-
detection) đều nhấn mạnh KHÔNG dùng random k-fold vì sẽ rò rỉ thông tin
tương lai vào quá khứ (leakage) và không phản ánh đúng bối cảnh production
(model luôn dự đoán cho giao dịch tương lai so với lúc train). Do đó:

  - Nếu config['time_column'] tồn tại trong df -> sort theo thời gian rồi
    cắt theo tỉ lệ (train = phần sớm nhất, val/test = phần muộn hơn).
  - Nếu không có cột thời gian -> fallback random stratified split.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_column = config.get("time_column")
    split_cfg = config["split"]
    train_ratio = split_cfg["train_ratio"]
    val_ratio = split_cfg["val_ratio"]

    if time_column and time_column in df.columns:
        logger.info(f"Chia dữ liệu theo thời gian dựa trên cột '{time_column}' (expanding split).")
        return _time_based_split(df, time_column, train_ratio, val_ratio)

    logger.info("Không có cột thời gian hợp lệ -> dùng random stratified split.")
    return _random_split(df, config)


def _time_based_split(
    df: pd.DataFrame, time_column: str, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    logger.info(f"Time-based split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def _random_split(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = config["split"]
    target_column = config["target_column"]
    seed = config["project"]["seed"]
    stratify_col = df[target_column] if split_cfg.get("stratify", True) else None

    train_ratio = split_cfg["train_ratio"]
    val_ratio = split_cfg["val_ratio"]
    test_ratio = split_cfg["test_ratio"]
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Tỉ lệ split phải cộng lại bằng 1.0"

    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, random_state=seed, stratify=stratify_col
    )
    remaining_ratio = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df[target_column] if split_cfg.get("stratify", True) else None
    val_df, test_df = train_test_split(
        temp_df, train_size=remaining_ratio, random_state=seed, stratify=stratify_temp
    )

    logger.info(f"Random split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def get_class_counts(df: pd.DataFrame, target_column: str) -> Dict[str, int]:
    counts = df[target_column].value_counts().to_dict()
    return {"negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0))}
