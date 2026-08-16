"""
src/models/rl/reward.py
==========================
Hàm tính reward cho RL agent: train một XGBoost "rút gọn" (ít cây, nông
hơn — xem rl.yaml::env.fast_xgboost_*) trên (dữ liệu train gốc + các mẫu
synthetic được agent CHỌN trong episode), rồi đánh giá trên tập validation
bằng AUC-PR (average_precision — nhất quán với metric chính trong
xgboost.yaml và evaluation/metrics.py).

Vì sao dùng XGBoost rút gọn thay vì XGBoost đầy đủ (2000 cây) làm reward?
  - Reward phải được tính lại ở MỖI episode trong quá trình train RL (có
    thể hàng trăm/nghìn lần) -> cần nhanh.
  - Baseline được cache 1 lần duy nhất (train trên dữ liệu gốc, không đổi
    trong suốt quá trình train RL) để tiết kiệm thời gian và làm điểm neo
    so sánh (delta reward) ổn định.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _train_fast_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]
) -> xgb.XGBClassifier:
    env_cfg = config["rl"]["env"]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=env_cfg["fast_xgboost_n_estimators"],
        max_depth=env_cfg["fast_xgboost_max_depth"],
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        random_state=config["project"]["seed"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_reward_metric(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any],
) -> float:
    """Train fast classifier trên (X_train, y_train), trả về metric trên (X_val, y_val)."""
    metric_name = config["rl"]["env"]["reward_metric"]
    model = _train_fast_classifier(X_train, y_train, config)
    proba = model.predict_proba(X_val)[:, 1]

    if metric_name == "average_precision":
        return float(average_precision_score(y_val, proba))
    if metric_name == "f1":
        preds = (proba >= 0.5).astype(int)
        return float(f1_score(y_val, preds))
    raise ValueError(f"reward_metric không được hỗ trợ: {metric_name}")


def compute_baseline_score(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, config: Dict[str, Any]
) -> float:
    """Điểm neo: classifier train CHỈ trên dữ liệu gốc (không có synthetic)."""
    score = evaluate_reward_metric(X_train, y_train, X_val, y_val, config)
    logger.info(f"[RL] Baseline reward metric (chỉ dữ liệu gốc) = {score:.4f}")
    return score


def compute_episode_reward(
    X_train_base: pd.DataFrame,
    y_train_base: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    selected_synthetic: pd.DataFrame,
    baseline_score: float,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Reward = score(train_gốc + mẫu đã chọn) - baseline_score.
    Trả về (reward, raw_score) để log/debug.
    """
    target_col = config["target_column"]
    if len(selected_synthetic) == 0:
        # Không chọn mẫu nào -> reward = 0 (tương đương baseline, không thưởng không phạt)
        return 0.0, baseline_score

    y_synth = selected_synthetic[target_col]
    X_synth = selected_synthetic.drop(columns=[target_col])

    X_combined = pd.concat([X_train_base, X_synth], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train_base, y_synth], axis=0, ignore_index=True)

    raw_score = evaluate_reward_metric(X_combined, y_combined, X_val, y_val, config)
    reward = raw_score - baseline_score

    if config["rl"]["env"].get("normalize_reward", True):
        reward = reward / max(abs(baseline_score), 1e-6)

    return float(reward), float(raw_score)
