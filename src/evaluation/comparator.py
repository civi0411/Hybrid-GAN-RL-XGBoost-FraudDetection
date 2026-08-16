"""
src/evaluation/comparator.py
===============================
Chạy ablation study: so sánh nhiều "kịch bản" xử lý mất cân bằng dữ liệu
trên CÙNG một tập test, dùng CÙNG một cấu hình XGBoost, để kết quả so
sánh công bằng.

Các kịch bản chuẩn được đề xuất (dựa trên các bài báo đã khảo sát —
xem README.md phần Research):
  1. "xgboost_only"      : XGBoost thuần trên dữ liệu gốc mất cân bằng
                            (chỉ dùng scale_pos_weight) — baseline chuẩn
                            của hầu hết solution IEEE-CIS trên Kaggle.
  2. "smote"              : Oversampling truyền thống (SMOTE) — baseline
                            phổ biến nhất trong các paper GAN-oversampling
                            để so sánh (Jiang et al. 2023; CTGAN-ENN 2024).
  3. "gan_only"           : Toàn bộ candidate pool GAN sinh ra được thêm
                            vào, KHÔNG qua RL lọc — tương đương "GAN
                            Augmentation" baseline trong Ye et al. 2020.
  4. "gan_rl" (full model): Chỉ thêm các mẫu GAN đã được RL agent chọn lọc.

So sánh 1 vs 2 cho biết GAN có tốt hơn SMOTE không (câu hỏi cốt lõi của
rất nhiều paper oversampling). So sánh 3 vs 4 cho biết RL-filtering có
thực sự cải thiện so với thêm toàn bộ dữ liệu GAN một cách "mù quáng"
không (đây là ablation quan trọng nhất, trực tiếp kiểm chứng đóng góp
của thành phần RL trong kiến trúc).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_classification_metrics
from src.models.xgb.trainer import XGBTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_ablation_study(
    scenarios: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Args:
        scenarios: dict {tên_kịch_bản: (X_train, y_train)}. Tất cả X_train phải
            có cùng bộ cột (đã qua transform_for_classifier với cùng preprocessor).
        X_val/y_val: dùng cho early stopping khi train XGBoost.
        X_test/y_test: tập test CHUNG, cố định, dùng để đánh giá công bằng.
    Returns:
        DataFrame so sánh, mỗi dòng là 1 kịch bản, các cột là các metrics.
    """
    rows = []
    for scenario_name, (X_train, y_train) in scenarios.items():
        logger.info(f"=== Ablation: đang train kịch bản '{scenario_name}' (n={len(X_train)}) ===")
        trainer = XGBTrainer(config)
        model = trainer.train(X_train, y_train, X_val, y_val)

        y_scores = model.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(y_test.to_numpy(), y_scores)
        metrics["scenario"] = scenario_name
        metrics["n_train_samples"] = len(X_train)
        metrics["n_positive_train"] = int((y_train == 1).sum())
        rows.append(metrics)

    df = pd.DataFrame(rows)
    ordered_cols = [
        "scenario", "n_train_samples", "n_positive_train",
        "auc_pr", "auc_roc", "f1", "precision", "recall", "mcc",
    ]
    ordered_cols += [c for c in df.columns if c.startswith("recall_at_fpr")]
    remaining = [c for c in df.columns if c not in ordered_cols]
    df = df[[c for c in ordered_cols if c in df.columns] + remaining]

    logger.info("=== Kết quả ablation study ===\n" + df.to_string(index=False))
    return df


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Ước lượng khoảng tin cậy (CI) cho 1 metric bằng bootstrap resampling —
    hữu ích khi so sánh 2 kịch bản có điểm số gần nhau, để biết chênh lệch
    có ý nghĩa thống kê hay chỉ do nhiễu của tập test.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_scores[idx]))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "mean": float(scores.mean()),
        "lower": float(np.quantile(scores, alpha)),
        "upper": float(np.quantile(scores, 1 - alpha)),
    }
