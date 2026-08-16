"""
src/evaluation/metrics.py
============================
Bộ metrics cho bài toán fraud detection (mất cân bằng nặng).

Cơ sở lựa chọn (research-grounded — xem thêm README.md phần "Research"):
  - AUC-PR (average precision) được ưu tiên làm metric CHÍNH thay vì
    AUC-ROC, vì AUC-ROC có thể "quá lạc quan" khi lớp dương chỉ chiếm
    một phần rất nhỏ (Saito & Rehmsmeier, 2015; nhiều bài khảo sát
    2024-2025 về imbalanced classification đều đồng thuận: "PR-AUC is
    the gold standard for rare-event prediction").
  - MCC (Matthews Correlation Coefficient) được thêm vào vì vẫn có ý
    nghĩa thống kê ngay cả khi mất cân bằng cực đoan (theo SAGE, 2026).
  - Recall@FPR cố định (vd FPR=1e-3) mô phỏng ràng buộc vận hành thực tế:
    đội ngũ fraud review chỉ chấp nhận một tỉ lệ false-positive nhất định.
  - F1-score làm điểm cân bằng dễ diễn giải giữa precision/recall.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float) -> float:
    """Recall đạt được khi giới hạn FPR <= target_fpr (nội suy tuyến tính trên ROC curve)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    fpr_targets: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Tính đầy đủ bộ metrics cho 1 tập dự đoán.

    Args:
        y_true: nhãn thật (0/1)
        y_scores: xác suất dự đoán (không phải nhãn cứng)
        threshold: ngưỡng để tính precision/recall/f1/confusion matrix
        fpr_targets: danh sách FPR cố định để tính Recall@FPR (mặc định [1e-3, 1e-4])
    """
    fpr_targets = fpr_targets or [1e-3, 1e-4]
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: Dict[str, Any] = {
        "auc_roc": float(roc_auc_score(y_true, y_scores)),
        "auc_pr": float(average_precision_score(y_true, y_scores)),   # metric chính
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    for target_fpr in fpr_targets:
        key = f"recall_at_fpr_{target_fpr:.0e}"
        metrics[key] = recall_at_fpr(y_true, y_scores, target_fpr)

    return metrics


def format_metrics_report(metrics: Dict[str, Any], model_name: str = "model") -> str:
    """In gọn các metrics chính dưới dạng text, phục vụ log nhanh."""
    lines = [f"=== Kết quả đánh giá: {model_name} ==="]
    for key in ["auc_pr", "auc_roc", "f1", "precision", "recall", "mcc"]:
        if key in metrics:
            lines.append(f"  {key:>12s}: {metrics[key]:.4f}")
    for key, value in metrics.items():
        if key.startswith("recall_at_fpr"):
            lines.append(f"  {key:>20s}: {value:.4f}")
    cm = metrics.get("confusion_matrix")
    if cm:
        lines.append(f"  confusion_matrix : TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
    return "\n".join(lines)
