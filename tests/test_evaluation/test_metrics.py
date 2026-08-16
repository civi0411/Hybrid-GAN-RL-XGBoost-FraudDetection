"""
tests/test_evaluation/test_metrics.py
========================================
Test cơ bản cho src/evaluation/metrics.py — đảm bảo các metric quan trọng
(đặc biệt AUC-PR, metric chính của toàn dự án) được tính đúng trên các
trường hợp biên (perfect classifier, random classifier).
"""
import numpy as np
import pytest

from src.evaluation.metrics import compute_classification_metrics, recall_at_fpr


def test_perfect_classifier_scores_are_maximal():
    y_true = np.array([0, 0, 0, 1, 1])
    y_scores = np.array([0.0, 0.1, 0.05, 0.9, 0.95])  # tách biệt hoàn hảo

    metrics = compute_classification_metrics(y_true, y_scores)

    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["auc_pr"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_confusion_matrix_counts_match_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.2, 0.6, 0.3, 0.8])  # 1 FP, 1 FN với threshold=0.5

    metrics = compute_classification_metrics(y_true, y_scores, threshold=0.5)
    cm = metrics["confusion_matrix"]

    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(y_true)
    assert cm["fp"] == 1
    assert cm["fn"] == 1


def test_recall_at_fpr_monotonic_with_looser_threshold():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_scores = np.clip(y_true * 0.6 + rng.normal(0, 0.3, size=500), 0, 1)

    recall_strict = recall_at_fpr(y_true, y_scores, target_fpr=1e-4)
    recall_loose = recall_at_fpr(y_true, y_scores, target_fpr=0.5)

    # FPR cho phép càng lớn thì recall đạt được không thể thấp hơn
    assert recall_loose >= recall_strict


def test_metrics_contain_required_keys_for_reporting():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.6, 0.4, 0.7, 0.3, 0.55])

    metrics = compute_classification_metrics(y_true, y_scores, fpr_targets=[1e-3, 1e-4])

    for key in ["auc_roc", "auc_pr", "f1", "precision", "recall", "mcc", "confusion_matrix"]:
        assert key in metrics
    assert "recall_at_fpr_1e-03" in metrics
    assert "recall_at_fpr_1e-04" in metrics
