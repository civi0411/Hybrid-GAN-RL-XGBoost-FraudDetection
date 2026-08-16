"""
src/evaluation/visualizer.py
===============================
Toàn bộ các hàm vẽ biểu đồ phục vụ đánh giá/so sánh mô hình. Mỗi hàm nhận
dữ liệu đã tính sẵn (không tự train/predict trong này) và trả về đường dẫn
file đã lưu trong artifacts/plots/, để dùng lại trong report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, plots_dir: str, filename: str) -> str:
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    path = Path(plots_dir) / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(path)


def plot_pr_curve(
    y_true: np.ndarray, scores_by_model: Dict[str, np.ndarray], plots_dir: str, filename: str = "pr_curve.png"
) -> str:
    """PR curve cho nhiều model chồng lên nhau — ưu tiên hơn ROC vì data mất cân bằng."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, y_scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax.plot(recall, precision, label=model_name, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (metric ưu tiên cho dữ liệu mất cân bằng)")
    ax.legend(loc="best")
    return _save(fig, plots_dir, filename)


def plot_roc_curve(
    y_true: np.ndarray, scores_by_model: Dict[str, np.ndarray], plots_dir: str, filename: str = "roc_curve.png"
) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, y_scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, label=model_name, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="best")
    return _save(fig, plots_dir, filename)


def plot_confusion_matrix(
    cm: Dict[str, int], plots_dir: str, filename: str = "confusion_matrix.png", model_name: str = "model"
) -> str:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Pred: Not Fraud", "Pred: Fraud"],
        yticklabels=["True: Not Fraud", "True: Fraud"], ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    return _save(fig, plots_dir, filename)


def plot_feature_importance(
    importance_df: pd.DataFrame, plots_dir: str, filename: str = "feature_importance.png",
    value_col: str = "mean_abs_shap", label_col: str = "feature",
) -> str:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importance_df))))
    df_sorted = importance_df.sort_values(value_col)
    ax.barh(df_sorted[label_col], df_sorted[value_col], color="#4C72B0")
    ax.set_xlabel(value_col)
    ax.set_title("Top features quan trọng nhất (SHAP)")
    return _save(fig, plots_dir, filename)


def plot_ablation_comparison(
    ablation_df: pd.DataFrame, plots_dir: str, filename: str = "ablation_comparison.png",
    metrics: Optional[List[str]] = None,
) -> str:
    """Bar chart so sánh nhiều metric giữa các kịch bản ablation (xgboost_only, smote, gan_only, gan_rl)."""
    metrics = metrics or ["auc_pr", "auc_roc", "f1", "recall"]
    plot_df = ablation_df.set_index("scenario")[metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("So sánh các kịch bản xử lý mất cân bằng dữ liệu")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20)
    return _save(fig, plots_dir, filename)


def plot_gan_training_curves(
    history: Dict[str, List[float]], plots_dir: str, filename: str = "gan_training_curves.png"
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss_d"], label="Critic loss (D)")
    ax.plot(history["loss_g"], label="Generator loss (G)")
    ax.plot(history["gp"], label="Gradient penalty", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("WGAN-GP Training Curves")
    ax.legend()
    return _save(fig, plots_dir, filename)


def plot_rl_reward_curve(
    rewards: List[float], plots_dir: str, filename: str = "rl_reward_curve.png", window: int = 20
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rewards, alpha=0.3, color="tab:blue", label="reward / episode")
    if len(rewards) >= window:
        smoothed = pd.Series(rewards).rolling(window).mean()
        ax.plot(smoothed, color="tab:blue", linewidth=2, label=f"moving avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (delta AUC-PR)")
    ax.set_title("RL Agent — Reward theo Episode")
    ax.legend()
    return _save(fig, plots_dir, filename)


def plot_real_vs_synthetic_distribution(
    real_numeric: np.ndarray,
    synthetic_numeric: np.ndarray,
    plots_dir: str,
    filename: str = "real_vs_synthetic_pca.png",
) -> str:
    """
    Chiếu dữ liệu numeric thật (fraud) và synthetic (fraud) xuống 2D bằng PCA
    để đánh giá trực quan mức độ "giống thật" của dữ liệu GAN sinh ra — cách
    làm phổ biến trong các paper CTGAN/CTAB-GAN+ để kiểm tra fidelity.
    """
    combined = np.vstack([real_numeric, synthetic_numeric])
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(combined)
    n_real = len(real_numeric)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(projected[:n_real, 0], projected[:n_real, 1], alpha=0.5, label="Real fraud", s=15)
    ax.scatter(projected[n_real:, 0], projected[n_real:, 1], alpha=0.5, label="Synthetic (GAN)", s=15)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title("Phân bố dữ liệu thật vs synthetic (PCA 2D)")
    ax.legend()
    return _save(fig, plots_dir, filename)
