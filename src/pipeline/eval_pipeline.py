"""
src/pipeline/eval_pipeline.py
================================
Load model + preprocessor đã lưu, đánh giá đầy đủ trên tập test/hold-out,
sinh toàn bộ plots + report (metrics, SHAP, PR/ROC curve, confusion matrix).

Chạy qua: python scripts/run_evaluation.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import xgboost as xgb

from src.data.loader import load_raw_data
from src.data.preprocessor import TabularPreprocessor
from src.data.splitter import split_data
from src.evaluation.metrics import compute_classification_metrics, format_metrics_report
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)
from src.models.xgb.explainer import XGBExplainer
from src.models.xgb.predictor import XGBPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    models_dir = config["paths"]["models_dir"]
    plots_dir = config["paths"]["plots_dir"]
    reports_dir = config["paths"]["reports_dir"]

    preprocessor = TabularPreprocessor.load(str(Path(models_dir) / "preprocessor.joblib"))
    predictor = XGBPredictor(str(Path(models_dir) / "xgboost_final.json"))

    df = load_raw_data(config)
    _, _, test_df = split_data(df, config)
    X_test, y_test = preprocessor.transform_for_classifier(test_df)

    y_scores = predictor.predict_proba(X_test)
    metrics = compute_classification_metrics(y_test.to_numpy(), y_scores)
    logger.info("\n" + format_metrics_report(metrics, model_name="xgboost_final"))

    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(reports_dir) / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_pr_curve(y_test.to_numpy(), {"xgboost_final": y_scores}, plots_dir)
    plot_roc_curve(y_test.to_numpy(), {"xgboost_final": y_scores}, plots_dir)
    plot_confusion_matrix(metrics["confusion_matrix"], plots_dir, model_name="xgboost_final")

    explainer = XGBExplainer(predictor.model, config)
    top_features_df = explainer.top_features(X_test)
    top_features_df.to_csv(Path(reports_dir) / "top_features_shap.csv", index=False)
    plot_feature_importance(top_features_df, plots_dir)

    return {"metrics": metrics, "top_features": top_features_df}
