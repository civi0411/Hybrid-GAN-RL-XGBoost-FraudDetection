"""
scripts/run_ablation_study.py
================================
Entry point CLI: chạy lại ablation study (xgboost_only / smote / gan_only /
gan_rl) dùng CÁC ARTIFACT ĐÃ LƯU từ lần train trước (GAN generator, RL agent
đã chọn mẫu) — không cần train lại GAN/RL từ đầu. Hữu ích khi bạn muốn thử
nghiệm ablation nhiều lần với các set hyperparameter XGBoost khác nhau mà
không tốn thời gian train lại GAN + RL.

Sử dụng:
    python scripts/run_ablation_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_raw_data
from src.data.preprocessor import TabularPreprocessor
from src.data.splitter import split_data
from src.evaluation.comparator import run_ablation_study
from src.evaluation.visualizer import plot_ablation_comparison
from src.pipeline.train_pipeline import make_smote_scenario
from src.utils.config_loader import load_all_configs
from src.utils.logger import get_logger


def main() -> None:
    config = load_all_configs()
    logger = get_logger(__name__, config=config)

    models_dir = config["paths"]["models_dir"]
    plots_dir = config["paths"]["plots_dir"]
    reports_dir = config["paths"]["reports_dir"]
    target_col = config["target_column"]

    preprocessor = TabularPreprocessor.load(str(Path(models_dir) / "preprocessor.joblib"))

    df = load_raw_data(config)
    train_df, val_df, test_df = split_data(df, config)
    X_train, y_train = preprocessor.transform_for_classifier(train_df)
    X_val, y_val = preprocessor.transform_for_classifier(val_df)
    X_test, y_test = preprocessor.transform_for_classifier(test_df)

    scenarios = {"xgboost_only": (X_train, y_train)}

    try:
        X_smote, y_smote = make_smote_scenario(X_train, y_train, config)
        scenarios["smote"] = (X_smote, y_smote)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Bỏ qua SMOTE: {exc}")

    rl_selected_path = Path(config["rl"]["output"]["selected_samples_dir"]) / "rl_selected_samples.parquet"
    if rl_selected_path.exists():
        rl_selected_df = pd.read_parquet(rl_selected_path)
        gan_rl_X = pd.concat([X_train, rl_selected_df.drop(columns=[target_col])], ignore_index=True)
        gan_rl_y = pd.concat([y_train, rl_selected_df[target_col]], ignore_index=True)
        scenarios["gan_rl"] = (gan_rl_X, gan_rl_y)
    else:
        logger.warning(
            f"Không tìm thấy {rl_selected_path}. Hãy chạy scripts/run_full_training.py trước."
        )

    ablation_df = run_ablation_study(scenarios, X_val, y_val, X_test, y_test, config)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(Path(reports_dir) / "ablation_study_standalone.csv", index=False)
    plot_ablation_comparison(ablation_df, plots_dir, filename="ablation_comparison_standalone.png")

    logger.info("Đã lưu kết quả tại artifacts/reports/ablation_study_standalone.csv")


if __name__ == "__main__":
    main()
