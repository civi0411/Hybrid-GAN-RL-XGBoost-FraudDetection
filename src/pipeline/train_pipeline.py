"""
src/pipeline/train_pipeline.py
=================================
Điểm vào DUY NHẤT để chạy toàn bộ quy trình:

    Data -> Preprocess -> Split (time-based)
         -> Train GAN (trên train set, điều kiện theo nhãn)
         -> Sinh candidate pool synthetic (fraud)
         -> Train RL agent để lọc candidate pool
         -> Ablation: xgboost_only vs smote vs gan_only vs gan_rl
         -> Train + lưu model cuối cùng (gan_rl, trừ khi ablation cho thấy
            kịch bản khác tốt hơn) + toàn bộ artifacts (plots, report)

Chạy qua: python scripts/run_full_training.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessor import TabularPreprocessor
from src.data.splitter import split_data
from src.evaluation.comparator import run_ablation_study
from src.evaluation.visualizer import (
    plot_ablation_comparison,
    plot_gan_training_curves,
    plot_real_vs_synthetic_distribution,
)
from src.models.gan.synthesizer import GANSynthesizer
from src.models.gan.trainer import GANTrainer
from src.models.rl.trainer import RLTrainer
from src.models.xgb.trainer import XGBTrainer
from src.utils.logger import get_logger
from src.utils.seed import get_device, set_global_seed
from src.utils.validators import validate_class_balance, validate_target_column

logger = get_logger(__name__)


def make_smote_scenario(
    X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Baseline oversampling truyền thống để so sánh với GAN — dùng SMOTENC vì
    dữ liệu có cả cột categorical (SMOTE thuần chỉ hoạt động đúng với numeric).
    """
    from imblearn.over_sampling import SMOTENC

    categorical_mask = [isinstance(X_train[c].dtype, pd.CategoricalDtype) for c in X_train.columns]
    if not any(categorical_mask):
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=config["project"]["seed"])
    else:
        cat_idx = [i for i, is_cat in enumerate(categorical_mask) if is_cat]
        sampler = SMOTENC(categorical_features=cat_idx, random_state=config["project"]["seed"])

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res


def run_training_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    set_global_seed(config["project"]["seed"])
    device = get_device(config["project"]["device"])
    logger.info(f"Đang chạy trên device: {device}")

    plots_dir = config["paths"]["plots_dir"]
    reports_dir = config["paths"]["reports_dir"]
    models_dir = config["paths"]["models_dir"]

    # ---------------------------------------------------------------- #
    # 1) Load + validate + split
    # ---------------------------------------------------------------- #
    df = load_raw_data(config)
    validate_target_column(df, config["target_column"])
    validate_class_balance(df, config["target_column"])

    train_df, val_df, test_df = split_data(df, config)

    # ---------------------------------------------------------------- #
    # 2) Preprocess (fit CHỈ trên train, tránh leakage)
    # ---------------------------------------------------------------- #
    preprocessor = TabularPreprocessor(config).fit(train_df)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    preprocessor.save(str(Path(models_dir) / "preprocessor.joblib"))

    X_train, y_train = preprocessor.transform_for_classifier(train_df)
    X_val, y_val = preprocessor.transform_for_classifier(val_df)
    X_test, y_test = preprocessor.transform_for_classifier(test_df)

    n_minority_real = int((y_train == 1).sum())
    logger.info(f"Số mẫu fraud thật trong train: {n_minority_real}/{len(y_train)}")

    # ---------------------------------------------------------------- #
    # 3) Train GAN (conditional WGAN-GP) trên toàn bộ train set
    # ---------------------------------------------------------------- #
    gan_data = preprocessor.transform_for_gan(train_df)
    gan_trainer = GANTrainer(preprocessor.spec, config, device=device)
    gan_history = gan_trainer.fit(gan_data)
    plot_gan_training_curves(gan_history, plots_dir)

    # ---------------------------------------------------------------- #
    # 4) Sinh candidate pool synthetic (fraud) từ Generator đã train
    # ---------------------------------------------------------------- #
    synthesizer = GANSynthesizer(gan_trainer.generator, preprocessor, config, device=device)
    candidate_pool_raw = synthesizer.sample_candidate_pool(n_minority_real)

    real_fraud_numeric = train_df.loc[train_df[config["target_column"]] == 1, preprocessor.spec.numeric_columns]
    synth_numeric = candidate_pool_raw[preprocessor.spec.numeric_columns]
    if preprocessor.spec.numeric_dim > 0:
        plot_real_vs_synthetic_distribution(
            real_fraud_numeric.to_numpy(), synth_numeric.to_numpy(), plots_dir
        )

    candidate_pool_df, _ = preprocessor.transform_for_classifier(candidate_pool_raw)
    candidate_pool_df[config["target_column"]] = 1
    candidate_pool_obs = preprocessor.to_numeric_array(
        candidate_pool_df.drop(columns=[config["target_column"]])
    )

    # ---------------------------------------------------------------- #
    # 5) Train RL agent để lọc candidate pool
    # ---------------------------------------------------------------- #
    rl_trainer = RLTrainer(config)
    rl_trainer.fit(
        candidate_pool_df=candidate_pool_df,
        candidate_pool_obs=candidate_pool_obs,
        base_train_X=X_train,
        base_train_y=y_train,
        val_X=X_val,
        val_y=y_val,
    )
    rl_selected_df = rl_trainer.select_final_samples(candidate_pool_df, candidate_pool_obs)

    # ---------------------------------------------------------------- #
    # 6) Ablation study: xgboost_only / smote / gan_only / gan_rl
    # ---------------------------------------------------------------- #
    scenarios: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "xgboost_only": (X_train, y_train),
    }

    try:
        X_smote, y_smote = make_smote_scenario(X_train, y_train, config)
        scenarios["smote"] = (X_smote, y_smote)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Bỏ qua kịch bản SMOTE do lỗi: {exc}")

    target_col = config["target_column"]
    gan_only_X = pd.concat([X_train, candidate_pool_df.drop(columns=[target_col])], ignore_index=True)
    gan_only_y = pd.concat([y_train, candidate_pool_df[target_col]], ignore_index=True)
    scenarios["gan_only"] = (gan_only_X, gan_only_y)

    if len(rl_selected_df) > 0:
        gan_rl_X = pd.concat([X_train, rl_selected_df.drop(columns=[target_col])], ignore_index=True)
        gan_rl_y = pd.concat([y_train, rl_selected_df[target_col]], ignore_index=True)
    else:
        logger.warning("RL agent không chọn mẫu nào -> kịch bản gan_rl trùng với xgboost_only.")
        gan_rl_X, gan_rl_y = X_train, y_train
    scenarios["gan_rl"] = (gan_rl_X, gan_rl_y)

    ablation_df = run_ablation_study(scenarios, X_val, y_val, X_test, y_test, config)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(Path(reports_dir) / "ablation_study.csv", index=False)
    plot_ablation_comparison(ablation_df, plots_dir)

    # ---------------------------------------------------------------- #
    # 7) Train + lưu model cuối cùng dựa trên kịch bản tốt nhất (theo AUC-PR)
    # ---------------------------------------------------------------- #
    best_scenario = ablation_df.sort_values("auc_pr", ascending=False).iloc[0]["scenario"]
    logger.info(f"Kịch bản tốt nhất theo AUC-PR: '{best_scenario}'")
    X_final, y_final = scenarios[best_scenario]

    final_trainer = XGBTrainer(config)
    final_model = final_trainer.train(X_final, y_final, X_val, y_val)
    final_trainer.save(str(Path(models_dir) / "xgboost_final.json"))

    return {
        "ablation_df": ablation_df,
        "best_scenario": best_scenario,
        "final_model": final_model,
        "preprocessor": preprocessor,
        "test_data": (X_test, y_test),
    }
