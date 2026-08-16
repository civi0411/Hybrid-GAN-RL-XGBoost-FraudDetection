"""
tests/test_models/test_xgb.py
================================
Smoke test cho XGBTrainer: train trên dữ liệu giả nhỏ, đảm bảo:
  - scale_pos_weight tự tính đúng công thức (n_neg / n_pos)
  - model train xong có thể predict_proba hợp lệ (trong khoảng [0, 1])
"""
import numpy as np
import pandas as pd

from src.models.xgb.trainer import XGBTrainer


def _build_dummy_config():
    return {
        "project": {"seed": 42},
        "xgboost": {
            "params": {
                "objective": "binary:logistic",
                "eval_metric": ["aucpr"],
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 20,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "random_state": 42,
            },
            "imbalance": {"scale_pos_weight": "auto"},
            "training": {"early_stopping_rounds": 5, "verbose_eval": False},
        },
    }


def _build_dummy_data(n=200, n_pos=20, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series([0] * (n - n_pos) + [1] * n_pos, name="isFraud").sample(frac=1, random_state=seed).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def test_scale_pos_weight_auto_computation():
    config = _build_dummy_config()
    trainer = XGBTrainer(config)
    y = pd.Series([0] * 90 + [1] * 10)

    weight = trainer._compute_scale_pos_weight(y)

    assert weight == 9.0  # 90 / 10


def test_train_produces_valid_probabilities():
    config = _build_dummy_config()
    X, y = _build_dummy_data()
    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_val, y_val = X.iloc[150:], y.iloc[150:]

    trainer = XGBTrainer(config)
    model = trainer.train(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)[:, 1]

    assert proba.shape[0] == len(X_val)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
