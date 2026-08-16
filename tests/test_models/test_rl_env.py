"""
tests/test_models/test_rl_env.py
===================================
Test "smoke test" cho SyntheticSampleSelectionEnv: đảm bảo API Gymnasium
(reset/step) hoạt động đúng chuẩn (obs shape, terminated đúng lúc, reward
chỉ khác 0 ở bước cuối episode).

Dùng dữ liệu giả (rất nhỏ) để bài test chạy nhanh — không nhằm kiểm tra
chất lượng lựa chọn của RL agent (việc đó cần train PPO thật).
"""
import numpy as np
import pandas as pd

from src.models.rl.environment import SyntheticSampleSelectionEnv


def _build_dummy_config():
    return {
        "project": {"seed": 42},
        "target_column": "isFraud",
        "rl": {
            "env": {
                "episode_batch_size": 4,
                "reward_metric": "average_precision",
                "fast_xgboost_n_estimators": 10,
                "fast_xgboost_max_depth": 2,
                "normalize_reward": False,
            }
        },
    }


def _build_dummy_data(n_train=60, n_val=20, n_candidates=10, n_features=3, seed=0):
    rng = np.random.default_rng(seed)

    def make_split(n):
        X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
        y = pd.Series(rng.integers(0, 2, size=n), name="isFraud")
        return X, y

    X_train, y_train = make_split(n_train)
    X_val, y_val = make_split(n_val)

    candidate_df = pd.DataFrame(
        rng.normal(size=(n_candidates, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    candidate_df["isFraud"] = 1
    candidate_obs = candidate_df.drop(columns=["isFraud"]).to_numpy(dtype="float32")

    return X_train, y_train, X_val, y_val, candidate_df, candidate_obs


def test_env_reset_returns_correct_obs_shape():
    config = _build_dummy_config()
    X_train, y_train, X_val, y_val, candidate_df, candidate_obs = _build_dummy_data()

    env = SyntheticSampleSelectionEnv(
        candidate_pool_df=candidate_df,
        candidate_pool_obs=candidate_obs,
        base_train_X=X_train,
        base_train_y=y_train,
        val_X=X_val,
        val_y=y_val,
        config=config,
    )
    obs, info = env.reset(seed=0)

    assert obs.shape == (env.obs_dim,)
    assert isinstance(info, dict)


def test_env_episode_terminates_after_batch_size_steps_and_reward_only_at_end():
    config = _build_dummy_config()
    X_train, y_train, X_val, y_val, candidate_df, candidate_obs = _build_dummy_data()

    env = SyntheticSampleSelectionEnv(
        candidate_pool_df=candidate_df,
        candidate_pool_obs=candidate_obs,
        base_train_X=X_train,
        base_train_y=y_train,
        val_X=X_val,
        val_y=y_val,
        config=config,
    )
    env.reset(seed=0)

    batch_size = env.episode_batch_size
    terminated = False
    n_steps = 0
    total_intermediate_reward = 0.0

    for _ in range(batch_size):
        obs, reward, terminated, truncated, info = env.step(1)  # luôn chọn "giữ lại"
        n_steps += 1
        if not terminated:
            total_intermediate_reward += reward

    assert n_steps == batch_size
    assert terminated is True
    assert total_intermediate_reward == 0.0  # reward chỉ có ở bước cuối (sparse reward)
    assert "n_selected" in info and info["n_selected"] == batch_size
