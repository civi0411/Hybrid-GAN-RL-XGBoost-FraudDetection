"""
src/models/rl/environment.py
===============================
Môi trường Gymnasium cho bài toán "chọn lọc mẫu synthetic" (xem thiết kế
chi tiết trong config/rl.yaml).

Một episode = duyệt qua 1 batch ngẫu nhiên (episode_batch_size) các mẫu
candidate GAN sinh ra, quyết định giữ/loại từng mẫu MỘT (sequential,
action Discrete(2)), reward chỉ được trả ở bước CUỐI episode (sparse
reward = delta AUC-PR khi retrain classifier nhanh trên train_gốc + mẫu
đã chọn — xem reward.py).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.models.rl.reward import compute_baseline_score, compute_episode_reward
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticSampleSelectionEnv(gym.Env):
    """
    Args:
        candidate_pool_df: DataFrame ứng viên synthetic (đã ở dạng classifier-ready,
            tức là output của TabularPreprocessor.transform_for_classifier, kèm cột target)
        candidate_pool_obs: np.ndarray [n_pool, obs_feature_dim] — biểu diễn số hoá của
            candidate_pool_df, dùng làm observation cho policy (categorical đã encode thành mã số)
        base_train_X / base_train_y: dữ liệu train gốc (không đổi trong suốt quá trình)
        val_X / val_y: dữ liệu validation dùng để tính reward
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        candidate_pool_df: pd.DataFrame,
        candidate_pool_obs: np.ndarray,
        base_train_X: pd.DataFrame,
        base_train_y: pd.Series,
        val_X: pd.DataFrame,
        val_y: pd.Series,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.candidate_pool_df = candidate_pool_df.reset_index(drop=True)
        self.candidate_pool_obs = candidate_pool_obs.astype(np.float32)
        self.base_train_X = base_train_X
        self.base_train_y = base_train_y
        self.val_X = val_X
        self.val_y = val_y
        self.config = config

        env_cfg = config["rl"]["env"]
        self.episode_batch_size = min(env_cfg["episode_batch_size"], len(self.candidate_pool_df))
        self.target_column = config["target_column"]

        obs_feature_dim = self.candidate_pool_obs.shape[1]
        # obs = [feature vector của candidate hiện tại] + [tiến độ episode, tỉ lệ đã chọn]
        self.obs_dim = obs_feature_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0 = loại bỏ, 1 = giữ lại

        logger.info("Đang tính baseline reward score (chỉ 1 lần, dùng làm điểm neo)...")
        self.baseline_score = compute_baseline_score(
            base_train_X, base_train_y, val_X, val_y, config
        )

        self._episode_indices: np.ndarray = np.array([], dtype=int)
        self._pointer = 0
        self._selected_mask: list = []
        self._last_raw_score = self.baseline_score

    def _current_obs(self) -> np.ndarray:
        if self._pointer >= len(self._episode_indices):
            return np.zeros(self.obs_dim, dtype=np.float32)
        idx = self._episode_indices[self._pointer]
        feature_vec = self.candidate_pool_obs[idx]
        progress = self._pointer / max(len(self._episode_indices), 1)
        selected_ratio = (sum(self._selected_mask) / max(len(self._selected_mask), 1)) if self._selected_mask else 0.0
        extra = np.array([progress, selected_ratio], dtype=np.float32)
        return np.concatenate([feature_vec, extra]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        n_pool = len(self.candidate_pool_df)
        self._episode_indices = rng.choice(n_pool, size=self.episode_batch_size, replace=False)
        self._pointer = 0
        self._selected_mask = []
        return self._current_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        keep = bool(action == 1)
        self._selected_mask.append(keep)
        self._pointer += 1

        terminated = self._pointer >= len(self._episode_indices)
        truncated = False
        info: Dict[str, Any] = {}

        if not terminated:
            return self._current_obs(), 0.0, terminated, truncated, info

        selected_local_idx = [
            self._episode_indices[i] for i, k in enumerate(self._selected_mask) if k
        ]
        selected_df = self.candidate_pool_df.iloc[selected_local_idx]

        reward, raw_score = compute_episode_reward(
            self.base_train_X,
            self.base_train_y,
            self.val_X,
            self.val_y,
            selected_df,
            self.baseline_score,
            self.config,
        )
        self._last_raw_score = raw_score
        info = {
            "n_selected": len(selected_local_idx),
            "n_candidates": len(self._episode_indices),
            "raw_score": raw_score,
            "baseline_score": self.baseline_score,
        }
        logger.debug(
            f"[RL-Env] episode kết thúc: n_selected={info['n_selected']}/{info['n_candidates']} "
            f"raw_score={raw_score:.4f} reward={reward:.4f}"
        )
        return self._current_obs(), reward, terminated, truncated, info
