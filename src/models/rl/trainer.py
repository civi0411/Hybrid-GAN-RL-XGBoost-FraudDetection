"""
src/models/rl/trainer.py
===========================
Orchestrate quá trình:
  1. Train PPO agent trên SyntheticSampleSelectionEnv (nhiều episode ngắn)
  2. Dùng policy đã train để duyệt (inference, deterministic) qua TOÀN BỘ
     candidate pool (không chỉ 1 batch/episode) và quyết định giữ/loại
     từng mẫu -> tập synthetic cuối cùng được đưa vào augment cho XGBoost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.rl.agent import build_ppo_agent, save_agent
from src.models.rl.environment import SyntheticSampleSelectionEnv
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RLTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rl_cfg = config["rl"]
        self.env: Optional[SyntheticSampleSelectionEnv] = None
        self.model = None

    def fit(
        self,
        candidate_pool_df: pd.DataFrame,
        candidate_pool_obs: np.ndarray,
        base_train_X: pd.DataFrame,
        base_train_y: pd.Series,
        val_X: pd.DataFrame,
        val_y: pd.Series,
    ):
        self.env = SyntheticSampleSelectionEnv(
            candidate_pool_df=candidate_pool_df,
            candidate_pool_obs=candidate_pool_obs,
            base_train_X=base_train_X,
            base_train_y=base_train_y,
            val_X=val_X,
            val_y=val_y,
            config=self.config,
        )
        self.model = build_ppo_agent(self.env, self.config)

        total_timesteps = self.rl_cfg["training"]["total_timesteps"]
        logger.info(f"[RL] Bắt đầu train PPO agent trong {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)

        save_agent(self.model, self.rl_cfg["output"]["save_dir"])
        return self.model

    def select_final_samples(
        self, candidate_pool_df: pd.DataFrame, candidate_pool_obs: np.ndarray
    ) -> pd.DataFrame:
        """
        Dùng policy đã train (deterministic) để duyệt qua TOÀN BỘ candidate pool
        (không giới hạn trong episode_batch_size) và trả về các mẫu được giữ lại.
        """
        if self.model is None or self.env is None:
            raise RuntimeError("Phải gọi .fit() trước khi select_final_samples().")

        n_pool = len(candidate_pool_df)
        keep_mask = np.zeros(n_pool, dtype=bool)

        selected_ratio_running = 0.0
        n_seen = 0
        for idx in range(n_pool):
            feature_vec = candidate_pool_obs[idx].astype(np.float32)
            progress = idx / max(n_pool, 1)
            extra = np.array([progress, selected_ratio_running], dtype=np.float32)
            obs = np.concatenate([feature_vec, extra]).astype(np.float32)

            action, _ = self.model.predict(obs, deterministic=True)
            keep = bool(action == 1)
            keep_mask[idx] = keep

            n_seen += 1
            n_kept = int(keep_mask[:n_seen].sum())
            selected_ratio_running = n_kept / n_seen

        selected_df = candidate_pool_df.iloc[keep_mask].reset_index(drop=True)
        logger.info(
            f"[RL] Đã chọn {len(selected_df)}/{n_pool} mẫu synthetic từ candidate pool "
            f"({100 * len(selected_df) / max(n_pool, 1):.1f}%)."
        )

        out_dir = self.rl_cfg["output"]["selected_samples_dir"]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / "rl_selected_samples.parquet"
        selected_df.to_parquet(out_path)
        logger.info(f"Đã lưu các mẫu đã chọn tại: {out_path}")

        return selected_df
