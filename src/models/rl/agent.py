"""
src/models/rl/agent.py
=========================
Wrapper mỏng quanh stable-baselines3 PPO — thuật toán được chọn vì tính ổn
định khi update policy với reward sparse/nhiễu (Schulman et al., 2017),
và đã được dùng trong các nghiên cứu "synthetic sample selection" liên
quan (Ye et al. 2020; Gowda 2023) — xem giải thích trong config/rl.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_ppo_agent(env, config: Dict[str, Any]) -> PPO:
    ppo_cfg = config["rl"]["ppo"]
    seed = config["rl"]["training"]["seed"]

    vec_env = DummyVecEnv([lambda: Monitor(env)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=dict(net_arch=list(ppo_cfg["policy_hidden_dims"])),
        seed=seed,
        verbose=1,
    )
    return model


def save_agent(model: PPO, save_dir: str, filename: str = "ppo_selector") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / filename
    model.save(str(path))
    logger.info(f"Đã lưu RL agent tại: {path}.zip")


def load_agent(save_dir: str, env, filename: str = "ppo_selector") -> PPO:
    path = Path(save_dir) / filename
    model = PPO.load(str(path), env=env)
    logger.info(f"Đã load RL agent từ: {path}.zip")
    return model
