"""
src/models/gan/trainer.py
===========================
Vòng lặp huấn luyện Conditional WGAN-GP cho dữ liệu bảng.

Thuật toán (Gulrajani et al., 2017 "Improved Training of Wasserstein GANs",
áp dụng cho tabular oversampling theo Engelmann & Lessmann, 2021):

  for epoch in epochs:
    for batch:
        # 1) Update critic n_critic lần
        for _ in range(n_critic):
            fake = G(z, c)  (soft, để differentiable)
            loss_D = E[D(fake)] - E[D(real)] + lambda_gp * gradient_penalty
            update D

        # 2) Update generator 1 lần
        fake = G(z, c)
        loss_G = -E[D(fake)]
        update G

Model chỉ được train có điều kiện (condition = nhãn fraud/không-fraud),
nhưng mục tiêu sử dụng cuối cùng là sinh thêm mẫu cho lớp fraud (thiểu số)
— xem sampling trong config/gan.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessor import TabularDataSpec
from src.models.gan.discriminator import Discriminator
from src.models.gan.generator import Generator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TabularGANDataset(Dataset):
    """Wrap output của TabularPreprocessor.transform_for_gan() thành Dataset."""

    def __init__(self, numeric: np.ndarray, categorical: List[np.ndarray], target: np.ndarray, n_classes: int = 2):
        self.numeric = torch.as_tensor(numeric, dtype=torch.float32)
        self.categorical = [torch.as_tensor(c, dtype=torch.float32) for c in categorical]
        self.condition = torch.nn.functional.one_hot(
            torch.as_tensor(target, dtype=torch.long), num_classes=n_classes
        ).float()

    def __len__(self) -> int:
        return self.numeric.size(0)

    def __getitem__(self, idx: int):
        cats = [c[idx] for c in self.categorical]
        return self.numeric[idx], cats, self.condition[idx]


def _collate(batch):
    numerics, cats_list, conditions = zip(*batch)
    numeric_batch = torch.stack(numerics)
    condition_batch = torch.stack(conditions)
    n_cat_cols = len(cats_list[0]) if cats_list[0] else 0
    categorical_batch = [
        torch.stack([sample[i] for sample in cats_list]) for i in range(n_cat_cols)
    ]
    return numeric_batch, categorical_batch, condition_batch


def gradient_penalty(
    critic: Discriminator,
    real_numeric: torch.Tensor,
    real_categorical: List[torch.Tensor],
    fake_numeric: torch.Tensor,
    fake_categorical: List[torch.Tensor],
    condition: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Gradient penalty chuẩn của WGAN-GP, áp dụng trên toàn bộ vector đã concat."""
    batch_size = real_numeric.size(0)
    eps = torch.rand(batch_size, 1, device=device)

    interp_numeric = (eps * real_numeric + (1 - eps) * fake_numeric).requires_grad_(True)
    interp_categorical = []
    for real_c, fake_c in zip(real_categorical, fake_categorical):
        interp_c = (eps * real_c + (1 - eps) * fake_c).requires_grad_(True)
        interp_categorical.append(interp_c)

    scores = critic(interp_numeric, interp_categorical, condition)

    inputs = [interp_numeric] + interp_categorical
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=inputs,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    grad_flat = torch.cat([g.reshape(batch_size, -1) for g in grads], dim=1)
    grad_norm = grad_flat.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


class GANTrainer:
    def __init__(self, spec: TabularDataSpec, config: Dict[str, Any], device: str = "cpu"):
        self.spec = spec
        self.config = config["gan"]
        self.device = device

        arch = self.config["architecture"]
        self.n_classes = arch["condition_dim"]

        self.generator = Generator(
            spec=spec,
            latent_dim=arch["latent_dim"],
            condition_dim=arch["condition_dim"],
            hidden_dims=arch["generator_hidden_dims"],
            use_batchnorm=arch["use_batchnorm"],
            gumbel_temperature=arch["gumbel_temperature"],
        ).to(device)

        self.critic = Discriminator(
            spec=spec,
            condition_dim=arch["condition_dim"],
            hidden_dims=arch["critic_hidden_dims"],
        ).to(device)

        train_cfg = self.config["training"]
        betas = tuple(train_cfg["betas"])
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=train_cfg["lr_generator"], betas=betas,
            weight_decay=train_cfg["weight_decay"],
        )
        self.opt_d = torch.optim.Adam(
            self.critic.parameters(), lr=train_cfg["lr_critic"], betas=betas,
            weight_decay=train_cfg["weight_decay"],
        )

        self.history: Dict[str, List[float]] = {"loss_d": [], "loss_g": [], "gp": []}

    def fit(self, gan_data: Dict[str, Any]) -> Dict[str, List[float]]:
        train_cfg = self.config["training"]
        dataset = TabularGANDataset(
            numeric=gan_data["numeric"],
            categorical=gan_data["categorical"],
            target=gan_data["target"],
            n_classes=self.n_classes,
        )
        loader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=_collate,
        )

        n_critic = train_cfg["n_critic"]
        gp_lambda = train_cfg["gradient_penalty_lambda"]

        for epoch in range(train_cfg["epochs"]):
            epoch_loss_d, epoch_loss_g, epoch_gp = [], [], []
            for step, (real_numeric, real_categorical, condition) in enumerate(loader):
                real_numeric = real_numeric.to(self.device)
                real_categorical = [c.to(self.device) for c in real_categorical]
                condition = condition.to(self.device)
                batch_size = real_numeric.size(0)

                # ---- (1) Update critic n_critic lần ----
                for _ in range(n_critic):
                    z = self.generator.sample_noise(batch_size, self.device)
                    with torch.no_grad():
                        fake_numeric, fake_categorical = self.generator(z, condition, hard=False)

                    self.opt_d.zero_grad()
                    real_score = self.critic(real_numeric, real_categorical, condition)
                    fake_score = self.critic(fake_numeric, fake_categorical, condition)
                    gp = gradient_penalty(
                        self.critic, real_numeric, real_categorical,
                        fake_numeric, fake_categorical, condition, self.device,
                    )
                    loss_d = fake_score.mean() - real_score.mean() + gp_lambda * gp
                    loss_d.backward()
                    self.opt_d.step()

                # ---- (2) Update generator 1 lần ----
                z = self.generator.sample_noise(batch_size, self.device)
                fake_numeric, fake_categorical = self.generator(z, condition, hard=False)
                self.opt_g.zero_grad()
                fake_score = self.critic(fake_numeric, fake_categorical, condition)
                loss_g = -fake_score.mean()
                loss_g.backward()
                self.opt_g.step()

                epoch_loss_d.append(loss_d.item())
                epoch_loss_g.append(loss_g.item())
                epoch_gp.append(gp.item())

                if step % train_cfg["log_every_n_steps"] == 0:
                    logger.debug(
                        f"[GAN] epoch={epoch} step={step} loss_D={loss_d.item():.4f} "
                        f"loss_G={loss_g.item():.4f} gp={gp.item():.4f}"
                    )

            mean_d, mean_g, mean_gp = np.mean(epoch_loss_d), np.mean(epoch_loss_g), np.mean(epoch_gp)
            self.history["loss_d"].append(float(mean_d))
            self.history["loss_g"].append(float(mean_g))
            self.history["gp"].append(float(mean_gp))
            logger.info(
                f"[GAN] Epoch {epoch + 1}/{train_cfg['epochs']} | "
                f"loss_D={mean_d:.4f} loss_G={mean_g:.4f} gp={mean_gp:.4f}"
            )

            if (epoch + 1) % train_cfg["checkpoint_every_n_epochs"] == 0:
                self.save(self.config["output"]["save_dir"], suffix=f"_epoch{epoch + 1}")

        self.save(self.config["output"]["save_dir"], suffix="_final")
        return self.history

    def save(self, save_dir: str, suffix: str = "") -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.generator.state_dict(), Path(save_dir) / f"generator{suffix}.pt")
        torch.save(self.critic.state_dict(), Path(save_dir) / f"critic{suffix}.pt")
        logger.info(f"Đã lưu GAN checkpoint tại {save_dir} (suffix={suffix})")

    def load(self, save_dir: str, suffix: str = "_final") -> None:
        self.generator.load_state_dict(
            torch.load(Path(save_dir) / f"generator{suffix}.pt", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(Path(save_dir) / f"critic{suffix}.pt", map_location=self.device)
        )
