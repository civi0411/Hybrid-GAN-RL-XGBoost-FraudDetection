"""
src/models/gan/discriminator.py
=================================
Critic (Discriminator) cho Conditional WGAN-GP.

Khác với GAN gốc, critic KHÔNG dùng sigmoid ở output — trả về 1 điểm số
thực (Wasserstein distance estimate), theo đúng công thức WGAN-GP
(Gulrajani et al., 2017; áp dụng cho tabular data bởi Engelmann & Lessmann,
2021 và CTGAN/CTAB-GAN+).

LƯU Ý: Critic không dùng BatchNorm (theo khuyến nghị gốc của WGAN-GP, vì
BatchNorm làm hỏng tính chất 1-Lipschitz cần thiết cho gradient penalty).
Dùng LayerNorm thay thế nếu cần chuẩn hoá.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.data.preprocessor import TabularDataSpec


def _build_critic_mlp(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        prev_dim = h
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(
        self,
        spec: TabularDataSpec,
        condition_dim: int,
        hidden_dims: List[int],
    ):
        super().__init__()
        input_dim = spec.numeric_dim + sum(spec.categorical_dims) + condition_dim
        self.backbone = _build_critic_mlp(input_dim, hidden_dims)
        last_hidden = hidden_dims[-1] if hidden_dims else input_dim
        self.score_head = nn.Linear(last_hidden, 1)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: List[torch.Tensor],
        condition: torch.Tensor,
    ) -> torch.Tensor:
        parts = [numeric] + list(categorical) + [condition]
        x = torch.cat(parts, dim=1)
        h = self.backbone(x)
        return self.score_head(h).squeeze(-1)  # [batch]
