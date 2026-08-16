"""
src/models/gan/generator.py
============================
Generator cho Conditional WGAN-GP trên dữ liệu bảng hỗn hợp (numeric +
categorical).

Kiến trúc:
  Input: noise z ~ N(0,1) [latent_dim]  ++  condition c (one-hot nhãn) [condition_dim]
  -> MLP với LeakyReLU + BatchNorm (giống CTGAN/CTAB-GAN+ generator)
  -> 2 nhánh output:
       (a) numeric_head: Linear -> giá trị liên tục (dữ liệu numeric đã được
           StandardScaler chuẩn hoá ở TabularPreprocessor nên không cần
           tanh/bound cứng — xem thảo luận trong config/gan.yaml)
       (b) categorical_heads: 1 Linear + Gumbel-Softmax cho MỖI cột
           categorical, để giữ được tính rời rạc nhưng vẫn differentiable
           (kỹ thuật chuẩn trong CTGAN/CTAB-GAN+ cho categorical columns)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.preprocessor import TabularDataSpec


def _build_mlp(input_dim: int, hidden_dims: List[int], use_batchnorm: bool) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        prev_dim = h
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(
        self,
        spec: TabularDataSpec,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: List[int],
        use_batchnorm: bool = True,
        gumbel_temperature: float = 0.5,
    ):
        super().__init__()
        self.spec = spec
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.gumbel_temperature = gumbel_temperature

        input_dim = latent_dim + condition_dim
        self.backbone = _build_mlp(input_dim, hidden_dims, use_batchnorm)
        last_hidden = hidden_dims[-1] if hidden_dims else input_dim

        self.numeric_head = (
            nn.Linear(last_hidden, spec.numeric_dim) if spec.numeric_dim > 0 else None
        )
        self.categorical_heads = nn.ModuleList(
            [nn.Linear(last_hidden, dim) for dim in spec.categorical_dims]
        )

    def forward(
        self, z: torch.Tensor, condition: torch.Tensor, hard: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            z: [batch, latent_dim] noise
            condition: [batch, condition_dim] one-hot label
            hard: nếu True, Gumbel-Softmax trả về one-hot cứng (dùng khi sample
                  để lưu dữ liệu cuối cùng); nếu False, trả về soft (dùng khi train,
                  để gradient chảy qua được discriminator).
        Returns:
            numeric_out: [batch, numeric_dim] (rỗng nếu không có cột numeric)
            categorical_outs: list các tensor [batch, cardinality_i]
        """
        x = torch.cat([z, condition], dim=1)
        h = self.backbone(x)

        numeric_out = self.numeric_head(h) if self.numeric_head is not None else torch.zeros(
            z.size(0), 0, device=z.device
        )

        categorical_outs = []
        for head in self.categorical_heads:
            logits = head(h)
            soft = F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=hard, dim=-1)
            categorical_outs.append(soft)

        return numeric_out, categorical_outs

    def sample_noise(self, batch_size: int, device: str) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)
