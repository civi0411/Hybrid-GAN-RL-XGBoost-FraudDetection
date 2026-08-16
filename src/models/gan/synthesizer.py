"""
src/models/gan/synthesizer.py
================================
Dùng Generator đã train để sinh ra các mẫu synthetic mới (thường là cho
lớp thiểu số - fraud). Đây là bước tạo ra "candidate pool" mà RL agent sẽ
lọc lại sau đó (xem config/gan.yaml::sampling.candidate_multiplier).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from src.data.preprocessor import TabularDataSpec, TabularPreprocessor
from src.models.gan.generator import Generator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GANSynthesizer:
    def __init__(
        self,
        generator: Generator,
        preprocessor: TabularPreprocessor,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.generator = generator.to(device)
        self.generator.eval()
        self.preprocessor = preprocessor
        self.config = config["gan"]
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples: int, target_class: int = 1) -> pd.DataFrame:
        """Sinh ra n_samples bản ghi mới thuộc lớp `target_class`, trả về DataFrame."""
        n_classes = self.config["architecture"]["condition_dim"]
        z = self.generator.sample_noise(n_samples, self.device)
        condition = torch.nn.functional.one_hot(
            torch.full((n_samples,), target_class, dtype=torch.long), num_classes=n_classes
        ).float().to(self.device)

        numeric_out, categorical_outs = self.generator(z, condition, hard=True)

        numeric_np = numeric_out.cpu().numpy()
        categorical_np = [c.cpu().numpy() for c in categorical_outs]

        df = self.preprocessor.inverse_transform_gan(numeric_np, categorical_np)
        df[self.preprocessor.target_column] = target_class
        df["__is_synthetic__"] = True
        logger.info(f"Đã sinh {n_samples} mẫu synthetic cho class={target_class}.")
        return df

    def sample_candidate_pool(self, n_minority_real: int) -> pd.DataFrame:
        """
        Sinh ra candidate pool = oversample_ratio * candidate_multiplier * n_minority_real
        mẫu, để RL agent (bước sau) có không gian lựa chọn/lọc phong phú thay vì
        phải nhận toàn bộ mẫu GAN sinh ra một cách "mù quáng".
        """
        sampling_cfg = self.config["sampling"]
        n_to_generate = int(
            n_minority_real * sampling_cfg["oversample_ratio"] * sampling_cfg["candidate_multiplier"]
        )
        return self.sample(n_to_generate, target_class=sampling_cfg["target_class"])
