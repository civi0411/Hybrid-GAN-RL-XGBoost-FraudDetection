"""
src/utils/seed.py
==================
Set seed thống nhất cho python random, numpy, torch (CPU & CUDA) để
đảm bảo khả năng tái lập kết quả (reproducibility).
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seed cho mọi thư viện có yếu tố ngẫu nhiên đang được dùng trong repo."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ưu tiên tính tái lập hơn tốc độ khi cần debug; có thể tắt nếu cần train nhanh.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """Trả về device string ("cpu" hoặc "cuda") theo preference trong config."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA được yêu cầu trong config nhưng không khả dụng trên máy này.")
        if preference == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return preference
    except ImportError:
        return "cpu"
