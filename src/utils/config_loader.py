"""
src/utils/config_loader.py
===========================
Load và merge các file YAML cấu hình (base.yaml + data/gan/xgboost/rl.yaml).

Sử dụng:
    from src.utils.config_loader import load_config
    cfg = load_config(config_dir="config", extra=["gan.yaml", "xgboost.yaml"])
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge đệ quy `override` vào `base`, override thắng nếu trùng key."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(
    config_dir: str = "config",
    base_filename: str = "base.yaml",
    extra: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load base.yaml rồi merge thêm các file trong `extra` (theo thứ tự).
    Mỗi file extra được lồng vào key riêng theo tên file (không có .yaml),
    ví dụ gan.yaml -> cfg["gan"], xgboost.yaml -> cfg["xgboost"], v.v.
    Đồng thời cfg["base"] chứa toàn bộ nội dung base.yaml ở top-level cho tiện.
    """
    config_dir_path = Path(config_dir)
    base_cfg = _read_yaml(config_dir_path / base_filename)

    merged: Dict[str, Any] = copy.deepcopy(base_cfg)

    for filename in extra or []:
        section_name = Path(filename).stem  # "gan.yaml" -> "gan"
        section_cfg = _read_yaml(config_dir_path / filename)
        if section_name in merged and isinstance(merged[section_name], dict):
            merged[section_name] = _deep_merge(merged[section_name], section_cfg)
        else:
            merged[section_name] = section_cfg

    return merged


def load_all_configs(config_dir: str = "config") -> Dict[str, Any]:
    """Tiện ích: load toàn bộ config chuẩn (base + data + gan + xgboost + rl)."""
    return load_config(
        config_dir=config_dir,
        extra=["data.yaml", "gan.yaml", "xgboost.yaml", "rl.yaml"],
    )
