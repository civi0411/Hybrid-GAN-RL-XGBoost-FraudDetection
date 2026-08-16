"""
scripts/run_full_training.py
===============================
Entry point CLI: chạy toàn bộ pipeline train (Data -> GAN -> RL -> XGBoost
-> Ablation -> Save).

Sử dụng:
    python scripts/run_full_training.py
    python scripts/run_full_training.py --config-dir config
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.train_pipeline import run_training_pipeline
from src.utils.config_loader import load_all_configs
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy full training pipeline")
    parser.add_argument("--config-dir", type=str, default="config", help="Thư mục chứa file YAML config")
    args = parser.parse_args()

    config = load_all_configs(config_dir=args.config_dir)
    logger = get_logger(__name__, config=config)
    logger.info(f"Loaded config từ '{args.config_dir}'. Project: {config['project']['name']}")

    results = run_training_pipeline(config)

    logger.info("=" * 60)
    logger.info(f"HOÀN TẤT. Kịch bản tốt nhất: {results['best_scenario']}")
    logger.info("Xem chi tiết tại artifacts/reports/ablation_study.csv và artifacts/plots/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
