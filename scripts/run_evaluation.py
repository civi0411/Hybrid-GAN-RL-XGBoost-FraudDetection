"""
scripts/run_evaluation.py
============================
Entry point CLI: đánh giá model đã train, sinh report + plots đầy đủ.

Sử dụng:
    python scripts/run_evaluation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.eval_pipeline import run_evaluation_pipeline
from src.utils.config_loader import load_all_configs
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá model đã train")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    config = load_all_configs(config_dir=args.config_dir)
    logger = get_logger(__name__, config=config)

    results = run_evaluation_pipeline(config)
    logger.info(f"AUC-PR trên test set: {results['metrics']['auc_pr']:.4f}")


if __name__ == "__main__":
    main()
