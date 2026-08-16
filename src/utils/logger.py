"""
src/utils/logger.py
====================
Cấu hình logging thống nhất cho toàn bộ dự án.

Sử dụng:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "artifacts/logs",
    log_filename: str = "run.log",
) -> None:
    """Cấu hình root logger một lần duy nhất cho cả process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    Trả về logger đã đặt tên. Nếu logging chưa được cấu hình, tự cấu hình
    với giá trị mặc định (hoặc theo `config['logging']` nếu được truyền vào).
    """
    if not _CONFIGURED:
        if config and "logging" in config:
            log_cfg = config["logging"]
            paths_cfg = config.get("paths", {})
            configure_logging(
                level=log_cfg.get("level", "INFO"),
                log_to_file=log_cfg.get("log_to_file", True),
                log_dir=paths_cfg.get("logs_dir", "artifacts/logs"),
                log_filename=log_cfg.get("log_filename", "run.log"),
            )
        else:
            configure_logging()
    return logging.getLogger(name)
