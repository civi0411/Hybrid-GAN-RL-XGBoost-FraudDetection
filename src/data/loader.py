"""
src/data/loader.py
===================
Load dữ liệu thô. Được giữ Ở MỨC GENERIC theo yêu cầu — chưa xử lý các
đặc thù của IEEE-CIS (join transaction+identity theo TransactionID, xử lý
riêng từng nhóm cột V1-V339, C1-C14, D1-D15, M1-M9, v.v.). Phần đó sẽ được
bổ sung sau khi hoàn thiện EDA trong notebooks/.

Hàm ở đây chỉ đảm nhiệm: đọc CSV, merge 2 bảng theo key (nếu có bảng thứ 2),
và trả về 1 DataFrame duy nhất.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Đọc dữ liệu thô theo config['data']['files'] và config['paths']['raw_dir'].
    Nếu có identity_file, merge với transaction_file theo merge_key (left join).
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    files_cfg = config["data"]["files"]

    transaction_path = raw_dir / files_cfg["transaction_file"]
    logger.info(f"Đang đọc transaction file: {transaction_path}")
    df = pd.read_csv(transaction_path)

    identity_filename: Optional[str] = files_cfg.get("identity_file")
    if identity_filename:
        identity_path = raw_dir / identity_filename
        if identity_path.exists():
            logger.info(f"Đang đọc identity file: {identity_path}")
            identity_df = pd.read_csv(identity_path)
            merge_key = files_cfg.get("merge_key", "TransactionID")
            df = df.merge(identity_df, on=merge_key, how="left")
            logger.info(f"Đã merge theo '{merge_key}'. Shape sau merge: {df.shape}")
        else:
            logger.warning(f"identity_file được khai báo nhưng không tồn tại: {identity_path}")

    logger.info(f"Shape dữ liệu thô cuối cùng: {df.shape}")
    return df


def load_processed_data(config: Dict[str, Any], filename: str = "processed.parquet") -> pd.DataFrame:
    """Đọc lại dữ liệu đã tiền xử lý (đã lưu bởi preprocessor.save_processed)."""
    processed_path = Path(config["paths"]["processed_dir"]) / filename
    logger.info(f"Đang đọc dữ liệu đã xử lý: {processed_path}")
    return pd.read_parquet(processed_path)
