"""
scripts/download_data.py
===========================
Script tải dataset IEEE-CIS Fraud Detection từ Kaggle về data/raw/.

Yêu cầu: đã cài `kaggle` CLI và cấu hình API token (~/.kaggle/kaggle.json).
Xem: https://github.com/Kaggle/kaggle-api#api-credentials
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
COMPETITION = "ieee-fraud-detection"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)]
    print(f"Đang chạy: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Lỗi khi tải dữ liệu từ Kaggle:", result.stderr, file=sys.stderr)
        print(
            "Gợi ý: kiểm tra ~/.kaggle/kaggle.json, và đảm bảo đã 'Join Competition' "
            "trên trang Kaggle trước khi tải bằng CLI.",
            file=sys.stderr,
        )
        sys.exit(1)

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Đang giải nén {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        print("Giải nén xong. Các file trong data/raw/:")
        for p in sorted(RAW_DIR.glob("*")):
            print(f"  - {p.name}")
    else:
        print("Không tìm thấy file zip sau khi tải — kiểm tra lại output của kaggle CLI.")


if __name__ == "__main__":
    main()
