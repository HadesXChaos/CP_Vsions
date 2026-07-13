"""
main.py
-------
Entry-point chính để thực thi toàn bộ pipeline Video Super Resolution.

Cách chạy (từ thư mục VideoSuperResolution/):
    python main.py --input-video data/videos/input.mp4

Tham số đầy đủ xem trong src/pipeline.py (argparse).
"""

import sys
import os

# Thêm src/ vào sys.path để import các module trong src/
_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pipeline import main

if __name__ == "__main__":
    main()
