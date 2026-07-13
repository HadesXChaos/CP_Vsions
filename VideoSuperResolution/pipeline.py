"""
pipeline.py
-----------
Kịch bản định nghĩa luồng luân chuyển dữ liệu toàn hệ thống.
Dùng để kiểm tra hoặc chạy thử nghiệm nhanh pipeline.

Cách chạy (từ thư mục VideoSuperResolution/):
    python pipeline.py --input-video data/videos/input.mp4

Tham số đầy đủ:
    --input-video   : Đường dẫn video đầu vào (bắt buộc)
    --no-extract    : Bỏ qua bước extract frames
    --no-denoise    : Bỏ qua bước denoise
    --no-canny      : Bỏ qua bước Canny edge detection
    --no-sr         : Bỏ qua bước super resolution
    --no-sharpen    : Bỏ qua bước sharpening
    --no-merge      : Bỏ qua bước merge video
    --segment       : Bật bước Mean Shift Segmentation
    --sr-method     : Chọn "bicubic" (mặc định) hoặc "real-esrgan"
    --scale         : Tỷ lệ phóng đại (mặc định: 4)
    --use-gpu       : Dùng GPU cho Real-ESRGAN

Ví dụ (chỉ test bicubic, skip merge):
    python pipeline.py --input-video data/raw/corridor.mp4 --sr-method bicubic --no-merge
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
