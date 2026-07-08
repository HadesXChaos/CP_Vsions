"""
extract_frames.py
------------------
Module tách video thành các frame riêng lẻ (bước đầu tiên của pipeline
Video Super Resolution).

Chức năng:
    - Đọc video đầu vào (mp4, avi, mov, ...)
    - Tách từng frame và lưu dưới dạng ảnh (png/jpg)
    - Hỗ trợ tách theo bước nhảy (frame_step) hoặc giới hạn số frame (max_frames)
    - Trả về thông tin metadata của video (fps, resolution, total_frames)
    - Có thể chạy độc lập qua CLI hoặc import làm module trong main.py

Output:
    data/frames/frame_0001.png
    data/frames/frame_0002.png
    ...
"""

import os
import cv2
import json
import logging
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_frames")


@dataclass
class VideoMetadata:
    """Thông tin metadata của video, dùng lại ở bước Merge Video sau này."""
    source_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    extracted_frames: int
    frame_step: int
    output_dir: str


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
    prefix: str = "frame",
    ext: str = "png",
    zero_pad: int = 6,
) -> VideoMetadata:
    """
    Tách video thành các frame riêng lẻ.

    Args:
        video_path: đường dẫn tới file video đầu vào.
        output_dir: thư mục lưu các frame đã tách (vd: data/frames/).
        frame_step: khoảng cách giữa các frame được lấy (1 = lấy tất cả,
                     2 = lấy mỗi frame thứ 2, ...). Hữu ích khi muốn giảm
                     số lượng frame cần xử lý qua Real-ESRGAN.
        max_frames: giới hạn số lượng frame tối đa được tách (None = không giới hạn).
        prefix: tiền tố tên file ảnh output.
        ext: định dạng ảnh output ("png" khuyến nghị vì không mất dữ liệu,
             "jpg" nếu cần tiết kiệm dung lượng).
        zero_pad: số chữ số đệm 0 trong tên file (vd: 6 -> frame_000001.png).

    Returns:
        VideoMetadata: thông tin video và số frame đã tách, dùng lại cho
        bước Merge Video ở cuối pipeline (để ghép lại đúng fps).

    Raises:
        FileNotFoundError: nếu video_path không tồn tại.
        RuntimeError: nếu không mở được video (file hỏng / codec không hỗ trợ).
    """
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    if frame_step < 1:
        raise ValueError("frame_step phải >= 1")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video (file hỏng hoặc codec không hỗ trợ): {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_path}")
    logger.info(f"FPS: {fps:.2f} | Resolution: {width}x{height} | Total frames (metadata): {total_frames}")

    frame_idx = 0        # index frame gốc trong video
    saved_idx = 0         # index frame đã lưu (liên tục, dùng đặt tên file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            saved_idx += 1
            filename = f"{prefix}_{saved_idx:0{zero_pad}d}.{ext}"
            filepath = out_dir / filename

            success = cv2.imwrite(str(filepath), frame)
            if not success:
                logger.warning(f"Lưu thất bại: {filepath}")
            else:
                if saved_idx % 100 == 0:
                    logger.info(f"Đã tách {saved_idx} frame...")

            if max_frames is not None and saved_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()

    logger.info(f"Hoàn tất: {saved_idx} frame đã được lưu tại '{out_dir}'")

    metadata = VideoMetadata(
        source_path=video_path,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        extracted_frames=saved_idx,
        frame_step=frame_step,
        output_dir=str(out_dir),
    )

    # Lưu metadata ra file json để bước Merge Video ở cuối pipeline dùng lại
    # (đặc biệt cần fps chính xác để ghép video đúng tốc độ gốc).
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=4, ensure_ascii=False)
    logger.info(f"Đã lưu metadata tại '{metadata_path}'")

    return metadata


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Tách video thành các frame riêng lẻ (bước Extract Frames của pipeline)."
    )
    parser.add_argument("--video", "-v", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument(
        "--output", "-o", default="data/frames",
        help="Thư mục lưu frame output (mặc định: data/frames)"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Lấy mỗi N frame (mặc định: 1, tức lấy tất cả)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Giới hạn số frame tối đa tách ra (mặc định: không giới hạn)"
    )
    parser.add_argument(
        "--ext", default="png", choices=["png", "jpg", "jpeg"],
        help="Định dạng ảnh output (mặc định: png)"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_step=args.step,
        max_frames=args.max_frames,
        ext=args.ext,
    )


if __name__ == "__main__":
    main()
