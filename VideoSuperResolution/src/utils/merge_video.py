import os
import cv2
import numpy as np
import imageio.v3 as iio
import imageio

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def merge_frames(frame_dir, output_path, fps):
    """
    Ghep cac frame trong frame_dir thanh video H.264 (.mp4) dung imageio-ffmpeg.
    Tuong thich voi Windows Media Player, VLC, Photos app.
    """
    files = sorted(
        f for f in os.listdir(frame_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise RuntimeError(f"Khong tim thay anh trong: {frame_dir}")

    # Dam bao output la .mp4
    base, _ = os.path.splitext(output_path)
    mp4_path = base + ".mp4"
    os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)

    # Doc kich thuoc tu frame dau
    first = cv2.imread(os.path.join(frame_dir, files[0]))
    if first is None:
        raise RuntimeError(f"Khong doc duoc: {files[0]}")
    h, w = first.shape[:2]

    print(f"[merge_video] {len(files)} frame  {w}x{h}  {fps}fps  ->  {mp4_path}")

    # imageio-ffmpeg writer: H.264, yuv420p (chay duoc tren moi player)
    # macro_block_size=1: tat tu dong resize khi chieu khong chia het cho 16
    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "23", "-preset", "fast"],
    )

    errors = 0
    for i, file in enumerate(files, 1):
        img_bgr = cv2.imread(os.path.join(frame_dir, file))
        if img_bgr is None:
            print(f"[merge_video] Bo qua: {file}")
            errors += 1
            continue
        # imageio can RGB, OpenCV tra ve BGR
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(img_rgb)
        if i % 10 == 0:
            print(f"[merge_video]   {i}/{len(files)} frame da xu ly...")

    writer.close()

    size = os.path.getsize(mp4_path)
    print(f"[merge_video] Hoan tat  {len(files)-errors}/{len(files)} frame  ({size/1024/1024:.1f} MB)  ->  {mp4_path}")
    return mp4_path