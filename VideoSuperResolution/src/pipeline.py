import argparse
import io
import json
import os
import sys
from pathlib import Path

# Buoc nay dam bao print() khong bi loi UnicodeEncodeError tren Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2

from extraction.extract_frames import extract_frames
from preprocessing.denoise import Denoiser
from feature.run_canny import run_canny_pipeline
from postprocessing.run_sharpening import run_sharpen_pipeline
from segmentation.mean_shift import MeanShiftSegmenter
from super_resolution.super_resolution import SuperResolutionModule
from utils.merge_video import merge_frames

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def read_video_metadata(frames_dir):
    metadata_path = Path(frames_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy metadata trong '{frames_dir}'. "
            "Hãy chạy lại bước extract frames hoặc cung cấp đúng thư mục chứa frame."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def denoise_frames(
    input_dir,
    output_dir,
    method="bilateral",
    gaussian_kernel=5,
    gaussian_sigma=0,
    bilateral_d=9,
    bilateral_sigma_color=75,
    bilateral_sigma_space=75,
):
    ensure_dir(output_dir)

    denoiser = Denoiser(
        method=method,
        gaussian_kernel=gaussian_kernel,
        gaussian_sigma=gaussian_sigma,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )

    files = list_images(input_dir)
    if not files:
        raise RuntimeError(f"Không tìm thấy ảnh trong thư mục: {input_dir}")

    print(f"Denoising {len(files)} frames using {method}...")

    for idx, filename in enumerate(files, start=1):
        input_path = Path(input_dir) / filename
        output_path = Path(output_dir) / filename

        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Không đọc được ảnh: {input_path}")
            continue

        result = denoiser.process(img)
        cv2.imwrite(str(output_path), result)

        if idx % 50 == 0:
            print(f"  - Đã xử lý {idx}/{len(files)} frame")

    print("Denoising hoàn tất!")
    return output_dir


def super_resolve_frames(
    input_dir,
    output_dir,
    scale=4,
    method="bicubic",
    use_gpu=False,
):
    ensure_dir(output_dir)

    sr = SuperResolutionModule(scale=scale, use_gpu=use_gpu)

    files = list_images(input_dir)
    if not files:
        raise RuntimeError(f"Không tìm thấy ảnh trong thư mục: {input_dir}")

    print(f"Super resolution ({method}) cho {len(files)} frame...")

    for idx, filename in enumerate(files, start=1):
        input_path = Path(input_dir) / filename
        output_path = Path(output_dir) / filename

        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Không đọc được ảnh: {input_path}")
            continue

        if method == "bicubic":
            result = sr.upscale_image(img, method=method)
        else:
            try:
                result = sr.upscale_image(img, method=method)
            except Exception as exc:
                print(f"Lỗi SR ở frame {filename}: {exc}. Chuyển sang bicubic.")
                result = sr.upscale_image(img, method="bicubic")

        cv2.imwrite(str(output_path), result)

        if idx % 10 == 0:
            print(f"  - Đã xử lý {idx}/{len(files)} frame")

    print("Super resolution hoàn tất!")
    return output_dir


def segment_frames(
    input_dir,
    segmented_dir,
    label_dir,
    spatial_radius=20,
    color_radius=30,
    quantization=16,
):
    ensure_dir(segmented_dir)
    ensure_dir(label_dir)

    segmenter = MeanShiftSegmenter(
        spatial_radius=spatial_radius,
        color_radius=color_radius,
        quantization=quantization,
    )

    files = list_images(input_dir)
    if not files:
        raise RuntimeError(f"Không tìm thấy ảnh trong thư mục: {input_dir}")

    print(f"Segmentation cho {len(files)} frame...")

    for idx, filename in enumerate(files, start=1):
        input_path = Path(input_dir) / filename
        segmented_path = Path(segmented_dir) / filename
        label_path = Path(label_dir) / filename

        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Không đọc được ảnh: {input_path}")
            continue

        segmented = segmenter.segment(img)
        labels = segmenter.segmentation_map(segmented)
        colored = segmenter.colorize(labels)

        cv2.imwrite(str(segmented_path), segmented)
        cv2.imwrite(str(label_path), colored)

        if idx % 50 == 0:
            print(f"  - Đã xử lý {idx}/{len(files)} frame")

    print("Segmentation hoàn tất!")
    return segmented_dir, label_dir


def run_pipeline(args):
    frames_dir = Path(args.frames_dir)
    denoised_dir = Path(args.denoised_dir)
    edges_dir = Path(args.edges_dir)
    sr_dir = Path(args.sr_dir)
    sharpened_dir = Path(args.sharpened_dir)
    segmented_dir = Path(args.segmented_dir)
    label_dir = Path(args.label_dir)
    output_video = Path(args.output_video)

    if args.extract_frames:
        frames_dir = ensure_dir(frames_dir)
        metadata = extract_frames(
            video_path=args.input_video,
            output_dir=str(frames_dir),
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            ext=args.frame_ext,
        )
        fps = metadata.fps
    else:
        frames_dir = Path(args.frames_dir)
        metadata = read_video_metadata(frames_dir)
        fps = float(metadata.get("fps", args.fps))

    if args.denoise:
        denoise_frames(
            input_dir=frames_dir,
            output_dir=denoised_dir,
            method=args.denoise_method,
            gaussian_kernel=args.gaussian_kernel,
            gaussian_sigma=args.gaussian_sigma,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )
    else:
        ensure_dir(denoised_dir)

    if args.canny:
        run_canny_pipeline(
            input_dir=str(denoised_dir),
            output_dir=str(edges_dir),
            overlay_dir=None,
            low_threshold=args.canny_low,
            high_threshold=args.canny_high,
            blur_kernel=args.canny_blur_kernel,
            l2_gradient=args.canny_l2_gradient,
        )
    else:
        ensure_dir(edges_dir)

    if args.segment:
        segment_frames(
            input_dir=denoised_dir,
            segmented_dir=segmented_dir,
            label_dir=label_dir,
            spatial_radius=args.segment_spatial_radius,
            color_radius=args.segment_color_radius,
            quantization=args.segment_quantization,
        )

    if args.sr:
        super_resolve_frames(
            input_dir=denoised_dir,
            output_dir=sr_dir,
            scale=args.scale,
            method=args.sr_method,
            use_gpu=args.use_gpu,
        )
    else:
        ensure_dir(sr_dir)

    if args.sharpen:
        sharpen_input_dir = sr_dir if args.sr else denoised_dir
        run_sharpen_pipeline(
            denoised_dir=str(sharpen_input_dir),
            edges_dir=str(edges_dir),
            output_dir=str(sharpened_dir),
            alpha=args.sharpen_alpha,
            blur_kernel=args.sharpen_blur_kernel,
            edge_dilate=args.sharpen_edge_dilate,
            w_edge=args.sharpen_w_edge,
            w_region=args.sharpen_w_region,
        )
    else:
        ensure_dir(sharpened_dir)

    final_dir = sharpened_dir if args.sharpen else sr_dir if args.sr else denoised_dir
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if args.merge:
        print(f"Ghép {final_dir} thành video: {output_video}")
        merge_frames(
            frame_dir=str(final_dir),
            output_path=str(output_video),
            fps=fps,
        )
        print("Merge video hoàn tất!")

    print("Pipeline đã chạy xong.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline Video Super Resolution: extract, denoise, canny, SR, sharpen, merge"
    )

    parser.add_argument("--input-video", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument(
        "--frames-dir",
        default="../data/frames",
        help="Thư mục lưu frame đầu vào"
    )
    parser.add_argument(
        "--denoised-dir",
        default="../data/processed/denoised",
        help="Thư mục lưu ảnh sau denoise"
    )
    parser.add_argument(
        "--edges-dir",
        default="../data/processed/edges",
        help="Thư mục lưu edge map"
    )
    parser.add_argument(
        "--sr-dir",
        default="../data/processed/sr",
        help="Thư mục lưu ảnh sau super resolution"
    )
    parser.add_argument(
        "--sharpened-dir",
        default="../data/processed/sharpened",
        help="Thư mục lưu ảnh sau sharpen"
    )
    parser.add_argument(
        "--segmented-dir",
        default="../data/processed/segmentation/segmented",
        help="Thư mục lưu ảnh segmentation"
    )
    parser.add_argument(
        "--label-dir",
        default="../data/processed/segmentation/label_map",
        help="Thư mục lưu label map"
    )
    parser.add_argument(
        "--output-video",
        default="../data/output/output.mp4",
        help="Đường dẫn video đầu ra"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Lấy mỗi N frame khi extract"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Giới hạn số frame khi extract"
    )
    parser.add_argument(
        "--frame-ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Định dạng frame khi extract"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="FPS mặc định khi skip extract nhưng không tìm thấy metadata"
    )

    parser.add_argument("--no-extract", dest="extract_frames", action="store_false")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false")
    parser.add_argument("--no-canny", dest="canny", action="store_false")
    parser.add_argument("--no-sr", dest="sr", action="store_false")
    parser.add_argument("--no-sharpen", dest="sharpen", action="store_false")
    parser.add_argument("--no-merge", dest="merge", action="store_false")
    parser.add_argument("--segment", dest="segment", action="store_true")

    parser.set_defaults(
        extract_frames=True,
        denoise=True,
        canny=True,
        sr=True,
        sharpen=True,
        merge=True,
        segment=False,
    )

    parser.add_argument(
        "--denoise-method",
        default="bilateral",
        choices=["gaussian", "bilateral"],
        help="Phương pháp khử nhiễu"
    )
    parser.add_argument("--gaussian-kernel", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=0)
    parser.add_argument("--bilateral-d", type=int, default=9)
    parser.add_argument("--bilateral-sigma-color", type=float, default=75)
    parser.add_argument("--bilateral-sigma-space", type=float, default=75)

    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--canny-blur-kernel", type=int, default=5)
    parser.add_argument("--canny-l2-gradient", action="store_true")

    parser.add_argument(
        "--sr-method",
        default="bicubic",
        choices=["bicubic", "real-esrgan"],
        help="Phương pháp super resolution"
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--use-gpu", action="store_true")

    parser.add_argument("--sharpen-alpha", type=float, default=1.0)
    parser.add_argument("--sharpen-blur-kernel", type=int, default=5)
    parser.add_argument("--sharpen-edge-dilate", type=int, default=2)
    parser.add_argument("--sharpen-w-edge", type=float, default=0.7)
    parser.add_argument("--sharpen-w-region", type=float, default=0.3)

    parser.add_argument("--segment-spatial-radius", type=int, default=20)
    parser.add_argument("--segment-color-radius", type=int, default=30)
    parser.add_argument("--segment-quantization", type=int, default=16)

    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
