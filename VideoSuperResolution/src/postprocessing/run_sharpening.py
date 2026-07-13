import os
import argparse
import cv2

from .sharpen import Sharpener


# ===========================
# PATHS (mặc định theo cấu trúc dự án)
# ===========================

DEFAULT_DENOISED = "../../data/processed/denoised"

DEFAULT_EDGES = "../../data/processed/edges"

DEFAULT_OUTPUT = "../../data/processed/sharpened"


# ===========================
# HELPERS
# ===========================

def reset_folder(folder):

    os.makedirs(folder, exist_ok=True)


def list_images(folder):

    exts = (".png", ".jpg", ".jpeg", ".bmp")

    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )


# ===========================
# MAIN
# ===========================

def run_sharpen_pipeline(
    denoised_dir,
    edges_dir,
    output_dir,
    alpha=1.0,
    blur_kernel=5,
    edge_dilate=2,
    w_edge=0.7,
    w_region=0.3
):

    reset_folder(output_dir)

    sharpener = Sharpener(
        alpha=alpha,
        blur_kernel=blur_kernel,
        edge_dilate=edge_dilate,
        w_edge=w_edge,
        w_region=w_region
    )

    files = list_images(denoised_dir)

    print(f"Found {len(files)} frames in {denoised_dir}")

    for file in files:

        denoised_path = os.path.join(denoised_dir, file)

        edges_path = os.path.join(edges_dir, file)

        denoised = cv2.imread(denoised_path)

        if denoised is None:

            print(f"Cannot read {denoised_path}")

            continue

        edges = cv2.imread(edges_path, cv2.IMREAD_GRAYSCALE)

        if edges is None:

            print(f"Cannot read {edges_path}")

            continue

        # Resize edge map neu kich thuoc khac voi frame (vd: SR da upscale 4x)
        if edges.shape[:2] != denoised.shape[:2]:
            edges = cv2.resize(
                edges,
                (denoised.shape[1], denoised.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        sharpened = sharpener.sharpen(denoised, edges)

        cv2.imwrite(
            os.path.join(output_dir, file),
            sharpened
        )

    print("Sharpening completed!")

    return output_dir


# ===========================
# CLI
# ===========================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Edge & Region-aware Sharpening - pipeline stage"
    )

    parser.add_argument(
        "--denoised",
        default=DEFAULT_DENOISED,
        help="Thư mục ảnh đã khử nhiễu (mặc định: data/processed/denoised)"
    )

    parser.add_argument(
        "--edges",
        default=DEFAULT_EDGES,
        help="Thư mục Edge Map (mặc định: data/processed/edges)"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Thư mục lưu ảnh đã sharpen (mặc định: data/processed/sharpened)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Cường độ sharpen (mặc định: 1.0)"
    )

    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Kích thước kernel Gaussian (mặc định: 5)"
    )

    parser.add_argument(
        "--edge-dilate",
        type=int,
        default=2,
        help="Số pixel giãn nở biên (mặc định: 2)"
    )

    parser.add_argument(
        "--w-edge",
        type=float,
        default=0.7,
        help="Trọng số ưu tiên vùng biên (mặc định: 0.7)"
    )

    parser.add_argument(
        "--w-region",
        type=float,
        default=0.3,
        help="Trọng số ưu tiên vùng texture (mặc định: 0.3)"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 60)
    print("Edge & Region-aware Sharpening")
    print("=" * 60)
    print(f"alpha        : {args.alpha}")
    print(f"blur_kernel  : {args.blur_kernel}")
    print(f"edge_dilate  : {args.edge_dilate}")
    print(f"w_edge       : {args.w_edge}")
    print(f"w_region     : {args.w_region}")

    run_sharpen_pipeline(
        denoised_dir=args.denoised,
        edges_dir=args.edges,
        output_dir=args.output,
        alpha=args.alpha,
        blur_kernel=args.blur_kernel,
        edge_dilate=args.edge_dilate,
        w_edge=args.w_edge,
        w_region=args.w_region
    )


if __name__ == "__main__":
    main()
