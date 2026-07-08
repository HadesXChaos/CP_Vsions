import os
import argparse
import cv2

from canny_edge import CannyEdgeDetector


# ===========================
# PATHS (mặc định theo cấu trúc dự án)
# ===========================

DEFAULT_INPUT_DIR = "../../data/processed/denoised"

DEFAULT_OUTPUT_DIR = "../../data/processed/edges"

DEFAULT_OVERLAY_DIR = "../../data/processed/edges_overlay"


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

def run_canny_pipeline(
    input_dir,
    output_dir,
    overlay_dir=None,
    low_threshold=50,
    high_threshold=150,
    blur_kernel=5,
    l2_gradient=False
):

    reset_folder(output_dir)

    if overlay_dir is not None:
        reset_folder(overlay_dir)

    detector = CannyEdgeDetector(
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_kernel=blur_kernel,
        l2_gradient=l2_gradient
    )

    files = list_images(input_dir)

    print(f"Found {len(files)} frames in {input_dir}")

    ratios = []

    for file in files:

        img_path = os.path.join(input_dir, file)

        img = cv2.imread(img_path)

        if img is None:
            print(f"Cannot read {img_path}")
            continue

        edges = detector.detect(img)

        cv2.imwrite(
            os.path.join(output_dir, file),
            edges
        )

        if overlay_dir is not None:

            overlay = detector.overlay(img, edges)

            cv2.imwrite(
                os.path.join(overlay_dir, file),
                overlay
            )

        ratios.append(detector.edge_ratio(edges))

    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Average edge ratio: {avg_ratio:.4f}")

    print("Canny edge detection completed!")

    return output_dir


# ===========================
# CLI
# ===========================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Canny Edge Detection - pipeline stage"
    )

    parser.add_argument("--input", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overlay", default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--low", type=int, default=50)
    parser.add_argument("--high", type=int, default=150)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--l2-gradient", action="store_true")

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 60)
    print("Canny Edge Detection")
    print("=" * 60)
    print(f"low_threshold  : {args.low}")
    print(f"high_threshold : {args.high}")
    print(f"blur_kernel    : {args.blur_kernel}")

    run_canny_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        overlay_dir=args.overlay,
        low_threshold=args.low,
        high_threshold=args.high,
        blur_kernel=args.blur_kernel,
        l2_gradient=args.l2_gradient
    )


if __name__ == "__main__":
    main()
