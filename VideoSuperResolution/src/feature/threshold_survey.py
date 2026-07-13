import os
import csv
import argparse
import cv2
import numpy as np

from .canny_edge import CannyEdgeDetector


# ===========================
# DEFAULT SURVEY GRID
# ===========================

LOW_THRESHOLDS = [30, 50, 80, 100]

HIGH_THRESHOLDS = [100, 150, 200, 250]


def build_grid_image(results, cell_size=(240, 180)):
    """Ghép các ảnh biên (kèm nhãn threshold) thành 1 lưới để so sánh trực quan."""

    w, h = cell_size

    n_rows = len(LOW_THRESHOLDS)
    n_cols = len(HIGH_THRESHOLDS)

    grid = np.zeros((h * n_rows, w * n_cols), dtype=np.uint8)

    for row, low in enumerate(LOW_THRESHOLDS):

        for col, high in enumerate(HIGH_THRESHOLDS):

            edges = results[(low, high)]["edges"]

            cell = cv2.resize(edges, (w, h))

            label = f"L{low}-H{high}"

            cv2.putText(
                cell,
                label,
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                255,
                1
            )

            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = cell

    return grid


def run_survey(
    image_path,
    output_dir,
    blur_kernel=5
):

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results = {}

    csv_path = os.path.join(output_dir, "threshold_survey.csv")

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["low_threshold", "high_threshold", "edge_ratio"])

        for low in LOW_THRESHOLDS:

            for high in HIGH_THRESHOLDS:

                if high <= low:
                    continue

                detector = CannyEdgeDetector(
                    low_threshold=low,
                    high_threshold=high,
                    blur_kernel=blur_kernel
                )

                edges = detector.detect(img)

                ratio = detector.edge_ratio(edges)

                results[(low, high)] = {
                    "edges": edges,
                    "ratio": ratio
                }

                writer.writerow([low, high, f"{ratio:.4f}"])

                print(f"low={low:<4} high={high:<4} edge_ratio={ratio:.4f}")

    # Ghép lưới so sánh (bỏ qua các cặp high <= low)
    for low in LOW_THRESHOLDS:
        for high in HIGH_THRESHOLDS:
            if (low, high) not in results:
                results[(low, high)] = {
                    "edges": np.zeros(img.shape[:2], dtype=np.uint8),
                    "ratio": 0.0
                }

    grid = build_grid_image(results)

    grid_path = os.path.join(output_dir, "threshold_survey_grid.png")

    cv2.imwrite(grid_path, grid)

    print(f"\nSaved grid comparison : {grid_path}")
    print(f"Saved CSV results     : {csv_path}")

    return csv_path, grid_path


def parse_args():

    parser = argparse.ArgumentParser(
        description="Khao sat anh huong cua nguong Canny (low/high threshold)"
    )

    parser.add_argument("--image", required=True, help="Duong dan anh mau dung de khao sat")
    parser.add_argument("--output", default="../../reports/canny_threshold_survey")
    parser.add_argument("--blur-kernel", type=int, default=5)

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 60)
    print("Canny Threshold Survey")
    print("=" * 60)

    run_survey(
        image_path=args.image,
        output_dir=args.output,
        blur_kernel=args.blur_kernel
    )


if __name__ == "__main__":
    main()
