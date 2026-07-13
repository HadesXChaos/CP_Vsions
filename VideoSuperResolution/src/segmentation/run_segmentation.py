import os
import csv
import cv2

from .mean_shift import MeanShiftSegmenter


INPUT_DIR = "../../data/processed/denoised"

OUTPUT_ROOT = "../../data/processed/segmentation"

SEGMENTED_DIR = os.path.join(OUTPUT_ROOT, "segmented")
LABEL_DIR = os.path.join(OUTPUT_ROOT, "label_map")


def main():

    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    segmenter = MeanShiftSegmenter(
        spatial_radius=20,
        color_radius=30,
        quantization=16
    )

    files = sorted(os.listdir(INPUT_DIR))

    statistics = []

    for file in files:

        if not file.endswith(".png"):
            continue

        img = cv2.imread(
            os.path.join(INPUT_DIR, file)
        )

        segmented = segmenter.segment(img)

        labels = segmenter.segmentation_map(segmented)

        colored = segmenter.colorize(labels)

        regions = segmenter.count_regions(labels)

        cv2.imwrite(
            os.path.join(
                SEGMENTED_DIR,
                file
            ),
            segmented
        )

        cv2.imwrite(
            os.path.join(
                LABEL_DIR,
                file
            ),
            colored
        )

        statistics.append([file, regions])

        print(f"{file}: {regions} regions")

    with open(
        os.path.join(OUTPUT_ROOT, "regions.csv"),
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["Frame", "Number of Regions"]
        )

        writer.writerows(statistics)

    print("Done")


if __name__ == "__main__":

    main()