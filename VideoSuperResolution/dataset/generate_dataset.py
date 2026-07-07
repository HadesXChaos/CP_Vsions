import os
import cv2

from extract_frames import FrameExtractor
from degrade import DatasetDegrader


RAW_VIDEO = "../../data/raw/video1.mp4"

GT_DIR = "../../data/ground_truth"

LR_DIR = "../../data/low_resolution"


def main():

    extractor = FrameExtractor(
        RAW_VIDEO,
        GT_DIR
    )

    fps, w, h = extractor.extract()

    degrader = DatasetDegrader(
        scale=0.5,
        noise_sigma=20,
        blur_kernel=5
    )

    os.makedirs(LR_DIR, exist_ok=True)

    files = sorted(os.listdir(GT_DIR))

    for file in files:

        img = cv2.imread(
            os.path.join(GT_DIR, file)
        )

        img = degrader.process(img)

        cv2.imwrite(
            os.path.join(LR_DIR, file),
            img
        )

    print("Done")


if __name__ == "__main__":

    main()