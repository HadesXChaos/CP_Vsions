import os
import cv2

from mean_shift import MeanShiftSegmenter


IMAGE_PATH = "../../data/processed/denoised/frame_000001.png"

OUTPUT_DIR = "../../reports/segmentation"


SP_LIST = [10, 20, 30]

SR_LIST = [20, 40, 60]


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    image = cv2.imread(IMAGE_PATH)

    for sp in SP_LIST:

        for sr in SR_LIST:

            segmenter = MeanShiftSegmenter(
                spatial_radius=sp,
                color_radius=sr
            )

            result = segmenter.segment(image)

            filename = f"sp{sp}_sr{sr}.png"

            cv2.imwrite(
                os.path.join(
                    OUTPUT_DIR,
                    filename
                ),
                result
            )

            print(filename)


if __name__ == "__main__":

    main()