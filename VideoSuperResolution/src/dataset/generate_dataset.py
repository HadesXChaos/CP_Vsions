import os
import shutil
import cv2

from VideoSuperResolution.src.utils.extract_frames import FrameExtractor
from VideoSuperResolution.src.dataset.degrade import DatasetDegrader
from VideoSuperResolution.src.utils.merge_video import merge_frames


# ===========================
# PATHS
# ===========================

RAW_DIR = "../data/raw"

GT_ROOT = "../data/ground_truth"

LR_ROOT = "../data/low_resolution"

INPUT_ROOT = "../data/videos/input"


# ===========================
# RESET FOLDER
# ===========================

def reset_folder(folder):

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)


# ===========================
# MAIN
# ===========================

def main():

    print("=" * 60)
    print("Generating Dataset")
    print("=" * 60)

    degrader = DatasetDegrader(
        scale=0.5,
        noise_sigma=20,
        blur_kernel=5
    )

    os.makedirs(GT_ROOT, exist_ok=True)
    os.makedirs(LR_ROOT, exist_ok=True)
    os.makedirs(INPUT_ROOT, exist_ok=True)

    videos = sorted(os.listdir(RAW_DIR))

    for video in videos:

        if not video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_name = os.path.splitext(video)[0]

        print(f"\nProcessing: {video_name}")

        raw_video = os.path.join(RAW_DIR, video)

        gt_dir = os.path.join(GT_ROOT, video_name)

        lr_dir = os.path.join(LR_ROOT, video_name)

        input_video = os.path.join(
            INPUT_ROOT,
            f"{video_name}.mp4"
        )

        # ------------------------
        # Reset folders
        # ------------------------

        reset_folder(gt_dir)
        reset_folder(lr_dir)

        # ------------------------
        # Extract frames
        # ------------------------

        extractor = FrameExtractor(
            raw_video,
            gt_dir
        )

        fps, width, height = extractor.extract()

        print(f"FPS       : {fps}")
        print(f"Resolution: {width}x{height}")

        # ------------------------
        # Degrade frames
        # ------------------------

        files = sorted(os.listdir(gt_dir))

        for file in files:

            img_path = os.path.join(gt_dir, file)

            img = cv2.imread(img_path)

            if img is None:
                print(f"Cannot read {img_path}")
                continue

            lr = degrader.process(img)

            cv2.imwrite(
                os.path.join(lr_dir, file),
                lr
            )

        print(f"Frames: {len(files)}")

        # ------------------------
        # Merge LR frames
        # ------------------------

        merge_frames(
            frame_dir=lr_dir,
            output_path=input_video,
            fps=fps
        )

        print("Input video created.")

    print("\n")
    print("=" * 60)
    print("Dataset generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()