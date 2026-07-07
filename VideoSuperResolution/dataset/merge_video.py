import cv2
import os


def merge_frames(
    frame_dir,
    output_path,
    fps
):

    files = sorted(os.listdir(frame_dir))

    first = cv2.imread(
        os.path.join(frame_dir, files[0])
    )

    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for file in files:

        img = cv2.imread(
            os.path.join(frame_dir, file)
        )

        writer.write(img)

    writer.release()