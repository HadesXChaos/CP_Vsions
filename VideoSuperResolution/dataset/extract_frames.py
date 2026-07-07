import cv2
import os

class FrameExtractor:

    def __init__(self, video_path, output_dir):

        self.video_path = video_path
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def extract(self):

        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        idx = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            filename = os.path.join(
                self.output_dir,
                f"{idx:05d}.png"
            )

            cv2.imwrite(filename, frame)

            idx += 1

        cap.release()

        return fps, width, height