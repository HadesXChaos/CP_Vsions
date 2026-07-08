import cv2
import numpy as np


class CannyEdgeDetector:

    def __init__(
        self,
        low_threshold=50,
        high_threshold=150,
        aperture_size=3,
        l2_gradient=False,
        blur_kernel=5
    ):

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient
        self.blur_kernel = blur_kernel

    def to_gray(self, img):

        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def smooth(self, gray):

        if self.blur_kernel <= 1:
            return gray

        return cv2.GaussianBlur(
            gray,
            (self.blur_kernel, self.blur_kernel),
            0
        )

    def detect(self, img):

        gray = self.to_gray(img)

        gray = self.smooth(gray)

        edges = cv2.Canny(
            gray,
            self.low_threshold,
            self.high_threshold,
            apertureSize=self.aperture_size,
            L2gradient=self.l2_gradient
        )

        return edges

    def overlay(self, img, edges, color=(0, 255, 0)):
        """Vẽ biên lên ảnh gốc để trực quan hóa."""

        result = img.copy()

        result[edges > 0] = color

        return result

    def edge_ratio(self, edges):
        """Tỉ lệ pixel biên / tổng số pixel, dùng để khảo sát ngưỡng."""

        return float(np.count_nonzero(edges)) / edges.size
