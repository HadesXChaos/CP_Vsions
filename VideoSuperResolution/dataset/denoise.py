import cv2
import numpy as np


class Denoiser:
    """
    Khử nhiễu ảnh bằng Gaussian Blur hoặc Bilateral Filter.
    Bilateral giữ biên tốt hơn Gaussian vì trọng số phụ thuộc
    cả khoảng cách không gian lẫn độ chênh lệch cường độ pixel.
    """

    def __init__(
        self,
        method="bilateral",
        gaussian_kernel=5,
        gaussian_sigma=0,
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75
    ):
        self.method = method

        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def gaussian_denoise(self, img):

        k = self.gaussian_kernel

        # kernel phải là số lẻ
        if k % 2 == 0:
            k += 1

        return cv2.GaussianBlur(
            img,
            (k, k),
            self.gaussian_sigma
        )

    def bilateral_denoise(self, img):

        return cv2.bilateralFilter(
            img,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )

    def process(self, img):

        if self.method == "gaussian":
            return self.gaussian_denoise(img)

        elif self.method == "bilateral":
            return self.bilateral_denoise(img)

        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                "Use 'gaussian' or 'bilateral'."
            )