import cv2
import numpy as np


class Sharpener:
    # Công thức:
    #     detail    = denoised - GaussianBlur(denoised)
    #     weight    = w_edge * edge_w + w_region * region_w
    #     sharpened = denoised + alpha * detail * weight

    def __init__(
        self,
        alpha=1.0,
        blur_kernel=5,
        edge_dilate=2,
        w_edge=0.7,
        w_region=0.3
    ):
        self.alpha = alpha
        self.blur_kernel = blur_kernel
        self.edge_dilate = edge_dilate
        self.w_edge = w_edge
        self.w_region = w_region

    def compute_detail(self, img):
        k = self.blur_kernel

        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(img, (k, k), 0)

        detail = img.astype(np.float32) - blurred.astype(np.float32)

        return detail

    def normalize(self, arr):
        a_min = arr.min()

        a_max = arr.max()

        return (arr - a_min) / (a_max - a_min + 1e-6)

    def edge_weight(self, edges):

        w = edges.astype(np.float32) / 255.0

        d = self.edge_dilate

        kernel = np.ones(
            (2 * d + 1, 2 * d + 1),
            np.uint8
        )

        w = cv2.dilate(w, kernel, iterations=1)

        return w

    def region_weight(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_64F)

        texture = np.abs(lap)

        return self.normalize(texture).astype(np.float32)

    def sharpen(self, denoised, edges):

        detail = self.compute_detail(denoised)

        edge_w = self.edge_weight(edges)

        region_w = self.region_weight(denoised)

        weight = self.w_edge * edge_w + self.w_region * region_w

        weight = np.clip(weight, 0.0, 1.0)

        weight_3c = np.stack([weight, weight, weight], axis=2)

        sharpened = denoised.astype(np.float32) \
            + self.alpha * detail * weight_3c

        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened
