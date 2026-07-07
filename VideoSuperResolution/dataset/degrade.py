import cv2
import numpy as np


class DatasetDegrader:

    def __init__(
        self,
        scale=0.5,
        noise_sigma=15,
        blur_kernel=5
    ):

        self.scale = scale
        self.noise_sigma = noise_sigma
        self.blur_kernel = blur_kernel

    def add_gaussian_noise(self, img):

        noise = np.random.normal(
            0,
            self.noise_sigma,
            img.shape
        )

        noisy = img.astype(np.float32) + noise

        noisy = np.clip(noisy, 0, 255)

        return noisy.astype(np.uint8)

    def resize(self, img):

        h, w = img.shape[:2]

        lr = cv2.resize(
            img,
            (
                int(w * self.scale),
                int(h * self.scale)
            ),
            interpolation=cv2.INTER_AREA
        )

        return lr

    def blur(self, img):

        return cv2.GaussianBlur(
            img,
            (self.blur_kernel, self.blur_kernel),
            0
        )

    def process(self, img):

        img = self.resize(img)

        img = self.add_gaussian_noise(img)

        img = self.blur(img)

        return img