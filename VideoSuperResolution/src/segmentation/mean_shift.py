import cv2
import numpy as np


class MeanShiftSegmenter:

    def __init__(
        self,
        spatial_radius=20,
        color_radius=30,
        quantization=16
    ):
        """
        Parameters
        ----------
        spatial_radius : int
            Bán kính không gian (sp)

        color_radius : int
            Bán kính màu (sr)

        quantization : int
            Mức lượng tử hóa màu sau Mean Shift.
        """

        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.quantization = quantization

    def segment(self, image):
        """
        Mean Shift Filtering.

        Parameters
        ----------
        image : np.ndarray

        Returns
        -------
        np.ndarray
            Ảnh sau Mean Shift.
        """

        filtered = cv2.pyrMeanShiftFiltering(
            image,
            sp=self.spatial_radius,
            sr=self.color_radius
        )

        # Lượng tử hóa màu để giảm số màu và làm rõ các vùng
        segmented = (
            filtered // self.quantization
        ) * self.quantization

        return segmented

    def segmentation_map(self, segmented):
        """
        Sinh Label Map từ ảnh đã Mean Shift.

        Parameters
        ----------
        segmented : np.ndarray

        Returns
        -------
        labels : np.ndarray
            Ma trận nhãn của các vùng.
        """

        # Gom các màu giống nhau thành cùng một nhãn
        pixels = segmented.reshape(-1, 3)

        _, labels = np.unique(
            pixels,
            axis=0,
            return_inverse=True
        )

        labels = labels.reshape(
            segmented.shape[:2]
        )

        return labels

    def colorize(self, labels):
        """
        Hiển thị Label Map bằng màu.

        Parameters
        ----------
        labels : np.ndarray

        Returns
        -------
        np.ndarray
        """

        labels = labels.astype(np.float32)

        labels = labels / labels.max()

        labels = (labels * 255).astype(np.uint8)

        colored = cv2.applyColorMap(
            labels,
            cv2.COLORMAP_JET
        )

        return colored

    def count_regions(self, labels):
        """
        Đếm số vùng phân đoạn.

        Parameters
        ----------
        labels : np.ndarray

        Returns
        -------
        int
        """

        return len(np.unique(labels))