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
        Sinh Label Map chuẩn: Kết hợp giá trị màu và tính liên thông không gian.
        """
        # Bước 1: Vẫn gom các màu độc nhất toàn cục để có danh sách màu (giống cũ của bạn)
        pixels = segmented.reshape(-1, 3)
        unique_colors, inverse_labels = np.unique(pixels, axis=0, return_inverse=True)
        
        # Mảng chứa nhãn cuối cùng (khởi tạo bằng 0)
        final_labels = np.zeros(segmented.shape[:2], dtype=np.int32)
        current_max_label = 0
        
        # Tạo mask cho ảnh nhãn tạm thời
        temp_labels = inverse_labels.reshape(segmented.shape[:2])
        
        # Bước 2: Duyệt qua từng màu độc nhất, tách các vùng bị cô lập bằng Connected Components
        for color_idx in range(len(unique_colors)):
            # Tạo mặt nạ nhị phân cho màu hiện tại
            color_mask = (temp_labels == color_idx).astype(np.uint8)
            
            # Tìm các thành phần liên thông độc lập của màu này
            num_labels, cc_labels = cv2.connectedComponents(color_mask, connectivity=8)
            
            # Nhãn 0 của cc_labels là vùng nền (không thuộc màu này), ta chỉ lấy từ nhãn 1 trở đi
            for i in range(1, num_labels):
                current_max_label += 1
                final_labels[cc_labels == i] = current_max_label
                
        return final_labels

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