import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class SuperResolutionHandler:
    def __init__(self, scale=4, use_gpu=True):
        """
        Khởi tạo bộ xử lý Siêu độ phân giải.
        :param scale: Tỷ lệ phóng đại (ví dụ: 4)
        :param use_gpu: Cho phép sử dụng cấu hình GPU nếu có cho AI
        """
        self.scale = scale
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.realesrgan_model = None
        
        # Kích hoạt sẵn mô hình AI ở chế độ "hộp đen"
        self._init_real_esrgan()

    def _init_real_esrgan(self):
        """Khởi tạo ngầm mô hình AI Real-ESRGAN"""
        try:
            model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                 num_block=23, num_grow_ch=32, scale=self.scale)
            self.realesrgan_model = RealESRGANer(
                scale=self.scale,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model_arch,
                tile=0,
                half=self.use_gpu
            )
            print("[SR Module] Đã cấu hình thành công mô hình AI Real-ESRGAN.")
        except Exception as e:
            print(f"[SR Module] Không thể tải Real-ESRGAN: {e}. Luồng AI sẽ tự động dùng Bicubic làm phương án dự phòng.")

    def upscale_image(self, frame, method='real-esrgan'):
        """
        LỰA CHỌN 1 TRONG 2 PHƯƠNG PHÁP CHO LUỒNG ẢNH CHÍNH
        :param frame: Khung hình độ phân giải thấp (RGB/BGR)
        :param method: 'bicubic' (Truyền thống) hoặc 'real-esrgan' (AI)
        """
        method = method.lower()
        
        # --- PHƯƠNG PHÁP 1: TRUYỀN THỐNG (Bicubic) ---
        if method == 'bicubic':
            h, w = frame.shape[:2]
            new_size = (int(w * self.scale), int(h * self.scale))
            return cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
            
        # --- PHƯƠNG PHÁP 2: CÔNG NGHỆ AI (Real-ESRGAN) ---
        elif method == 'real-esrgan':
            if self.realesrgan_model is not None:
                # Gọi API hộp đen
                output, _ = self.realesrgan_model.enhance(frame, outscale=self.scale)
                return output
            else:
                # Fallback nếu AI lỗi
                h, w = frame.shape[:2]
                new_size = (int(w * self.scale), int(h * self.scale))
                return cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError("Phương pháp không hợp lệ. Chỉ chọn 'bicubic' hoặc 'real-esrgan'.")

    def upscale_map_nearest(self, map_img):
        """
        HÀM DÀNH RIÊNG CHO EDGE MAP VÀ SEG MAP
        Phóng đại bằng Nearest Neighbor để giữ nguyên các đường biên nét đứt/khối màu thô ở độ phân giải thấp
        """
        h, w = map_img.shape[:2]
        new_size = (int(w * self.scale), int(h * self.scale))
        return cv2.resize(map_img, new_size, interpolation=cv2.INTER_NEAREST)