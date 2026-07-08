import torch
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class RealESRGANPredictor:
    def __init__(self, scale=4, use_gpu=True):
        self.scale = scale
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = self._load_model()

    def _load_model(self):
        try:
            # Định nghĩa kiến trúc mạng
            model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                 num_block=23, num_grow_ch=32, scale=self.scale)
            # Khởi tạo API hộp đen
            return RealESRGANer(
                scale=self.scale,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model_arch,
                tile=0,
                half=self.use_gpu
            )
        except Exception as e:
            print(f"[Real-ESRGAN] Lỗi tải mô hình: {e}")
            return None

    def predict(self, frame):
        if self.model is None:
            return None
        output, _ = self.model.enhance(frame, outscale=self.scale)
        return output