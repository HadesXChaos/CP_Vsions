import cv2
from real_esrgan import RealESRGANPredictor  # Import từ file anh em cùng thư mục

class SuperResolutionModule:
    def __init__(self, scale=4, use_gpu=True):
        self.scale = scale
        # Khởi tạo sẵn nhánh AI
        self.ai_upsampler = RealESRGANPredictor(scale=scale, use_gpu=use_gpu)

    def upscale_image(self, frame, method='real-esrgan'):
        """Luồng chính cho ảnh RGB (Tùy chọn Bicubic hoặc AI)"""
        method = method.lower()
        
        if method == 'bicubic':
            h, w = frame.shape[:2]
            return cv2.resize(frame, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_CUBIC)
            
        elif method == 'real-esrgan':
            ai_output = self.ai_upsampler.predict(frame)
            if ai_output is not None:
                return ai_output
            # Fallback nếu AI lỗi
            return cv2.resize(frame, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_CUBIC)
            
        else:
            raise ValueError("Chỉ hỗ trợ 'bicubic' hoặc 'real-esrgan'")

    def upscale_map_nearest(self, map_img):
        """Hàm dùng riêng cho Edge Map và Segmentation Map"""
        h, w = map_img.shape[:2]
        return cv2.resize(map_img, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_NEAREST)