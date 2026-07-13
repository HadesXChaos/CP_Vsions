import cv2


class SuperResolutionModule:
    def __init__(self, scale=4, use_gpu=False):
        self.scale = scale
        self.use_gpu = use_gpu
        # Lazy-load: AI model chi duoc khoi tao khi thuc su can
        # Tranh import torch khi chi dung bicubic
        self._ai_upsampler = None
        self._ai_loaded = False

    def _get_ai_upsampler(self):
        """Khoi tao Real-ESRGAN model lan dau tien khi can."""
        if not self._ai_loaded:
            try:
                from .real_esrgan import RealESRGANPredictor
                self._ai_upsampler = RealESRGANPredictor(
                    scale=self.scale, use_gpu=self.use_gpu
                )
            except ImportError as e:
                print(f"[SR] Khong the tai Real-ESRGAN (thieu torch?): {e}")
                self._ai_upsampler = None
            except Exception as e:
                print(f"[SR] Loi khoi tao Real-ESRGAN: {e}")
                self._ai_upsampler = None
            self._ai_loaded = True
        return self._ai_upsampler

    def upscale_image(self, frame, method='real-esrgan'):
        """Luong chinh cho anh RGB (Tuy chon Bicubic hoac AI)."""
        method = method.lower()
        h, w = frame.shape[:2]

        if method == 'bicubic':
            return cv2.resize(
                frame,
                (int(w * self.scale), int(h * self.scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        elif method == 'real-esrgan':
            ai = self._get_ai_upsampler()
            if ai is not None:
                ai_output = ai.predict(frame)
                if ai_output is not None:
                    return ai_output
            # Fallback neu AI loi hoac khong co torch
            print(f"[SR] Fallback sang bicubic.")
            return cv2.resize(
                frame,
                (int(w * self.scale), int(h * self.scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        else:
            raise ValueError("Chi ho tro 'bicubic' hoac 'real-esrgan'")

    def upscale_map_nearest(self, map_img):
        """Ham dung rieng cho Edge Map va Segmentation Map."""
        h, w = map_img.shape[:2]
        return cv2.resize(
            map_img,
            (int(w * self.scale), int(h * self.scale)),
            interpolation=cv2.INTER_NEAREST,
        )