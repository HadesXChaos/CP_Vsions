import cv2
import time
from super_resolution import SuperResolutionModule

def test_sr():
    img_path = "../../data/processed/denoised/frame_000001.png"  # Đường dẫn ảnh test của bạn
    img = cv2.imread(img_path)
    if img is None:
        print("Không tìm thấy ảnh test!")
        return

    # Khởi tạo module với scale = 4
    sr = SuperResolutionModule(scale=4, use_gpu=True)

    # 1. Test Bicubic
    start = time.time()
    bicubic_out = sr.upscale_image(img, method='bicubic')
    print(f"Bicubic Runtime: {time.time() - start:.4f}s")

    # 2. Test Real-ESRGAN
    start = time.time()
    ai_out = sr.upscale_image(img, method='real-esrgan')
    print(f"Real-ESRGAN Runtime: {time.time() - start:.4f}s")

    # Lưu kết quả kiểm tra
    cv2.imwrite("test_output_bicubic.jpg", bicubic_out)
    cv2.imwrite("test_output_ai.jpg", ai_out)
    print("Đã xuất ảnh test thành công!")

if __name__ == "__main__":
    test_sr()