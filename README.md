# Video Super Resolution using Classical Computer Vision and AI

## Đề tài

**Khôi phục chất lượng video độ phân giải thấp bằng pipeline xử lý ảnh và AI Super Resolution**

## Giới thiệu

Nhiều video thu được từ camera giám sát, điện thoại đời cũ hoặc video nén có chất lượng thấp, xuất hiện hiện tượng nhiễu, mờ và mất chi tiết. Điều này làm giảm khả năng quan sát cũng như ảnh hưởng đến các bài toán thị giác máy tính phía sau.

Dự án này xây dựng một pipeline kết hợp giữa các kỹ thuật xử lý ảnh truyền thống và mô hình AI Super Resolution nhằm cải thiện chất lượng video.

Thay vì chỉ sử dụng một mô hình Deep Learning như một "hộp đen", dự án xây dựng toàn bộ quy trình xử lý ảnh, trong đó mô hình AI chỉ là một thành phần của pipeline.

---

# Mục tiêu

* Khử nhiễu video đầu vào.
* Bảo toàn biên và chi tiết quan trọng.
* Tăng độ phân giải cho từng frame.
* Cải thiện độ sắc nét của video đầu ra.
* Đánh giá kết quả bằng các chỉ số định lượng.

---

# Pipeline

```
                         Video
                           │
                           ▼
                    Extract Frames
                           │
                           ▼
             Gaussian / Bilateral Denoising
                           │
                           ▼
                 Denoised Frame (RGB)
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
      ▼                    ▼                    ▼
Canny Edge Detection  Mean Shift Segmentation  Bicubic / Real-ESRGAN
      │                    │                    │
      ▼                    ▼                    ▼
  Edge Map         Segmentation Map        SR Frame
      └────────────────────┼────────────────────┘
                           ▼
            Edge & Region-aware Sharpening
                           │
                           ▼
         PSNR + SSIM + Runtime Evaluation
                           │
                           ▼
               Merge Frames to Video
```
---

# Giải thích từng bước

## 1. Extract Frames

Video được tách thành các frame riêng biệt để thuận tiện cho việc xử lý từng ảnh.

Output:

```
frame0001.png
frame0002.png
...
```

---

## 2. Gaussian / Bilateral Denoising

Khử nhiễu trước khi tăng độ phân giải.

Hai phương pháp sẽ được khảo sát:

* Gaussian Blur
* Bilateral Filter

Mục tiêu:

* giảm nhiễu
* vẫn giữ được biên

Các tham số sẽ được khảo sát:

* kernel size
* sigma

---

## 3. Canny Edge Detection

Phát hiện các cạnh quan trọng của ảnh.

Kết quả biên được sử dụng để:

* hỗ trợ bước làm sắc nét
* phân tích ảnh trung gian
* khảo sát ảnh hưởng của ngưỡng Canny

Các tham số khảo sát:

* low threshold
* high threshold

---

## 4. Mean Shift Segmentation

Phân đoạn ảnh thành các vùng có màu sắc và kết cấu tương đồng.

Mục đích:

* xác định vùng đối tượng
* hỗ trợ bước tăng cường ảnh
* giảm ảnh hưởng của nền

Các tham số khảo sát:

* spatial radius
* color radius

---

## 5. Real-ESRGAN Super Resolution

Sử dụng mô hình Real-ESRGAN để tăng độ phân giải cho từng frame.

Đây là thành phần AI duy nhất trong pipeline.

Input:

* ảnh sau tiền xử lý

Output:

* ảnh độ phân giải cao

---

## 6. Edge-aware Sharpening

Sau khi tăng độ phân giải, ảnh thường bị mềm hoặc mất độ sắc nét.

Bước này sử dụng thông tin cạnh để:

* tăng độ sắc nét
* hạn chế tạo halo
* giữ chi tiết biên

Các phương pháp dự kiến khảo sát:

* Unsharp Mask
* Laplacian Sharpening
* Edge-aware Sharpening

---

## 7. Evaluation

Kết quả được đánh giá bằng:

* PSNR
* SSIM
* Runtime

Ngoài ra sẽ so sánh trực quan:

* Input
* Sau khử nhiễu
* Sau Super Resolution
* Kết quả cuối cùng

---

## 8. Merge Video

Ghép toàn bộ frame đã xử lý thành video hoàn chỉnh.

Output:

```
output.mp4
```

---

# Cấu trúc dự án

```
VideoSuperResolution/

│── data/
│   ├── raw/
│   ├── frames/
│   ├── processed/
│   └── output/
│
│── models/
│
│── notebooks/
│
│── src/
│   ├── extraction/
│   ├── preprocessing/
│   ├── feature/
│   ├── segmentation/
│   ├── super_resolution/
│   ├── postprocessing/
│   ├── evaluation/
│   └── utils/
│
│── reports/
│
│── requirements.txt
│── README.md
│── main.py
```

---

# Thư viện sử dụng

* Python 3.11+
* OpenCV
* NumPy
* Matplotlib
* Scikit-image
* SciPy
* PyTorch
* Real-ESRGAN

---

# Cài đặt

Tạo môi trường ảo:

```
python -m venv .venv
```

Kích hoạt:

Windows

```
.venv\Scripts\activate
```

Linux/Mac

```
source .venv/bin/activate
```

Cài thư viện:

```
pip install -r requirements.txt
```

---

# Chạy chương trình

```
python main.py
```

Hoặc chạy từng notebook trong thư mục `notebooks/`.

---

# Kết quả mong đợi

* Video có độ phân giải cao hơn.
* Giảm nhiễu so với đầu vào.
* Biên rõ nét hơn.
* PSNR và SSIM được cải thiện so với video gốc.
* Toàn bộ pipeline có thể chạy lại trên dữ liệu mới.

---

# Công việc dự kiến

* [ ] Xây dựng module Extract Frames
* [ ] Xây dựng module Denoising
* [ ] Xây dựng module Canny
* [ ] Xây dựng module Mean Shift
* [ ] Tích hợp Real-ESRGAN
* [ ] Xây dựng Edge-aware Sharpening
* [ ] Đánh giá PSNR, SSIM
* [ ] Xuất video kết quả
