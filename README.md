# Video Super Resolution using Classical Computer Vision and AI

Kho lưu trữ này chứa mã nguồn của dự án **"Khôi phục chất lượng video độ phân giải thấp bằng pipeline xử lý ảnh truyền thống kết hợp AI Super Resolution"**. 

Thay vì chỉ sử dụng các mô hình Deep Learning như một "hộp đen" độc lập, dự án này xây dựng một pipeline xử lý ảnh toàn diện. Trong đó, kỹ thuật thị giác máy tính cổ điển đóng vai trò tiền/hậu xử lý định hướng (Edge & Region-aware), đồng thời tích hợp thuật toán nội suy truyền thống (**Bicubic Interpolation**) để làm mốc đối chiếu (Baseline) đánh giá hiệu quả vượt trội của mô hình AI (**Real-ESRGAN**).

---

## 📌 Đề tài & Đặt vấn đề

Nhiều video thu được từ camera giám sát, điện thoại đời cũ hoặc video nén có chất lượng thấp, xuất hiện hiện tượng nhiễu, mờ và mất chi tiết. Điều này làm giảm khả năng quan sát cũng như ảnh hưởng tiêu cực đến các bài toán thị giác máy tính phía sau (như nhận diện khuôn mặt, biển số xe, theo dõi đối tượng).

**Mục tiêu của dự án:**
- Khử nhiễu video đầu vào nhưng vẫn bảo toàn biên và chi tiết quan trọng.
- Tăng độ phân giải đồng thời bằng 2 nhánh song song để đối chiếu: Nội suy hình học truyền thống (**Bicubic Interpolation**) và Học sâu (**Real-ESRGAN**).
- Cải thiện độ sắc nét dựa trên bản đồ cạnh (Edge map) và phân đoạn vùng (Segmentation map).
- Đánh giá và so sánh kết quả định lượng giữa các phương pháp bằng các chỉ số tiêu chuẩn ($PSNR$, $SSIM$, $Runtime$).

---

## 🗺️ Kiến trúc Pipeline Hệ thống

Quy trình xử lý tuần tự và song song của từng frame trong hệ thống được mô tả qua sơ đồ dưới đây:

```text
                         Video Đầu Vào
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
      ┌────────────────────────┼────────────────────────┐
      │                        │                        │
      ▼                        ▼                        ▼
Canny Edge Detection    Mean Shift Segmentation    [Bicubic vs Real-ESRGAN]
      │                        │                        │
      │                        │                        │    
Upscale (Nearest)        Upscale (Nearest)              │
      │                        │                        │
      ▼                        ▼                        ▼
   Edge Map             Segmentation Map            SR Frame
      └────────────────────────┼────────────────────────┘
                               ▼
                 Edge & Region-aware Sharpening
                               │
                               ▼
                PSNR + SSIM + Runtime Evaluation
             (Đối chiếu Bicubic Baseline vs AI SR)
                               │
                               ▼
                     Merge Frames to Video
                               │
                               ▼
                         Video Kết Quả

```

---

## 🔍 Chi Tiết Các Bước Xử Lý

1. **Extract Frames:** Tách video thành các frame riêng biệt để thuận tiện cho việc xử lý từng ảnh.
2. **Gaussian / Bilateral Denoising:** Khảo sát hiệu năng giữa Gaussian Blur và Bilateral Filter nhằm giảm nhiễu hạt, nhiễu nén nhưng vẫn cố gắng giữ lại cấu trúc biên nguyên bản. Các tham số khảo sát gồm `kernel size` và `sigma`.
3. **Canny Edge Detection:** Trích xuất ma trận biên (Edge map) để hỗ trợ bước làm nét hậu xử lý và khảo sát ảnh hưởng của ngưỡng Canny (`low threshold`, `high threshold`).
4. **Mean Shift Segmentation:** Phân đoạn ảnh thành các vùng màu sắc đồng nhất (Segmentation map) dựa trên `spatial radius` và `color radius` nhằm hỗ trợ tăng cường ảnh và giảm nhiễu nền.
5. **Super Resolution & Baseline Comparison:** Tiến hành tăng độ phân giải frame ảnh theo 2 nhánh để đối chiếu:
* **Nhánh Baseline:** Thuật toán nội suy hình học truyền thống (**Bicubic Interpolation**).
* **Nhánh AI:** Mô hình học sâu nâng cao (**Real-ESRGAN**) đóng vai trò mắt xích cốt lõi xử lý ảnh sau tiền xử lý sạch nhiễu.


6. **Edge-aware Sharpening:** Khắc phục hiện tượng mờ biên sau siêu phân giải, hạn chế tạo halo và giữ chi tiết biên. Các phương pháp khảo sát gồm: Unsharp Mask, Laplacian Sharpening, Edge-aware Sharpening.
7. **Evaluation:** Đánh giá chất lượng bằng hai chỉ số khách quan $PSNR$ và $SSIM$ của cả 2 phương pháp so với ảnh gốc chất lượng cao (Ground Truth), kết hợp đo lường thời gian thực thi ($Runtime$) của từng module. So sánh trực quan các trạng thái ảnh: Input $\rightarrow$ Denoised $\rightarrow$ Super Resolution $\rightarrow$ Final Result.
8. **Merge Video:** Hợp nhất các frame cấu trúc cao thành video đầu ra hoàn chỉnh định dạng `output.mp4`.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
CP_VSIONS/
└── VideoSuperResolution/
    │
    ├── data/
    │   ├── frames/             # Các frame ảnh được trích xuất từ video
    │   ├── ground_truth/       # Dữ liệu ảnh gốc chất lượng cao dùng để đối chiếu đánh giá
    │   ├── low_resolution/     # Các frame ảnh độ phân giải thấp
    │   ├── processed/          # Các frame ảnh trong quá trình xử lý trung gian
    │   ├── raw/                # Dữ liệu ảnh thô ban đầu
    │   └── videos/             # Chứa các file video đầu vào và video kết quả
    │
    ├── notebook/               # Các file Jupyter Notebook (.ipynb) để thử nghiệm từng bước
    │
    ├── reports/                # Biểu đồ phân tích và báo cáo kết quả khảo sát tham số
    │
    ├── src/                    # Mã nguồn cốt lõi của dự án chia theo module
    │   ├── dataset/            # Module quản lý, phân loại và tải dữ liệu (Dataloader)
    │   ├── evaluation/         # Module tính toán chỉ số PSNR, SSIM, Runtime
    │   ├── extraction/         # Module trích xuất frame từ video và gộp frame thành video
    │   ├── feature/            # Module xử lý đặc trưng và phát hiện cạnh (Canny Edge)
    │   ├── postprocessing/     # Module làm nét thích nghi hậu xử lý (Sharpening)
    │   ├── preprocessing/      # Module khử nhiễu tiền xử lý (Denoising)
    │   ├── segmentation/       # Module phân đoạn ảnh (Mean Shift)
    │   ├── super_resolution/   # Module tích hợp thuật toán Bicubic và AI Real-ESRGAN
    │   └── utils/              # Các hàm bổ trợ cấu hình hệ thống, đọc/ghi file
    │
    ├── main.py                 # File thực thi chính/Giao diện dòng lệnh chạy chương trình
    └── pipeline.py             # Kịch bản định nghĩa luồng luân chuyển dữ liệu toàn hệ thống

```

---

## 🛠️ Công Nghệ & Thư Viện Sử Dụng

Dự án yêu cầu cài đặt môi trường chạy **Python 3.11+** cùng các thư viện cốt lõi:

* **Xử lý ảnh & Nội suy truyền thống:** `OpenCV`, `NumPy`, `SciPy`, `Scikit-image`
* **Học sâu (AI):** `PyTorch`, `Real-ESRGAN`
* **Hiển thị & Trực quan hóa:** `Matplotlib`

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Chương Trình

### 1. Khởi tạo môi trường ảo

**Trên Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate

```

**Trên Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate

```

### 2. Cài đặt các thư viện phụ thuộc

*(Đảm bảo bạn đã tạo file `requirements.txt` ở thư mục gốc trước khi chạy lệnh này)*

```bash
pip install -r requirements.txt

```

*(Lưu ý: Nếu bạn sử dụng GPU để tăng tốc mô hình Real-ESRGAN, hãy đảm bảo cài đặt phiên bản PyTorch tương thích với phiên bản CUDA trên máy của bạn).*

### 3. Thực thi hệ thống

Để kiểm tra hoặc chạy thử nghiệm luồng xử lý (pipeline):

```bash
python pipeline.py

```

Để thực thi toàn bộ project với cấu hình chính thức:

```bash
python main.py

```

Hoặc bạn có thể chạy và khảo sát trực quan từng phân đoạn thuật toán thông qua các tệp tương ứng nằm trong thư mục `notebook/`.

---

## 📈 Kết Quả Mong Đợi & So Sánh Đối Chiếu

* **Độ trực quan:** Ảnh xử lý bằng Real-ESRGAN giữ được chi tiết bề mặt (texture) và độ nét biên vượt trội, không bị hiện tượng mờ nhòe (blur) thô ở các vùng tần số cao như khi xử lý bằng thuật toán nội suy Bicubic thuần túy.
* **Chỉ số định lượng:** Báo cáo xuất ra từ hệ thống sẽ chứng minh bằng biểu đồ rằng giá trị cấu trúc mạng học sâu AI cải thiện điểm số số học $PSNR$ và cấu trúc tương đồng $SSIM$ đáng kể so với mốc đối chiếu Baseline, đổi lại là sự đánh đổi hợp lý về mặt thời gian xử lý ($Runtime$).
* Toàn bộ pipeline có tính linh hoạt cao, có thể tái cấu hình và chạy lại ổn định trên các tập dữ liệu video mới.

---

## 📝 Nhật Ký Tiến Độ (Roadmap)

* [x] Xây dựng module Extract Frames (`src/extraction`)
* [x] Xây dựng module Denoising (Gaussian / Bilateral) (`src/preprocessing`)
* [x] Xây dựng module Canny Edge Detection (`src/feature`)
* [x] Xây dựng module Mean Shift Segmentation (`src/segmentation`)
* [x] Tích hợp thuật toán nội suy Bicubic và mô hình học sâu Real-ESRGAN (`src/super_resolution`)
* [x] Xây dựng bộ lọc thích nghi Edge-aware Sharpening (`src/postprocessing`)
* [x] Quản lý luồng dữ liệu thông qua `src/dataset` và `pipeline.py`
* [x] Viết bộ công cụ đánh giá so sánh song song đối chiếu ($PSNR$, $SSIM$, $Runtime$) (`src/evaluation`)
* [ ] Xuất và đóng gói Video kết quả hoàn chỉnh (cần video đầu vào thực tế)