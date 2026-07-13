"""
metrics.py
----------
Module tính toán chỉ số đánh giá chất lượng ảnh cho pipeline Video Super Resolution.

Chức năng:
    - PSNR  (Peak Signal-to-Noise Ratio): Đo độ sai khác pixel giữa ảnh SR và Ground Truth.
    - SSIM  (Structural Similarity Index): Đo độ tương đồng cấu trúc (sáng, tương phản, kết cấu).
    - Runtime: Đo thời gian xử lý của từng module.
    - Hỗ trợ đánh giá so sánh song song 2 nhánh (Bicubic Baseline vs AI Real-ESRGAN).
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


# ============================================================
# Core metrics
# ============================================================

def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """
    Tính PSNR giữa ảnh dự đoán và Ground Truth.

    Args:
        img_pred: Ảnh kết quả (uint8, BGR hoặc grayscale).
        img_gt:   Ảnh Ground Truth (uint8, cùng shape với img_pred).

    Returns:
        Giá trị PSNR (dB). Càng cao càng tốt.
        Trả về float('inf') nếu hai ảnh giống hệt nhau.

    Raises:
        ValueError: Nếu img_pred và img_gt khác shape.
    """
    if img_pred.shape != img_gt.shape:
        raise ValueError(
            f"Shape không khớp: pred={img_pred.shape}, gt={img_gt.shape}"
        )

    pred = img_pred.astype(np.float64)
    gt   = img_gt.astype(np.float64)

    mse = np.mean((pred - gt) ** 2)
    if mse == 0.0:
        return float("inf")

    max_pixel = 255.0
    return 20.0 * np.log10(max_pixel / np.sqrt(mse))


def compute_ssim(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """
    Tính SSIM giữa ảnh dự đoán và Ground Truth.

    Triển khai theo công thức Wang et al. (2004) với:
        C1 = (0.01 * 255)^2, C2 = (0.03 * 255)^2
    Chuyển sang grayscale trước khi tính (theo chuẩn đánh giá SR phổ biến).

    Args:
        img_pred: Ảnh kết quả (uint8, BGR hoặc grayscale).
        img_gt:   Ảnh Ground Truth (uint8, cùng shape với img_pred).

    Returns:
        Giá trị SSIM trong [0, 1]. Càng cao càng tốt.

    Raises:
        ValueError: Nếu img_pred và img_gt khác shape.
    """
    if img_pred.shape != img_gt.shape:
        raise ValueError(
            f"Shape không khớp: pred={img_pred.shape}, gt={img_gt.shape}"
        )

    # Chuyển sang grayscale nếu là ảnh màu
    if img_pred.ndim == 3:
        pred_gray = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gt_gray   = cv2.cvtColor(img_gt,   cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        pred_gray = img_pred.astype(np.float64)
        gt_gray   = img_gt.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1    = cv2.GaussianBlur(pred_gray, (11, 11), 1.5)
    mu2    = cv2.GaussianBlur(gt_gray,   (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(pred_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gt_gray   ** 2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(pred_gray * gt_gray, (11, 11), 1.5) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


# ============================================================
# Per-image evaluation
# ============================================================

def evaluate_image(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    runtime: Optional[float] = None,
) -> Dict[str, float]:
    """
    Đánh giá một cặp ảnh (dự đoán, ground truth).

    Args:
        img_pred:  Ảnh dự đoán (uint8).
        img_gt:    Ảnh Ground Truth (uint8, cùng shape).
        runtime:   Thời gian xử lý (giây) của bước sinh ra img_pred (tùy chọn).

    Returns:
        dict với các key: "psnr", "ssim", và tùy chọn "runtime".
    """
    # Resize gt về cùng kích thước pred nếu khác nhau (vd: GT gốc vs SR 4x)
    if img_pred.shape != img_gt.shape:
        img_gt = cv2.resize(
            img_gt,
            (img_pred.shape[1], img_pred.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    result = {
        "psnr": compute_psnr(img_pred, img_gt),
        "ssim": compute_ssim(img_pred, img_gt),
    }
    if runtime is not None:
        result["runtime"] = runtime
    return result


# ============================================================
# Folder-level evaluation (so sánh 2 nhánh song song)
# ============================================================

def evaluate_folder(
    pred_dir: str,
    gt_dir: str,
    label: str = "Method",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Đánh giá toàn bộ thư mục ảnh dự đoán so với Ground Truth.

    Args:
        pred_dir: Thư mục chứa ảnh kết quả (Bicubic hoặc AI SR).
        gt_dir:   Thư mục chứa ảnh Ground Truth có cùng tên file.
        label:    Tên phương pháp (dùng khi in kết quả).
        verbose:  In từng file nếu True.

    Returns:
        dict: {"avg_psnr", "avg_ssim", "avg_runtime", "n_frames"}
    """
    pred_files = sorted(
        f for f in os.listdir(pred_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    psnr_list: List[float] = []
    ssim_list: List[float] = []
    runtime_list: List[float] = []

    for fname in pred_files:
        pred_path = os.path.join(pred_dir, fname)
        gt_path   = os.path.join(gt_dir,   fname)

        if not os.path.exists(gt_path):
            logger.warning(f"Không tìm thấy GT cho: {fname}, bỏ qua.")
            continue

        pred_img = cv2.imread(pred_path)
        gt_img   = cv2.imread(gt_path)

        if pred_img is None or gt_img is None:
            logger.warning(f"Không đọc được ảnh: {fname}")
            continue

        t0 = time.perf_counter()
        metrics = evaluate_image(pred_img, gt_img)
        elapsed = time.perf_counter() - t0

        psnr_list.append(metrics["psnr"])
        ssim_list.append(metrics["ssim"])
        runtime_list.append(elapsed)

        if verbose:
            print(
                f"  {fname}: PSNR={metrics['psnr']:.2f}dB  "
                f"SSIM={metrics['ssim']:.4f}  ({elapsed*1000:.1f}ms)"
            )

    n = len(psnr_list)
    if n == 0:
        logger.warning(f"[{label}] Không có frame nào được đánh giá.")
        return {"avg_psnr": 0.0, "avg_ssim": 0.0, "avg_runtime": 0.0, "n_frames": 0}

    avg_psnr    = sum(psnr_list) / n
    avg_ssim    = sum(ssim_list) / n
    avg_runtime = sum(runtime_list) / n

    if verbose:
        print(f"\n  [{label}] Tổng kết {n} frame:")
        print(f"    Avg PSNR    = {avg_psnr:.2f} dB")
        print(f"    Avg SSIM    = {avg_ssim:.4f}")
        print(f"    Avg Runtime = {avg_runtime*1000:.1f} ms/frame")

    return {
        "avg_psnr":    avg_psnr,
        "avg_ssim":    avg_ssim,
        "avg_runtime": avg_runtime,
        "n_frames":    n,
    }


# ============================================================
# So sánh 2 nhánh song song: Bicubic vs AI
# ============================================================

def compare_methods(
    bicubic_dir: str,
    ai_dir: str,
    gt_dir: str,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    So sánh song song kết quả 2 nhánh: Bicubic Baseline vs AI Real-ESRGAN.

    Args:
        bicubic_dir: Thư mục ảnh kết quả nhánh Bicubic.
        ai_dir:      Thư mục ảnh kết quả nhánh Real-ESRGAN.
        gt_dir:      Thư mục Ground Truth.
        verbose:     In kết quả chi tiết.

    Returns:
        dict: {"bicubic": {...metrics}, "real-esrgan": {...metrics}}
    """
    print("\n" + "=" * 60)
    print("  Đánh giá so sánh: Bicubic Baseline vs Real-ESRGAN")
    print("=" * 60)

    print("\n[Nhánh 1: Bicubic Interpolation]")
    bicubic_metrics = evaluate_folder(bicubic_dir, gt_dir, label="Bicubic", verbose=verbose)

    print("\n[Nhánh 2: Real-ESRGAN AI]")
    ai_metrics = evaluate_folder(ai_dir, gt_dir, label="Real-ESRGAN", verbose=verbose)

    # In bảng so sánh
    print("\n" + "=" * 60)
    print("  Bảng so sánh tổng kết")
    print("=" * 60)
    print(f"  {'Chỉ số':<20} {'Bicubic':>12} {'Real-ESRGAN':>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*14}")
    print(f"  {'PSNR (dB)':<20} {bicubic_metrics['avg_psnr']:>12.2f} {ai_metrics['avg_psnr']:>14.2f}")
    print(f"  {'SSIM':<20} {bicubic_metrics['avg_ssim']:>12.4f} {ai_metrics['avg_ssim']:>14.4f}")
    print(f"  {'Runtime (ms/frame)':<20} {bicubic_metrics['avg_runtime']*1000:>12.1f} {ai_metrics['avg_runtime']*1000:>14.1f}")
    print("=" * 60)

    return {
        "bicubic":     bicubic_metrics,
        "real-esrgan": ai_metrics,
    }


# ============================================================
# CLI standalone
# ============================================================

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Đánh giá chất lượng SR: PSNR, SSIM, Runtime (Bicubic vs Real-ESRGAN)"
    )
    parser.add_argument("--bicubic-dir", required=True,
                        help="Thư mục kết quả Bicubic SR")
    parser.add_argument("--ai-dir",      required=True,
                        help="Thư mục kết quả Real-ESRGAN SR")
    parser.add_argument("--gt-dir",      required=True,
                        help="Thư mục Ground Truth")
    parser.add_argument("--verbose",     action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compare_methods(
        bicubic_dir=args.bicubic_dir,
        ai_dir=args.ai_dir,
        gt_dir=args.gt_dir,
        verbose=args.verbose,
    )
