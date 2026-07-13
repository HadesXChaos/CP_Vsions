# -*- coding: utf-8 -*-
"""
test_pipeline.py
================
Smoke-test toàn bộ pipeline Video Super Resolution.

Cach chay (tu bat ky thu muc nao):
    python VideoSuperResolution/src/test_pipeline.py

Mac dinh dung video  VideoSuperResolution/data/raw/corridor.mp4  (5 frame, bicubic SR).
Dung --sr-method real-esrgan de test nhanh AI.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import io
import cv2
import numpy as np

# Đảm bảo stdout dùng UTF-8 trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Màu terminal ────────────────────────────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"

OK   = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
SKIP = f"{YELLOW}[SKIP]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def header(title: str):
    bar = "-" * 60
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{CYAN}  {title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")


def check(label: str, fn, *args, **kwargs):
    """Chạy fn, in PASS/FAIL, trả về (ok, result)."""
    try:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  {OK} {label}  ({elapsed:.2f}s)")
        return True, result
    except Exception as exc:
        print(f"  {FAIL} {label}")
        print(f"       {RED}{exc}{RESET}")
        traceback.print_exc()
        return False, None


# ────────────────────────────────────────────────────────────────
# Test từng module
# ────────────────────────────────────────────────────────────────

def test_imports(skip_sr: bool = False):
    header("1. Import kiem tra")
    # Modules yeu cau torch - chi test khi khong --skip-sr
    torch_modules = {"super_resolution.super_resolution"}

    modules = {
        "cv2":                       lambda: cv2.__version__,
        "numpy":                     lambda: np.__version__,
        "extraction.extract_frames": lambda: __import__("extraction.extract_frames", fromlist=["extract_frames"]),
        "preprocessing.denoise":     lambda: __import__("preprocessing.denoise",     fromlist=["Denoiser"]),
        "feature.run_canny":         lambda: __import__("feature.run_canny",         fromlist=["run_canny_pipeline"]),
        "postprocessing.run_sharpening":lambda: __import__("postprocessing.run_sharpening",fromlist=["run_sharpen_pipeline"]),
        "segmentation.mean_shift":   lambda: __import__("segmentation.mean_shift",   fromlist=["MeanShiftSegmenter"]),
        "super_resolution.super_resolution": lambda: __import__(
            "super_resolution.super_resolution", fromlist=["SuperResolutionModule"]),
        "utils.merge_video":         lambda: __import__("utils.merge_video",         fromlist=["merge_frames"]),
    }
    results = {}
    all_ok = True
    for name, fn in modules.items():
        if skip_sr and name in torch_modules:
            print(f"  {SKIP} import {name}  (--skip-sr, torch not required)")
            results[name] = None
            continue
        ok, _ = check(f"import {name}", fn)
        results[name] = ok
        if not ok:
            all_ok = False
    return all_ok, results


def test_extract(video_path: Path, out_dir: Path, max_frames: int = 5):
    header("2. Extract Frames")
    from extraction.extract_frames import extract_frames

    out_dir.mkdir(parents=True, exist_ok=True)

    ok, metadata = check(
        f"extract_frames  (max={max_frames} frame)",
        extract_frames,
        video_path=str(video_path),
        output_dir=str(out_dir),
        frame_step=1,
        max_frames=max_frames,
        ext="png",
    )
    if not ok:
        return False, None

    frames = sorted(out_dir.glob("frame_*.png"))
    ok2 = len(frames) == max_frames
    status = OK if ok2 else FAIL
    print(f"  {status} Số frame đã lưu: {len(frames)} / {max_frames} mong đợi")

    meta_file = out_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  {INFO} fps={meta.get('fps')}, {meta.get('width')}x{meta.get('height')}, extracted={meta.get('extracted_frames')}")

    return ok and ok2, frames


def test_denoise(frames_dir: Path, out_dir: Path, method="bilateral"):
    header("3. Denoise")
    from preprocessing.denoise import Denoiser

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(frames_dir.glob("frame_*.png"))

    denoiser = Denoiser(method=method)
    errors = 0
    t0 = time.perf_counter()
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  {FAIL} Không đọc được: {f.name}")
            errors += 1
            continue
        result = denoiser.process(img)
        cv2.imwrite(str(out_dir / f.name), result)

    elapsed = time.perf_counter() - t0
    ok = errors == 0
    status = OK if ok else FAIL
    print(f"  {status} Denoise {len(files)} frame  [{method}]  ({elapsed:.2f}s, lỗi={errors})")

    if files:
        orig     = cv2.imread(str(files[0]))
        denoised = cv2.imread(str(out_dir / files[0].name))
        shape_ok = orig.shape == denoised.shape
        print(f"  {OK if shape_ok else FAIL} Shape giữ nguyên: {orig.shape} → {denoised.shape}")
        ok = ok and shape_ok

    return ok


def test_canny(frames_dir: Path, out_dir: Path):
    header("4. Canny Edge Detection")
    from feature.run_canny import run_canny_pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    ok, _ = check(
        "run_canny_pipeline",
        run_canny_pipeline,
        input_dir=str(frames_dir),
        output_dir=str(out_dir),
        overlay_dir=None,
        low_threshold=50,
        high_threshold=150,
        blur_kernel=5,
        l2_gradient=False,
    )
    if not ok:
        return False

    edges = list(out_dir.glob("*.png"))
    ok2 = len(edges) > 0
    print(f"  {OK if ok2 else FAIL} Edge map đã lưu: {len(edges)} file")

    if edges:
        e = cv2.imread(str(edges[0]), cv2.IMREAD_GRAYSCALE)
        has_edges = e is not None and e.max() > 0
        print(f"  {OK if has_edges else FAIL} Edge map có pixel trắng (phát hiện cạnh): max={e.max() if e is not None else 'N/A'}")
        ok2 = ok2 and has_edges

    return ok and ok2


def test_sr(frames_dir: Path, out_dir: Path, method="bicubic", scale=4, use_gpu=False):
    header(f"5. Super Resolution  [{method}]")
    from super_resolution.super_resolution import SuperResolutionModule

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(frames_dir.glob("frame_*.png"))

    ok_all = True
    t0 = time.perf_counter()
    try:
        sr = SuperResolutionModule(scale=scale, use_gpu=use_gpu)
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w = img.shape[:2]
            result = sr.upscale_image(img, method=method)
            expected_h, expected_w = h * scale, w * scale
            size_ok = (result.shape[0] == expected_h and result.shape[1] == expected_w)
            if not size_ok:
                print(f"  {FAIL} {f.name}: mong {expected_w}x{expected_h}, nhận {result.shape[1]}x{result.shape[0]}")
                ok_all = False
            cv2.imwrite(str(out_dir / f.name), result)
        elapsed = time.perf_counter() - t0
        print(f"  {OK if ok_all else FAIL} SR {len(files)} frame  ({elapsed:.2f}s)")
    except Exception as exc:
        print(f"  {FAIL} Super Resolution lỗi: {exc}")
        traceback.print_exc()
        ok_all = False

    return ok_all


def test_sharpen(denoised_dir: Path, edges_dir: Path, out_dir: Path):
    header("6. Edge-aware Sharpening")
    from postprocessing.run_sharpening import run_sharpen_pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    ok, _ = check(
        "run_sharpen_pipeline",
        run_sharpen_pipeline,
        denoised_dir=str(denoised_dir),
        edges_dir=str(edges_dir),
        output_dir=str(out_dir),
        alpha=1.0,
        blur_kernel=5,
        edge_dilate=2,
        w_edge=0.7,
        w_region=0.3,
    )
    if ok:
        sharpened = list(out_dir.glob("*.png"))
        print(f"  {OK} Sharpened frame: {len(sharpened)} file")
    return ok


def test_merge(frames_dir: Path, output_video: Path, fps: float = 25.0):
    header("7. Merge Frames -> Video")
    from utils.merge_video import merge_frames

    output_video.parent.mkdir(parents=True, exist_ok=True)
    ok, result_path = check(
        "merge_frames",
        merge_frames,
        frame_dir=str(frames_dir),
        output_path=str(output_video),
        fps=fps,
    )
    if ok:
        # merge_frames tra ve duong dan thuc te (co the doi extension)
        actual_path = Path(result_path) if result_path else output_video
        exists = actual_path.exists()
        size = actual_path.stat().st_size if exists else 0
        print(f"  {OK if exists else FAIL} Output video: {actual_path}  ({size/1024:.1f} KB)")
        ok = ok and exists and size > 0
    return ok



# ────────────────────────────────────────────────────────────────
# Tổng kết
# ────────────────────────────────────────────────────────────────

def summary(results: dict):
    header("Tổng kết")
    for name, ok in results.items():
        if ok is None:
            print(f"  {SKIP}  {name}")
        else:
            print(f"  {OK if ok else FAIL}  {name}")
    print()
    counted = {k: v for k, v in results.items() if v is not None}
    passed  = sum(counted.values())
    total   = len(counted)
    color   = GREEN if passed == total else RED
    print(f"  {color}{passed}/{total} bước PASS{RESET}")
    return passed == total


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

# Default video path anchored to script location (works from any cwd)
_DEFAULT_VIDEO = str(
    Path(__file__).resolve().parent.parent / "data" / "raw" / "corridor.mp4"
)

def parse_args():
    p = argparse.ArgumentParser(description="Smoke-test pipeline Video SR")
    p.add_argument("--video",        default=_DEFAULT_VIDEO,
                   help="Video đầu vào để test")
    p.add_argument("--max-frames",   type=int, default=5,
                   help="Số frame tối đa cần extract (mặc định: 5)")
    p.add_argument("--sr-method",    default="bicubic",
                   choices=["bicubic", "real-esrgan"],
                   help="Phương pháp SR (mặc định: bicubic)")
    p.add_argument("--scale",        type=int, default=4)
    p.add_argument("--use-gpu",      action="store_true")
    p.add_argument("--skip-sr",      action="store_true",
                   help="Bỏ qua bước SR")
    p.add_argument("--skip-sharpen", action="store_true",
                   help="Bỏ qua bước Sharpen")
    p.add_argument("--skip-merge",   action="store_true",
                   help="Bỏ qua bước ghép video")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    # Dùng đường dẫn tuyệt đối tính từ vị trí script
    # để tránh sai khi chạy từ thư mục khác
    SRC_DIR = Path(__file__).resolve().parent          # .../VideoSuperResolution/src
    BASE_DIR = SRC_DIR.parent / "data" / "_test_run"  # .../VideoSuperResolution/data/_test_run
    base = BASE_DIR

    frames_dir    = base / "frames"
    denoised_dir  = base / "denoised"
    edges_dir     = base / "edges"
    sr_dir        = base / "sr"
    sharpened_dir = base / "sharpened"
    output_video  = base / "output" / "test_output.mp4"

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  Video SR Pipeline - Smoke Test{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"  {INFO} Video      : {video_path}")
    print(f"  {INFO} Max frames : {args.max_frames}")
    print(f"  {INFO} SR method  : {args.sr_method}")

    if not video_path.exists():
        print(f"\n  {FAIL} Không tìm thấy video: {video_path}")
        print(f"  {YELLOW}  Dùng --video <path> để chỉ định video khác.{RESET}")
        sys.exit(1)

    results = {}

    # 1. Imports
    ok_imports, _ = test_imports(skip_sr=args.skip_sr)
    results["Imports"] = ok_imports
    if not ok_imports:
        print(f"\n  {RED}Import thất bại → dừng sớm.{RESET}")
        summary(results)
        sys.exit(1)

    # 2. Extract
    ok_ext, frames = test_extract(video_path, frames_dir, args.max_frames)
    results["Extract Frames"] = ok_ext
    if not ok_ext or not frames:
        summary(results)
        sys.exit(1)

    fps = 25.0
    meta_file = frames_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            fps = float(json.load(f).get("fps", 25.0))

    # 3. Denoise
    results["Denoise (bilateral)"] = test_denoise(frames_dir, denoised_dir, method="bilateral")

    # 4. Canny
    results["Canny Edge"] = test_canny(denoised_dir, edges_dir)

    # 5. SR
    if args.skip_sr:
        print(f"\n  {SKIP} SR bị bỏ qua (--skip-sr)")
        results["Super Resolution"] = None
        sr_dir = denoised_dir
    else:
        results["Super Resolution"] = test_sr(
            denoised_dir, sr_dir,
            method=args.sr_method,
            scale=args.scale,
            use_gpu=args.use_gpu,
        )

    # 6. Sharpen
    if args.skip_sharpen:
        print(f"\n  {SKIP} Sharpen bị bỏ qua (--skip-sharpen)")
        results["Sharpen"] = None
        final_dir = sr_dir
    else:
        results["Sharpen"] = test_sharpen(sr_dir, edges_dir, sharpened_dir)
        final_dir = sharpened_dir

    # 7. Merge
    if args.skip_merge:
        print(f"\n  {SKIP} Merge bị bỏ qua (--skip-merge)")
        results["Merge Video"] = None
    else:
        results["Merge Video"] = test_merge(final_dir, output_video, fps=fps)

    all_ok = summary(results)
    print()
    if all_ok:
        print(f"  {GREEN}✓ Pipeline hoạt động tốt!{RESET}\n")
    else:
        print(f"  {RED}✗ Có bước bị lỗi, xem log bên trên.{RESET}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
