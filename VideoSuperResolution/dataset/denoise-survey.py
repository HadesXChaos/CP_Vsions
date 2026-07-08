import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from denoise import Denoiser


def evaluate(gt_img, test_img):

    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    p = psnr(gt_gray, test_gray)
    s = ssim(gt_gray, test_gray)

    return p, s


def survey_gaussian(noisy_img, gt_img, kernels=(3, 5, 7, 9), sigmas=(0, 1, 2)):

    results = []

    for k in kernels:
        for sigma in sigmas:

            denoiser = Denoiser(
                method="gaussian",
                gaussian_kernel=k,
                gaussian_sigma=sigma
            )

            out = denoiser.process(noisy_img)

            p, s = evaluate(gt_img, out)

            results.append({
                "kernel": k,
                "sigma": sigma,
                "psnr": p,
                "ssim": s
            })

    return sorted(results, key=lambda r: r["ssim"], reverse=True)


def survey_bilateral(noisy_img, gt_img, d_list=(5, 9, 15), sigma_list=(25, 50, 75, 100)):

    results = []

    for d in d_list:
        for sigma in sigma_list:

            denoiser = Denoiser(
                method="bilateral",
                bilateral_d=d,
                bilateral_sigma_color=sigma,
                bilateral_sigma_space=sigma
            )

            out = denoiser.process(noisy_img)

            p, s = evaluate(gt_img, out)

            results.append({
                "d": d,
                "sigma": sigma,
                "psnr": p,
                "ssim": s
            })

    return sorted(results, key=lambda r: r["ssim"], reverse=True)