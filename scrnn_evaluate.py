

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from phasepack import phasecong

# ✅ Function to Compute FSIM Manually
def compute_fsim(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Compute phase congruency (PC)
    pc1 = phasecong(gray1)[0]
    pc2 = phasecong(gray2)[0]

    # Compute gradient magnitude similarity (G)
    gradient1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1, ksize=3)
    gradient2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1, ksize=3)

    # Compute similarity maps
    T = 0.85  # Threshold for similarity
    similarity_map = (2 * pc1 * pc2 + T) / (pc1 ** 2 + pc2 ** 2 + T)
    gradient_map = (2 * gradient1 * gradient2 + T) / (gradient1 ** 2 + gradient2 ** 2 + T)

    # Compute FSIM
    fsim_score = np.sum(similarity_map * gradient_map) / np.sum(similarity_map)
    return fsim_score

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")  # Perfect match
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ✅ Load images (ensure same dimensions)
img1 = cv2.imread("output_hr.png", cv2.IMREAD_COLOR)  # Reference HR image
img2 = cv2.imread("dataset/test/hr/portrait/1.png", cv2.IMREAD_COLOR)  # Super-resolved image


# ✅ Convert to grayscale (for SSIM and FSIM)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ✅ Compute SSIM
ssim_score, _ = ssim(gray1, gray2, full=True)

# ✅ Compute PSNR
psnr_score = compute_psnr(img1, img2)

# ✅ Compute MSE
mse_score = mean_squared_error(gray1, gray2)

# ✅ Compute FSIM (from sewar)
fsim_score = compute_fsim(img1, img2)

# ✅ Print results
print(f"SSIM: {ssim_score:.4f}")
print(f"PSNR: {psnr_score:.2f} dB")
print(f"MSE: {mse_score:.2f}")
print(f"FSIM: {fsim_score:.4f}")