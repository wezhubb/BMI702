
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from phasepack import phasecong
import matplotlib.pyplot as plt

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
# img1 = cv2.imread("output_hr.png", cv2.IMREAD_COLOR)  # Reference HR image
# img2 = cv2.imread("dataset/test/hr/portrait/1.png", cv2.IMREAD_COLOR)  # Super-resolved image


def statistics(img1, img2):

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
    # print(f"SSIM: {ssim_score:.4f}")
    # print(f"PSNR: {psnr_score:.2f} dB")
    # print(f"MSE: {mse_score:.2f}")
    # print(f"FSIM: {fsim_score:.4f}")
    return ssim_score, psnr_score, mse_score, fsim_score


input_dir = "dataset/test/hr/portrait/"
output_dir = "dataset/test/result/portrait/"

count = 0
ssim_total = 0
psnr_total = 0
mse_total = 0
fsim_total = 0
result = {}

for filename in os.listdir(output_dir):
    if filename.endswith('.png'):
        out = os.path.join(output_dir, filename)
        ground_truth = os.path.join(input_dir, filename)
        # print(out, ground_truth)
        img1 = cv2.imread(out, cv2.IMREAD_COLOR)
        img2 = cv2.imread(ground_truth, cv2.IMREAD_COLOR)
        ssim_score, psnr_score, mse_score, fsim_score = statistics(img1, img2)
        ssim_total += ssim_score
        psnr_total += psnr_score
        mse_total += mse_score
        fsim_total += fsim_score
        count += 1
        result[filename] = {
            'SSIM': ssim_score,
            'PSNR': psnr_score,
            'MSE': mse_score,
            'FSIM': fsim_score
        }

print(f"Average SSIM: {ssim_total / count:.4f}")
print(f"Average PSNR: {psnr_total / count:.2f} dB")
print(f"Average MSE: {mse_total / count:.2f}")
print(f"Average FSIM: {fsim_total / count:.4f}")

plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)
# Extract metrics from result dict
ssim_scores = [v['SSIM'] for v in result.values()]
psnr_scores = [v['PSNR'] for v in result.values()]
mse_scores = [v['MSE'] for v in result.values()]
fsim_scores = [v['FSIM'] for v in result.values()]

# =====================
# 📦 BOXPLOT
# =====================
plt.figure(figsize=(10, 6))
plt.boxplot([ssim_scores, psnr_scores, mse_scores, fsim_scores],
            labels=['SSIM', 'PSNR', 'MSE', 'FSIM'],
            patch_artist=True)
plt.title('Boxplot of Image Quality Metrics')
plt.ylabel('Metric Value')
plt.grid(True)
plt.tight_layout()
boxplot_path = os.path.join(plot_dir, "metrics_boxplot.png")
plt.savefig(boxplot_path)
plt.close()
print(f"Boxplot saved to: {boxplot_path}")

# =====================
# 🎯 PSNR vs SSIM SCATTERPLOT
# =====================
plt.figure(figsize=(8, 6))
plt.scatter(ssim_scores, psnr_scores, alpha=0.7)
plt.title('PSNR vs SSIM Scatter Plot')
plt.xlabel('SSIM')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.tight_layout()
scatter_path = os.path.join(plot_dir, "psnr_vs_ssim_scatter.png")
plt.savefig(scatter_path)
plt.close()
print(f"Scatter plot saved to: {scatter_path}")