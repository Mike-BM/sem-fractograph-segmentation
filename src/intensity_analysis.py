
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Brittle"
    / "100Cr6_AR_05_11.png"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "intensity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

    mean_intensity = float(np.mean(image))
    std_intensity = float(np.std(image))
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()

    # Find local histogram peaks using a simple neighborhood comparison.
    smoothed_hist = np.convolve(histogram, np.ones(5) / 5, mode="same")
    peaks = [
        i for i in range(1, 255)
        if smoothed_hist[i] >= smoothed_hist[i - 1]
        and smoothed_hist[i] > smoothed_hist[i + 1]
    ]
    peaks = sorted(peaks, key=lambda i: smoothed_hist[i], reverse=True)[:10]
    peaks.sort()

    print(f"Image mean intensity: {mean_intensity:.2f}")
    print(f"Image standard deviation: {std_intensity:.2f}")
    print(f"Brightest histogram peaks (intensity levels): {peaks}")

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(256), histogram)
    plt.xlabel("Grayscale intensity (0–255)")
    plt.ylabel("Pixel count")
    plt.title("SEM Image Intensity Histogram")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "intensity_histogram.png"
    plt.savefig(output_path, dpi=150)
    print(f"Histogram saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
