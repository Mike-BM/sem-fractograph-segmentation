
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "void_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_candidate_voids(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_float = image.astype(np.float32)

    # Estimate local intensity statistics.
    local_mean = cv2.GaussianBlur(image_float, (31, 31), 0)
    local_mean_square = cv2.GaussianBlur(image_float ** 2, (31, 31), 0)

    local_variance = np.maximum(
        local_mean_square - local_mean ** 2, 0
    )
    local_std = np.sqrt(local_variance)

    # Initial exploratory threshold: local mean minus half a
    # local standard deviation, following the paper's threshold form.
    threshold = local_mean - 0.5 * local_std

    # Dark pixels are candidates only, not confirmed voids.
    candidate_mask = (image_float < threshold).astype(np.uint8) * 255

    # Remove isolated pixels with a small morphological opening.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN, kernel
    )

    return image, candidate_mask, cleaned_mask, local_mean, local_std


def main():
    image_path = RAW_DIR / "Brittle" / "100Cr6_AR_05_11.png"

    image, candidates, cleaned, local_mean, local_std = (
        detect_candidate_voids(image_path)
    )

    cv2.imwrite(str(OUTPUT_DIR / "candidate_voids.png"), candidates)
    cv2.imwrite(str(OUTPUT_DIR / "candidate_voids_cleaned.png"), cleaned)

    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Candidate pixels before cleanup: {np.count_nonzero(candidates)}")
    print(f"Candidate pixels after cleanup: {np.count_nonzero(cleaned)}")
    print(f"Candidate pixel percentage: {100 * np.mean(cleaned > 0):.2f}%")
    print(f"Saved masks to: {OUTPUT_DIR}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original SEM image")
    axes[1].imshow(candidates, cmap="gray")
    axes[1].set_title("Candidate dark regions")
    axes[2].imshow(cleaned, cmap="gray")
    axes[2].set_title("After small-noise cleanup")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
