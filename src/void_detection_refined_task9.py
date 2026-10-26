
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "task9_refinement"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experimental parameters to compare, not validated scientific settings.
LOCAL_WINDOW = 31
THRESHOLD_FACTORS = [0.8, 1.0, 1.2]
MIN_COMPONENT_AREA = 10


def detect_candidate_voids(image_path, factor):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_float = image.astype(np.float32)

    local_mean = cv2.GaussianBlur(
        image_float, (LOCAL_WINDOW, LOCAL_WINDOW), 0
    )
    local_mean_square = cv2.GaussianBlur(
        image_float ** 2, (LOCAL_WINDOW, LOCAL_WINDOW), 0
    )

    local_variance = np.maximum(
        local_mean_square - local_mean ** 2, 0
    )
    local_std = np.sqrt(local_variance)

    # More selective: larger factor requires pixels to be darker
    # than their local mean by a larger fraction of local variation.
    threshold = local_mean - factor * local_std
    candidate = (image_float < threshold).astype(np.uint8) * 255

    # Remove isolated pixels and tiny connected components.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    opened = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, kernel
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (opened > 0).astype(np.uint8), connectivity=8
    )

    filtered = np.zeros_like(opened)

    for component_id in range(1, count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            filtered[labels == component_id] = 255

    return image, candidate, filtered


def main():
    image_path = RAW_DIR / "Brittle" / "100Cr6_AR_05_11.png"

    results = []

    for factor in THRESHOLD_FACTORS:
        image, raw_mask, refined_mask = detect_candidate_voids(
            image_path, factor
        )

        tag = str(factor).replace(".", "_")
        cv2.imwrite(
            str(OUTPUT_DIR / f"candidates_factor_{tag}.png"),
            raw_mask,
        )
        cv2.imwrite(
            str(OUTPUT_DIR / f"refined_factor_{tag}.png"),
            refined_mask,
        )

        pixels = int(np.count_nonzero(refined_mask))
        percentage = 100 * pixels / refined_mask.size

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (refined_mask > 0).astype(np.uint8), connectivity=8
        )
        component_count = count - 1

        results.append((factor, pixels, percentage, component_count))

        print(
            f"Factor={factor:.1f} | pixels={pixels} | "
            f"coverage={percentage:.2f}% | "
            f"components={component_count}"
        )

    # Side-by-side comparison for visual inspection.
    fig, axes = plt.subplots(
        1, len(THRESHOLD_FACTORS) + 1, figsize=(16, 5)
    )
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original SEM")

    for index, factor in enumerate(THRESHOLD_FACTORS, start=1):
        tag = str(factor).replace(".", "_")
        mask = cv2.imread(
            str(OUTPUT_DIR / f"refined_factor_{tag}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        axes[index].imshow(mask, cmap="gray")
        axes[index].set_title(f"Factor {factor}")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    comparison_path = OUTPUT_DIR / "threshold_comparison.png"
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"\nComparison saved to: {comparison_path}")
    print("All masks are experimental candidate detections, not verified voids.")


if __name__ == "__main__":
    main()
