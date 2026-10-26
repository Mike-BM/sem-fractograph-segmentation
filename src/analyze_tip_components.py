
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MASK_PATH = ROOT / "data" / "processed" / "paper_void_tips" / "void_tip_mask.png"
OUTPUT_DIR = ROOT / "data" / "processed" / "paper_void_tips"


def main():
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Cannot read tip mask: {MASK_PATH}")

    # Convert the white candidate pixels into a binary mask.
    binary = (mask > 0).astype(np.uint8)

    # Measure connected candidate regions without modifying the mask.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    # Exclude component 0, which represents the background.
    component_stats = stats[1:]
    areas = component_stats[:, cv2.CC_STAT_AREA]

    total_pixels = binary.size
    candidate_pixels = int(binary.sum())

    print(f"Total image pixels: {total_pixels}")
    print(f"Candidate-tip pixels: {candidate_pixels}")
    print(f"Connected components: {count - 1}")

    if len(areas) == 0:
        print("No candidate components found.")
        return

    print(f"Smallest component area: {areas.min()} pixels")
    print(f"Largest component area: {areas.max()} pixels")
    print(f"Median component area: {np.median(areas):.1f} pixels")

    # Summarize the size distribution without deleting anything.
    for minimum_area in [1, 2, 5, 10, 20, 50]:
        selected = areas >= minimum_area
        number = int(selected.sum())
        pixels = int(areas[selected].sum())

        print(
            f"Components with area >= {minimum_area:2}: "
            f"{number:5} components, "
            f"{pixels:6} pixels "
            f"({100 * pixels / total_pixels:.2f}% of image)"
        )

    # Create a diagnostic image showing only components of at least 5 pixels.
    # This is a comparison output, not a validated segmentation.
    filtered = np.zeros_like(binary)

    for component_id in range(1, count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area >= 5:
            filtered[labels == component_id] = 255

    output_path = OUTPUT_DIR / "tip_components_area5.png"
    cv2.imwrite(str(output_path), filtered)

    print(f"Saved comparison mask: {output_path}")


if __name__ == "__main__":
    main()
