
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data/raw/Brittle/100Cr6_AR_05_11.png"
MASK_PATH = (
    ROOT / "data/processed/task9_refinement/refined_factor_1_0.png"
)
OUTPUT_DIR = ROOT / "data/processed/task9_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Original image not found: {IMAGE_PATH}")
if mask is None:
    raise FileNotFoundError(f"Candidate mask not found: {MASK_PATH}")
if image.shape != mask.shape:
    raise ValueError("Image and mask dimensions do not match.")

binary = (mask > 0).astype(np.uint8)
count, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

csv_path = OUTPUT_DIR / "shape_analysis_factor_1_0.csv"
overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "component_id", "area_pixels", "width", "height",
        "aspect_ratio", "perimeter_pixels", "circularity",
        "centroid_x", "centroid_y"
    ])

    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        circularity = (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter > 0 else 0
        )
        aspect_ratio = w / h if h > 0 else 0

        writer.writerow([
            i, area, w, h, round(aspect_ratio, 3),
            round(perimeter, 2), round(circularity, 3),
            round(float(centroids[i][0]), 2),
            round(float(centroids[i][1]), 2)
        ])

        # Draw component boundaries for visual review.
        cv2.drawContours(
            overlay, contours, -1, (0, 255, 255), 1,
            offset=(0, 0)
        )

overlay_path = OUTPUT_DIR / "shape_analysis_overlay.png"
cv2.imwrite(str(overlay_path), overlay)

areas = stats[1:, cv2.CC_STAT_AREA]
print(f"Components analyzed: {count - 1}")
print(f"Components under 10 pixels: {int((areas < 10).sum())}")
print(f"Components from 10 to 49 pixels: {int(((areas >= 10) & (areas < 50)).sum())}")
print(f"Components from 50 to 199 pixels: {int(((areas >= 50) & (areas < 200)).sum())}")
print(f"Components of 200 pixels or more: {int((areas >= 200).sum())}")
print(f"Largest component area: {int(areas.max()) if len(areas) else 0}")
print(f"CSV report: {csv_path}")
print(f"Overlay: {overlay_path}")
print("Shape measurements describe candidates, not verified physical voids.")
