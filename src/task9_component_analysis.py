
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MASK_PATH = ROOT / "data/processed/void_detection/binary_void_mask.png"
OUTPUT_DIR = ROOT / "data/processed/task9_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

binary = (mask > 0).astype(np.uint8)
count, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

csv_path = OUTPUT_DIR / "component_analysis.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "component_id", "area_pixels", "x", "y",
        "width_pixels", "height_pixels", "centroid_x", "centroid_y"
    ])

    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        writer.writerow([
            i, area, x, y, w, h,
            round(float(centroids[i][0]), 2),
            round(float(centroids[i][1]), 2)
        ])

areas = stats[1:, cv2.CC_STAT_AREA]

print(f"Components analyzed: {len(areas)}")
print(f"Components under 10 pixels: {int((areas < 10).sum())}")
print(f"Components from 10 to 99 pixels: {int(((areas >= 10) & (areas < 100)).sum())}")
print(f"Components from 100 to 499 pixels: {int(((areas >= 100) & (areas < 500)).sum())}")
print(f"Components from 500 to 999 pixels: {int(((areas >= 500) & (areas < 1000)).sum())}")
print(f"Components of 1000 pixels or more: {int((areas >= 1000).sum())}")
print(f"Report saved to: {csv_path}")
