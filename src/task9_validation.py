
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT / "data/raw/Brittle/100Cr6_AR_05_11.png"
MASK_PATH = ROOT / "data/processed/void_detection/binary_void_mask.png"
OUTPUT_DIR = ROOT / "data/processed/task9_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Original image not found: {IMAGE_PATH}")

if mask is None:
    raise FileNotFoundError(f"Void mask not found: {MASK_PATH}")

if image.shape != mask.shape:
    raise ValueError(
        f"Image and mask dimensions differ: "
        f"{image.shape} vs {mask.shape}"
    )

binary = (mask > 0).astype(np.uint8)
total_pixels = binary.size
candidate_pixels = int(binary.sum())

count, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

areas = stats[1:, cv2.CC_STAT_AREA]
minimum_area = 5
kept = areas[areas >= minimum_area]

overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
overlay[binary == 1] = (0, 0, 255)

overlay_path = OUTPUT_DIR / "validation_overlay.png"
mask_path = OUTPUT_DIR / "validation_mask.png"

cv2.imwrite(str(overlay_path), overlay)
cv2.imwrite(str(mask_path), binary * 255)

print("TASK 9 VALIDATION SUMMARY")
print("------------------------")
print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
print(f"Candidate mask pixels: {candidate_pixels}")
print(f"Candidate mask coverage: {candidate_pixels / total_pixels * 100:.2f}%")
print(f"Connected components including background: {count}")
print(f"Candidate components excluding background: {count - 1}")
print(f"Components with area >= {minimum_area} pixels: {len(kept)}")
print(f"Smallest retained component: {int(kept.min()) if len(kept) else 0} pixels")
print(f"Largest retained component: {int(kept.max()) if len(kept) else 0} pixels")
print(f"Overlay saved to: {overlay_path}")
print(f"Mask saved to: {mask_path}")
print()
print("NOTE: These checks assess mask structure, not scientific accuracy.")
print("Visual comparison with the original SEM image is still required.")
