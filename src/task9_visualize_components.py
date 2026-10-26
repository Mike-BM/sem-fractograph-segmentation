
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

if image is None or mask is None:
    raise FileNotFoundError("Could not load the original image or void mask.")

if image.shape != mask.shape:
    raise ValueError("Original image and mask dimensions do not match.")

binary = (mask > 0).astype(np.uint8)
count, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for i in range(1, count):
    area = int(stats[i, cv2.CC_STAT_AREA])
    x = int(stats[i, cv2.CC_STAT_LEFT])
    y = int(stats[i, cv2.CC_STAT_TOP])
    w = int(stats[i, cv2.CC_STAT_WIDTH])
    h = int(stats[i, cv2.CC_STAT_HEIGHT])

    # Different colors indicate component-size groups, not accuracy.
    if area < 10:
        color = (180, 180, 180)
    elif area < 100:
        color = (0, 220, 255)
    elif area < 500:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    component = labels[y:y+h, x:x+w] == i
    roi = overlay[y:y+h, x:x+w]
    roi[component] = color

    if area >= 500:
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (255, 255, 0), 1)
        cv2.putText(
            overlay, str(i), (x, max(12, y-3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1
        )

output_path = OUTPUT_DIR / "labelled_components.png"
cv2.imwrite(str(output_path), overlay)

print(f"Components found: {count - 1}")
print(f"Labelled image saved to: {output_path}")
print("Grey: <10 px | Yellow: 10-99 px | Orange: 100-499 px | Red: >=500 px")
print("Cyan boxes identify larger components; colors do not indicate correctness.")
