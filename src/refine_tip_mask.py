
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MASK_PATH = (
    ROOT / "data" / "processed" / "paper_void_tips"
    / "void_tip_mask.png"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "paper_void_tips"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

binary = (mask > 0).astype(np.uint8)

# Step 1: Remove connected components smaller than 5 pixels.
count, labels, stats, _ = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

filtered = np.zeros_like(binary)

for component_id in range(1, count):
    area = stats[component_id, cv2.CC_STAT_AREA]
    if area >= 5:
        filtered[labels == component_id] = 1

# Step 2: Apply gentle morphological closing.
# A 3x3 kernel tests whether very nearby fragments can be connected.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))


closed = cv2.morphologyEx(
    filtered,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)

# Save masks as 0/255 grayscale images.
filtered_path = OUTPUT_DIR / "tip_mask_min_area5.png"
closed_path = OUTPUT_DIR / "tip_mask_min_area5_closed3x3.png"
comparison_path = OUTPUT_DIR / "tip_mask_refinement_comparison.png"

cv2.imwrite(str(filtered_path), filtered * 255)
cv2.imwrite(str(closed_path), closed * 255)

# Produce a side-by-side visual comparison.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(binary, cmap="gray")
axes[0].set_title("Original candidate mask")

axes[1].imshow(filtered, cmap="gray")
axes[1].set_title("Minimum component area: 5 px")

axes[2].imshow(closed, cmap="gray")
axes[2].set_title("5 px filter + 3x3 closing")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig(comparison_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Original candidate pixels: {int(binary.sum())}")
print(f"After 5 px filter:         {int(filtered.sum())}")
print(f"After closing:             {int(closed.sum())}")
print(f"Saved comparison: {comparison_path}")
