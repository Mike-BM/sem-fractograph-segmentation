
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "raw" / "Brittle" / "100Cr6_AR_05_11.png"
MASK_PATH = ROOT / "data" / "processed" / "paper_void_tips" / "void_tip_mask.png"
OUTPUT_DIR = ROOT / "data" / "processed" / "paper_void_tips"
OUTPUT_PATH = OUTPUT_DIR / "tip_filter_comparison.png"

# Load the original SEM image and candidate-tip mask
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Cannot read SEM image: {IMAGE_PATH}")

if mask is None:
    raise FileNotFoundError(
        f"Cannot read tip mask: {MASK_PATH}\n"
        "Check the mask filename in data/processed/paper_void_tips/."
    )

# Convert the mask to binary: white = candidate tip pixels
binary_mask = (mask > 0).astype(np.uint8)

if image.shape != binary_mask.shape:
    raise ValueError(
        f"Image and mask dimensions differ: "
        f"{image.shape} versus {binary_mask.shape}"
    )

# Find connected components in the original candidate mask
count, labels, stats, _ = cv2.connectedComponentsWithStats(
    binary_mask, connectivity=8
)

areas = stats[1:, cv2.CC_STAT_AREA]
minimum_areas = [1, 2, 5, 10, 20, 50]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot original image plus filters at different minimum areas
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original SEM image")

for ax, minimum_area in zip(axes[1:], minimum_areas):
    keep = np.zeros_like(binary_mask, dtype=np.uint8)

    for component_id in range(1, count):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area >= minimum_area:
            keep[labels == component_id] = 1

    retained_pixels = int(keep.sum())
    retained_components = int(np.sum(areas >= minimum_area))
    percentage = 100 * retained_pixels / binary_mask.size

    ax.imshow(keep, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"Min area {minimum_area} px\n"
        f"{retained_components} components, "
        f"{percentage:.2f}% pixels"
    )

for ax in axes:
    ax.axis("off")

fig.suptitle(
    "Candidate-tip mask: connected-component filter comparison",
    fontsize=14
)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Original SEM image: {image.shape[1]} x {image.shape[0]}")
print(f"Original candidate pixels: {int(binary_mask.sum())}")
print("\nFilter comparison:")

for minimum_area in minimum_areas:
    kept = areas >= minimum_area
    pixels = int(areas[kept].sum())
    components = int(kept.sum())
    percentage = 100 * pixels / binary_mask.size

    print(
        f"Minimum area {minimum_area:>2} px: "
        f"{components:>4} components, "
        f"{pixels:>6} pixels ({percentage:.2f}%)"
    )

print(f"\nSaved comparison image: {OUTPUT_PATH}")

