
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "raw" / "Brittle" / "100Cr6_AR_05_11.png"
OUTPUT_DIR = ROOT / "data" / "processed" / "contour_refinement"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONTOUR_AREA = 5.0
LOCAL_THRESHOLD_FACTOR = 0.5
PADDING = 3
CLOSING_KERNEL_SIZE = 3
MIN_REGION_PIXELS = 20


# --------------------------------------------------
# 2. Helpers
# --------------------------------------------------
def make_candidate_mask(shape, contour, x0, y0):
    """Create a filled candidate mask in ROI coordinates."""
    mask = np.zeros(shape, dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)
    return mask


def contour_mask(shape, contour):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask


def overlap_ratio(contour_a, contour_b, shape):
    """Measure overlap relative to the smaller filled contour."""
    mask_a = contour_mask(shape, contour_a)
    mask_b = contour_mask(shape, contour_b)

    area_a = cv2.countNonZero(mask_a)
    area_b = cv2.countNonZero(mask_b)

    if min(area_a, area_b) == 0:
        return 0.0

    intersection = cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b))
    return intersection / min(area_a, area_b)


def to_full_coordinates(contour, x0, y0):
    result = contour.copy()
    result[:, 0, 0] += x0
    result[:, 0, 1] += y0
    return result


# --------------------------------------------------
# 3. Load image and detect initial contours
# --------------------------------------------------
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

blurred = cv2.GaussianBlur(image, (5, 5), 0)

global_threshold, initial_binary = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

initial_edges = cv2.Canny(initial_binary, 50, 150)

initial_contours, _ = cv2.findContours(
    initial_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
)

initial_contours = [
    c for c in initial_contours
    if cv2.contourArea(c) >= MIN_CONTOUR_AREA
]

if not initial_contours:
    raise RuntimeError("No initial contours were detected.")

areas = np.array(
    [cv2.contourArea(c) for c in initial_contours],
    dtype=float,
)

mean_area = float(areas.mean())
std_area = float(areas.std())
candidate_limit = mean_area + std_area

candidates = [
    c for c in initial_contours
    if cv2.contourArea(c) > candidate_limit
]

print(f"Image: {IMAGE_PATH.name}")
print(f"Dimensions: {image.shape[1]} x {image.shape[0]}")
print(f"Global Otsu threshold: {global_threshold:.2f}")
print(f"Initial contours: {len(initial_contours)}")
print(f"Mean contour area: {mean_area:.2f} px^2")
print(f"Contour-area standard deviation: {std_area:.2f} px^2")
print(f"Candidate area limit: {candidate_limit:.2f} px^2")
print(f"Candidate regions: {len(candidates)}")


# --------------------------------------------------
# 4. Prepare overlays and metrics
# --------------------------------------------------
initial_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(initial_overlay, initial_contours, -1, (0, 255, 0), 1)

refined_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
refined_mask = np.zeros_like(image)

accepted_contours = []
metrics = []
counts = {
    "accepted": 0,
    "too_small": 0,
    "duplicate": 0,
    "empty_candidate": 0,
}


# --------------------------------------------------
# 5. Refine each candidate region
# --------------------------------------------------
for region_id, candidate in enumerate(candidates, start=1):
    candidate_area = float(cv2.contourArea(candidate))
    x, y, w, h = cv2.boundingRect(candidate)

    x0 = max(0, x - PADDING)
    y0 = max(0, y - PADDING)
    x1 = min(image.shape[1], x + w + PADDING)
    y1 = min(image.shape[0], y + h + PADDING)

    roi = image[y0:y1, x0:x1]
    candidate_mask = make_candidate_mask(
        roi.shape, candidate, x0, y0
    )

    pixels = roi[candidate_mask > 0]

    if pixels.size == 0:
        counts["empty_candidate"] += 1
        continue

    local_mean = float(pixels.mean())
    local_std = float(pixels.std())

    threshold = float(np.clip(
        local_mean + LOCAL_THRESHOLD_FACTOR * local_std,
        0, 255
    ))

    # Threshold only inside the original candidate region.
    binary = np.zeros_like(roi, dtype=np.uint8)
    selected = (roi >= threshold) & (candidate_mask > 0)
    binary[selected] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE),
    )

    linked = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    # Enforce the original candidate-region constraint after closing.
    linked[candidate_mask == 0] = 0

    local_contours, _ = cv2.findContours(
        linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Save the local threshold result for diagnostic inspection.
    cv2.imwrite(
        str(OUTPUT_DIR / f"local_mask_{region_id:03d}.png"),
        linked,
    )

    for local_contour in local_contours:
        area = float(cv2.contourArea(local_contour))
        full = to_full_coordinates(local_contour, x0, y0)
        bx, by, bw, bh = cv2.boundingRect(full)

        reason = "accepted"

        if area < MIN_CONTOUR_AREA:
            reason = "below_minimum_contour_area"
            counts["too_small"] += 1

        elif cv2.countNonZero(contour_mask(roi.shape, local_contour)) < MIN_REGION_PIXELS:
            reason = "below_minimum_region_pixels"
            counts["too_small"] += 1

        else:
            # Only compare with accepted contours from the same candidate.
            # Separate candidates may overlap because initial contours can
            # be nested; don't automatically treat those as duplicates.
            for existing in accepted_contours:
                if existing["region_id"] != region_id:
                    continue

                ratio = overlap_ratio(
                    full, existing["contour"], image.shape
                )

                if ratio >= 0.85:
                    reason = "duplicate_overlap"
                    counts["duplicate"] += 1
                    break

        if reason == "accepted":
            accepted_contours.append({
                "region_id": region_id,
                "contour": full,
            })
            cv2.drawContours(
                refined_overlay, [full], -1, (0, 0, 255), 1
            )
            cv2.drawContours(
                refined_mask, [full], -1, 255, cv2.FILLED
            )
            counts["accepted"] += 1

        metrics.append({
            "candidate_region": region_id,
            "candidate_area": candidate_area,
            "local_mean_intensity": local_mean,
            "local_std_intensity": local_std,
            "local_threshold": threshold,
            "refined_contour_area": area,
            "bounding_box_x": bx,
            "bounding_box_y": by,
            "bounding_box_width": bw,
            "bounding_box_height": bh,
            "status": reason,
        })


# --------------------------------------------------
# 6. Save outputs and metrics
# --------------------------------------------------
cv2.imwrite(str(OUTPUT_DIR / "initial_contours.png"), initial_overlay)
cv2.imwrite(str(OUTPUT_DIR / "refined_contours.png"), refined_overlay)
cv2.imwrite(str(OUTPUT_DIR / "refined_regions_mask.png"), refined_mask)

fields = [
    "candidate_region",
    "candidate_area",
    "local_mean_intensity",
    "local_std_intensity",
    "local_threshold",
    "refined_contour_area",
    "bounding_box_x",
    "bounding_box_y",
    "bounding_box_width",
    "bounding_box_height",
    "status",
]

with open(
    OUTPUT_DIR / "refinement_metrics.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(metrics)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original SEM image")

axes[1].imshow(cv2.cvtColor(initial_overlay, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Initial contours ({len(initial_contours)})")

axes[2].imshow(cv2.cvtColor(refined_overlay, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Accepted refined contours ({counts['accepted']})")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
comparison_path = OUTPUT_DIR / "contour_comparison.png"
fig.savefig(comparison_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nRefinement summary")
print(f"Accepted contours: {counts['accepted']}")
print(f"Rejected as too small: {counts['too_small']}")
print(f"Rejected as duplicates: {counts['duplicate']}")
print(f"Candidates with no usable pixels: {counts['empty_candidate']}")
print(f"All contour records: {len(metrics)}")
print(f"Saved comparison: {comparison_path}")
print(f"Saved metrics: {OUTPUT_DIR / 'refinement_metrics.csv'}")
